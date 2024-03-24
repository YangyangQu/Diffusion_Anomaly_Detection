import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h

    
class ScoreNet1(nn.Module):
    """A modified time-dependent score-based model with reduced layers and early downsampling."""
    
    def __init__(self, marginal_prob_std, channels=[32, 64], embed_dim=256):
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        # Decoding layers where the resolution increases
        self.tconv1 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense3 = Dense(embed_dim, channels[0])
        self.tgnorm1 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv2 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # Swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        embed = self.act(self.embed(t))
        
        # Encoding path
        h1 = self.conv1(x)
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        # Decoding path
        h = self.tconv1(h2)
        h += self.dense3(embed)
        h = self.tgnorm1(h)
        h = self.act(h)
        
        h = self.tconv2(torch.cat([h, h1], dim=1))

        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h



# class ScoreNetMLP(nn.Module):
#     def __init__(self, marginal_prob_std, input_dim=28*28, hidden_dim=256, output_dim=28*28):
#         super(ScoreNetMLP, self).__init__()
#         self.input_dim = input_dim
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)
#         self.act = nn.ReLU()

#         self.marginal_prob_std = marginal_prob_std

#     def forward(self, x, t):
#         x = x.view(x.size(0), -1)  # 将输入 x 展平

#         # 前向传播过程
#         x = self.act(self.fc1(x))
#         x = self.act(self.fc2(x))
#         x = self.fc3(x)
        
#         # 将输出形状调整为 [batch_size, 1, 28, 28]
#         x = x.view(-1, 1, 28, 28)
        
#         x = x / self.marginal_prob_std(t)[:, None, None, None]  # 标准化输出
#         return x

    

# 定义一个基本的 ResNet 块，包含一个卷积层和标准化层
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class ShallowResNet(nn.Module):
    def __init__(self, marginal_prob_std, input_channels=1, num_blocks=3, num_channels=32, output_dim=28*28):
        super(ShallowResNet, self).__init__()
        self.input_channels = input_channels
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.output_dim = output_dim
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 构建基本块
        self.blocks = []
        for _ in range(num_blocks):
            block = BasicBlock(num_channels, num_channels)
            self.blocks.append(block)
        
        # 将基本块列表转化为 nn.ModuleList
        self.blocks = nn.ModuleList(self.blocks)
        
        # 新增卷积层，将输出通道数设置为1
        self.conv2 = nn.Conv2d(num_channels, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # 初始卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 基本块
        for block in self.blocks:
            x = block(x)
        
        # 新增的卷积层
        x = self.conv2(x)
        
        x = x / self.marginal_prob_std(t)[:, None, None, None]  # 标准化输出
        return x


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, use_batch_norm=True, dropout_prob=0.0):
        super(MLPBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.use_batch_norm = use_batch_norm
        self.use_dropout = dropout_prob > 0.0

        # # Batch Normalization layer
        # if self.use_batch_norm:
        #     self.batch_norm = nn.BatchNorm1d(output_dim)
        # # Batch Normalization layer

        if self.use_batch_norm:
            # 在训练期间将 track_running_stats 设置为 True
            self.batch_norm = nn.BatchNorm1d(output_dim, track_running_stats=True)

        # if self.use_batch_norm:
        #     self.batch_norm = nn.BatchNorm1d(output_dim, track_running_stats=False)  # Added track_running_stats=False
        # Dropout layer
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_prob)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        if self.training and x.size(0) > 1:  # 检查样本数是否大于 1
            x = self.batch_norm(x)
        x = self.act(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class ScoreNetMLP(nn.Module):
    def __init__(self, marginal_prob_std, input_dim, hidden_dims, output_dim, embed_dim=256, use_batch_norm=True,
                 dropout_prob=0.0):
        super(ScoreNetMLP, self).__init__()

        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        # 构建编码器路径
        encoder_layers = [MLPBlock(input_dim + embed_dim, hidden_dims[0], use_batch_norm, dropout_prob)]
        for i in range(1, len(hidden_dims)):
            encoder_layers.append(MLPBlock(hidden_dims[i - 1], hidden_dims[i], use_batch_norm, dropout_prob))
        self.encoder = nn.Sequential(*encoder_layers)

        # 构建解码器路径
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(MLPBlock(hidden_dims[i], hidden_dims[i - 1], use_batch_norm, dropout_prob))
        decoder_layers.append(MLPBlock(hidden_dims[0], output_dim, use_batch_norm, dropout_prob))
        self.decoder = nn.Sequential(*decoder_layers)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        embed = self.act(self.embed(t))
        x_with_time = torch.cat([x, embed], dim=-1)
        encoded = self.encoder(x_with_time)
        decoded = self.decoder(encoded)
        decoded = decoded / self.marginal_prob_std(t)[:, None, None, None]
        return decoded


