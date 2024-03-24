import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from Sde import marginal_prob_std, diffusion_coeff
from Network import ScoreNet
import functools
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont
from ode_sampler import ode_sampler
from ode_likelihood import ode_likelihood
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

batch_size = 32  # @param {'type':'integer'}
device = 'cuda'  # @param ['cuda', 'cpu'] {'type':'string'}
sigma = 25.0  # @param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)
## Load the pre-trained checkpoint from disk.
device = 'cuda'  # @param ['cuda', 'cpu'] {'type':'string'}
ckpt = torch.load('ckpt1_9.pth', map_location=device)
score_model.load_state_dict(ckpt)

dataset = MNIST('.', train=False, transform=transforms.ToTensor(), download=True)

# 分别选择标签为0的样本和其他标签的样本
label_0_indices = [i for i, (image, label) in enumerate(dataset) if label == 0]
other_label_indices = [i for i, (image, label) in enumerate(dataset) if label != 0]

# 创建两个Subset数据集
dataset_label_0 = Subset(dataset, label_0_indices)
dataset_other_labels = Subset(dataset, other_label_indices)

# 计算10%的数据集大小
subset_size = int(len(dataset_other_labels) * 0.10)

# 随机划分数据集，获取10%的子集
subset_dataset = random_split(dataset_other_labels, [subset_size, len(dataset_other_labels) - subset_size])[0]

# 创建数据加载器
data_loader_other_labels = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 在代码的开始处初始化一个空的DataFrame
df = pd.DataFrame(columns=['Label', 'Prior_LogP', 'Later_LogP', 'Prior_LogP1', 'Later_LogP1'])

# 创建保存图像的文件夹
save_dir = 'sample_images19_using1_z'
os.makedirs(save_dir, exist_ok=True)

tqdm_data = tqdm(data_loader_other_labels, desc="Processing")
k = 0
for x, batch_label in tqdm_data:
    x = x.to(device)
    batch_label = batch_label.to(device)
    k = k + 1
    for i in range(x.size(0)):
        x_sample = x[i:i + 1]  # 选择一个样本
        x_ode_sample = ode_sampler(score_model,
                                   marginal_prob_std_fn,
                                   diffusion_coeff_fn,
                                   batch_size=1,
                                   device=device, x_input=x_sample)
        x_sample = (x_sample * 255. + torch.rand_like(x_sample)) / 256
        x_ode_sample = (x_ode_sample * 255. + torch.rand_like(x_ode_sample)) / 256
        z, prior_logp, later_lopg = ode_likelihood(x_sample, score_model, marginal_prob_std_fn,
                                                   diffusion_coeff_fn,
                                                   x_sample.shape[0], device=device, eps=1e-5)
        _, prior_logp1, later_lopg1 = ode_likelihood(x_ode_sample, score_model, marginal_prob_std_fn,
                                                     diffusion_coeff_fn,
                                                     x_sample.shape[0], device=device, eps=1e-5)
        label = batch_label[i].cpu().tolist()

        # 在DataFrame中添加两个不同的值，一个用于Prior_LogP，一个用于Later_LogP
        df.loc[len(df)] = [label, prior_logp.item(), later_lopg.item(), prior_logp1.item(), later_lopg1.item()]

        z_ode_sample = z.cpu().numpy()
        # 计算直方图
        hist, bins = np.histogram(z_ode_sample, bins=100, density=True)

        # 计算直方图的中心值
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # 定义高斯分布函数
        def gaussian(z, amplitude, mean, stddev):
            return amplitude * np.exp(-((z - mean) / stddev) ** 2 / 2)

        valid_data = hist[np.isfinite(hist)]
        valid_bin_centers = bin_centers[np.isfinite(hist)]
        # 初始参数估计
        init_params = [1.0, np.mean(z_ode_sample), np.std(z_ode_sample)]

        # 拟合高斯分布
        try:
            params, params_covariance = curve_fit(gaussian, valid_bin_centers, valid_data, p0=init_params, maxfev=10000)
        except RuntimeError:
            # 拟合失败，可以在这里添加处理失败情况的代码，或者跳过这个样本的处理
            continue

        # 获取拟合的参数
        amplitude, mean, stddev = params

        # 将 x_ode_sample 展平为一维数组
        z_ode_sample = z_ode_sample.flatten()

        # 将 PyTorch 张量转换为 NumPy 数组
        z_numpy = z.squeeze().cpu().numpy()  # 去掉批次和通道维度，然后转换为 NumPy 数组

        # 将 NumPy 数组保存为图像
        z_ode_sample_image = Image.fromarray((z_numpy * 255).astype(np.uint8))
        # z_ode_sample_image.save(f'{save_dir}/z_ode_sample_image_{k}_{i}.png')

        # 绘制直方图和拟合曲线
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(x_sample[0, 0].cpu().numpy(), cmap='gray')
        plt.title('x_sample')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(x_ode_sample[0, 0].cpu().numpy(), cmap='gray')
        plt.title('x_ode_sample')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.hist(z_ode_sample, bins=100, density=True, label='Histogram', alpha=0.7)
        plt.plot(valid_bin_centers, gaussian(valid_bin_centers, amplitude, mean, stddev), 'r-', label='Fit')
        plt.xlabel('z_ode_sample')
        plt.ylabel('Density')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{save_dir}/combined_image_{k}_{i}.png')
        plt.close()

# 保存到Excel文件时，分别指定列名
df.to_excel('results19.xlsx', index=False, columns=['Label', 'Prior_LogP', 'Later_LogP', 'Prior_LogP1', 'Later_LogP1'])

print("Results saved")
