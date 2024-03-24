import torch
import functools
from torch.optim import Adam
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from Sde import marginal_prob_std, diffusion_coeff
from Network import ScoreNetMLP
from loss import loss_fn
import argparse
from dataGen import OutlierSampler

def main(args):
    device = args.device
    sigma = args.sigma
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    checkpoint_name = args.checkpoint_name
    patience = args.patience  # 添加一个耐心参数，表示在多少个连续的 epoch 内没有性能改进时停止训练
    best_validation_loss = float('inf')  # 用于存储验证集上的最佳损失值
    no_improvement_count = 0  # 记录连续没有性能改进的 epoch 数量

    # Define the marginal_prob_std and diffusion_coeff functions with the specified sigma
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    # Create the score model and move it to the specified device
    score_model = torch.nn.DataParallel(ScoreNetMLP(
        marginal_prob_std = marginal_prob_std_fn,
        input_dim = args.input_dim,
        hidden_dims = args.hidden_dims,
        output_dim = args.output_dim,
        embed_dim = args.embed_dim
    )).to(device)
    
    optimizer_mlp = Adam(score_model.parameters(), lr=args.lr)

    dataOriginal = OutlierSampler(args.data_file)  # Assuming you have a dataset loader
    X_train, X_test = dataOriginal.X_train, dataOriginal.data_label_test
    dataset = torch.Tensor(X_train).float()
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Training loop
    tqdm_epoch = tqdm(range(n_epochs))
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x in train_dataloader:
            x = x.to(device)
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer_mlp.zero_grad()
            loss.backward()
            optimizer_mlp.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        # 打印平均训练损失和验证损失
        tqdm_epoch.set_description(f'Avg Train Loss: {avg_loss / num_items:.5f}')

    # Save the trained model checkpoint
    torch.save(score_model.state_dict(), checkpoint_name)

def save_checkpoint(model, checkpoint_name):
    torch.save(model.state_dict(), checkpoint_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with hyperparameters')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device for training (cuda or cpu)')
    parser.add_argument('--sigma', type=float, default=25.0,
                        help='Sigma hyperparameter')
    parser.add_argument('--n_epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint_name', type=str, default='ckpt0_big_mlp.pth',
                        help='Name of the checkpoint file to save')
    parser.add_argument('--patience', type=int, default=100,
                        help='Number of consecutive epochs with no improvement to trigger early stopping')
    
    parser.add_argument('--input_dim', type=int, default=100,
                        help='Input dimension')
    parser.add_argument('--hidden_dims', type=list, default=[512, 256, 128],
                        help='List of hidden layer dimensions')
    parser.add_argument('--output_dim', type=int, default=100,
                        help='Output dimension')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension for GaussianFourierProjection')
    parser.add_argument('--data_file', type=str, default='mnist.npz',
                        help='Name of the data file')

    args = parser.parse_args()
    main(args)
