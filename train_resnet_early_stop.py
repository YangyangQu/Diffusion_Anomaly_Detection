import torch
import functools
from torch.optim import Adam
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from Sde import marginal_prob_std, diffusion_coeff
from Network import ShallowResNet
from loss import loss_fn
import argparse

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
    score_model = torch.nn.DataParallel(ShallowResNet(marginal_prob_std=marginal_prob_std_fn)).to(device)

    # Load the MNIST dataset and select specific classes
    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    selected_classes = [1,2,3,4,5,6,7,8,9]
    selected_indices = [i for i in range(len(dataset)) if dataset.targets[i] in selected_classes]
    selected_dataset = Subset(dataset, selected_indices)
    data_loader = DataLoader(selected_dataset, batch_size=batch_size, shuffle=True)

    # Define the optimizer
    optimizer = Adam(score_model.parameters(), lr=lr)

    # Training loop
    tqdm_epoch = tqdm(range(n_epochs))
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = x.to(device)
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        # 在每个 epoch 结束后计算验证集上的损失
        validation_loss = validate_model(score_model, data_loader, device, marginal_prob_std_fn)
        
        # # 如果验证损失小于当前最佳损失，更新最佳损失和保存模型
        # if validation_loss < best_validation_loss:
        #     best_validation_loss = validation_loss
        #     save_checkpoint(score_model, checkpoint_name)
        #     no_improvement_count = 0
        # else:
        #     no_improvement_count += 1
        #
        # # 如果连续 patience 个 epoch 都没有性能改进，提前停止训练
        # if no_improvement_count >= patience:
        #     print(f"Early stopping: No improvement for {patience} epochs.")
        #     break

        # 打印平均训练损失和验证损失
        tqdm_epoch.set_description(f'Avg Train Loss: {avg_loss / num_items:.5f}, Validation Loss: {validation_loss:.5f}')

    # Save the trained model checkpoint
    torch.save(score_model.state_dict(), checkpoint_name)

def validate_model(model, data_loader, device, marginal_prob_std_fn):
    model.eval()
    total_loss = 0.
    num_items = 0

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            loss = loss_fn(model, x, marginal_prob_std_fn)
            total_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

    model.train()
    return total_loss / num_items

def save_checkpoint(model, checkpoint_name):
    torch.save(model.state_dict(), checkpoint_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with hyperparameters')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device for training (cuda or cpu)')
    parser.add_argument('--sigma', type=float, default=25.0,
                        help='Sigma hyperparameter')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint_name', type=str, default='ckpt0.pth',
                        help='Name of the checkpoint file to save')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of consecutive epochs with no improvement to trigger early stopping')

    args = parser.parse_args()
    main(args)
