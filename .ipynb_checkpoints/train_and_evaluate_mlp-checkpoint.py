from torch.optim import Adam
from loss import loss_fn
import argparse
import torch
from tqdm import tqdm
from Sde import marginal_prob_std, diffusion_coeff
import functools
from ode_likelihood import ode_likelihood
from dataGen import OutlierSampler
from Network import ScoreNetMLP
from sklearn.metrics import roc_auc_score
import itertools
import os
import numpy as np

def train_and_evaluate(args, input_dim, hidden_dims, output_dim, embed_dim, dropout_prob, n_epochs, lr, sigma):
    device = args.device
    checkpoint_name = args.checkpoint_name
    # Define the marginal_prob_std and diffusion_coeff functions with the specified sigma
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
    # Create the score model and move it to the specified device
    score_model = torch.nn.DataParallel(ScoreNetMLP(
        marginal_prob_std=marginal_prob_std_fn,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        embed_dim=embed_dim,
        use_batch_norm=True,
        dropout_prob=dropout_prob  # Add dropout_prob parameter
    )).to(device)

    optimizer_mlp = Adam(score_model.parameters(), lr=lr)

    dataOriginal = OutlierSampler(args.data_file)
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

        tqdm_epoch.set_description(f'Avg Train Loss: {avg_loss / num_items:.5f}')

    # Save the trained model checkpoint

    # 保存模型的文件夹路径（相对于当前目录）
    checkpoint_folder = 'model_dic'  # 设置保存文件夹的路径

    # 确保文件夹存在，如果不存在则创建
    os.makedirs(checkpoint_folder, exist_ok=True)

    # 保存模型的文件名
    checkpoint_name = f'{checkpoint_folder}/{args.data_file}_h{"_".join(map(str, args.hidden_dims))}_embed{args.embed_dim}_dropout{args.dropout_prob}_epochs{args.n_epochs}_lr{args.lr}_sigma{args.sigma}.pth'

    torch.save(score_model.state_dict(), checkpoint_name)
    score_model = score_model.to(device)
    ckpt = torch.load(checkpoint_name, map_location=device)
    score_model.load_state_dict(ckpt)

    X_train, X_test_true, X_test_false = dataOriginal.X_train, dataOriginal.data_test_true, dataOriginal.data_test_false
    dataset_true = torch.Tensor(X_test_true).float()
    test_true_dataloader = torch.utils.data.DataLoader(dataset_true, batch_size=64, shuffle=True)

    tqdm_true_data = tqdm(test_true_dataloader, desc="Processing")
    prior_logp_val_true = []
    later_lopg_val_true = []
    for x in tqdm_true_data:
        x = x.to(device)
        for i in range(x.size(0)):
            x_sample = x[i:i + 1]
            z, prior_logp, later_lopg = ode_likelihood(x_sample, score_model, marginal_prob_std_fn,
                                                       diffusion_coeff_fn,
                                                       x_sample.shape[0], device=device, eps=1e-5)
            prior_logp_val_true.append(prior_logp.item())
            later_lopg_val_true.append(later_lopg.item())

    dataset_false = torch.Tensor(X_test_false).float()
    test_false_dataloader = torch.utils.data.DataLoader(dataset_false, batch_size=64, shuffle=True)

    tqdm_false_data = tqdm(test_false_dataloader, desc="Processing")
    prior_logp_val_false = []
    later_lopg_val_false = []
    for x in tqdm_false_data:
        x = x.to(device)
        for i in range(x.size(0)):
            x_sample = x[i:i + 1]
            z, prior_logp, later_lopg = ode_likelihood(x_sample, score_model, marginal_prob_std_fn,
                                                       diffusion_coeff_fn,
                                                       x_sample.shape[0], device=device, eps=1e-5)
            prior_logp_val_false.append(prior_logp.item())
            later_lopg_val_false.append(later_lopg.item())



    # 构造真实标签
    true_labels_true = np.ones_like(later_lopg_val_true)
    true_labels_false = np.zeros_like(later_lopg_val_false)

    # 构造模型预测概率或分数
    predicted_scores_true = later_lopg_val_true
    predicted_scores_false = later_lopg_val_false

    # 将两组数据合并
    true_labels = np.concatenate([true_labels_true, true_labels_false])
    predicted_scores = np.concatenate([predicted_scores_true, predicted_scores_false])

    # 计算总的 AUC 分数
    auc_score = roc_auc_score(true_labels, predicted_scores)

    # 返回总的 AUC 分数
    return auc_score


def main(args):
    # 定义超参数的候选值
    hidden_dims_values = [[256, 128, 64], [512, 256, 128], [1024, 512, 256, 128]]
    embed_dim_values = [128, 256, 512, 1024]
    # dropout_prob_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # Add dropout_prob_values
    dropout_prob_values = [0.3, 0.5]  # Add dropout_prob_values
    n_epochs_values = [200, 400]
    lr_values = [1e-4, 1e-5]
    sigma_values = [25.0, 30.0, 35.0, 40.0]
    # n_epochs_values = [100, 200, 300, 400]
    # lr_values = [1e-3, 1e-4, 1e-5]
    # sigma_values = [20.0, 25.0, 30.0, 35.0, 40.0]

    # 初始化最佳AUC分数和对应的超参数组合
    best_auc_score = float('-inf')
    best_params = None

    # 循环遍历每个超参数组合
    for hidden_dims, embed_dim, dropout_prob, n_epochs, lr, sigma in itertools.product(
            hidden_dims_values, embed_dim_values, dropout_prob_values,
            n_epochs_values, lr_values, sigma_values):
        # 修改args中的超参数
        args.hidden_dims = hidden_dims
        args.embed_dim = embed_dim
        args.dropout_prob = dropout_prob  # 将 dropout_prob 添加到 args
        args.n_epochs = n_epochs
        args.lr = lr
        args.sigma = sigma

        # 调用训练和评估函数
        auc_score = train_and_evaluate(args, args.input_dim, args.hidden_dims, args.output_dim,
                                       args.embed_dim, args.dropout_prob, args.n_epochs, args.lr,
                                       args.sigma)
        # 更新最佳结果
        print("AUC Score:", auc_score)
        if auc_score > best_auc_score:
            best_auc_score = auc_score
            best_params = {
                'hidden_dims': hidden_dims,
                'embed_dim': embed_dim,
                'dropout_prob': dropout_prob,
                'n_epochs': n_epochs,
                'lr': lr,
                'sigma': sigma
            }
    
    # 输出最佳超参数和AUC分数
    print("Best Parameters:", best_params)
    print("Best AUC Score:", best_auc_score)


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
    parser.add_argument('--dropout_prob', type=float, default=0.3,  # Add dropout_prob
                        help='Dropout probability')
    parser.add_argument('--data_file', type=str, default='mnist.npz',
                        help='Name of the data file')

    args = parser.parse_args()
    main(args)
