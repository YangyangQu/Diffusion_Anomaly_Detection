from torch.utils.data import DataLoader, Subset
import pandas as pd
from ode_likelihood import ode_likelihood
import torch
import functools
from torch.optim import Adam
from tqdm import tqdm
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
    save_tab_name = args.save_tab_name
    # Define the marginal_prob_std and diffusion_coeff functions with the specified sigma
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    # Create the score model and move it to the specified device
    score_model = torch.nn.DataParallel(ScoreNetMLP(
        marginal_prob_std=marginal_prob_std_fn,
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
        embed_dim=args.embed_dim
    )).to(device)

    score_model = score_model.to(device)
    ckpt = torch.load(checkpoint_name, map_location=device)
    score_model.load_state_dict(ckpt)

    dataOriginal = OutlierSampler(args.data_file)  # Assuming you have a dataset loader
    X_train, X_test_true, X_test_false = dataOriginal.X_train, dataOriginal.data_test_true, dataOriginal.data_test_false
    dataset_true = torch.Tensor(X_test_true).float()
    test_dataloader = torch.utils.data.DataLoader(dataset_true, batch_size=64, shuffle=True)

    # 在代码的开始处初始化一个空的DataFrame
    df = pd.DataFrame(columns=['Prior_LogP', 'Later_LogP'])

    tqdm_data = tqdm(test_dataloader, desc="Processing")
    k = 0
    for x in tqdm_data:
        x = x.to(device)
        k = k + 1
        for i in range(x.size(0)):
            x_sample = x[i:i + 1]  # 选择一个样本
            z, prior_logp, later_lopg = ode_likelihood(x_sample, score_model, marginal_prob_std_fn,
                                                       diffusion_coeff_fn,
                                                       x_sample.shape[0], device=device, eps=1e-5)
            # 在DataFrame中添加两个不同的值，一个用于Prior_LogP，一个用于Later_LogP
            df.loc[len(df)] = [prior_logp.item(), later_lopg.item()]

    # 保存到Excel文件时，分别指定列名
    df.to_excel(save_tab_name, index=False, columns=['Prior_LogP', 'Later_LogP'])

    print("Results saved")

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
    parser.add_argument('--save_tab_name', type=str, default='mamm_false_1109.xlsx',
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

b