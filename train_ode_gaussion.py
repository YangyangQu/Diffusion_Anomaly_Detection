import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
from Sde import marginal_prob_std, diffusion_coeff
from Network import ScoreNetMLP
from dataGen import OutlierSampler
import util
import argparse
from loss import loss_fn
from tqdm import tqdm
import numpy as np

class GMM_indep_sampler(object):
    def __init__(self, N, sd, dim, n_components, weights=None, bound=1):
        np.random.seed(1024)
        self.total_size = N
        self.dim = dim
        self.sd = sd
        self.n_components = n_components
        self.bound = bound
        self.centers = np.linspace(-bound, bound, n_components)
        self.X = np.vstack([self.generate_gmm() for _ in range(dim)]).T
        self.X_train, self.X_val, self.X_test = self.split(self.X)
        self.nb_train = self.X_train.shape[0]
        self.Y = None

    def generate_gmm(self, weights=None):
        if weights is None:
            weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        Y = np.random.choice(self.n_components, size=self.total_size, replace=True, p=weights)
        return np.array([np.random.normal(self.centers[i], self.sd) for i in Y], dtype='float64')

    def split(self, data):
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test

    def get_density(self, data):
        assert data.shape[1] == self.dim
        from scipy.stats import norm
        centers = np.linspace(-self.bound, self.bound, self.n_components)
        prob = []
        for i in range(self.dim):
            p_mat = np.zeros((self.n_components, len(data)))
            for j in range(len(data)):
                for k in range(self.n_components):
                    p_mat[k, j] = norm.pdf(data[j, i], loc=centers[k], scale=self.sd)
            prob.append(np.mean(p_mat, axis=0))
        prob = np.stack(prob)
        return np.prod(prob, axis=0)

    def train(self, batch_size):
        indx = np.random.randint(low=0, high=self.nb_train, size=batch_size)
        return self.X_train[indx, :]

    def load_all(self):
        return self.X, self.Y

def main(args):
    device = args.device
    checkpoint_name = args.checkpoint_name
    n_epochs = args.n_epochs
    # 根据数据集选择相应的超参数
    if args.dataset == "involute":
        sigma = args.sigma1
    elif args.dataset == "indep_gmm":
        sigma = args.sigma2
    elif args.dataset == "eight_octagon_gmm":
        sigma = args.sigma3
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # 定义 marginal_prob_std 和 diffusion_coeff 函数
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    # 创建 ScoreNetMLP 模型 并移动到指定的设备
    score_model = torch.nn.DataParallel(ScoreNetMLP(
        marginal_prob_std=marginal_prob_std_fn,
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
        embed_dim=args.embed_dim
    )).to(device)

    optimizer_mlp = Adam(score_model.parameters(), lr=args.lr)

    # 获取数据集信息
    ys = GMM_indep_sampler(N=20000, sd=0.1, dim=2, n_components=3, bound=1)
    X_train, X_val, X_test = ys.X_train, ys.X_val, ys.X_test

    dataset = torch.Tensor(X_train).float()
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    # 训练循环
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
    # 保存训练好的模型
    torch.save(score_model.state_dict(), checkpoint_name)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with hyperparameters')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device for training (cuda or cpu)')
    parser.add_argument('--n_epochs', type=int, default=300,
                        help='Number of training epochs')
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
    parser.add_argument('--dataset', type=str, required=True,
                        choices=["involute", "indep_gmm", "eight_octagon_gmm"],
                        help='Dataset name')
    parser.add_argument('--sigma1', type=float, default=25.0,
                        help='Sigma hyperparameter for dataset 1 (involute)')
    parser.add_argument('--sigma2', type=float, default=25.0,
                        help='Sigma hyperparameter for dataset 2 (indep_gmm)')
    parser.add_argument('--sigma3', type=float, default=25.0,
                        help='Sigma hyperparameter for dataset 3 (eight_octagon_gmm)')

    args = parser.parse_args()
    main(args)