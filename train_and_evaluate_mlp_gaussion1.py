from torch.optim import Adam
from loss import loss_fn
import argparse
import torch
from tqdm import tqdm
from Sde import marginal_prob_std, diffusion_coeff
import functools
from ode_likelihood import ode_likelihood
from dataGen import OutlierSampler
from Network import ScoreNetMLP,ScoreNetMLP1
import matplotlib.pyplot as plt
from torch.nn import functional as F
from scipy.stats import gaussian_kde
import os
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

class GMM_sampler(object):
    def __init__(self, N, mean=None, n_components=None, cov=None, sd=None, dim=None, weights=None):
        np.random.seed(1024)
        self.total_size = N
        self.n_components = n_components
        self.dim = dim
        self.sd = sd
        self.weights = weights
        if mean is None:
            assert n_components is not None and dim is not None and sd is not None
            self.mean = np.random.uniform(-5,5,(self.n_components,self.dim))
        else:
            assert cov is not None
            self.mean = mean
            self.n_components = self.mean.shape[0]
            self.dim = self.mean.shape[1]
            self.cov = cov
        if weights is None:
            self.weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        self.Y = np.random.choice(self.n_components, size=N, replace=True, p=self.weights)
        if mean is None:
            self.X = np.array([np.random.normal(self.mean[i],scale=self.sd) for i in self.Y],dtype='float64')
        else:
            self.X = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        self.X_train, self.X_val,self.X_test = self.split(self.X)

    def split(self,data):
        #N_test = int(0.1*data.shape[0])
        N_test = 2000
        data_test = data[-N_test:]
        data = data[0:-N_test]
        #N_validate = int(0.1*data.shape[0])
        N_validate = 2000
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = len(self.X_train), size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]

    def load_all(self):
        return self.X, self.Y


# Swiss roll (r*sin(scale*r),r*cos(scale*r)) + Gaussian noise
class Swiss_roll_sampler(object):
    def __init__(self, N, theta=2 * np.pi, scale=2, sigma=0.4):
        np.random.seed(1024)
        self.total_size = N
        self.theta = theta
        self.scale = scale
        self.sigma = sigma
        params = np.linspace(0, self.theta, self.total_size)
        self.X_center = np.vstack((params * np.sin(scale * params), params * np.cos(scale * params)))
        self.X = self.X_center.T + np.random.normal(0, sigma, size=(self.total_size, 2))
        np.random.shuffle(self.X)
        self.X_train, self.X_val, self.X_test = self.split(self.X)
        self.Y = None
        self.mean = 0
        self.sd = 0

    def split(self, data):
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test

    def train(self, batch_size, label=False):
        indx = np.random.randint(low=0, high=self.total_size, size=batch_size)
        return self.X[indx, :]

    def get_density(self, x):
        assert len(x) == 2
        a = 1. / (np.sqrt(2 * np.pi) * self.sigma)
        de = 2 * self.sigma ** 2
        nu = -np.sum((np.tile(x, [self.total_size, 1]) - self.X_center.T) ** 2, axis=1)
        return np.mean(a * np.exp(nu / de))

    def load_all(self):
        return self.X, self.Y


# Gaussian mixture + normal + uniform distribution
class GMM_Uni_sampler(object):
    def __init__(self, N, mean, cov, norm_dim=2, uni_dim=10, weights=None):
        self.total_size = N
        self.mean = mean
        self.n_components = self.mean.shape[0]
        self.norm_dim = norm_dim
        self.uni_dim = uni_dim
        self.cov = cov
        np.random.seed(1024)
        if weights is None:
            weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        self.Y = np.random.choice(self.n_components, size=self.total_size, replace=True, p=weights)
        # self.X = np.array([np.random.normal(self.mean[i],scale=self.sd) for i in self.Y],dtype='float64')
        self.X_gmm = np.array([np.random.multivariate_normal(mean=self.mean[i], cov=self.cov[i]) for i in self.Y],
                              dtype='float64')
        self.X_normal = np.random.normal(0.5, np.sqrt(0.1), (self.total_size, self.norm_dim))
        self.X_uni = np.random.uniform(-0.5, 0.5, (self.total_size, self.uni_dim))
        self.X = np.concatenate([self.X_gmm, self.X_normal, self.X_uni], axis=1)

    def train(self, batch_size, label=False):
        indx = np.random.randint(low=0, high=self.total_size, size=batch_size)
        if label:
            return self.X[indx, :], self.Y[indx].flatten()
        else:
            return self.X[indx, :]

    def load_all(self):
        return self.X, self.Y
class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        device = 'cpu'
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise.to(device)

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        device = 'cpu'
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise.to(device)

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    def main(args):
        device = args.device
        sigma = args.sigma
        n_epochs = args.n_epochs
        batch_size = args.batch_size
        lr = args.lr
        embed_dim = args.embed_dim
        eval_batch_size = args.eval_batch_size
        checkpoint_name = args.checkpoint_name
        num_timesteps = args.num_timesteps
        beta_schedule = args.beta_schedule
        save_images_step= args.save_images_step
        experiment_name = args.experiment_name
        num_epochs = args.num_epochs
        # Define the marginal_prob_std and diffusion_coeff functions with the specified sigma
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
        # Create the score model and move it to the specified device

        score_model = ScoreNetMLP1(
            marginal_prob_std=marginal_prob_std_fn,
            embed_dim=embed_dim
            # Add dropout_prob parameter
        ).to(device)
        optimizer_mlp = Adam(score_model.parameters(), lr=lr)

        ys = GMM_indep_sampler(N=20000, sd=0.1, dim=2, n_components=3, bound=1)
        #ys = Swiss_roll_sampler(N=20000)
        X_train, X_val, X_test = ys.X_train, ys.X_val, ys.X_test
        dataset = torch.Tensor(X_train).float()
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        # 确保文件夹存在，如果不存在则创建
        # 保存模型的文件夹路径（相对于当前目录）
        checkpoint_folder = 'model_ode_guassion'  # 设置保存文件夹的路径
        os.makedirs(checkpoint_folder, exist_ok=True)

        outdir = f"exps/{experiment_name}"

        imgdir = f"{outdir}/images"
        os.makedirs(imgdir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)
        noise_scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule)
        frames = []
        losses = []
        # Training loop
        tqdm_epoch = tqdm(range(n_epochs))
        xmin, xmax = -10, 10
        ymin, ymax = -10, 10
        for epoch in tqdm_epoch:
            score_model.train()
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
            score_model.eval()
            sample = torch.randn(eval_batch_size, 2)
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(timesteps):
                score_model.eval()
                t = torch.from_numpy(np.repeat(t, eval_batch_size)).long()
                with torch.no_grad():
                    residual = score_model(sample, t)
                sample = noise_scheduler.step(residual, t[0], sample)
            frames.append(sample.numpy())

            # 保存图像（每隔一定步数保存）
            if epoch % save_images_step == 0 or epoch == num_epochs - 1:

                frames_np = np.stack(frames)
                xmin, xmax = -10, 10
                ymin, ymax = -10, 10
                #
                # # 获取测试集数据
                # frames_test = ys.X_test
                #
                # # 计算密度估计
                # density = gaussian_kde(frames_test.T)
                # z = density.evaluate(frames_test.T)
                # # 创建绘图对象
                # fig, ax = plt.subplots()
                #
                # # 绘制测试集数据点，根据密度设置颜色深浅
                # scatter = ax.scatter(frames_test[:, 0], frames_test[:, 1], c=z, cmap='cool', s=5)
                # # 设置颜色条
                # cbar = plt.colorbar(scatter)
                # cbar.set_label('Density')

                plt.figure(figsize=(10, 10))
                for i, frame in enumerate(frames_np):
                    plt.scatter(frame[:, 0], frame[:, 1])
                    plt.xlim(xmin, xmax)
                    plt.ylim(ymin, ymax)
                    if i % 10 == 0:
                        plt.savefig(f"{imgdir}/{i:04}.png")
                        plt.close()

            tqdm_epoch.set_description(f'Avg Train Loss: {avg_loss / num_items:.5f}')

        # Save the trained model checkpoint

        # 保存模型的文件名
        checkpoint_name = f'{checkpoint_folder}/{args.data_file}_h{"_".join(map(str, args.hidden_dims))}_embed{args.embed_dim}_dropout{args.dropout_prob}_epochs{args.n_epochs}_lr{args.lr}_sigma{args.sigma}.pth'

        torch.save(score_model.state_dict(), checkpoint_name)

        print("Saving loss as numpy array...")
        np.save(f"{outdir}/loss.npy", np.array(losses))

        # dataset = torch.Tensor(X_test).float()
        # test_true_dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        #
        # tqdm_true_data = tqdm(test_true_dataloader, desc="Processing")
        # prior_logp_val_true = []
        # later_lopg_val_true = []
        # for x in tqdm_true_data:
        #     x = x.to(device)
        #     for i in range(x.size(0)):
        #         x_sample = x[i:i + 1]
        #         z, prior_logp, later_lopg = ode_likelihood(x_sample, score_model, marginal_prob_std_fn,
        #                                                    diffusion_coeff_fn,
        #                                                    x_sample.shape[0], device=device, eps=1e-5)
        #         prior_logp_val_true.append(prior_logp.item())
        #         later_lopg_val_true.append(later_lopg.item())

            # dataset_false = torch.Tensor(X_test_false).float()
            # test_false_dataloader = torch.utils.data.DataLoader(dataset_false, batch_size=64, shuffle=True)
            #
            # tqdm_false_data = tqdm(test_false_dataloader, desc="Processing")
            # prior_logp_val_false = []
            # later_lopg_val_false = []
            # for x in tqdm_false_data:
            #     x = x.to(device)
            #     for i in range(x.size(0)):
            #         x_sample = x[i:i + 1]
            #         z, prior_logp, later_lopg = ode_likelihood(x_sample, score_model, marginal_prob_std_fn,
            #                                                    diffusion_coeff_fn,
            #                                                    x_sample.shape[0], device=device, eps=1e-5)
            #         prior_logp_val_false.append(prior_logp.item())
            #         later_lopg_val_false.append(later_lopg.item())
            #
            #
            # # 构造模型预测概率或分数
            # predicted_scores = np.concatenate([later_lopg_val_true, later_lopg_val_false])
            #
            # # 构造真实标签（二进制形式）
            # true_labels = np.hstack([np.ones_like(later_lopg_val_true), np.zeros_like(later_lopg_val_false)])
            #
            # # 计算总的 AUC 分数
            # auc_score = roc_auc_score(true_labels, predicted_scores)
            # # 返回总的 AUC 分数
            # return auc_score
            #

    #     def main(args):
    #         # 定义超参数的候选值
    #         hidden_dims_values = [[512, 256, 128], [1024, 512, 256, 128]]
    #         # embed_dim_values = [128, 256, 512]
    #         # dropout_prob_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # Add dropout_prob_values
    #         dropout_prob_values = [0.3, 0.5]  # Add dropout_prob_values
    #         n_epochs_values = [300, 400]
    #         lr_values = [1e-4, 1e-5]
    #         sigma_values = [25.0, 30.0, 35.0, 40.0]
    #         # n_epochs_values = [100, 200, 300, 400]
    #         # lr_values = [1e-3, 1e-4, 1e-5]
    #         # sigma_values = [20.0, 25.0, 30.0, 35.0, 40.0]
    #
    #         # 初始化最佳AUC分数和对应的超参数组合
    #         best_auc_score = float('-inf')
    #         best_params = None
    #
    #         # 循环遍历每个超参数组合
    #         for hidden_dims, dropout_prob, n_epochs, lr, sigma in itertools.product(
    #                 hidden_dims_values, dropout_prob_values,
    #                 n_epochs_values, lr_values, sigma_values):
    #             # 修改args中的超参数
    #             args.hidden_dims = hidden_dims
    #             args.dropout_prob = dropout_prob  # 将 dropout_prob 添加到 args
    #             args.n_epochs = n_epochs
    #             args.lr = lr
    #             args.sigma = sigma
    #
    #             # 调用训练和评估函数
    #             auc_score = train_and_evaluate(args, args.input_dim, args.hidden_dims, args.output_dim,
    #                                            args.embed_dim, args.dropout_prob, args.n_epochs, args.lr,
    #                                            args.sigma)
    #             # 更新最佳结果
    #             print("----------------------------AUC Score------------------------------:", auc_score)
    #             if auc_score > best_auc_score:
    #                 best_auc_score = auc_score
    #                 best_params = {
    #                     'hidden_dims': hidden_dims,
    #                     'dropout_prob': dropout_prob,
    #                     'n_epochs': n_epochs,
    #                     'lr': lr,
    #                     'sigma': sigma
    #                 }
    #
    #         # 输出最佳超参数和AUC分数
    #         print("Best Parameters:", best_params)
    #         print("Best AUC Score:", best_auc_score)
    # print(prof)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with hyperparameters')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device for training (cuda or cpu)')
    parser.add_argument('--sigma', type=float, default=25.0,
                        help='Sigma hyperparameter')
    parser.add_argument('--n_epochs', type=int, default=10000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint_name', type=str, default='ckpt0_big_mlp.pth',
                        help='Name of the checkpoint file to save')
    parser.add_argument('--patience', type=int, default=100,
                        help='Number of consecutive epochs with no improvement to trigger early stopping')
    parser.add_argument('--eval_batch_size', type=int, default=100)
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
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--save_images_step", type=int, default=1)
    parser.add_argument("--experiment_name", type=str, default="base_ode1")
    parser.add_argument("--num_epochs", type=int, default=10000)
    args = parser.parse_args()
    main(args)
