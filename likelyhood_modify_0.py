import argparse
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from Sde import marginal_prob_std, diffusion_coeff
from Network import ScoreNet
import functools
from torch.utils.data import DataLoader, Subset, random_split
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont
from ode_sampler import ode_sampler
from ode_likelihood import ode_likelihood
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='ODE Solver with Hyperparameters')

# 添加超参数选项
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Device (cuda or cpu)')
parser.add_argument('--sigma', type=float, default=25.0, help='Value for sigma')
parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--ckpt_path', type=str, default='model_n400_lr0.0001_sigma25.0.pth', help='Path to the checkpoint file')

# 解析命令行参数
args = parser.parse_args()

# 使用命令行参数来设置超参数
batch_size = args.batch_size
device = args.device
sigma = args.sigma
n_epochs = args.n_epochs
lr = args.lr
ckpt_path = args.ckpt_path

# 创建目录以保存结果
results_dir = f'results_sigma{sigma}_epochs{n_epochs}_lr{lr}'
os.makedirs(results_dir, exist_ok=True)

# 加载MNIST数据集
dataset = MNIST('.', train=False, transform=transforms.ToTensor(), download=True)

# 分别选择标签为0的样本和其他标签的样本
label_0_indices = [i for i, (image, label) in enumerate(dataset) if label == 0]
other_label_indices = [i for i, (image, label) in enumerate(dataset) if label != 0]

# 创建两个Subset数据集
dataset_label_0 = Subset(dataset, label_0_indices)
dataset_other_labels = Subset(dataset, other_label_indices)

# 计算10%的数据集大小
subset_size = int(len(dataset_label_0) * 0.5)
# 随机划分数据集，获取10%的子集
subset_dataset0 = random_split(dataset_label_0, [subset_size, len(dataset_label_0) - subset_size])[0]

# 创建数据加载器
dataset_label_0 = DataLoader(subset_dataset0, batch_size=batch_size, shuffle=True, num_workers=4)
dataset_label_others = DataLoader(subset_dataset0, batch_size=batch_size, shuffle=True, num_workers=4)
# 在代码的开始处初始化一个空的DataFrame
df = pd.DataFrame(columns=['Label', 'Prior_LogP', 'Later_LogP', 'Prior_LogP1', 'Later_LogP1'])

# 创建保存图像的文件夹
save_dir = f'{results_dir}/sample_images0_sigma{sigma}_epochs{n_epochs}_lr{lr}'
os.makedirs(save_dir, exist_ok=True)

# 加载预训练模型
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=functools.partial(marginal_prob_std, sigma=sigma)))
score_model = score_model.to(device)
ckpt = torch.load(ckpt_path, map_location=device)
score_model.load_state_dict(ckpt)

# 数据处理和模型推断部分
tqdm_data = tqdm(dataset_label_0, desc="Processing")
k = 0
for x, batch_label in tqdm_data:
    x = x.to(device)
    batch_label = batch_label.to(device)
    k = k + 1
    for i in range(x.size(0)):
        x_sample = x[i:i + 1]  # 选择一个样本
        z_x_sample, prior_logp, later_lopg = ode_likelihood(x_sample, score_model, functools.partial(marginal_prob_std, sigma=sigma),
                                                   functools.partial(diffusion_coeff, sigma=sigma),
                                                   x_sample.shape[0], device=device, eps=1e-5)
        # z_tmp = torch.randn(1, 1, 28, 28, device=device)
        # z_x_sample = z_x_sample* 0.70 + z_tmp* 0.3
        x_ode_sample = ode_sampler(score_model,
                                   functools.partial(marginal_prob_std, sigma=sigma),
                                   functools.partial(diffusion_coeff, sigma=sigma),
                                   batch_size=1,
                                   device=device, z=z_x_sample)
        _, prior_logp1, later_lopg1 = ode_likelihood(x_ode_sample, score_model, functools.partial(marginal_prob_std, sigma=sigma),
                                                     functools.partial(diffusion_coeff, sigma=sigma),
                                                     x_sample.shape[0], device=device, eps=1e-5)
        label = batch_label[i].cpu().tolist()

        # 在DataFrame中添加两个不同的值，一个用于Prior_LogP，一个用于Later_LogP
        df.loc[len(df)] = [label, prior_logp.item(), later_lopg.item(), prior_logp1.item(), later_lopg1.item()]

        z_ode_sample = z_x_sample.cpu().numpy()
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
        z_numpy = z_x_sample.squeeze().cpu().numpy()  # 去掉批次和通道维度，然后转换为 NumPy 数组

#         # 将 NumPy 数组保存为图像
#         z_ode_sample_image = Image.fromarray((z_numpy * 255).astype(np.uint8))
#         # z_ode_sample_image.save(f'{save_dir}/z_ode_sample_image_{k}_{i}.png')
#         true_gauss_noise = torch.randn(z_ode_sample_image.size)
        
#         z_ode_sample_array = np.array(z_ode_sample_image)
#         diffusion_mean = np.mean(z_ode_sample_array)
#         diffusion_std = np.std(z_ode_sample_array)

#         true_gauss_mean = true_gauss_noise.mean()
#         true_gauss_std = true_gauss_noise.std()
#         psnr_z_ode_sample = compute_psnr(np.array(z_ode_sample_image), true_gauss_noise.cpu().numpy())
#         print(f"--------------------------PSNR (z_ode_sample_image vs. true_gauss_noise): {psnr_z_ode_sample:.2f} dB")

#         _, p_value = stats.shapiro(z_ode_sample_image)
#         if p_value > 0.05:
#             print("--------------------------Data appears to be normally distributed (p-value =", p_value, ")")
#         else:
#             print("--------------------------Data does not appear to be normally distributed (p-value =", p_value, ")")


        # 来自标准正态分布
        # 绘制直方图和拟合曲线
        plt.figure(figsize=(5, 2))
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
results_filename = f'{results_dir}/results_0_sigma{sigma}_epochs{n_epochs}_lr{lr}.xlsx'
df.to_excel(results_filename, index=False, columns=['Label', 'Prior_LogP', 'Later_LogP', 'Prior_LogP1', 'Later_LogP1'])

print(f"Results saved in {results_filename}")
