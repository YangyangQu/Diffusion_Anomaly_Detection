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
from ode_sampler import ode_sampler_tab
from ode_likelihood import ode_likelihood
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from dataGen import OutlierSampler
from Network import ScoreNetMLP

batch_size = 32  # @param {'type':'integer'}
device = 'cuda'  # @param ['cuda', 'cpu'] {'type':'string'}
sigma = 25.0  # @param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

score_model = torch.nn.DataParallel(ScoreNetMLP(
        marginal_prob_std = marginal_prob_std_fn,
        input_dim = 100,
        hidden_dims = [512, 256, 128],
        output_dim = 100,
        embed_dim = 256
    )).to(device)
score_model = score_model.to(device)
ckpt = torch.load('ckpt0_big_mlp.pth', map_location=device)
score_model.load_state_dict(ckpt)


dataOriginal = OutlierSampler('mammography.npz')  # Assuming you have a dataset loader
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