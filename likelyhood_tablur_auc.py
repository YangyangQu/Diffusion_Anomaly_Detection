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
from sklearn.metrics import roc_auc_score

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


dataOriginal = OutlierSampler('mnist.npz')  # Assuming you have a dataset loader
X_train, X_test_true,  X_test_false = dataOriginal.X_train, dataOriginal.data_test_true, dataOriginal.data_test_false
dataset_true = torch.Tensor(X_test_true).float()
test_dataloader = torch.utils.data.DataLoader(dataset_true, batch_size=64, shuffle=True)


tqdm_data = tqdm(test_dataloader, desc="Processing")
k = 0
prior_logp_val = []
later_lopg_val = []
for x in tqdm_data:
    x = x.to(device)
    k = k + 1
    for i in range(x.size(0)):
        x_sample = x[i:i + 1]  # 选择一个样本        
        z, prior_logp, later_lopg = ode_likelihood(x_sample, score_model, marginal_prob_std_fn,
                                                   diffusion_coeff_fn,
                                                   x_sample.shape[0], device=device, eps=1e-5)
        # 在DataFrame中添加两个不同的值，一个用于Prior_LogP，一个用于Later_LogP
        prior_logp_val.append(prior_logp.item())
        later_lopg_val.append(later_lopg.item())

# 计算AUC
auc_score = roc_auc_score(prior_logp_val, later_lopg_val)

print("AUC Score:", auc_score)

