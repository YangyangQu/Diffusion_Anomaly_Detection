import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from dataGen import OutlierSampler
from Network import ScoreNetMLP
from torch.utils.data import DataLoader
import functools
import numpy as np
from Sde import marginal_prob_std, diffusion_coeff
from ode_likelihood import ode_likelihood
import os
import re
from multiprocessing import Pool, cpu_count

def process_data(args):
    x, score_model, marginal_prob_std_fn, diffusion_coeff_fn, device = args
    x = x.to(device)
    prior_logp_val = []
    later_lopg_val = []
    for i in range(x.size(0)):
        x_sample = x[i:i + 1]
        z, prior_logp, later_lopg = ode_likelihood(x_sample, score_model, marginal_prob_std_fn,
                                                   diffusion_coeff_fn,
                                                   x_sample.shape[0], device=device, eps=1e-5)
        prior_logp_val.append(prior_logp.item())
        later_lopg_val.append(later_lopg.item())
    return prior_logp_val, later_lopg_val

def calculate_auc_parallel(score_model, marginal_prob_std_fn, diffusion_coeff_fn):
    device = 'cuda'  # @param ['cuda', 'cpu'] {'type':'string'}
    dataOriginal = OutlierSampler('mammography.npz')  # Assuming you have a dataset loader
    X_train, X_test_true, X_test_false = dataOriginal.X_train, dataOriginal.data_test_true, dataOriginal.data_test_false

    dataset_true = torch.Tensor(X_test_true).float()
    test_true_dataloader = torch.utils.data.DataLoader(dataset_true, batch_size=64, shuffle=True)

    dataset_false = torch.Tensor(X_test_false).float()
    test_false_dataloader = torch.utils.data.DataLoader(dataset_false, batch_size=64, shuffle=True)

    # 并行处理 true 数据集
    with Pool(cpu_count()) as pool:
        tqdm_true_data = tqdm(pool.imap(process_data, [(x, score_model, marginal_prob_std_fn, diffusion_coeff_fn, device) for x in test_true_dataloader]), desc="Processing True Data")
        prior_logp_val_true, later_lopg_val_true = zip(*tqdm_true_data)

    # 并行处理 false 数据集
    with Pool(cpu_count()) as pool:
        tqdm_false_data = tqdm(pool.imap(process_data, [(x, score_model, marginal_prob_std_fn, diffusion_coeff_fn, device) for x in test_false_dataloader]), desc="Processing False Data")
        prior_logp_val_false, later_lopg_val_false = zip(*tqdm_false_data)

    # Flatten 结果
    prior_logp_val_true = [item for sublist in prior_logp_val_true for item in sublist]
    later_lopg_val_true = [item for sublist in later_lopg_val_true for item in sublist]
    prior_logp_val_false = [item for sublist in prior_logp_val_false for item in sublist]
    later_lopg_val_false = [item for sublist in later_lopg_val_false for item in sublist]

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

    return auc_score

# 主循环
def main():

    model_folder = 'mm_model_list'
    model_files = os.listdir(model_folder)

    # 循环加载并运行模型
    for model_file in model_files:
        model_path = os.path.join(model_folder, model_file)
        # 定义正则表达式模式
        pattern = re.compile(
            r'.*_h(\d+)_(\d+)_(\d+)_embed(\d+)_dropout([\d.]+)_epochs(\d+)_lr([\d.]+)_sigma([\d.]+).pth')

        # 使用正则表达式匹配模式
        match = pattern.match(model_path)

        # 提取匹配到的参数
        hidden_dims = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
        embed_dim = int(match.group(4))
        dropout = float(match.group(5))
        sigma = float(match.group(8))
        device = 'cuda'  # @param ['cuda', 'cpu'] {'type':'string'}
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

        score_model = torch.nn.DataParallel(ScoreNetMLP(
            marginal_prob_std=marginal_prob_std_fn,
            input_dim=6,
            hidden_dims=hidden_dims,
            output_dim=6,
            embed_dim=embed_dim,
            use_batch_norm=True,
            dropout_prob=dropout  # Add dropout_prob parameter
        )).to(device)

        score_model = score_model.to(device)
        ckpt = torch.load(model_path, map_location=device)
        score_model.load_state_dict(ckpt)
        print("000000000000000000000")
        auc_score = calculate_auc_parallel(score_model, marginal_prob_std_fn, diffusion_coeff_fn)
        print(f'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMModel: {model_path}, AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUC: {auc_score}')

if __name__ == "__main__":
    main()



