from torchvision.utils import make_grid
import torch
import numpy as np
import functools
from ode_sampler import ode_sampler
from Sde import marginal_prob_std, diffusion_coeff
from Network import ScoreNet
import matplotlib.pyplot as plt

device = 'cuda'  # @param ['cuda', 'cpu'] {'type':'string'}
sigma = 25.0  # @param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)
## Load the pre-trained checkpoint from disk.
device = 'cuda'  # @param ['cuda', 'cpu'] {'type':'string'}
ckpt = torch.load('model_n100_lr0.0001_sigma35.0.pth', map_location=device)
score_model.load_state_dict(ckpt)

sample_batch_size = 64  # @param {'type':'integer'}
sampler = ode_sampler  # @param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

## Generate samples using the specified sampler.
samples = sampler(score_model,
                  marginal_prob_std_fn,
                  diffusion_coeff_fn,
                  sample_batch_size,
                  device=device)

## Sample visualization.
# samples = samples.clamp(0.0, 1.0)

sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.savefig('unet_100_sigma35.0.png')
plt.show()
