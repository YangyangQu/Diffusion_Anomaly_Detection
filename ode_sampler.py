from scipy import integrate
import torch
import numpy as np
from ode_likelihood import ode_likelihood
import functools
from Sde import marginal_prob_std, diffusion_coeff

sigma = 25.0  # @param {'type':'number'}
error_tolerance = 1e-5  # @param {'type': 'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64,
                atol=error_tolerance,
                rtol=error_tolerance,
                device='cuda',
                z=None,
                eps=1e-5):
    t = torch.ones(batch_size, device=device)

    if z is None:
        z = torch.randn(batch_size, 1, 28, 28, device=device) \
                 * marginal_prob_std(t)[:, None, None, None]
    shape = z.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return -0.5 * (g ** 2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (1., eps), z.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol,
                              method='RK45')
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x


def ode_sampler_tab(score_model,
                marginal_prob_std,
                diffusion_coeff,
                data_shape,
                batch_size=64,
                atol=error_tolerance,
                rtol=error_tolerance,
                device='cuda',
                z=None,
                eps=1e-5):
    t = torch.ones(batch_size, device=device)

    if z is None:
        # 根据数据集的形状生成相应形状的随机 z
        z = torch.randn(batch_size, *data_shape, device=device) \
                 * marginal_prob_std(t)[:, None, None, None]

    shape = z.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return -0.5 * (g ** 2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (1., eps), z.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol,
                              method='RK45')
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x
