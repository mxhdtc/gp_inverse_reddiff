# Adapted from RED-diff algos/adam_vvgp.py
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from .forward_model import GPPredictionModel
from .ddim import DDIM

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .score_estimator import ScoreEstimator
from ..utils.diffusion import Diffusion
 

class ReverseDiffusionModel:
    def __init__(self, model: ScoreEstimator, diffusion: Diffusion, cfg: DictConfig):
        self.model = model
        self.diffusion = diffusion
        self.cfg = cfg
    
    def __call__(self, xt, y, alpha_t):
        # Returns both the noise value (score function scaled) and the predicted x0.
        # alpha_t = self.diffusion.alpha(t).view(-1, 1)
        et = self.model(xt, y, alpha_t)
        x0_pred = (xt - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
        return et, x0_pred

class ADAM(DDIM):
    def __init__(self, model: ReverseDiffusionModel, forward_model: GPPredictionModel, cfg: DictConfig):
        self.model = model
        self.diffusion = model.diffusion
        self.forward_model = forward_model
        self.cfg = cfg
        self.awd = cfg.algo.awd
        self.batch_size = cfg.algo.batch_size
        self.grad_term_weight = cfg.algo.grad_term_weight
        self.obs_weight = cfg.algo.obs_weight
        self.eta = cfg.algo.eta
        self.lr = cfg.algo.lr
        self.denoise_term_weight = cfg.algo.denoise_term_weight
        self.columns = cfg.dataset.columns
        self.sigma_x0 = cfg.algo.sigma_x0
        self.decay_rate = getattr(cfg.algo, 'decay_rate', 0.9)

        print('self.lr', self.lr)
        print('self.sigma_x0', self.sigma_x0)
        print('self.denoise_term_weight', cfg.algo.denoise_term_weight)
        print('self.optim', cfg.algo.optim)

    def sample(self, x, y, ts, **kwargs):
        """sample by minimizing the RED-diff loss

        argmin |y - GP(mu)|^{2} + EE [2 w(t) (sgima_u / simga_t)^{2} |score(x_t, t) - eps|^{2} ]

        Args:
            x (torch.Tensor): Initial point
            y (torch.Tensor): Observed trajectory
            ts (torch.Tensor): Time steps
        """
        rank = int(os.environ["LOCAL_RANK"]) if kwargs.get('rank') else 0
        world_size = kwargs.get('world_size', 0)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        n = x.size(0)
        # Convert ts to tensor on device if not already
        if not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts, device=device)
        else:
            ts = ts.to(device)
        true_x = x.clone().to(device)
        x = self.initialize(x, ts).to(device)
        ss = torch.cat([torch.tensor([-1], device=device), ts[:-1]])
        delay_steps = self.cfg.algo.delay_schedule
        self.diffusion.alphas = self.diffusion.alphas.to(device)
        # Move models to device once
        self.forward_model = self.forward_model.to(device)
        # if self.model:
            # self.model = self.model.to(device)
        if delay_steps == 0:
            ss = ts.clone()
        else:
            ts, ss = ts[:-delay_steps], ts[delay_steps:]
            delta_t = torch.ones(n, device=device).long() * delay_steps
            ts = torch.repeat_interleave(ts, self.cfg.algo.repeat)
            ss = torch.repeat_interleave(ss, self.cfg.algo.repeat)

        #optimizer
        mu = torch.autograd.Variable(x, requires_grad=True)
        if self.cfg.algo.optim == 'Adam':
            optimizer = torch.optim.Adam([mu], lr=self.lr, betas=(0.9, 0.99), weight_decay=0.0)
        else:
            optimizer = torch.optim.SGD([mu], lr=1e-3)

        total_steps = len(ts)
        mu_list = []
        mse_results = []
        rmse_results = []
        y = y.to(device)
        with tqdm(zip(reversed(ts), reversed(ss)), position=rank, total=total_steps, desc="RED-diff sampling") as progress_bar:
            for step_idx, (ti, si) in enumerate(progress_bar):
                t = torch.ones(n, device=device).long() * ti

                alpha_t = self.diffusion.alpha(t).view(-1, 1)
                alpha_t = alpha_t[0].to(device) #1d torch tensor
                sigma_x0 = self.sigma_x0

                noise_x0 = torch.randn_like(mu, device=device)

                x0_pred = mu + sigma_x0*noise_x0
                e_obs = y - torch.cat(self.forward_model(x0_pred), dim=-1)
                loss_obs = (e_obs**2).mean()/2

                v_t = self.obs_weight
                loss = v_t*loss_obs
                loss += 0.5 * mu.square().mean()
                loss_obs_scalar = (e_obs**2).mean().sqrt().item()

                e_gp = torch.cat(self.forward_model(true_x), dim=-1) - torch.cat(self.forward_model(mu), dim=-1)
                gp_rmse = torch.mean(e_gp.square(), dim=1).sqrt().mean().item()

                mse = torch.mean((mu[:, self.columns] - true_x[:, self.columns]) ** 2, dim=1).item()
                progress_bar.set_description(f"{kwargs['idx']}:RED-diff sampling(obs_loss={loss_obs_scalar:.6f})(mse = {mse:.6f})(gp_loss={gp_rmse:.6f})")

                #adam step
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                #Save testing results
                mu_list.append(mu.detach().cpu().numpy())
                mse_results.append(mse)
                rmse_results.append(loss_obs_scalar)

        return x0_pred, mu, mu_list, mse_results, rmse_results

    def initialize(self, x, ts, **kwargs):
        # The initial guess point is zero
        x_0 = torch.zeros_like(x).detach()
        return x_0

    def plot_weight_den(self, ts, **kwargs):
        # Use the same device as diffusion alphas
        device = self.diffusion.alphas.device
        alpha = self.diffusion.alpha(torch.tensor(ts, device=device))

        snr_inv = (1-alpha).sqrt()/alpha.sqrt()
        snr_inv = snr_inv.detach().cpu().numpy()

        # plot lines
        plt.plot(ts, snr_inv, label="1/snr", linewidth=2)
        plt.plot(ts, np.sqrt(snr_inv), label="sqrt(1/snr)", linewidth=2)
        plt.plot(ts, np.square(snr_inv), label="square(1/snr)", linewidth=2)
        plt.plot(ts, np.log(snr_inv+1), label="log(1+1/snr)", linewidth=2)
        plt.plot(ts, np.clip(snr_inv, None, 1), label="clip(1/snr,max=1)", linewidth=2)
        plt.plot(ts, np.power(snr_inv, 0.0), label="const", linewidth=2)

        plt.legend()
        plt.yscale('log')
        plt.xlim(max(ts), min(ts))
        plt.xlabel("timestep", fontsize=15)
        plt.ylabel("denoiser weight", fontsize=15)

        plt.legend(fontsize=13)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        plt.savefig('weight_type_vs_step.png')
        return 0