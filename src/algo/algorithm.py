"""
# Modified based on code from red-diff

@article{mardani2023variational,
  title={A Variational Perspective on Solving Inverse Problems with Diffusion Models},
  author={Mardani, Morteza and Song, Jiaming and Kautz, Jan and Vahdat, Arash},
  journal={arXiv preprint arXiv:2305.04391},
  year={2023}
}
"""


import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import torch.distributed as dist
from .forward_model import GPPredictionModel
from .ddim import DDIM

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .score_estimator import ScoreEstimator
from ..utils.diffusion import Diffusion
from .score_estimator import ReverseDiffusionModel



class REDDIFF(DDIM):
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
        # self.projection = getattr(cfg.algo, 'projection', False)
        # self.moving_delay = getattr(cfg.algo, 'moving_delay', False)
        self.truncate = getattr(cfg.algo, 'truncate', False)
        
        print('self.lr', self.lr)
        print('self.sigma_x0', self.sigma_x0)
        print('self.denoise_term_weight', cfg.algo.denoise_term_weight)
        print('self.batch_size', cfg.algo.batch_size)
        # print('self.projection', cfg.algo.projection)
        # print('self.moving_delay', cfg.algo.moving_delay)
        print('self.truncate', cfg.algo.truncate)






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
        logger = kwargs.get("logger", None)
        n = x.size(0)
        true_x = x.clone()
        x = self.initialize(x, ts)
        self.diffusion.alphas = self.diffusion.alphas.to(device)
        delay_steps = self.cfg.algo.delay_schedule

        if delay_steps == 0:
            ss = list(ts)
            scale = 1.0
        else:
            ts, ss = ts[:-delay_steps], ts[delay_steps:]
            # mu/\alpha_{\delta} is the initial point for evaluationg the KL term gradient
            delta = torch.ones(n).to(device).long() * delay_steps
            scale = self.diffusion.alpha(delta).detach().view(-1).sqrt()[0]
            scale.to(device)
            ts, ss = torch.repeat_interleave(torch.tensor(ts), self.cfg.algo.repeat), torch.repeat_interleave(torch.tensor(ss), self.cfg.algo.repeat)

        mu = torch.autograd.Variable(x.to(device), requires_grad=True)   
        mu_ema = mu.detach().clone()
        beta = 0.9
        mu.to(device)

        #optimizer
        if self.cfg.algo.optim == 'Adam':
            print('self.optim Adam')
            optimizer = torch.optim.Adam([mu], lr=self.lr, betas=(0.9, 0.99), weight_decay=0.0)   #original: 0.999

        else:
            print('self.optim SGD')
            optimizer = torch.optim.SGD([mu], lr=self.lr)   #original: 0.999
        
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 1000, eta_min=0.1)

        total_steps = len(ts)
        mu_list = []
        mse_results = []
        rmse_results = []
        # ts: The time step for the SNR, ss: The time step for the score estimation
        with tqdm(zip(reversed(ts), reversed(ss)), position=rank, total=total_steps, desc=f"Rank:{rank} RED-diff sampling") as progress_bar:
            for step_idx, (ti, si) in enumerate(progress_bar):

                s = torch.ones(n).to(device).long() * si
                delta_t = torch.ones(n).to(device).long() * delay_steps

                alpha_s = self.diffusion.alpha(s).view(-1, 1)
                alpha_s = alpha_s[0]
                alpha_delta = self.diffusion.alpha(delta_t).view(-1, 1)
                alpha_delta = alpha_delta[0]
                sigma_x0 = self.sigma_x0  #0.0001 
                alpha_t = alpha_s / alpha_delta

                noise_x0 = torch.randn_like(mu)
                noise_x0 = noise_x0.to(device)

                x0_pred = mu + sigma_x0 * noise_x0
                noise_xt = torch.randn_like(torch.repeat_interleave(mu, self.batch_size, dim=0))
                xt = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise_xt # Sampling the diffusion without time shift
                y = y.to(device)
                variance = (alpha_delta + alpha_t - 2.0 * alpha_s) / (alpha_delta)
                variance = torch.clip(variance, min=0.1)
                et, x0_hat = self.model(xt, y, alpha_t, variance)

                # Clip the estimated score
                # et = torch.clip(et, -10.0, 10.0)
                et = et.detach()
                if delay_steps > 0:
                    x0_hat = (xt - (1 - alpha_t).sqrt() * et ) / alpha_t.sqrt() # The posterior mean at inital time is re-calculated with unshifted time schedule.
                et = et.to(device)
                score_error = (et - noise_xt)
                if self.truncate:
                    score_error = torch.clip(score_error, -3.0, 3.0)
                else:
                    loss_noise = torch.einsum('ij,kj->ik', score_error.detach(), x0_pred).mean()

                e_obs = y - torch.cat(self.forward_model(mu), dim=-1)

                loss_obs = (e_obs**2).mean()/2
                
                snr_inv = (1-alpha_t).sqrt()/alpha_t.sqrt()  
                snr = alpha_t.sqrt() / (1 - alpha_t).sqrt()
                if self.denoise_term_weight == "linear":
                    snr_inv = snr_inv # w(t) = t, w'(t) = 1
                elif self.denoise_term_weight == "EDM":
                    snr_inv = 1.0 / (snr + 0.25)
                    # snr_inv = snr_inv / (snr_inv + 1.0)
                elif self.denoise_term_weight == "sqrt":
                    snr_inv = torch.sqrt(snr_inv)
                elif self.denoise_term_weight == "square":
                    snr_inv = torch.square(snr_inv)
                elif self.denoise_term_weight == "log":
                    snr_inv = torch.log(snr_inv + 1.0)
                elif self.denoise_term_weight == "trunc_linear":
                    snr_inv = torch.clip(snr_inv, max=10.0, min=0.1)
                elif self.denoise_term_weight == "power2over3":
                    snr_inv = torch.pow(snr_inv, 2/3)
                elif self.denoise_term_weight == "const":
                    snr_inv = torch.pow(snr_inv, 0.0)
                elif self.denoise_term_weight == "cosine_decay_reverse":
                    # cosine decay
                    if total_steps > 1:
                        decay_factor = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * step_idx / (total_steps - 1)).to(snr_inv.device)))
                    else:
                        decay_factor = 1.0
                    snr_inv = snr * decay_factor


                w_t = self.grad_term_weight*snr_inv   #0.25
                
                v_t = self.obs_weight


                loss = w_t*loss_noise + v_t*loss_obs
                loss_obs_scalar = (e_obs**2).mean().sqrt().item()
                mse = torch.mean((mu_ema.cpu()[:, self.columns] - true_x.cpu()[:, self.columns]) ** 2, dim=1).item()

                true_x = true_x.to(device)
                e_gp = torch.cat(self.forward_model(true_x), dim=-1) - torch.cat(self.forward_model(mu), dim=-1)
                gp_rmse = torch.mean(e_gp.square(), dim=1).sqrt().mean().item()
                # progress_bar.set_description(f"{kwargs['idx']}:RED-diff sampling (obs_loss={loss_obs_scalar:.6f})(mse = {mse:.6f})(SNR_INV = {snr_inv.item():.6f})(gp_loss={gp_rmse:.6f})")
                progress_bar.set_description(f"rank:{rank}:RED-diff sampling (obs_loss={loss_obs_scalar:.6f})(mse = {mse:.6f})(SNR_INV = {snr_inv.item():.6f})(gp_loss={gp_rmse:.6f})")



                #adam step
                # if rank == 0:
                optimizer.zero_grad()  #initialize

                loss.backward(retain_graph=True)
                # if world_size > 0:
                #     # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                #     dist.all_reduce(mu_world.grad, op=dist.ReduceOp.SUM)

                # loss = loss.to('cuda:0')
                # if rank == 0:
                optimizer.step()
                with torch.no_grad():
                    # Formula: ema = beta * ema + (1 - beta) * current
                    # We use in-place operations (.mul_ and .add_) to save memory
                    mu_ema.mul_(beta).add_(mu, alpha=(1 - beta))
                # scheduler.step()

                # if logger is not None:
                #     logger.info(f"Current estimate param:{mu.cpu().detach().numpy()}")
                
                #Save testing results
                # mu_list.append(mu.cpu().detach().numpy())
                mu_list.append(mu_ema.cpu().detach().numpy())
                mse_results.append(mse)
                rmse_results.append(loss_obs_scalar)
                # score_rmse_list.append(score_rmse.cpu().item())

                # if world_size > 0:
                # # with torch.no_grad():
                #     dist.broadcast(mu_world, src=0)
        # if self.cfg.algo.EMA:
        #     mu = mu_ema

        return x0_pred, mu, mu_list, mse_results, rmse_results
    # , score_rmse_list

        
    def initialize(self, x, ts, **kwargs):
        # The initial guess point is zero
        x_0 = torch.zeros_like(x).detach()
        return x_0   #alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)    #x_0


    def plot_weight_den(self, ts, **kwargs):
    
        #ts.reverse()
        alpha = self.diffusion.alpha(torch.tensor(ts).cuda())
        
        snr_inv = (1-alpha).sqrt()/alpha.sqrt()  #1d torch tensor
        snr_inv = snr_inv.detach().cpu().numpy()
            
        # plot lines
        plt.plot(ts, snr_inv, label = "1/snr", linewidth=2)
        plt.plot(ts, np.sqrt(snr_inv), label = "sqrt(1/snr)", linewidth=2)
        #plt.plot(ts, np.power(snr_inv, 2/3), label = "(1/snr)^2/3")
        plt.plot(ts, np.square(snr_inv), label = "square(1/snr)", linewidth=2)
        plt.plot(ts, np.log(snr_inv+1), label = "log(1+1/snr)", linewidth=2)   #ln
        plt.plot(ts, np.clip(snr_inv, None, 1), label = "clip(1/snr,max=1)", linewidth=2)
        plt.plot(ts, np.power(snr_inv, 0.0), label = "const", linewidth=2)

        plt.legend()
        #plt.xscale('log')
        plt.yscale('log')
        plt.xlim(max(ts), min(ts))
        plt.xlabel("timestep", fontsize = 15)
        plt.ylabel("denoiser weight", fontsize = 15)
        
        plt.legend(fontsize = 13)
        plt.xticks(fontsize = 13)
        plt.yticks(fontsize = 13)

        plt.savefig('weight_type_vs_step.png')

        return 0








