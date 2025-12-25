# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

# from models.classifier_guidance_model import ClassifierGuidanceModel
# from utils.degredations import build_degredation_model
from .forward_model import GPPredictionModel
from .ddim import DDIM

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# score_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'slips')
# sys.path.append("/home/xim22003/Diffusion_CLM/slips/slips")
# sys.path.insert(0, "/home/xim22003/Diffusion_CLM/slips")
# sys.path.insert(1, "/home/xim22003/Diffusion_CLM/sde_sampler/sde_sampler")
# sys.path.insert(0, score_path)
from .score_estimator import ScoreEstimator
# from ..reddiff_gp import ReverseDiffusionModel
from ..utils.diffusion import Diffusion
from .score_estimator import ReverseDiffusionModel
 

# sys.path.append("sde_sampler/sde_sampler/distr")
# from distr.VVGP import GPPredictionModel



    

class REDDIFF_VVGP(DDIM):
    def __init__(self, model: ReverseDiffusionModel, forward_model: GPPredictionModel, cfg: DictConfig):
        self.model = model
        self.diffusion = model.diffusion
        self.forward_model = forward_model
        # self.forward_model = build_forward_model(cfg)
        # self.H = build_degredation_model(cfg)
        self.cfg = cfg
        self.awd = cfg.algo.awd
        self.batch_size = cfg.algo.batch_size
        # self.cond_awd = cfg.algo.cond_awd
        self.grad_term_weight = cfg.algo.grad_term_weight
        self.obs_weight = cfg.algo.obs_weight
        self.eta = cfg.algo.eta
        self.lr = cfg.algo.lr
        self.denoise_term_weight = cfg.algo.denoise_term_weight
        self.columns = cfg.dataset.columns
        self.sigma_x0 = cfg.algo.sigma_x0
        self.decay_rate = getattr(cfg.algo, 'decay_rate', 0.9)
        # if self.cfg.algo.batch_size > 1:
        #             self.lr *= self.cfg.algo.batch_size
        
        print('self.lr', self.lr)
        print('self.sigma_x0', self.sigma_x0)
        print('self.denoise_term_weight', cfg.algo.denoise_term_weight)
        print('self.batch_size', cfg.algo.batch_size)




    def sample(self, x, y, ts, **kwargs):
        """sample by minimizing the RED-diff loss

        argmin |y - GP(mu)|^{2} + EE [2 w(t) (sgima_u / simga_t)^{2} |score(x_t, t) - eps|^{2} ]
        
        Args:
            x (torch.Tensor): Initial point
            y (torch.Tensor): Observed trajectory
            ts (torch.Tensor): Time steps
        """
        n = x.size(0)
        true_x = x.clone()
        x = self.initialize(x, ts)
        delay_steps = self.cfg.algo.delay_schedule
        if delay_steps == 0:
            # ss = [-1] + list(ts[:-1])
            ss = list(ts)
            scale = 1.0
        else:
            ts, ss = ts[:-delay_steps], ts[delay_steps:]
            # mu/\alpha_{\delta} is the initial point for evaluationg the KL term gradient
            delta_t = torch.ones(n).to(x.device).long() * delay_steps
            scale = self.diffusion.alpha(delta_t).detach().view(-1).sqrt()[0]
            ts, ss = torch.repeat_interleave(torch.tensor(ts), self.cfg.algo.repeat), torch.repeat_interleave(torch.tensor(ss), self.cfg.algo.repeat)
        
        factor = 1-1e-8

        #optimizer
        mu = torch.autograd.Variable(x, requires_grad=True)   #, device=device).type(dtype)
        if self.cfg.algo.optim == 'Adam':
            print('self.optim Adam')
            optimizer = torch.optim.Adam([mu], lr=self.lr, betas=(0.9, 0.99), weight_decay=0.0)   #original: 0.999
        else:
            print('self.optim SGD')
            optimizer = torch.optim.SGD([mu], lr=1e-3)   #original: 0.999
        # optimizer = torch.optim.Adam([mu], lr=self.lr, betas=(0.9, 0.99), weight_decay=0.0)   #original: 0.999
        # optimizer = torch.optim.SGD([mu], lr=self.lr, momentum=0.999)   #original: 0.999
        # optimizer = torch.optim.SGD([mu], lr=1e-3)   #original: 0.999


        total_steps = len(ts)
        mu_list = []
        mse_results = []
        rmse_results = []
        cos_sim_results = []
        score_matching_losses = []
        # ts: The time step for the SNR, ss: The time step for the score estimation
        with tqdm(zip(reversed(ts), reversed(ss)), total=total_steps, desc="RED-diff sampling") as progress_bar:
            for step_idx, (ti, si) in enumerate(progress_bar):
        # for ti, si in tqdm(zip(reversed(ts), reversed(ss)), total=len(ts), desc="RED-diff sampling"):
                
                t = torch.ones(n).to(x.device).long() * ti
                s = torch.ones(n).to(x.device).long() * si
                alpha_t = self.diffusion.alpha(t).view(-1, 1)
                alpha_t = alpha_t[0] #1d torch tensor
                alpha_s = self.diffusion.alpha(s).view(-1, 1)
                alpha_s = alpha_s[0]
                sigma_x0 = self.sigma_x0  #0.0001 

                noise_x0 = torch.randn_like(mu)
                # noise_xt = torch.randn_like(mu)

                x0_pred = mu * scale + (sigma_x0 * scale)*noise_x0
                # x0_pred = mu * scale + (sigma_x0 * torch.sqrt(torch.tensor(scale).detach().clone()))*noise_x0
                # xt = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise_xt
                # et, x0_hat = self.model(xt, y, alpha_t.sqrt())   #et, x0_pred
                
                # if not self.awd:
                #     et = (xt - x0_hat * alpha_t.sqrt()) / (1 - alpha_t).sqrt()
                
                # et = et.detach()

                # noise_xt = torch.randn((self.batch_size, *mu.size()))
                noise_xt = torch.randn_like(torch.repeat_interleave(mu, self.batch_size, dim=0))
                xt = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise_xt # Sampling the diffusion without time shift
                et, x0_hat = self.model(xt, y, alpha_s.sqrt())   #et, x0_pred, estimate the score function at shifted time.
                et = et.detach()
                if delay_steps > 0:
                    x0_hat = (xt - (1 - alpha_t).sqrt() * et ) / alpha_t.sqrt() # The posterior mean at inital time is re-calculated with unshifted time schedule.
                    # et = (xt - x0_hat * alpha_t.sqrt()) / (1 - alpha_t).sqrt() 
                loss_noise = torch.einsum('ij,kj->ik', (et - noise_xt).detach(), x0_pred).mean()
                loss_noise = loss_noise / scale
                
                
                
                # grad_reg = et - noise_xt  # shape (batch_size * n, ...)
                # batch_size = self.batch_size
                # # reshape to (batch_size, n, -1)
                # grad_reg_flat = grad_reg.view(batch_size, n, -1)
                # grad_reg_mean = grad_reg_flat.mean(dim=0)  # (n, flattened_features)

                # diff = true_x - x0_pred  # (n, ...)
                # diff_flat = diff.view(n, -1)

                # grad_reg_mean, diff_flat = grad_reg_mean[:, self.columns], diff_flat[:, self.columns]

                # # cosine similarity per sample
                # cos_sim = F.cosine_similarity(grad_reg_mean, diff_flat, dim=1)
                # cos_sim_mean = cos_sim.mean().item()
                # cos_sim_results.append(cos_sim_mean)

                # score_erroe = torch.mean(torch.square(et - noise_xt)).cpu().item()
                # score_matching_losses.append(score_erroe)

                # e_obs = y - torch.cat(self.forward_model(x0_pred), dim=-1)
                e_obs = y - torch.cat(self.forward_model(mu), dim=-1)



                # noise_xt = torch.randn_like(torch.repeat_interleave(mu, self.batch_size, dim=0))
                # xt = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise_xt
                # # print(xt.size())
                # et, x0_hat = self.model(xt, y, alpha_t.sqrt())   #et, x0_pred
                # et = et.detach()
                # # loss_noise = torch.mul((et - noise_xt).detach(), x0_pred).mean()
                # loss_noise = torch.einsum('ij,kj->ik', (et - noise_xt).detach(), x0_pred).mean()
                # # loss_noise = torch.einsum('ij,kj->ik', et, x0_pred).mean()

                # Compute cosine similarity between gradient (et - noise_xt) and (true_x - x0_pred)
                # grad_reg = et - noise_xt  # shape (batch_size * n, ...)
                # batch_size = self.batch_size
                # # reshape to (batch_size, n, -1)
                # grad_reg_flat = grad_reg.view(batch_size, n, -1)
                # grad_reg_mean = grad_reg_flat.mean(dim=0)  # (n, flattened_features)

                # diff = true_x - x0_pred  # (n, ...)
                # diff_flat = diff.view(n, -1)

                # grad_reg_mean, diff_flat = grad_reg_mean[:, self.columns], diff_flat[:, self.columns]

                # # cosine similarity per sample
                # cos_sim = F.cosine_similarity(grad_reg_mean, diff_flat, dim=1)
                # cos_sim_mean = cos_sim.mean().item()
                # cos_sim_results.append(cos_sim_mean)

                # # e_obs = y - torch.cat(self.forward_model(x0_pred), dim=-1)
                # e_obs = y - torch.cat(self.forward_model(mu), dim=-1)

                loss_obs = (e_obs**2).mean()/2
                # loss_noise = torch.mul((et - noise_xt).detach(), x0_pred).mean()
                
                # snr_inv = (1-alpha_t[0]).sqrt()/alpha_t[0].sqrt()  #1d torch tensor
                snr_inv = (1-alpha_t).sqrt()/alpha_t.sqrt()  
                snr = alpha_t.sqrt() / (1 - alpha_t).sqrt()
                factor *= factor
                if self.denoise_term_weight == "linear":
                    snr_inv = snr_inv # w(t) = t, w'(t) = 1
                elif self.denoise_term_weight == "sqrt":
                    snr_inv = torch.sqrt(snr_inv)
                elif self.denoise_term_weight == "square":
                    snr_inv = torch.square(snr_inv)
                elif self.denoise_term_weight == "log":
                    snr_inv = torch.log(snr_inv + 1.0)
                elif self.denoise_term_weight == "trunc_linear":
                    snr_inv = torch.clip(snr_inv, max=1.0)
                elif self.denoise_term_weight == "power2over3":
                    snr_inv = torch.pow(snr_inv, 2/3)
                elif self.denoise_term_weight == "const":
                    snr_inv = torch.pow(snr_inv, 0.0)
                elif self.denoise_term_weight == "log_sigmoid":
                    # snr_inv = torch.sigmoid(torch.log(snr_inv)) * (1 - torch.sigmoid(torch.log(snr_inv)))
                    # snr_inv = snr * torch.sigmoid(torch.log(snr)) * (1 - torch.sigmoid(torch.log(snr)))
                    snr_inv = snr_inv * torch.sigmoid(torch.log(torch.tensor(step_idx / (total_steps - 1)))) * (1 - torch.sigmoid(torch.log(torch.tensor(step_idx / (total_steps - 1)))))


                elif self.denoise_term_weight == "linear_reverse":
                    snr_inv = snr
                elif self.denoise_term_weight == "linear_reverse_truncate":
                    snr_inv = torch.clip(snr, max=1.0)
                elif self.denoise_term_weight == "linear_reverse_decay":
                    snr_inv *= factor
                    snr_inv = torch.clip(snr_inv, min=0.1)
                elif self.denoise_term_weight == "exponential_decay":
                    # exponential decay over iterations
                    decay_factor = self.decay_rate ** step_idx
                    snr_inv = snr_inv * decay_factor
                elif self.denoise_term_weight == "linear_decay":
                    # linear decay from 1 to decay_rate over iterations
                    if total_steps > 1:
                        decay_factor = 1 - (1 - self.decay_rate) * (step_idx / (total_steps - 1))
                    else:
                        decay_factor = 1.0
                    snr_inv = snr_inv * decay_factor
                elif self.denoise_term_weight == "cosine_decay":
                    # cosine decay
                    if total_steps > 1:
                        decay_factor = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * step_idx / (total_steps - 1)).to(snr_inv.device)))
                    else:
                        decay_factor = 1.0
                    snr_inv = snr_inv * decay_factor
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
                mse = torch.mean((mu.cpu()[:, self.columns] - true_x.cpu()[:, self.columns]) ** 2, dim=1).item()
                e_gp = torch.cat(self.forward_model(true_x), dim=-1) - torch.cat(self.forward_model(mu), dim=-1)
                gp_rmse = torch.mean(e_gp.square(), dim=1).sqrt().mean().item()
                # progress_bar.set_description(f"RED-diff sampling (obs_loss={loss_obs_scalar:.6f})(mse = {mse:.6f})(SNR_INV = {snr_inv.item():.6f})(cos_sim={cos_sim_mean:.6f})")
                # progress_bar.set_description(f"RED-diff sampling (obs_loss={loss_obs_scalar:.6f})(mse = {mse:.6f})(SNR_INV = {snr_inv.item():.6f})(score_matching_loss={score_erroe:.6f})")
                progress_bar.set_description(f"RED-diff sampling (obs_loss={loss_obs_scalar:.6f})(mse = {mse:.6f})(SNR_INV = {snr_inv.item():.6f})(gp_loss={gp_rmse:.6f})")

                #adam step
                optimizer.zero_grad()  #initialize
                loss.backward(retain_graph=True)
                optimizer.step()

                #Save testing results
                mu_list.append(mu.cpu().detach().numpy())
                mse_results.append(mse)
                rmse_results.append(loss_obs_scalar)
            

        return x0_pred, mu, mu_list, mse_results, rmse_results

        
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








