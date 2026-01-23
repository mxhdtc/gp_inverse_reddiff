# Modified based on Reverse Diffusion Monte Carlo from arXiv:2307.02037

import torch
import torch.nn as nn
import math
from tqdm import trange
import os
import sys
from omegaconf import DictConfig
from ..utils import  (
    sample_multivariate_normal_diag,
    log_prob_multivariate_normal_diag,
    heuristics_step_size,
    heuristics_step_size_vectorized
)
from ..utils.diffusion import Diffusion


def ula_mcmc(x0, step_size, score, n_steps, n_warmup_steps=0, return_intermediates=False,
             return_intermediates_gradients=False):
    """Perform multiple steps of the ULA algorithm

        X_{k+1} = X_k + steps_size * score(X_k) + sqrt(2 * step_size) * Z_k

    Args:
        x0 (torch.Tensor of shape (batch_size, *data_shape)): Initial sample
        step_size (float): Step size for Langevin
        score (function): Gradient of the log-likelihood of the target distribution
        n_steps (int): Number of steps of the algorithm
        n_warmup_steps (int): Number of warmup steps (default is 0)
        return_intermediates (bool): Whether to return intermediates states
        return_intermediates_gradients (bool): Whether to return intermediates gradients
            (only is return_intermediates is True)

    Returns:
        x (torch.Tensor of shape (batch_size, *data_shape)): Final sample of the Langevin chain
    """

    x = x0
    if return_intermediates:
        xs = torch.empty((n_steps, *x.shape), device=x.device)
        if return_intermediates_gradients:
            grad_xs = torch.empty_like(xs)
    for i in range(n_warmup_steps+n_steps):
        grad_x = score(x)
        x += step_size * grad_x + torch.sqrt(2. * step_size) * torch.randn_like(x)
        if return_intermediates and (i >= n_warmup_steps):
            xs[i - n_warmup_steps] = x.clone()
            if return_intermediates_gradients and i >= 1:
                grad_xs[i - 1 - n_warmup_steps] = grad_x.clone()
    if return_intermediates:
        if return_intermediates_gradients:
            grad_xs[-1] = score(x).clone()
            return xs, grad_xs
        else:
            return xs
    else:
        return x


def mala_mcmc(
        x0,
        step_size,
        log_prob_and_grad,
        n_steps,
        n_warmup_steps=0,
        per_chain_step_size=True,
        return_intermediates=False,
        return_intermediates_gradients=False,
        target_acceptance=0.75) -> tuple:
    """Perform multiple steps of the MALA algorithm

        X_{k+1} = X_k + steps_size * score(X_k) + sqrt(2 * step_size) * Z_k

    Args:
        x0 (torch.Tensor of shape (batch_size, *data_shape)): Initial sample
        step_size (float): Step size for Langevin
        score (function): Gradient of the log-likelihood of the target distribution
        n_steps (int): Number of steps of the algorithm
        n_warmup_steps (int): Number of warmup steps (default is 0)
        per_chain_step_size (bool): Use a per chain step size (default is True)
        return_intermediates (bool): Whether to return intermediates steps
        target_acceptance (float): Default is 0.75
        return_intermediates_gradients (bool): Whether to return intermediates gradients
            (only is return_intermediates is True)

    Returns:
        x (torch.Tensor of shape (batch_size, *data_shape)): Final sample of the Langevin chain
    """

    sum_indexes = (1,) * (len(x0.shape) - 1)
    x = x0
    log_prob_x, grad_x = log_prob_and_grad(x)
    if return_intermediates:
        xs = torch.empty((n_steps, *x.shape), device=x.device)
        if return_intermediates_gradients:
            grad_xs = torch.empty_like(xs)
    # Reshape the step size if it hasn't been done yet
    if per_chain_step_size and not (isinstance(step_size, torch.Tensor) and len(step_size.shape) > 0):
        step_size = step_size * torch.ones((x.shape[0], *(1,) * (len(x.shape) - 1)), device=x.device)
    for i in range(n_steps+n_warmup_steps):
        # Sample the proposal
        x_prop = sample_multivariate_normal_diag(
            batch_size=x.shape[0],
            mean=x + step_size * grad_x,
            variance=2.0 * step_size
        )
        # Compute log-densities at the proposal
        log_prob_x_prop, grad_x_prop = log_prob_and_grad(x_prop)
        # Compute the MH ratio
        with torch.no_grad():
            joint_prop = log_prob_x_prop - \
                log_prob_multivariate_normal_diag(
                    x_prop,
                    mean=x + step_size * grad_x,
                    variance=2.0 * step_size,
                    sum_indexes=sum_indexes)
            joint_orig = log_prob_x - log_prob_multivariate_normal_diag(x,
                                                                        mean=x_prop + step_size * grad_x_prop,
                                                                        variance=2.0 * step_size,
                                                                        sum_indexes=sum_indexes)
        # Acceptance step
        log_acc = joint_prop - joint_orig
        mask = torch.log(torch.rand_like(log_prob_x_prop, device=x.device)) < log_acc
        x.data[mask] = x_prop[mask]
        log_prob_x.data[mask] = log_prob_x_prop[mask]
        grad_x.data[mask] = grad_x_prop[mask]
        # Update the step size
        if per_chain_step_size:
            step_size = heuristics_step_size_vectorized(step_size,
                                                        torch.minimum(torch.ones_like(log_acc), torch.exp(log_acc)), target_acceptance=target_acceptance)
        else:
            step_size = heuristics_step_size(step_size,
                                             torch.minimum(torch.ones_like(log_acc), torch.exp(log_acc)).mean(), target_acceptance=target_acceptance)
        # Save the sample
        if return_intermediates and (i >= n_warmup_steps):
            xs[i-n_warmup_steps] = x.clone()
            if return_intermediates_gradients:
                grad_xs[i-n_warmup_steps] = grad_x.clone()
    if return_intermediates:
        if return_intermediates_gradients:
            return xs, grad_xs, step_size
        else:
            return xs, step_size
    else:
        return x, step_size

class  ScoreEstimator(nn.Module):
    def __init__(self, target_log_prob, target_log_prob_and_grad, n_chains:int =16, 
                 n_mcmc_steps: int = 32, n_is_samples: int = 128) -> None:
        super(ScoreEstimator, self).__init__()
        self.T = -math.log(0.95)
        self.target_log_prob = target_log_prob
        self.target_log_prob_and_grad = target_log_prob_and_grad
        self.n_chains = n_chains
        self.n_mcmc_steps = n_mcmc_steps
        self.n_is_samples = n_is_samples
    
        # Example: Initialize RED-diff diffusion model if needed
        # self.diffusion = Diffusion(beta_schedule="linear", num_diffusion_timesteps=1000)
    
    def forward(self, xt, y, alpha_t, variance) -> torch.Tensor:
        con_target_log_prob = self.target_log_prob(y)
        con_target_log_prob_and_grad = self.target_log_prob_and_grad(y)
        step_size = torch.tensor(1.0 - math.exp(-2.0 * self.T)) / 2.0
        et, _ = self.score_estimation(alpha_t=alpha_t, variance=variance, x=xt, 
                                   step_size=step_size, n_is_samples=self.n_is_samples, n_mcmc_steps=self.n_mcmc_steps, n_chains=self.n_chains,
                                   target_log_prob=con_target_log_prob, target_log_prob_and_grad=con_target_log_prob_and_grad,)
        return et

    def posterior_log_prob_and_grad(self, y, xt, alpha_t, variance, target_log_prob_and_grad):
        """Compute the posterior distribution of RDMC and its gradient

            q_t(y|x_t) = pi(x_t) N(x_t;alpha_t*y, [1 - alpha_t^2] I)
        Args:
            t (torch.Tensor): Current time
            y (torch.Tensor of shape (batch_size, dim)): Evaluation point
            x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            T (torch.Tensor): Limit time
            target_log_prob_and_grad (function): Log-likelihood of the target and its gradient

        Returns:
            log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the posterior
            grad (torch.Tensor of shape (batch_size, dim)): Score of the posterior
        """
        # Evaluate the target
        target_log_prob, target_grad = target_log_prob_and_grad(y)

        # Compute the log_prob
        log_prob = target_log_prob
        log_prob -= 0.5 * torch.sum(torch.square(xt - alpha_t * y), dim=-1, keepdim=True) / variance 
        # Compute the gradient
        grad = target_grad
        grad += alpha_t * (xt - alpha_t * y) / variance
        # Return everything
        return log_prob, grad
    
    def posterior_importance_sampling(self, n_chains, x, alpha_t, variance, target_log_prob, n_mc_samples=128):
        """Sample the importance distribution associated with the posterior of RDMC
        ***Fix the issue that the importance weights does not include the proposal log-prob ***
        Args:
            n_chains (int): Number of samples to output per batch_size
            t (torch.Tensor): Current time
            x (torch.Tensor for shape (batch_size, dim)): Conditioning point
            T (torch.Tensor): Limit time
            target_log_prob (function): Target log-likelihood
            n_mc_samples (int): Number of particles (default is 128)

        Returns:
            samples (torch.Tensor for shape (n_chains * batch_size, dim)): Approximate samples from the posterior
        """

        # Compute the variance and the mean
        # variance = (1. - torch.square(alpha_t))
        mean = alpha_t * x
        # variance = (1. - torch.exp(-2. * (T - t))) / torch.exp(-2. * (T - t))
        # mean = torch.exp((T - t)) * x
        # Generate particles
        z = torch.sqrt(variance) * torch.randn((n_mc_samples, *x.shape), device=x.device)
        z += mean.unsqueeze(0)
        # Compute the importance weights
        log_weight = target_log_prob(z.view((-1, *x.shape[1:]))).view((n_mc_samples, -1))
        diff = z - mean.unsqueeze(0)
        proposal_log_prob = -0.5 * torch.sum(torch.square(diff.view((-1, *x.shape[1:]))), dim=-1) / variance
        log_weight -= proposal_log_prob.view((n_mc_samples, -1))
        weights = torch.nn.functional.softmax(log_weight, dim=0)
        # Sample the importance weights
        idx = torch.multinomial(weights.T, n_chains).T
        return torch.gather(z, 0, idx.unsqueeze(-1).expand((-1, -1, z.shape[-1]))).view((-1, z.shape[-1]))
    
    def score_estimation(self, x, alpha_t, variance, target_log_prob, target_log_prob_and_grad, step_size, n_is_samples,
                        n_mcmc_steps, n_chains, warmup_fraction=0.5):
        """Estimate the score of RDMC using IS followed by MCMC

        Args:
            t (torch.Tensor): Current time
            x (torch.Tensor of shape (batch_size, dim)): Evaluation point for the score
            target_log_prob (function): Log-likelihood of the target distribution
            target_log_prob_and_grad (function): Log-likelihood of the target distribution and its gradient
            step_size (torch.Tensor): Step size for MALA
            n_is_samples (int): Number of particles for IS
            n_mcmc_steps (int): Number of MCMC steps
            n_chains (int): Number of parrallel MCMC chains
            warmup_fraction (float): Warmup proportion (between 0.0 and 1.0 stricly) (default is 0.5)

        Returns:
            score (torch.Tensor of shape (batch_size, dim)): Score at time t and state x
            step_size (torch.Tensor): Updated step size for MALA
        """

        # Sample the importance distribution associated to the posterior
        y_langevin_start = self.posterior_importance_sampling(n_chains, x, alpha_t, variance, target_log_prob, n_is_samples)
        # Run Langevin on the posterior from the IS warm-start
        x_reshaped = x.unsqueeze(0).repeat((n_chains, 1, 1)).view((-1, x.shape[-1]))
        # print(y_langevin_start.size(), x_reshaped.size())
        def current_posterior_log_prob_and_grad(y): return self.posterior_log_prob_and_grad( y, x_reshaped,
                                                                                    alpha_t, variance, target_log_prob_and_grad)
        
        ys_langevin, step_size = mala_mcmc(y_langevin_start, step_size, current_posterior_log_prob_and_grad,
                                        n_mcmc_steps, per_chain_step_size=True, return_intermediates=True)
        ys_langevin = ys_langevin[-int(warmup_fraction * n_mcmc_steps):]
        ys_langevin = ys_langevin.view((int(warmup_fraction * n_mcmc_steps) * n_chains, -1, x.shape[-1]))
        # Compute the approximate score
        return - alpha_t * (x - alpha_t * ys_langevin.mean(dim=0)) / variance, step_size


class ReverseDiffusionModel:
    def __init__(self, model: ScoreEstimator, diffusion: Diffusion, cfg: DictConfig):
        self.model = model
        self.diffusion = diffusion
        self.cfg = cfg
    
    def __call__(self, xt, y, alpha_t, variance):
        # Returns both the noise value (score function scaled) and the predicted x0.
        # alpha_t = self.diffusion.alpha(t).view(-1, 1)
        et = self.model(xt, y, alpha_t, variance)
        x0_pred = (xt - et * (1 - alpha_t).sqrt()) / variance
        return et, x0_pred

