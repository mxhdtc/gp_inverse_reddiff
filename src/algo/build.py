import torch
import logging
import os
import sys
from hydra.utils import call
from omegaconf import DictConfig
from .score_estimator import ScoreEstimator

from .algorithm import REDDIFF
from .benchmark import ADAM
# from .normal_vvgp import NORMAL_REG

# sys.path.insert(0, os.path.abspath(os.path.join(f"{__file__}/", "../../")))

from ..utils.diffusion import Diffusion






def build_gp_algo(cg_model, forward_model, cfg):
    if cfg.algo.name == 'reddiff_vvgp':
        return REDDIFF(cg_model, forward_model, cfg)
    elif cfg.algo.name == 'MAP':
        return ADAM(cg_model, forward_model, cfg)
    else:
        raise ValueError(f'No algorithm named {cfg.algo.name}')


def target_log_prob_and_grad(model, grad=True):
    """
    model: GPPredictionModel
    """
    standard_normal = torch.distributions.Normal(0., 1.)
    model.eval() 
    def _target_log_prob(y):
        """
        y: Tensor: [Batch_size, Dim], the observed trajectory
        return: Tensor: [Batch_size, 1], the log-prob of the observed trajectory
        """
        y_P, y_Q = y[:, :600], y[:, 600:]

        def log_normal_loss(x):
            # model.eval()

            mu_P, mu_Q = model(x)
            diff_P = torch.sqrt(torch.mean(torch.square(y_P - mu_P), dim=1, keepdim=True))
            diff_Q = torch.sqrt(torch.mean(torch.square(y_Q - mu_Q), dim=1, keepdim=True))
            # diff_P =  torch.mean(torch.square(y_P - mu_P), dim=1, keepdim=True)
            # diff_Q =  torch.mean(torch.square(y_Q - mu_Q), dim=1, keepdim=True)
            diff = torch.cat([diff_P, diff_Q], dim=1)
            return -1.0 * torch.mean(diff, dim=1, keepdim=True) + standard_normal.log_prob(x).sum(dim=1, keepdim=True)

        return log_normal_loss

    def _target_log_prob_and_grad(y):
        """
        y: Tensor: [Batch_size, Dim], the observed trajectory
        return: Tensor: [Batch_size, 1], the log-prob of the observed trajectory
        """
        y_P, y_Q = y[:, :600], y[:, 600:]
        # model.eval()

        def log_normal_loss(x):

            mu_P, mu_Q = model(x)
            diff_P = torch.sqrt(torch.mean(torch.square(y_P - mu_P), dim=1, keepdim=True))
            diff_Q = torch.sqrt(torch.mean(torch.square(y_Q - mu_Q), dim=1, keepdim=True))
            # diff_P =  torch.mean(torch.square(y_P - mu_P), dim=1, keepdim=True)
            # diff_Q =  torch.mean(torch.square(y_Q - mu_Q), dim=1, keepdim=True)
            diff = torch.cat([diff_P, diff_Q], dim=1)
            return -1.0 * torch.mean(diff, dim=1, keepdim=True) + standard_normal.log_prob(x).sum(dim=1, keepdim=True)

        def log_prob_grad(x):
            x_ = torch.autograd.Variable(x, requires_grad=True)
            log_prob_y = log_normal_loss(x_)
            return torch.autograd.grad(log_prob_y.sum(), x_, retain_graph=True)[0].detach()

        def wrapper(x):
            return log_normal_loss(x), log_prob_grad(x)
        return wrapper

    if grad:
        return _target_log_prob_and_grad
    else:
        return _target_log_prob


def create_gp_model(cfg):
    model = call(cfg.model)  # This will return a Gaussian Process model to generate the predicted trajectory
    model.cuda()
    # model.eval()
    logging.info(f"Loading GP model from {cfg.model.model_path}..")
    target_log_prob_wrapper = target_log_prob_and_grad(model, grad=False)
    target_log_prob_and_grad_wrapper = target_log_prob_and_grad(model, grad=True)
    return model, target_log_prob_wrapper, target_log_prob_and_grad_wrapper


def create_score_model(cfg):
    model, target_log_prob_wrapper, target_log_prob_and_grad_wrapper = create_gp_model(cfg)
    score_estimator = ScoreEstimator(target_log_prob_wrapper,
                                     target_log_prob_and_grad_wrapper,
                                     cfg.algo.n_chains,
                                     cfg.algo.n_mcmc_steps,
                                     cfg.algo.n_is_samples)
    return model, score_estimator


def build_model(cfg):
    """
    Build GP model and score estimator for RED-diff GP.
    Returns:
        model: GPPredictionModel (GP forward model)
        score_estimator: ScoreEstimator (score estimation model)
    """
    model, score_estimator = create_score_model(cfg)
    score_estimator.cuda()
    model.cuda()
    logging.info(f"Building score model from {cfg.model.model_path}..")
    return model, score_estimator