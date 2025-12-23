import torch
import torch.nn.functional as F
from omegaconf import DictConfig

# from models.classifier_guidance_model import ClassifierGuidanceModel
# from utils.degredations import build_degredation_model
from forward_model import GPPredictionModel
from .ddim import DDIM

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class NORMAL_REG(DDIM):
    def __init__(self, model: GPPredictionModel, forward_model: GPPredictionModel, cfg: DictConfig):
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
        
        print('self.lr', self.lr)
        print('self.sigma_x0', self.sigma_x0)
        print('self.denoise_term_weight', cfg.algo.denoise_term_weight)



