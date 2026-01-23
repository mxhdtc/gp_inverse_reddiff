import torch
import torch.nn as nn
import gpytorch
from collections import namedtuple
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from ..utils.vvgp import MLPFeatureExtractor, DKLModel

# LAMBDA = 1e-1
LAMBDA = 1e0
SIGMA = 1e0
Delta = 2.5e1


class GPPredictionModel(nn.Module):
    """
    A PyTorch Module for making predictions using the loaded GP models.
    This class takes the saved model parameters and creates a prediction model
    that returns the predicted mean values from the likelihood(model(x)) computation.
    """
    
    def __init__(self, model_path, inducing_dim, grid_bounds = (-1.0, 1.0), USE_CUDA = True):
        """
        Initialize the prediction model with saved parameters.
        
        Args:
            model_path (str): Path to the saved model .pth file
        """
        super(GPPredictionModel, self).__init__()
        
        # Load the saved model
        self.model_info = torch.load(model_path)
        
        # Extract model parameters
        self.NUM_TASKS = self.model_info['NUM_TASKS']
        self.NUM_LATENTS = self.model_info['NUM_LATENTS']
        self.INPUT_DIM = self.model_info['INPUT_DIM']
        
        # Recreate the feature extractors
        self.feature_encoder_P = MLPFeatureExtractor(
            input_dim=self.INPUT_DIM,
            hidden_dim=1024,
            output_dim=inducing_dim,
            num_layers=8
        )
        
        self.feature_encoder_Q = MLPFeatureExtractor(
            input_dim=self.INPUT_DIM,
            hidden_dim=1024,
            output_dim=inducing_dim,
            num_layers=8
        )
        
        # Create inducing points
        # inducing_points_P = torch.rand(self.NUM_LATENTS, 512, 2) * 2 - 1.0
        # inducing_points_Q = torch.rand(self.NUM_LATENTS, 512, 2) * 2 - 1.0
        inducing_points_P = self.model_info['inducing_points_P']
        inducing_points_Q = self.model_info['inducing_points_Q']
        

        # Recreate the models
        self.model_P = DKLModel(
            feature_encoder=self.feature_encoder_P,
            inducing_points=inducing_points_P,
            num_tasks=self.NUM_TASKS,
            num_latents=self.NUM_LATENTS,
            grid_bounds=grid_bounds
            # grid_bounds=(-1., 1.)
        )
        
        self.model_Q = DKLModel(
            feature_encoder=self.feature_encoder_Q,
            inducing_points=inducing_points_Q,
            num_tasks=self.NUM_TASKS,
            num_latents=self.NUM_LATENTS,
            grid_bounds=grid_bounds
            # grid_bounds=(-1., 1.)
        )
        
        # Recreate the likelihoods
        self.likelihood_P = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.NUM_TASKS)
        self.likelihood_Q = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.NUM_TASKS)
        
        # Load the state dicts
        self.model_P.load_state_dict(self.model_info['model_P_state_dict'])
        self.model_Q.load_state_dict(self.model_info['model_Q_state_dict'])
        self.likelihood_P.load_state_dict(self.model_info['likelihood_P_state_dict'])
        self.likelihood_Q.load_state_dict(self.model_info['likelihood_Q_state_dict'])
        # self.feature_encoder_P.load_state_dict(self.model_info['feature_encoder_P_state_dict'])
        # self.feature_encoder_Q.load_state_dict(self.model_info['feature_encoder_Q_state_dict'])
        
        # Set models to evaluation mode
        # self.model_P.eval()
        # self.model_Q.eval()
        # self.likelihood_P.eval()
        # self.likelihood_Q.eval()

        # if USE_CUDA:
        #     self.model_P = self.model_P.to(device)
        #     self.model_Q = self.model_Q.to(device)
        #     self.likelihood_P = self.likelihood_P.to(device)
        #     self.likelihood_Q = self.likelihood_Q.to(device)
        
        # # Create MLLs (for completeness, not used in prediction)
        # self.mll_P = gpytorch.mlls.VariationalELBO(
        #     self.likelihood_P, self.model_P.gp_layer, num_data=1)
        # self.mll_Q = gpytorch.mlls.VariationalELBO(
        #     self.likelihood_Q, self.model_Q.gp_layer, num_data=1)

    def forward(self, x):
        """
        Forward pass to get predictions.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            tuple: (predictions_P, predictions_Q) where each is a tensor of predicted means
        """
        # Get predictions from both models
        pred_P = self.likelihood_P(self.model_P(x))
        pred_Q = self.likelihood_Q(self.model_Q(x))
            
        # Return the means
        return pred_P.mean, pred_Q.mean
    
    def predict(self, x):
        """
        Convenience method for making predictions.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            tuple: (predictions_P, predictions_Q) where each is a tensor of predicted means
        """
        return self.forward(x)



def build_forward_model(cfg: DictConfig):
    """
    Build a forward model from the configuration.
    """
    if cfg.algo.model == "vvgp":
        model_path = cfg.algo.model_path
        inducing_dim = cfg.algo.inducing_dim        
        grid_bounds = -cfg.algo.grid_bound, cfg.algo.grid_bound

        forward_model = GPPredictionModel(model_path=model_path, inducing_dim=inducing_dim, grid_bounds=grid_bounds)
        concat_ouput = lambda x: torch.cat(forward_model(x), dim=-1)
        return concat_ouput

        # return GaussianProcess(cfg)
    else:
        raise ValueError(f"Unknown model: {cfg.algo.model}")