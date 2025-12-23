from __future__ import annotations
import torch
from torch import nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.kernels import RBFKernel, Kernel
from gpytorch.variational import CholeskyVariationalDistribution, MeanFieldVariationalDistribution
from gpytorch.variational import VariationalStrategy, LMCVariationalStrategy
from torch.distributions.kl import kl_divergence, register_kl

NUM_TASKS = 3362
NUM_LATENTS = 128



# residual_target = torch.tensor(pd.read_csv("/home/xim22003/Diffusion_CLM/Open-source-power-dataset/Code/Joint Simulation/code/pvmodel/default_para_result.csv").values, dtype=torch.float32)
# residual_target = residual_target.cuda()



class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(MLPFeatureExtractor, self).__init__()
        
        # Create list for network layers
        layers = []

        # First layer (input to hidden)
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Intermediate hidden layers
        for _ in range(num_layers - 2):  # Subtract 2 because we already added first layer
            layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
            layers.append(nn.ReLU())
            hidden_dim = hidden_dim // 2

        # Final layer (hidden to output)
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Wrap layers in Sequential
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)
    

class LMCVectorGP(ApproximateGP):
    """
    Use Linear Model of Coeregionalization to define a Gaussian Process. Varaitional Optimization with CholeskyVariationalDistribution.
    This class will use forward() to generate the prior distribution of latent functions and variational distribution.
    # The __call__() in ApproximateGP will call its variational_strategy.call() to calculate the prior distribution of variational varaibles, which is .
    # The forward() in ApproximateGP will be called in its variational_strategy.forward() method, to create the prior distribution of the inducing points and train data. 
    # will calculate the optimal posterior distribution of variational varaibles under current ELBO.
    Model(): __call__() will return the conditional distribution of latend function under current variational distribution of inducing points.
    Likelihood(): __call__() will return the predictive distribution of the data points with model noise.
    mll(): 
    """
    def __init__(self, inducing_points,  num_tasks=3362, num_latents=128):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([num_latents]))
        # variational_distribution = MeanFieldVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([num_latents]))
        variational_strategy = LMCVariationalStrategy(VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=False),
                                                                            num_tasks=num_tasks,
                                                                            num_latents =num_latents,
                                                                            latent_dim=-1, 
                                                                            )
        super(LMCVectorGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=torch.Size([num_latents]),
                outputscale_constraint = [0.25, 0.75],
                # lengthscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
            ),
            batch_shape=torch.Size([num_latents]),
        )
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #    NN_RBFKernel(feature_extractor = feature_encoder, batch_shape=torch.Size([num_latents])),
        #     batch_shape=torch.Size([num_latents]), )   
        # self.feature_encoder = feature_encoder if feature_encoder is not None else IdentityMap()
        # if isinstance(feature_encoder, BatchedMLPFeatureExtractor) or isinstance(feature_encoder, IdentityMap):
        #     self.flag = 1
        # elif isinstance(feature_encoder, MLPFeatureExtractor):
        #     self.flag = 2
        # elif isinstance(feature_encoder, TransformerFeatureExtractor):
        #     self.flag = 3
        # else:
        #     self.flag = -1
        #     raise ValueError("Invalid feature encoder")

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class IdentityMap(nn.Module):
    """
    Customer identity map that takes in a tensor of shape (batch_size, seq_len, 1)
    and returns a tensor of shape (batch_size, seq_len). This is required because the
    input to the Transformer is of shape (batch_size, seq_len, embed_dim)
    """
    def forward(self, x):
        return x

class DKLModel(gpytorch.Module):
    def __init__(self, feature_encoder, inducing_points,  num_tasks=3362, num_latents=128, grid_bounds=(-1., 1.)):
        super(DKLModel, self).__init__()
        self.feature_encoder = feature_encoder if feature_encoder is not None else IdentityMap()
        self.gp_layer = LMCVectorGP(inducing_points, num_tasks, num_latents)
        self.grid_bounds = grid_bounds
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])
    def forward(self, x):
        features = self.feature_encoder(x)
        features = self.scale_to_bounds(features)
        res = self.gp_layer(features)
        return res
        


# class GP_likelihood(Distribution):
#     """
#     A distribution class that uses the GPPredictionModel to compute log probabilities.
#     The log PDF is computed as: logN(y - mu(x), I) + logN(bar{x}, I)
#     where:
#     - y is the observed measurement trajectory
#     - mu(x) is the conditional mean prediction from the GP model
#     - bar{x} is the default input (mean of training data)
#     """
    
#     def __init__(
#         self,
#         gp_model_path,
#         default_x,
#         default_y,
#         input_dim=15,
#         output_dim=1200,
#         inducing_dim=8,
#         dim=None,
#         log_norm_const=0.0,
#         n_reference_samples=int(1e6),
#     ):
#         """
#         Initialize the GP likelihood distribution.
        
#         Args:
#             gp_model_path (str): Path to the saved GP model
#             default_x (torch.Tensor): Default input x̄, if None will be initialized as zeros
#             input_dim (int): Dimension of input x
#             output_dim (int): Dimension of output y (should match GP model output)
#             dim (int): Total dimension (input_dim + output_dim)
#         """
#         if dim is None:
#             dim = input_dim + output_dim
    
#         super().__init__(
#             dim=dim,
#             log_norm_const=log_norm_const,
#             n_reference_samples=n_reference_samples,
#         )
        
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         # Load the GP prediction model
#         self.gp_model = GPPredictionModel(gp_model_path, inducing_dim)
#         # self.gp_model.eval()  # Set to evaluation mode
#         device = torch.device('cuda')
#         self.gp_model = self.gp_model.to(device)
#         # self.standard_normal = torch.distributions.Normal(
#         #     torch.tensor(0.0, device=device), 
#         #     torch.tensor(1.0, device=device)
#         # )

#         for param in self.gp_model.parameters():
#             # param.requires_grad = False
#             param.requires_grad_(False)
        
#         self.gp_model.eval()
        
#         # Register default_x as a buffer so it's part of the model state
#         # if default_x is None:
#         #     # Initialize default_x as zeros if not provided
#         #     default_x = torch.zeros(input_dim) + 1.0
#         default_x = torch.zeros(input_dim).cuda()
#         self.register_buffer('default_x', default_x.clone().detach())
#         self.register_buffer('y', default_y.clone().detach())
        
#         # # Register default_y as a buffer if provided
#         # if default_y is not None:
#         #     # Convert pandas DataFrame to tensor if needed
#         #     default_y = torch.tensor(default_y.values, dtype=torch.float32)
#         #     self.register_buffer('default_y', default_y.clone().detach())
#         # else:
#         #     self.register_buffer('default_y', torch.zeros(output_dim))

#         # Standard normal distribution for computing log probabilities
#         self.standard_normal = torch.distributions.Normal(0., 1.)
#         # self.standard_normal = torch.distributions.Normal(0., .1)


#         # self.prior_normal = torch.distributions.Normal(0., .25)
#         # self.prior_normal = torch.distributions.Normal(0., 1.)
#         self.prior_normal = torch.distributions.Normal(0., 1.)

    
#     def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the unnormalized log probability density function: logN(y - mu(x), I) + logN(x̄, I)
        
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, input_dim) or (batch_size, dim)
#                          If shape is (batch_size, dim), we assume first input_dim elements are x
#                          and last output_dim elements are y.
            
#         Returns:
#             torch.Tensor: Unnormalized log probability of shape (batch_size, 1)
#         """
#         # if x.shape[-1] == self.dim:
#         #     # Split x into input (x_in) and output (y) parts
#         #     x_in = x[..., :self.input_dim]
#         #     y = x[..., self.input_dim:]
#         # elif x.shape[-1] == self.input_dim:
#         #     # Only input is provided, assume y=0 (this is just for interface compatibility)
#         #     x_in = x
#         #     y = self.y
#         # else:
#         #     raise ValueError(f"Input dimension mismatch. Expected {self.input_dim} or {self.dim}, got {x.shape[-1]}")
        
#         # x_in = x
#         y = self.y
#         # Get predictions from GP model
#         # with torch.no_grad():
#         mu_P, mu_Q = self.gp_model(x)
        
#         # Remove interval with low predictive ability
#         # mu_P = mu_P[:, 50:550]
#         # mu_Q = mu_Q[:, 50:550]

#         # mu_Q = mu_Q[:, :50]

#         # Concatenate P and Q predictions to match y shape
#         mu = torch.cat([mu_P, mu_Q], dim=1)
#         y = y.unsqueeze(0).expand(mu.shape[0], -1)

#         y_1, y_2 = y[:, :600], y[:, 600:] 

#         # y_1 = y_1[:, 50:550]
#         # y_2 = y_2[:, 50:550]

#         # y_2 = y_2[:, :50]

#         # y = torch.cat([y_1, y_2], dim=1)

#         # Compute logN(y - mu(x), I) - this is a multivariate normal with identity covariance
#         # Which is equivalent to sum of independent standard normal distributions of (y - mu(x))
#         # Use mean squared error instead
#         # diff = torch.sqrt(torch.mean(torch.square(y - mu)))
#         diff_P = y_1 - mu_P
#         diff_Q = y_2 - mu_Q
#         # diff_P[:, 0:50]  = diff_P[:, 0:50] * LAMBDA 
#         # diff_Q[:, 50:] = diff_Q[:, 50:] * LAMBDA
#         log_prob_data = self.standard_normal.log_prob(torch.cat([diff_P, diff_Q], dim=1)).sum(dim=1, keepdim=True)

#         # diff_P = torch.mean(torch.square(diff_P), dim=1, keepdim=True)
#         # diff_Q = torch.mean(torch.square(diff_Q), dim=1, keepdim=True)
#         diff_P = torch.sqrt(torch.mean(torch.square(diff_P), dim=1, keepdim=True))
#         diff_Q = torch.sqrt(torch.mean(torch.square(diff_Q), dim=1, keepdim=True))
#         # diff_Q = diff_Q * LAMBDA
#         # diff_Q = diff_Q * 0.0
#         # diff_P[:, 50:550], diff_Q[:, 50:550] = diff_P[:, 50:550] * LAMBDA, diff_Q[:, 50:550] * LAMBDA
#         # diff_P[:, 0:50], diff_P[:, 550:] = diff_P[:, 0:50] * LAMBDA, diff_P[:, 550:] * LAMBDA
#         # diff_Q[:, 50:] = diff_Q[:, 50:] * LAMBDA
#         # diff_P[:, 0:50]  = diff_P[:, 0:50] * LAMBDA 

#         diff = torch.cat([diff_P, diff_Q], dim=1)

#         diff = diff * LAMBDA
#         log_prob_RMSE = -1.0 * torch.mean(diff, dim=1, keepdim=True)
#         # log_prob_RMSE_Hubber = -1.0 * nn.HuberLoss(log_prob_RMSE, delta=Delta)
#         log_prob_RMSE_Hubber = -1.0 * nn.functional.huber_loss(log_prob_RMSE, torch.zeros_like(log_prob_RMSE), delta=Delta)
        
#         # Compute logN(\bar{x}, \sigma) - log probability of input under standard normal
#         diff_x = x - self.default_x
#         log_prob_prior = self.prior_normal.log_prob(diff_x).sum(dim=1, keepdim=True)
        
#         # Return total unnormalized log probability
#         # return log_prob_RMSE_Hubber
#         # return log_prob_RMSE_Hubber + log_prob_prior
#         return log_prob_RMSE + log_prob_prior
#         # return log_prob_data + log_prob_prior
#         # return log_prob_data 


#     # def conditional_unnorm_log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     #     """
#     #     Compute the unnormalized log probability density function conditioned on y:
#     #     logN(y - mu(x), I) + logN(x̄, I)
        
#     #     This method is used for training diffusion models where y is fixed and only x is perturbed.
        
#     #     Args:
#     #         x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
#     #         y (torch.Tensor): Output tensor of shape (batch_size, output_dim) - the conditioning values
            
#     #     Returns:
#     #         torch.Tensor: Unnormalized log probability of shape (batch_size, 1)
#     #     """
#     #     x_in = x
        
#     #     # Get predictions from GP model
#     #     with torch.no_grad():
#     #         mu_P, mu_Q = self.gp_model(x_in)
        
#     #     # Concatenate P and Q predictions to match y shape
#     #     mu = torch.cat([mu_P, mu_Q], dim=1)
        
#     #     # Compute logN(y - mu(x), I) - this is a multivariate normal with identity covariance
#     #     # Which is equivalent to sum of independent standard normal distributions of (y - mu(x))
#     #     diff = y - mu
#     #     diff = diff * LAMBDA 
#     #     log_prob_data = self.standard_normal.log_prob(diff).sum(dim=1, keepdim=True)
        
#     #     # Compute logN(\bar{x}, I) - log probability of input under standard normal
#     #     diff_x = x_in - self.default_x
#     #     log_prob_prior = self.standard_normal.log_prob(diff_x).sum(dim=1, keepdim=True)
        
#     #     # Return total unnormalized log probability
#     #     return log_prob_data + log_prob_prior

#     def _predic_error(self, x: torch.Tensor):
#         # x_in = x
#         y = self.y
#         # Get predictions from GP model
#         with torch.no_grad():
#             mu_P, mu_Q = self.gp_model(x)
        
#         # Remove interval with low predictive ability

#         # Concatenate P and Q predictions to match y shape
#         # mu = torch.cat([mu_P, mu_Q], dim=1)
#         # y = y.unsqueeze(0).expand(mu.shape[0], -1)
#         y = y.unsqueeze(0)


#         y_1, y_2 = y[:, :600], y[:, 600:] 


#         # y = torch.cat([y_1, y_2], dim=1)

#         # Compute logN(y - mu(x), I) - this is a multivariate normal with identity covariance
#         # Which is equivalent to sum of independent standard normal distributions of (y - mu(x))
#         # Use mean squared error instead
#         # diff = torch.sqrt(torch.mean(torch.square(y - mu)))
#         # print()
#         # print(mu_P.shape, y_1.shape)
#         # print(torch.sqrt(torch.mean((mu_P[0] -  y_1[0])**2)))
#         diff_P = torch.sqrt(torch.mean(torch.square(mu_P - y_1), dim=1, keepdim=True))
#         diff_Q = torch.sqrt(torch.mean(torch.square(mu_Q - y_2), dim=1, keepdim=True))
        
#         return diff_P, diff_Q

#         # return diff_P, diff_Q, mu_P, mu_Q, y_1, y_2
    
#     # def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
#     #     fn = lambda t, x: self.unnorm_log_prob(x)
#     #     div, outputs = compute_divx(fn, x, *args, **kwargs)
#     #     return div
    
#     def _initialize_distr(self):
#         """
#         Reinitialize distributions when transferring between devices.
#         """
#         # Make sure the GP model is moved to the correct device
#         device = self.default_x.device
#         self.gp_model = self.gp_model.to(device)
#         # Reinitialize the standard normal distribution on the correct device
#         self.standard_normal = torch.distributions.Normal(
#             torch.tensor(0.0, device=device), 
#             torch.tensor(1.0, device=device)
#         )

#     def to(self, device, *args, **kwargs):
#         """
#         Move the distribution to the specified device.
#         """
#         # Move the parent distribution
#         super().to(device, *args, **kwargs)
        
#         # Move the GP model and reinitialize distributions
#         self._initialize_distr()
         
#         # Reinitialize prior normal distribution
#         self.prior_normal = torch.distributions.Normal(
#             torch.tensor(0.0, device=device), 
#             torch.tensor(1.0, device=device)
#         )
#         self.gp_model = self.gp_model.to(device)
        
#         return self

