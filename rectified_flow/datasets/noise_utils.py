#######################################################################################
#
# Utils for generating the D0 (noise distributions)
#
#######################################################################################

import torch
import torch.nn as nn
import torch.distributions as dist


# Circular GMM Class
class CircularGMM(dist.MixtureSameFamily):
    def __init__(self, n_components=6, radius=10, dim=2, std=1.0, device=torch.device("cpu")):
        self.device = device
        angles = torch.linspace(0, 2 * torch.pi, n_components + 1)[:-1].to(device)
        means = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1).to(device)
        stds = std * torch.ones(n_components, dim).to(device)
        weights = torch.ones(n_components).to(device) / n_components

        # Initialize the MixtureSameFamily distribution
        super().__init__(dist.Categorical(weights), dist.Independent(dist.Normal(means, stds), 1))


# Two-point GMM Class
class TwoPointGMM(dist.MixtureSameFamily):
    def __init__(self, x=10.0, y=10.0, std=1.0, device=torch.device("cpu")):
        self.device = device
        means = torch.tensor([[x, y], [x, -y]]).to(device)
        stds = torch.ones(2, 2).to(device) * std
        weights = torch.ones(2).to(device) / 2

        # Initialize the MixtureSameFamily distribution
        super().__init__(dist.Categorical(weights), dist.Independent(dist.Normal(means, stds), 1))