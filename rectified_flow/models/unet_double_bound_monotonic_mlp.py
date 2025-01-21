import numpy as np
import torch
import os
import json
import torch.nn as nn

from torch.nn.functional import silu
from dataclasses import dataclass, asdict, field
from rectified_flow.utils import match_dim_with_data
from rectified_flow.models.unet import SongUNet, SongUNetConfig

class MonotonicLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MonotonicLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        # Apply the absolute value to the weights to ensure non-negativity
        weight = torch.abs(self.linear.weight)
        return nn.functional.linear(input, weight, self.linear.bias)


class MLP(nn.Module):
    def __init__(self, layer_sizes, activation, linear_layer):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(linear_layer(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PathMLP(nn.Module):
    def __init__(self, layer_sizes):
        super(PathMLP, self).__init__()
        self.mlp_plus = MLP(layer_sizes, activation=nn.ReLU, linear_layer=MonotonicLinear)
        self.mlp_minus = MLP(layer_sizes, activation=nn.ReLU, linear_layer=MonotonicLinear)

    def forward(self, t):
        """
        Args:
            t: Input time points of shape (batch_size,)

        Returns:
            y: alpha_t, of shape (batch_size, mlp_out_dim)
        """
        t = torch.cat([torch.tensor([0.0], device=t.device), # (batch_size + 2, 1), with [0, 1] at the beginning
                       torch.tensor([1.0], device=t.device), t], dim=0).unsqueeze(1)  
        y = self.mlp_plus(t) - self.mlp_minus(-t)  # (batch_size + 2, mlp_out_dim), monotonic and concave
        y = (y[2:] - y[0]) / (y[1] - y[0] + 1e-10) # (batch_size, mlp_out_dim), normalize to [0, 1]
        return y


class DoubleBoundMonotonicMLP(nn.Module):
    def __init__(
        self,
        config: SongUNetConfig,
        data_mean: torch.Tensor,
        use_monotonics: bool = True,
    ):
        super().__init__()
        self.unet = SongUNet(config)
        if use_monotonics:
            # The output shape of PathMLP is [B, 1]
            self.decreasing_path = PathMLP([1, 128, 256, 128, 1]) # data_mean - x_t
            self.increasing_path = PathMLP([1, 128, 256, 128, 1]) # x_t
        else:
            self.decreasing_mlp = MLP([1, 128, 256, 128, 1], activation=nn.ReLU, linear_layer=nn.Linear)
            self.increasing_mlp = MLP([1, 128, 256, 128, 1], activation=nn.ReLU, linear_layer=nn.Linear)
            self.decreasing_path = lambda t: torch.cos(torch.pi / 2. * t) + t * (1. - t) * self.decreasing_mlp(t.unsqueeze(1)).squeeze(-1)
            self.increasing_path = lambda t: torch.sin(torch.pi / 2. * t) + t * (1. - t) * self.increasing_mlp(t.unsqueeze(1)).squeeze(-1)

        self.data_mean = data_mean

    def forward(self, x_t, time, class_labels=None, augment_labels=None):
        model_t = self.unet(x_t, time, class_labels, augment_labels)
        decreasing_coeff = 1. - self.decreasing_path(time).squeeze(-1)
        increasing_coeff = self.increasing_path(time).squeeze(-1)

        decreasing_coeff = match_dim_with_data(decreasing_coeff, x_t.shape, x_t.device, x_t.dtype)
        increasing_coeff = match_dim_with_data(increasing_coeff, x_t.shape, x_t.device, x_t.dtype)

        return (
            (self.data_mean - x_t) * decreasing_coeff +
            x_t * increasing_coeff + 
            model_t * (decreasing_coeff.detach() * increasing_coeff.detach())
        )
