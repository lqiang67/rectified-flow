import torch
import torch.nn as nn


# Generic MLP model
class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU, linear_layer=nn.Linear):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(linear_layer(in_dim, out_dim))
            layers.append(activation())
        layers.pop()  # Remove the activation after the last layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.shape[0], -1))


# Base velocity model
class MLPVelocityBase(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super().__init__()
        self.mlp = MLP([input_dim, *hidden_sizes, output_dim])

    def forward(self, x, t, labels=None):
        t = t.view(t.shape[0], -1)
        x = x.view(x.shape[0], -1)
        inputs = [x, t]
        if labels is not None:
            labels = labels.view(labels.shape[0], -1).float()
            inputs.append(labels)
        return self.mlp(torch.cat(inputs, dim=1))


# Model for velocity field v(x, t)
class MLPVelocity(MLPVelocityBase):
    def __init__(self, dim, hidden_sizes=[128, 128, 128], output_dim=None):
        super().__init__(dim + 1, hidden_sizes, output_dim or dim)


# Model for velocity field v(x, t) with label conditioning
class MLPVelocityConditioned(MLPVelocityBase):
    def __init__(self, dim, hidden_sizes=[128, 128, 128], output_dim=None, label_dim=1):
        super().__init__(dim + 1 + label_dim, hidden_sizes, output_dim or dim)