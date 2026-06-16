import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim = 784, hidden_dims = None, num_classes = 10, activation = "relu", use_batchnorm = False, dropout_p = 0.0,):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        layers = [nn.Flatten()]
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(_get_activation(activation))

            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))

        self.network = nn.Sequential(*layers)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.network(x)

def _get_activation(name : str) -> nn.Module:
    name.lower()
    activations = {
        "relu" : nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu" : nn.GELU,
        "elu" : nn.ELU
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation : {name}"
                         f"Choose from {list(activations.keys())}")
    return activations[name]()