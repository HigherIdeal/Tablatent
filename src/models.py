from __future__ import annotations

import torch
from torch import nn


def _mlp(dims: list[int], dropout: float, final_activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(dims) - 2 or final_activation:
            layers.extend([nn.LayerNorm(right), nn.GELU(), nn.Dropout(dropout)])
    return nn.Sequential(*layers)


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float):
        super().__init__()
        self.encoder = _mlp([input_dim, *hidden_dims, latent_dim], dropout, final_activation=False)
        self.decoder = _mlp([latent_dim, *reversed(hidden_dims), input_dim], dropout, final_activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class ProbabilityHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        self.network = _mlp([input_dim, *hidden_dims, 1], dropout, final_activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

