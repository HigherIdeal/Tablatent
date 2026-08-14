from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _mlp(dims: list[int], dropout: float, final_activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(left, right))
        is_last = i == len(dims) - 2
        if (not is_last) or final_activation:
            layers.extend([nn.LayerNorm(right), nn.GELU(), nn.Dropout(dropout)])
    return nn.Sequential(*layers)


def embedding_dim(cardinality: int, max_dim: int) -> int:
    return max(2, min(max_dim, int(round(1.6 * math.sqrt(max(cardinality, 2))))))


class ContextAutoencoder(nn.Module):
    """Categorical embedding + numeric input -> latent -> mixed reconstruction."""

    def __init__(
        self,
        cardinalities: list[int],
        numeric_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        embedding_dim_max: int,
        dropout: float,
    ):
        super().__init__()
        self.cardinalities = list(cardinalities)
        self.numeric_dim = int(numeric_dim)

        emb_dims = [embedding_dim(c, embedding_dim_max) for c in self.cardinalities]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, dim) for cardinality, dim in zip(self.cardinalities, emb_dims)]
        )
        input_dim = sum(emb_dims) + self.numeric_dim

        self.encoder = _mlp([input_dim, *hidden_dims, latent_dim], dropout, final_activation=False)
        self.decoder_trunk = _mlp(
            [latent_dim, *reversed(hidden_dims)], dropout, final_activation=True
        )
        decoder_dim = hidden_dims[0] if hidden_dims else latent_dim
        self.category_heads = nn.ModuleList(
            [nn.Linear(decoder_dim, cardinality) for cardinality in self.cardinalities]
        )
        self.numeric_head = nn.Linear(decoder_dim, self.numeric_dim)

    def encode(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        embedded = [layer(categorical[:, i]) for i, layer in enumerate(self.embeddings)]
        return self.encoder(torch.cat([*embedded, numeric], dim=1))

    def decode(self, z: torch.Tensor):
        h = self.decoder_trunk(z)
        return [head(h) for head in self.category_heads], self.numeric_head(h)

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor):
        return self.decode(self.encode(categorical, numeric))


class HistoryAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float):
        super().__init__()
        self.encoder = _mlp([input_dim, *hidden_dims, latent_dim], dropout, final_activation=False)
        self.decoder = _mlp(
            [latent_dim, *reversed(hidden_dims), input_dim], dropout, final_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def context_reconstruction_loss(
    cat_logits: list[torch.Tensor],
    target_cat: torch.Tensor,
    pred_num: torch.Tensor,
    target_num: torch.Tensor,
    categorical_weight: float,
    numeric_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if cat_logits:
        cat_losses = [
            F.cross_entropy(logits, target_cat[:, i])
            for i, logits in enumerate(cat_logits)
        ]
        cat_loss = torch.stack(cat_losses).mean()
    else:
        cat_loss = pred_num.new_tensor(0.0)
    num_loss = F.smooth_l1_loss(pred_num, target_num)
    total = categorical_weight * cat_loss + numeric_weight * num_loss
    return total, {
        "categorical": float(cat_loss.detach()),
        "numeric": float(num_loss.detach()),
    }
