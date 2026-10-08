"""Small residual MLPs for low-dimensional continuous variables."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.dropout = dropout

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.norm(value)
        residual = F.silu(self.linear1(residual))
        residual = F.dropout(self.linear2(residual), p=self.dropout, training=self.training)
        return value + residual


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, hidden_dim, depth) <= 0:
            raise ValueError("all network dimensions and depth must be positive")
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(ResidualBlock(hidden_dim, dropout) for _ in range(depth))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.input(value))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.output_norm(hidden))


__all__ = ["ResidualMLP"]

