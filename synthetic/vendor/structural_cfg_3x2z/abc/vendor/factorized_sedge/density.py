from __future__ import annotations
import math
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from .networks import ResidualMLP

class ConditionalGaussianMixture(nn.Module):
    """Diagonal-Gaussian MDN implementing ``log p(target | context)``."""

    def __init__(
        self,
        context_dim: int,
        target_dim: int,
        components: int,
        hidden_dim: int,
        depth: int,
        min_scale: float,
    ) -> None:
        super().__init__()
        if min(context_dim, target_dim, components) <= 0:
            raise ValueError("context_dim, target_dim, and components must be positive")
        if min_scale <= 0:
            raise ValueError("min_scale must be positive")
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.components = components
        self.min_scale = float(min_scale)
        output_dim = components * (1 + 2 * target_dim)
        self.network = ResidualMLP(context_dim, output_dim, hidden_dim, depth)

    def parameters_for(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError(
                f"context must have shape [N, {self.context_dim}], got {tuple(context.shape)}"
            )
        output = self.network(context)
        logits_end = self.components
        means_end = logits_end + self.components * self.target_dim
        logits = output[:, :logits_end]
        means = output[:, logits_end:means_end].reshape(-1, self.components, self.target_dim)
        raw_scales = output[:, means_end:].reshape(-1, self.components, self.target_dim)
        log_weights = F.log_softmax(logits, dim=-1)
        scales = F.softplus(raw_scales) + self.min_scale
        return log_weights, means, scales

    def log_prob(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        if target.ndim != 2 or target.shape != (context.shape[0], self.target_dim):
            raise ValueError(
                f"target must have shape [{context.shape[0]}, {self.target_dim}], "
                f"got {tuple(target.shape)}"
            )
        log_weights, means, scales = self.parameters_for(context)
        standardized = (target[:, None, :] - means) / scales
        component_log_prob = (
            -0.5 * standardized.square()
            - scales.log()
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        result = torch.logsumexp(log_weights + component_log_prob, dim=-1)
        return result

    @torch.no_grad()
    def diagnostics(self, context: torch.Tensor) -> dict[str, Any]:
        log_weights, _, scales = self.parameters_for(context)
        average_weights = log_weights.exp().mean(dim=0)
        return {
            "min_scale": float(scales.min()),
            "median_scale": float(scales.median()),
            "max_scale": float(scales.max()),
            "collapsed_components": int((average_weights < 1e-3).sum()),
            "component_weights": average_weights.detach().cpu().tolist(),
        }

