"""Token cross-entropy plus 0.1 Gaussian property NLL, computed in FP32."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

PROPERTY_WEIGHT = 0.1


def _next_token_view(logits: torch.Tensor, sequences: torch.Tensor):
    """Align logits[:, :-1] with next-token targets sequences[:, 1:]."""
    return logits[:, :-1], sequences[:, 1:]


def token_cross_entropy(logits: torch.Tensor, sequences: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Average next-token cross-entropy over non-PAD targets, including EOS."""
    lg, tgt = _next_token_view(logits, sequences)
    lg = lg.float()                                     # fp32
    return F.cross_entropy(
        lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
        ignore_index=pad_id, reduction="mean",
    )


def property_nll(mu: torch.Tensor, sigma: torch.Tensor, z_true: torch.Tensor,
                 sequences: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Gaussian NLL including log(sigma) and the normalization constant.

    Average over the same non-PAD next-token positions as token loss,
    giving both property heads equal weight."""
    mu, sigma = mu[:, :-1].float(), sigma[:, :-1].float()
    tgt_valid = (sequences[:, 1:] != pad_id)
    z = z_true.unsqueeze(1).float()                     # (B,1,2)
    nll = 0.5 * math.log(2 * math.pi) + torch.log(sigma) + 0.5 * ((z - mu) / sigma) ** 2
    m = tgt_valid.unsqueeze(-1)
    denom = m.sum() * nll.size(-1)
    if denom == 0:
        return nll.sum() * 0.0
    return (nll * m).sum() / denom


def total_loss(out: dict, sequences: torch.Tensor, z_true: torch.Tensor, pad_id: int):
    """Return the total loss and separately recorded loss components."""
    l_token = token_cross_entropy(out["logits"], sequences, pad_id)
    l_prop = property_nll(out["property_mu"], out["property_sigma"], z_true, sequences, pad_id)
    total = l_token + PROPERTY_WEIGHT * l_prop
    return total, {
        "loss_total": float(total.detach()),
        "loss_token": float(l_token.detach()),
        "loss_property": float(l_prop.detach()),
        "property_weight": PROPERTY_WEIGHT,
    }
