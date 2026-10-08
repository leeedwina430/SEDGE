"""Paired 100/1000-step models and source-compatible raw training losses.

The trainer owns minibatch draws.  Diffusion losses then draw a canonical
integer time in [0,1000), Gaussian noise, and (for CFG) a condition mask, in that
order.  This is the old raw trainer's exact random sequence when T=1000.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from factorized_sedge.density import ConditionalGaussianMixture
from raw_diffusion_guidance.common import COUNTS, RECIPE, ROOT, WIDTHS, keyed
from raw_diffusion_guidance.models import DDPM, Forward, NoisyDensity


PARENT = ROOT / "runs/raw_diffusion_guidance_20260913_v1"
COHORTS = ("narrow", "wide", "cov_all")
SEEDS = tuple(range(42, 47))
RAW_ROLES = ("prior", "forward", "density", "cfg", "direct")
CANONICAL_STEPS = 1000




def _recipe(job):
    recipe = dict(RECIPE)
    recipe.update(timesteps=int(job["steps"]), updates=int(job["max_budget"]),
                  lr=float(job["lr"]), validate_every=int(job["validate_every"]))
    if job["ema"] is not None:
        recipe["ema"] = float(job["ema"])
    return recipe


def make_model(job: dict, device="cpu") -> nn.Module:
    """Construct the declared architecture without loading or fitting weights."""
    kind, family = job["kind"], job["family"]
    width = int(job["width"])
    if family == "raw":
        if kind not in RAW_ROLES or job["role"] != kind:
            raise ValueError("Raw role and model kind must agree")
        d, k = (9, 4) if job["cohort"] == "cov_all" else (3, 2)
        if job["steps"] not in ((0,) if kind == "forward" else (100, 1000)):
            raise ValueError("Forward is schedule-independent; other raw models require T=100 or 1000")
    elif family == "neural":
        if kind not in ("neural_ddpm", "neural_mdn"):
            raise ValueError("Unknown Neural21 model kind")
        d, k = len(job["target"]), len(job["context"])
        if d < 1 or (kind == "neural_mdn" and k < 1):
            raise ValueError("Neural factor dimensions are invalid")
        if kind == "neural_ddpm" and job["steps"] not in (100, 1000):
            raise ValueError("Neural DDPM requires T=100 or 1000")
        if kind == "neural_mdn" and (job["steps"] != 0 or job["ema"] is not None):
            raise ValueError("Neural MDN is schedule-independent and does not use EMA")
    else:
        raise ValueError(f"Unknown model family: {family}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(job["init_seed"]))
        if kind == "forward":
            model = Forward(d, k, width)
        elif kind == "density":
            model = NoisyDensity(d, k, width, _recipe(job))
        elif kind == "neural_mdn":
            if width != 128:
                raise ValueError("Neural MDN width is frozen at 128")
            model = ConditionalGaussianMixture(
                context_dim=k, target_dim=d, components=8, hidden_dim=width,
                depth=3, min_scale=.001)
        else:
            model = DDPM(d, k, "direct" if kind == "neural_ddpm" else kind,
                         width, _recipe(job))
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != int(job["parameters"]):
        raise AssertionError(f"Parameter count mismatch for {job['cohort']}/{job['role']}: "
                             f"expected {job['parameters']}, constructed {actual}")
    return model.to(device)


def _columns(x, z, fields):
    columns = []
    for field in fields:
        if len(field) != 2 or field[0] not in ("x", "z"):
            raise ValueError(f"Invalid factor field: {field!r}")
        array, index = (x if field[0] == "x" else z), field[1]
        if not isinstance(index, int) or not 0 <= index < array.shape[1]:
            raise ValueError(f"Invalid factor column: {field!r}")
        columns.append(array[:, index:index+1])
    return torch.cat(columns, dim=1) if columns else x.new_empty((len(x), 0))


def training_data(job: dict, data: dict) -> tuple[torch.Tensor, ...]:
    """Return normalized (train target, context, validation target, context).

Raw forward/density have a special internal interface: x always means complete
normalized X and z means complete normalized Z.  They predict Z from X inside
the loss.  Neural factors use the declared target/context columns instead.
    """
    values = []
    for split in ("train", "validation"):
        x = (data[f"{split}_x"] - data["x_mean"]) / data["x_std"]
        z = (data[f"{split}_z"] - data["z_mean"]) / data["z_std"]
        if job["family"] == "raw":
            values.extend((x, z))
        elif job["family"] == "neural":
            values.extend((_columns(x, z, job["target"]), _columns(x, z, job["context"])))
        else:
            raise ValueError("Unknown training-data family")
    return tuple(values)


def canonical_times(count, steps, *, device, rng):
    """Use a common integer quantile while retaining old T=1000 RNG calls."""
    if steps not in (100, 1000):
        raise ValueError("Only the paired T=100 and T=1000 schedules are defined")
    base = torch.randint(CANONICAL_STEPS, (count,), device=device, generator=rng)
    return torch.div(base * steps, CANONICAL_STEPS, rounding_mode="floor")


def training_loss(model, job, x, z, rng):
    """Loss after the trainer's minibatch draw; no hidden/global random draws."""
    kind = job["kind"]
    if kind == "forward":
        return F.mse_loss(model(x), z)
    if kind == "neural_mdn":
        return -model.log_prob(x, z).mean()
    if int(model.steps) != int(job["steps"]):
        raise ValueError("Model schedule disagrees with the job")
    t = canonical_times(len(x), model.steps, device=x.device, rng=rng)
    noise = torch.randn(x.shape, device=x.device, generator=rng, dtype=x.dtype)
    if kind == "density":
        return -model.log_prob(z, model.corrupt(x, t, noise), t).mean()
    present = ((torch.rand((len(x), 1), device=x.device, generator=rng)
                >= model.recipe["condition_dropout"]).to(x.dtype) if kind == "cfg" else True)
    return F.mse_loss(model.predict_noise(model.corrupt(x, t, noise), t, z, present), noise)


@torch.no_grad()
def validation_values(model, job, vx, vz, vt=None, vn=None) -> dict[str, torch.Tensor]:
    """Per-row validation losses; vt is already mapped to the model schedule.

For exact legacy CFG aggregation, average the conditional and unconditional
vectors separately, then combine their scalar means as .9*conditional+.1*null.
    """
    kind = job["kind"]
    if kind == "forward":
        return {"loss": (model(vx) - vz).square().mean(-1)}
    if kind == "neural_mdn":
        return {"loss": -model.log_prob(vx, vz)}
    if vt is None or vn is None or vn.shape != vx.shape or vt.shape != (len(vx),):
        raise ValueError("Diffusion validation requires aligned fixed times and Gaussian noise")
    noisy = model.corrupt(vx, vt, vn)
    if kind == "density":
        return {"loss": -model.log_prob(vz, noisy, vt)}
    conditional = (model.predict_noise(noisy, vt, vz, True) - vn).square().mean(-1)
    if kind != "cfg":
        return {"loss": conditional}
    unconditional = (model.predict_noise(noisy, vt, vz, False) - vn).square().mean(-1)
    return {"loss": .9 * conditional + .1 * unconditional,
            "conditional": conditional, "unconditional": unconditional}

