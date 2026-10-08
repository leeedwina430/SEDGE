"""Frozen TRAIN kernels, complete-panel scoring and exact-target diagnostics."""

import math

import numpy as np
import torch

from .data import Normalizer, subset
from .graphs import graph
from .runtime import seed_for


def mmd_squared(a, b, kernel):
    a, b = a.double(), b.double()
    if a.shape != b.shape or len(a) < 2 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("MMD requires complete finite paired rows")
    def distances(x, y):
        return (x.square().sum(1)[:, None]+y.square().sum(1)[None, :]-2*x@y.T).clamp_min(0)
    xx, yy, xy = distances(a, a), distances(b, b), distances(a, b)
    xx.fill_diagonal_(0)
    yy.fill_diagonal_(0)
    xy.diagonal().copy_((a-b).square().sum(1))
    raw, paired = [], []
    for multiplier in kernel["multipliers"]:
        gamma = 1/(2*(kernel["base_bandwidth"]*multiplier)**2)
        difference = (-gamma*xx).exp()+(-gamma*yy).exp()-(-gamma*xy).exp()-(-gamma*xy.T).exp()
        raw.append(float(difference.mean()))
        paired.append(float((difference.sum()-difference.diag().sum())/(len(a)*(len(a)-1))))
    if min(raw) < -1e-12:
        raise FloatingPointError("Material negative MMD-squared V-statistic")
    return {"mmd2_v": float(np.mean(np.maximum(raw, 0))), "mmd2_v_raw": float(np.mean(raw)),
            "mmd2_paired_u": float(np.mean(paired)), "mmd2_v_per_bandwidth_raw": raw,
            "roundoff_corrections": sum(value < 0 for value in raw)}


def ranked_indices(cohort, rows, purpose, count):
    return sorted(range(len(rows["ids"])), key=lambda i: seed_for(cohort["graph"], cohort["seed"], purpose, int(rows["ids"][i])))[:count]


def score(cohort, rows, result, *, full=True):
    valid = result["valid"] & torch.isfinite(result["x"]).all(1)
    base = {"count": len(valid), "valid_count": int(valid.sum()), "failed_count": int((~valid).sum()),
            "sampling_seconds": result["elapsed_seconds"]}
    if not bool(valid.all()):
        return {**base, "status": "incomplete", "mmd2_v": None, "mmd2_paired_u": None}
    g = graph(cohort["graph"])
    x, ref, z = result["x"].double(), rows["x"].double(), rows["z"].double()
    norm = Normalizer(**cohort["metric_state"])
    ax, bx, zz = norm.normalize_x(x), norm.normalize_x(ref), norm.normalize_z(z)
    metric = mmd_squared(torch.cat((ax, zz), 1), torch.cat((bx, zz), 1), cohort["kernel"])
    values = {**base, "status": "complete", **metric}
    if not full:
        return values
    for label, settings in cohort["kernel"]["subspaces"].items():
        dims = settings["dims"]
        result_mmd = mmd_squared(ax[:, dims], bx[:, dims], {**settings, "multipliers": cohort["kernel"]["multipliers"]})
        values[f"mmd2_{label}"] = result_mmd["mmd2_v"]
    for j in range(g.x_dim):
        values[f"w1_x{j+1}"] = float((torch.sort(x[:, j]).values-torch.sort(ref[:, j]).values).abs().mean())
    values["x_mean_error_norm"] = float((ax.mean(0)-bx.mean(0)).norm())
    values["x_covariance_error_norm"] = float((torch.cov(ax.T)-torch.cov(bx.T)).norm())
    directions = np.random.default_rng(seed_for(g.name, cohort["seed"], "swd_directions")).standard_normal((512, g.x_dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    selected = ranked_indices(cohort, rows, "swd_rows", min(800, len(x)))
    projections = np.sort(x[selected].numpy()@directions.T, axis=0)-np.sort(ref[selected].numpy()@directions.T, axis=0)
    values["swd_x_rms"] = float(np.sqrt(np.square(projections).mean()))
    excess = torch.maximum((-x).clamp_min(0), (x-1).clamp_min(0))
    values.update(support_rate=float((excess == 0).all(1).double().mean()),
                  support_rate_tolerance_1e6=float((excess <= 1e-6).all(1).double().mean()),
                  support_excess_mean=float(excess.mean()), support_excess_max=float(excess.max()))
    calibration = cohort["contract"]["calibration"]
    for label, value in (("generated", x), ("reference", ref)):
        signal = torch.stack([value[:, parents].square().sum(1) for parents in g.parents], 1)
        means, scales = torch.tensor(calibration["mean"], dtype=torch.float64), torch.tensor(calibration["scale"], dtype=torch.float64)
        residual = (z-(signal-means)/scales)/.2
        for j in range(g.z_dim):
            values[f"{label}_residual{j+1}_mean"] = float(residual[:, j].mean())
            values[f"{label}_residual{j+1}_std"] = float(residual[:, j].std(correction=0))
    if not all(math.isfinite(v) for v in values.values() if isinstance(v, float)):
        raise FloatingPointError("Nonfinite full-panel metric")
    return values


def repeated_rows(cohort, rows, cfg):
    chosen = ranked_indices(cohort, rows, "repeated_targets", min(cfg.repeated_targets, len(rows["ids"])))
    index = torch.tensor(chosen).repeat_interleave(cfg.repeated_draws)
    expanded = subset(rows, index)
    draws = list(range(1, cfg.repeated_draws+1))*len(chosen)
    return chosen, expanded, draws


def score_repeated(cohort, original, chosen, generated, cfg):
    norm = Normalizer(**cohort["metric_state"])
    x = norm.normalize_x(generated["x"].double()).reshape(len(chosen), cfg.repeated_draws, -1)
    valid = (generated["valid"] & torch.isfinite(generated["x"]).all(1)).reshape(len(chosen), cfg.repeated_draws).all(1)
    reference = norm.normalize_x(original["x"][chosen].double())
    records = []
    for i, source in enumerate(chosen):
        row = {"target_id": int(original["ids"][source]), "valid": bool(valid[i])}
        if bool(valid[i]):
            samples, m = x[i], cfg.repeated_draws
            energy = (samples-reference[i]).norm(dim=1).mean()-torch.pdist(samples).sum()/(m*(m-1))
            row.update(energy_score=float(energy), variance=samples.var(0, correction=1).tolist())
        else:
            row.update(energy_score=None, variance=None)
        records.append(row)
    complete = bool(valid.all())
    return {"status": "complete" if complete else "incomplete", "targets": len(chosen),
            "draws": cfg.repeated_draws, "failed_groups": int((~valid).sum()), "groups": records,
            "energy_score": float(np.mean([r["energy_score"] for r in records])) if complete else None}
