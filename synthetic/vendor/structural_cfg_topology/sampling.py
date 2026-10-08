"""Target-keyed diffusion and bounded multi-specification SIR, with no fallback."""

import math
import time

import numpy as np
import torch

from .data import Normalizer
from .graphs import graph
from .runtime import seed_for


def random_values(cohort, target_id, draw_id, purpose, step, stop, dimension, *, normal=True):
    seed = seed_for(cohort["graph"], cohort["seed"]+81, int(target_id), int(draw_id), purpose, step)
    rng = np.random.default_rng(seed)
    shape = (stop, dimension)
    return rng.standard_normal(shape) if normal else rng.random(shape)


def noise_arrays(cohort, ids, draw_ids, purpose, dimension, steps=100, start=0, stop=1):
    # Each step has a prefix-stable proposal stream, independent of target chunks.
    result = np.empty((steps, len(ids), stop-start, dimension), dtype=np.float32)
    for step in range(steps):
        for i, (target, draw) in enumerate(zip(ids, draw_ids)):
            result[step, i] = random_values(cohort, target, draw, purpose, step, stop, dimension)[start:stop]
    return torch.from_numpy(result.reshape(steps, -1, dimension))


@torch.no_grad()
def ddpm_samples(model, context, noises):
    parameter = next(model.parameters())
    context = context.to(device=parameter.device, dtype=parameter.dtype)
    noises = noises.to(device=parameter.device, dtype=parameter.dtype)
    if noises.shape != (model.steps, len(context), model.d):
        raise ValueError("Invalid canonical DDPM noise shape")
    # Noise row 0 initializes X_T; rows 1..T-1 drive the stochastic steps.
    x = noises[0].clone()
    for offset, step in enumerate(reversed(range(model.steps)), 1):
        t = torch.full((len(x),), step, dtype=torch.long, device=x.device)
        eps = model.predict_noise(x, t, context)
        x = (x-model.betas[step]*eps/(1-model.alpha_bars[step]).sqrt())/model.alphas[step].sqrt()
        if step:
            x = x+model.posterior_variance[step].sqrt()*noises[offset]
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"Nonfinite DDPM step {step}")
    return x.cpu()


def gaussian_log_prob(model, target, context):
    p = next(model.parameters())
    weights, means, scales = model.parameters_for(context.to(device=p.device, dtype=p.dtype))
    weights, means, scales = weights.double(), means.double(), scales.double()
    error = (target.to(device=p.device, dtype=torch.float64)[:, None, :]-means)/scales
    component = (-.5*error.square()-scales.log()-.5*math.log(2*math.pi)).sum(-1)
    return torch.logsumexp(weights+component, -1)


def importance_summary(log_weights):
    if log_weights.ndim != 1 or not torch.isfinite(log_weights).all():
        raise FloatingPointError("Nonfinite importance log weights")
    weights = torch.softmax(log_weights.double(), 0)
    if not torch.isfinite(weights).all() or float(weights.sum()) <= 0:
        raise FloatingPointError("Invalid normalized importance weights")
    ess = float(1/weights.square().sum())
    return weights, ess


def resample_one(weights, uniform):
    cdf = weights.detach().cpu().double().cumsum(0)
    cdf[-1] = 1.
    return int(torch.searchsorted(cdf, torch.tensor(uniform, dtype=torch.float64), right=False))


def numeric_summary(value):
    value = value.detach().cpu().double().flatten()
    finite = value[torch.isfinite(value)]
    return {"count": len(value), "nonfinite": len(value)-len(finite),
            "mean": float(finite.mean()) if len(finite) else None,
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None}


@torch.no_grad()
def central_proposals(cohort, models, method, z, ids, draws, start, stop, cfg):
    g = graph(cohort["graph"])
    parameter = next(next(iter(models.values())).parameters())
    device = parameter.device
    count, batch = stop-start, len(ids)
    target = z.to(device=device, dtype=parameter.dtype)
    expanded_z = target.repeat_interleave(count, 0)
    factor_logs = {}
    if method != "neural21":
        raise ValueError(method)
    model = models["anchor_1"]
    noises = noise_arrays(cohort, ids, draws, "anchor_1", model.d, cfg.timesteps, start, stop)
    proposed = ddpm_samples(model, expanded_z[:, :1], noises).to(device)
    c = proposed[:, :len(g.separator)]
    logw = torch.zeros(len(c), dtype=torch.float64, device=device)
    for j in range(1, g.z_dim):
        component_log = gaussian_log_prob(models[f"likelihood_{j+1}"], expanded_z[:, j:j+1], c)
        factor_logs[f"likelihood_{j+1}"] = component_log
        logw += component_log
    return (proposed.reshape(batch, count, -1).cpu(), logw.reshape(batch, count).cpu(),
            {key: value.reshape(batch, count).cpu() for key, value in factor_logs.items()})


def isolated_proposals(cohort, models, method, z, ids, draws, active, start, stop, cfg):
    def call(indices):
        return central_proposals(cohort, models, method, z[indices], [ids[i] for i in indices],
                                 [draws[i] for i in indices], start, stop, cfg)
    try:
        proposed, logw, logs = call(active)
        for row, i in enumerate(active):
            yield i, proposed[row], logw[row], {key: value[row] for key, value in logs.items()}, None
    except (FloatingPointError, torch.linalg.LinAlgError):
        for i in active:
            try:
                proposed, logw, logs = call([i])
                yield i, proposed[0], logw[0], {key: value[0] for key, value in logs.items()}, None
            except (FloatingPointError, torch.linalg.LinAlgError) as error:
                yield i, None, None, {}, str(error)


@torch.no_grad()
def formula_samples(cohort, arm, method, models, rows, draws, cfg):
    g = graph(cohort["graph"])
    norm = Normalizer(**cohort["normalizers"][arm])
    z = norm.normalize_z(rows["z"]).double()
    ids = rows["ids"].tolist()
    output = torch.full((len(ids), g.x_dim), float("nan"), dtype=torch.float64)
    valid = torch.ones(len(ids), dtype=torch.bool)
    diagnostics = [{"target_id": int(tid), "draw_id": int(draw)} for tid, draw in zip(ids, draws)]
    central = torch.empty(len(ids), len(g.separator), dtype=torch.float64)
    anchor_private = None
    if g.separator:
        picked = torch.full((len(ids), len(g.separator)+(len(g.private[0]) if method == "neural21" else 0)), float("nan"), dtype=torch.float64)
        for batch_start in range(0, len(ids), cfg.target_batch):
            active = list(range(batch_start, min(batch_start+cfg.target_batch, len(ids))))
            values, weights = {i: [] for i in active}, {i: [] for i in active}
            factor_history = {i: {} for i in active}
            start, stop = 0, cfg.candidate_initial
            while active:
                remaining = []
                for i, proposed, logw, logs, error in isolated_proposals(cohort, models, method, z, ids, draws, active, start, stop, cfg):
                    if error is not None:
                        valid[i] = False
                        diagnostics[i].update(failure="nonfinite_proposals", error=str(error), proposals=stop)
                        continue
                    values[i].append(proposed.double())
                    weights[i].append(logw.double())
                    for key, value in logs.items():
                        factor_history[i].setdefault(key, []).append(value)
                    diagnostics[i]["proposal_log_factors"] = {key: numeric_summary(torch.cat(parts))
                                                              for key, parts in factor_history[i].items()}
                    try:
                        pool = torch.cat(values[i])
                        if not torch.isfinite(pool).all():
                            raise FloatingPointError("Nonfinite proposal states")
                        w, ess = importance_summary(torch.cat(weights[i]))
                    except FloatingPointError as error:
                        valid[i] = False
                        diagnostics[i].update(failure="nonfinite_weights", error=str(error), proposals=stop)
                        continue
                    diagnostics[i].update(proposals=stop, ess=ess, ess_fraction=ess/stop,
                                          low_fraction=ess/stop < .05)
                    if ess >= cfg.target_ess:
                        u = random_values(cohort, ids[i], draws[i], method+"_resample", 0, 1, 1, normal=False)[0, 0]
                        chosen = resample_one(w, u)
                        picked[i] = pool[chosen]
                        diagnostics[i]["proposal_index"] = chosen
                    elif stop == cfg.candidate_max:
                        valid[i] = False
                        diagnostics[i]["failure"] = "ess_budget_exhausted"
                    else:
                        remaining.append(i)
                active = remaining
                start, stop = stop, min(2*stop, cfg.candidate_max)
        central = picked[:, :len(g.separator)]
        if method == "neural21":
            anchor_private = picked[:, len(g.separator):]
        output[:, g.separator] = central
    else:
        for row in diagnostics:
            row.update(proposals=0, ess=None, coincident_formulas=True)
    indices = torch.nonzero(valid, as_tuple=True)[0]
    if len(indices):
        for j, private in enumerate(g.private):
            indices = torch.nonzero(valid, as_tuple=True)[0]
            if not len(indices):
                break
            if not private:
                continue
            if j == 0 and anchor_private is not None:
                output[indices[:, None], torch.tensor(private)] = anchor_private[indices]
                continue
            component = f"private_{j+1}"
            context = torch.cat((central[indices], z[indices, j:j+1]), 1)
            noises = noise_arrays(cohort, [ids[i] for i in indices], [draws[i] for i in indices], component,
                                  len(private), cfg.timesteps)
            try:
                sampled = ddpm_samples(models[component], context, noises).cpu().double()
                if not torch.isfinite(sampled).all():
                    raise FloatingPointError("Nonfinite private samples")
                output[indices[:, None], torch.tensor(private)] = sampled
            except (FloatingPointError, torch.linalg.LinAlgError):
                for k, i in enumerate(indices.tolist()):
                    try:
                        sample = ddpm_samples(models[component], context[k:k+1], noises[:, k:k+1]).cpu().double()
                        if not torch.isfinite(sample).all():
                            raise FloatingPointError("Nonfinite private samples")
                        output[i, torch.tensor(private)] = sample[0]
                    except (FloatingPointError, torch.linalg.LinAlgError) as error:
                        valid[i] = False
                        diagnostics[i].update(failure="private_sampler_failure", component=component, error=str(error))
    output = norm.inverse_x(output)
    valid &= torch.isfinite(output).all(1)
    return {"x": output, "valid": valid, "diagnostics": diagnostics}


@torch.no_grad()
def generate(cohort, arm, method, models, rows, cfg, draw_ids=None):
    if "x" in rows:
        raise ValueError("Generation must not receive reference X")
    began = time.monotonic()
    ids = rows["ids"].tolist()
    draws = [0]*len(ids) if draw_ids is None else list(draw_ids)
    if len(ids) != len(draws) or len(set(zip(ids, draws))) != len(ids):
        raise ValueError("Each output requires a unique (target ID, draw ID)")
    if not torch.isfinite(rows["z"]).all():
        raise ValueError("Nonfinite target Z")
    if method == "neural21":
        result = formula_samples(cohort, arm, method, models, rows, draws, cfg)
    else:
        if method not in ("structural_cfg", "direct_cd"):
            raise ValueError(method)
        norm = Normalizer(**cohort["normalizers"][arm])
        z = norm.normalize_z(rows["z"]).float()
        model = models[method]
        noise = noise_arrays(cohort, ids, draws, "common_full_x", model.d, cfg.timesteps)
        values, valid, diagnostics = [], [], []
        def call(index):
            return ddpm_samples(model, z[index], noise[:, index]).double()
        for start in range(0, len(ids), 256):
            indices = list(range(start, min(start+256, len(ids))))
            try:
                values.append(call(indices))
                valid.extend([True]*len(indices))
            except (FloatingPointError, torch.linalg.LinAlgError):
                for i in indices:
                    try:
                        values.append(call([i]))
                        valid.append(True)
                    except (FloatingPointError, torch.linalg.LinAlgError) as error:
                        values.append(torch.full((1, model.d), float("nan"), dtype=torch.float64))
                        valid.append(False)
                        diagnostics.append(dict(target_id=ids[i], draw_id=draws[i], failure=str(error)))
        result = dict(x=norm.inverse_x(torch.cat(values)), valid=torch.tensor(valid), diagnostics=diagnostics)
    result.update(target_ids=rows["ids"].clone(), draw_ids=torch.tensor(draws),
                  elapsed_seconds=time.monotonic()-began)
    return result

