"""Neural21 factor jobs and reference-free matched-schedule SIR sampling.

Training jobs share the same factor across anchors, diffusion schedules where
applicable (MDNs), and budget snapshots. This module never loads reference X.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

COHORTS = ("narrow", "wide", "cov_all")
SEEDS = (42, 43, 44, 45, 46)
SCHEDULES = (100, 1000)
BUDGET_LABELS = (20000, 40000)


def _keyed(*parts: Any) -> int:
    encoded = json.dumps(list(parts), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return int(hashlib.sha256(encoded.encode()).hexdigest()[:15], 16) % (2**31 - 1)


def _columns(kind: str, *indices: int) -> list[list]:
    return [[kind, index] for index in indices]


def neural_jobs() -> list[dict]:
    """Return the 130 unique, fresh factor trajectories approved for this study.

    Public budget labels 20k/40k mean DDPM 20k/40k and MDN 10k/20k.
    MDNs have steps=0 and are shared by the two diffusion schedules. A1/A2
    additionally share l3/l4 and p3/p4. Seeds exclude schedule and budget.
    """
    c = _columns
    small = [
        ("g12", "neural_ddpm", c("x", 0, 1), c("z", 0), 109058),
        ("g3", "neural_ddpm", c("x", 2), c("x", 1) + c("z", 1), 108929),
        ("l2", "neural_mdn", c("z", 1), c("x", 1), 103448),
    ]
    separator = c("x", 2, 3, 4)
    large = [
        ("a1", "neural_ddpm", c("x", 0, 2, 3, 4), c("z", 0), 109572),
        ("a2", "neural_ddpm", c("x", 1, 2, 3, 4), c("z", 1), 109572),
        ("p1", "neural_ddpm", c("x", 0), separator + c("z", 0), 109185),
        ("p2", "neural_ddpm", c("x", 1), separator + c("z", 1), 109185),
        ("p3", "neural_ddpm", c("x", 5, 6), separator + c("z", 2), 109442),
        ("p4", "neural_ddpm", c("x", 7, 8), separator + c("z", 3), 109442),
        *[(f"l{i+1}", "neural_mdn", c("z", i), separator, 103704) for i in range(4)],
    ]
    result = []
    for cohort in COHORTS:
        for seed in SEEDS:
            for role, kind, target, context, parameters in (large if cohort == "cov_all" else small):
                diffusion = kind == "neural_ddpm"
                for steps in (SCHEDULES if diffusion else (0,)):
                    result.append(dict(family="neural", cohort=cohort, seed=seed, role=role, kind=kind,
                        steps=steps, width=128, parameters=parameters,
                        base_budget=20000 if diffusion else 10000,
                        max_budget=40000 if diffusion else 20000,
                        lr=.0003 if diffusion else .001, validate_every=250 if diffusion else 100,
                        ema=.999 if diffusion else None,
                        target=[list(v) for v in target], context=[list(v) for v in context],
                        init_seed=_keyed("raw-diffusion-followup-v1", "neural", cohort, seed, role, "initialization"),
                        fit_seed=_keyed("raw-diffusion-followup-v1", "neural", cohort, seed, role, "fit"),
                        parent_dir=None))
    assert len(result) == 130
    assert len({(j["cohort"], j["seed"], j["role"], j["steps"]) for j in result}) == 130
    return result

# The job inventory above is frozen independently from the sampling implementation.
import math
import time
from dataclasses import dataclass
from typing import Callable

import torch

CANONICAL_NOISE_LENGTH = 1001
MAX_MODEL_ROWS = 16384
INITIAL_3X_CANDIDATES = 64
MAXIMUM_3X_CANDIDATES = 4096
TARGET_3X_ESS = 32.
MINIMUM_ESS_FRACTION = .05
FIXED_9X_CANDIDATES = 8192
NOISE_NAMESPACE = "raw-followup-neural-sampling-v1"


def _identities(values) -> list[int | str]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError("Sampling identities must be integer row IDs or explicit pool names")
        result.append(value)
    return result


def canonical_noise(cohort: str, seed: int, split: str, role: str,
                    target_ids: list, draw_indices: list[int], dimension: int,
                    steps: int) -> torch.Tensor:
    """CPU [rows, steps+1, d] prefix of a fixed [1001,d] stream per draw.

    A draw owns its stream; chunking, T, budgets, other targets and adaptive
    proposal growth cannot advance that stream. The same model role has the
    same noise under both schedules/budgets. A1/A2 share private-role streams.
    """
    if not 1 <= steps < CANONICAL_NOISE_LENGTH or dimension < 1:
        raise ValueError("Invalid canonical diffusion-noise dimensions")
    if len(target_ids) != len(draw_indices) or any(type(i) is not int or i < 0 for i in draw_indices):
        raise ValueError("Noise draw indices must align with target identities")
    target_ids = _identities(target_ids)
    noise = torch.empty((len(target_ids), steps + 1, dimension), dtype=torch.float32)
    sampling_seed = 123 + seed - 42
    for row, (identity, draw) in enumerate(zip(target_ids, draw_indices)):
        rng = torch.Generator().manual_seed(_keyed(NOISE_NAMESPACE, cohort, sampling_seed, split, role, identity, draw, "noise"))
        stream = torch.randn((CANONICAL_NOISE_LENGTH, dimension), generator=rng)
        noise[row].copy_(stream[:steps + 1])
    return noise


def _uniform(cohort: str, seed: int, split: str, method: str, identity: int | str) -> float:
    rng = torch.Generator().manual_seed(_keyed(NOISE_NAMESPACE, cohort, 123 + seed - 42,
                                             split, method, identity, "systematic-resampling"))
    return float(torch.rand((), generator=rng, dtype=torch.float64))


def _synchronize(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _draw_ddpm(model, context: torch.Tensor, *, cohort: str, seed: int,
               split: str, role: str, target_ids: list, draw_indices: list[int],
               max_model_rows: int = MAX_MODEL_ROWS) -> torch.Tensor:
    """Batched ancestral DDPM using the declared per-draw canonical noise."""
    if len(context) != len(target_ids) or len(context) != len(draw_indices) or max_model_rows < 1:
        raise ValueError("DDPM context, noise identities and chunk bound must align")
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    steps, dimension = int(model.steps), int(model.d)
    chunks = []
    for start in range(0, len(context), max_model_rows):
        stop = min(start + max_model_rows, len(context))
        noise = canonical_noise(cohort, seed, split, role, target_ids[start:stop],
                                draw_indices[start:stop], dimension, steps).transpose(0, 1).contiguous().to(device=device, dtype=dtype)
        z = context[start:stop].to(device=device, dtype=dtype)
        x = noise[0].clone()
        for step in reversed(range(steps)):
            t = torch.full((len(x),), step, device=device, dtype=torch.long)
            epsilon = model.predict_noise(x, t, z)
            beta, alpha, alpha_bar = model.betas[step], model.alphas[step], model.alpha_bars[step]
            x = (x - beta * epsilon / (1 - alpha_bar).sqrt()) / alpha.sqrt()
            if step > 0:
                x = x + model.posterior_variance[step].sqrt() * noise[step + 1]
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError(f"Nonfinite Neural21 DDPM samples: {cohort}/{seed}/{split}/{role}")
        chunks.append(x.detach().cpu())
    return torch.cat(chunks) if chunks else torch.empty((0, dimension), dtype=dtype)


@torch.no_grad()
def _density_log_prob(model, target: torch.Tensor, context: torch.Tensor,
                      *, raw_z_std: torch.Tensor | None = None,
                      max_model_rows: int = MAX_MODEL_ROWS) -> torch.Tensor:
    if len(target) != len(context):
        raise ValueError("Density target/context rows differ")
    parameter = next(model.parameters())
    values = []
    jacobian = -float(raw_z_std.double().log().sum()) if raw_z_std is not None else None
    for start in range(0, len(target), max_model_rows):
        stop = start + max_model_rows
        value = model.log_prob(target[start:stop].to(parameter), context[start:stop].to(parameter))
        if jacobian is not None:
            value = value + jacobian
        value = value.detach().cpu()
        # Original 9X scores each raw density in model precision, then sums in double.
        values.append(value.double() if jacobian is not None else value)
    return torch.cat(values)


def normalized_row_weights(log_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rowwise equivalent of the historical normalization and double-precision ESS."""
    if log_weights.ndim != 2 or min(log_weights.shape) < 1:
        raise ValueError("Importance log weights must be [targets,candidates]")
    if torch.isnan(log_weights).any() or torch.isposinf(log_weights).any():
        raise FloatingPointError("Importance log weights contain NaN or +infinity")
    if not bool(torch.isfinite(log_weights).any(1).all()):
        raise FloatingPointError("At least one target has no finite importance weight")
    weights = torch.exp(log_weights - torch.logsumexp(log_weights, 1, keepdim=True))
    weights = weights / weights.sum(1, keepdim=True)
    if not bool(torch.isfinite(weights).all()):
        raise FloatingPointError("Nonfinite normalized importance weights")
    ess = 1 / weights.double().square().sum(1)
    return weights, ess


def systematic_rows(weights: torch.Tensor, uniforms: torch.Tensor, output_count: int) -> torch.Tensor:
    """One systematic offset per target; complete anchor rows remain intact."""
    if weights.ndim != 2 or uniforms.shape != (len(weights),) or output_count < 1:
        raise ValueError("Systematic resampling requires aligned offsets and a positive draw count")
    if not bool(((uniforms >= 0) & (uniforms < 1)).all()):
        raise ValueError("Systematic offsets must lie in [0,1)")
    cumulative = weights.double().cumsum(1)
    cumulative[:, -1] = 1.
    positions = (torch.arange(output_count, dtype=torch.float64)[None, :] + uniforms.double()[:, None]) / output_count
    return torch.searchsorted(cumulative.contiguous(), positions.contiguous(), right=False).clamp_max(weights.shape[1] - 1)


@dataclass
class ChunkCache:
    """Sealed, small partial results; never store all canonical Gaussian streams."""
    directory: Path
    contract_fingerprint: str

    def run(self, name: str, function: Callable[[], dict]) -> dict:
        from .common import artifact, read_json, save, verify_artifact, write_json
        path = self.directory / f"{name}.pt"
        receipt = path.with_suffix(".json")
        if path.is_file() and receipt.is_file():
            metadata = read_json(receipt)
            if metadata["contract_fingerprint"] != self.contract_fingerprint:
                raise ValueError(f"Neural partial-sample contract changed: {receipt}")
            verify_artifact(metadata["artifact"])
            payload = torch.load(path, map_location="cpu", weights_only=True)
            return {**payload, "cache_reused": True}
        payload = function()
        save(path, payload)
        write_json(receipt, dict(contract_fingerprint=self.contract_fingerprint, artifact=artifact(path)))
        return {**payload, "cache_reused": False}


def _cached(cache: ChunkCache | None, name: str, function: Callable[[], dict]) -> dict:
    return cache.run(name, function) if cache else {**function(), "cache_reused": False}


def _proposal_identity(ids: list, count: int, start: int = 0) -> tuple[list, list[int]]:
    return ([identity for identity in ids for _ in range(count)],
            list(range(start, start + count)) * len(ids))


def _weight_diagnostic(identity, weights, ess, ancestor, count) -> dict:
    return dict(target_id=identity, candidate_count=int(count), ess=float(ess),
                ess_fraction=float(ess / count), max_normalized_weight=float(weights.max()),
                ancestor_index=int(ancestor))


@torch.no_grad()
def _adaptive_3x_anchor(models: dict, normalized_z: torch.Tensor, target_ids: list,
                        *, cohort: str, seed: int, split: str,
                        max_model_rows: int = MAX_MODEL_ROWS,
                        initial_candidates: int = INITIAL_3X_CANDIDATES,
                        maximum_candidates: int = MAXIMUM_3X_CANDIDATES,
                        target_ess: float = TARGET_3X_ESS) -> tuple[torch.Tensor, list[dict]]:
    """Vectorized independent target pools; grow only the still-active pools."""
    if initial_candidates < 1 or maximum_candidates < initial_candidates or target_ess <= 0:
        raise ValueError("Invalid adaptive SIR budget")
    count = len(target_ids)
    active = torch.arange(count)
    pools = torch.empty((count, 0, 2), dtype=torch.float32)
    selected = torch.empty((count, 2), dtype=torch.float32)
    diagnostics: list[dict | None] = [None] * count
    desired = initial_candidates
    while len(active):
        previous = pools.shape[1]
        ids = [target_ids[int(i)] for i in active]
        noise_ids, draw_indices = _proposal_identity(ids, desired - previous, previous)
        context = normalized_z[active, 0:1].repeat_interleave(desired - previous, dim=0)
        proposed = _draw_ddpm(models["g12"], context, cohort=cohort, seed=seed, split=split,
                              role="g12", target_ids=noise_ids, draw_indices=draw_indices,
                              max_model_rows=max_model_rows).reshape(len(active), desired - previous, 2)
        pools = torch.cat((pools, proposed), dim=1)
        log_weights = _density_log_prob(models["l2"], normalized_z[active, 1:2].repeat_interleave(desired, dim=0),
                                         pools[:, :, 1:2].reshape(-1, 1), max_model_rows=max_model_rows).reshape(len(active), desired)
        weights, ess = normalized_row_weights(log_weights)
        done = (ess >= target_ess) | (desired >= maximum_candidates)
        if bool(done.any()):
            completed = active[done]
            uniforms = torch.tensor([_uniform(cohort, seed, split, "neural21", target_ids[int(i)]) for i in completed], dtype=torch.float64)
            ancestors = systematic_rows(weights[done], uniforms, 1).squeeze(1)
            completed_pools = pools[done]
            selected[completed] = completed_pools[torch.arange(len(completed)), ancestors]
            for local, original in enumerate(completed.tolist()):
                d = _weight_diagnostic(target_ids[original], weights[done][local], ess[done][local], ancestors[local], desired)
                d.update(target_ess=target_ess, low_ess=bool(ess[done][local] < target_ess or ess[done][local] / desired < MINIMUM_ESS_FRACTION),
                         budget_exhausted=desired >= maximum_candidates, max_candidates=maximum_candidates,
                         sir_status="weighted", resampling_duplicate_rate=0.,
                         resampling_duplicate_rate_note="one draw per target; intact-row resampling")
                diagnostics[original] = d
        active, pools = active[~done], pools[~done]
        desired = min(maximum_candidates, max(desired * 2, desired + 1))
    if any(d is None for d in diagnostics):
        raise AssertionError("Adaptive SIR did not resolve every target")
    return selected, diagnostics


@torch.no_grad()
def sample_3x(models: dict, normalized_z: torch.Tensor, target_ids,
              *, cohort: str, seed: int, split: str,
              max_model_rows: int = MAX_MODEL_ROWS, target_batch_size: int | None = None,
              cache: ChunkCache | None = None) -> tuple[torch.Tensor, dict]:
    """Normalized full X[rows,3], with unchanged per-target adaptive SIR rules."""
    ids = _identities(target_ids)
    if normalized_z.shape != (len(ids), 2) or len(set(ids)) != len(ids):
        raise ValueError("3X targets must have unique IDs and shape [N,2]")
    target_batch_size = target_batch_size or max(1, max_model_rows // INITIAL_3X_CANDIDATES)
    outputs, per_target, stage_seconds, reused = [], [], 0., 0
    for start in range(0, len(ids), target_batch_size):
        stop = min(start + target_batch_size, len(ids))
        def compute():
            device = next(models["g12"].parameters()).device
            _synchronize(device)
            began = time.perf_counter()
            anchor, diagnostic = _adaptive_3x_anchor(models, normalized_z[start:stop], ids[start:stop],
                cohort=cohort, seed=seed, split=split, max_model_rows=max_model_rows)
            context = torch.cat((anchor[:, 1:2], normalized_z[start:stop, 1:2]), 1)
            private = _draw_ddpm(models["g3"], context, cohort=cohort, seed=seed, split=split, role="g3",
                target_ids=ids[start:stop], draw_indices=[0] * (stop - start), max_model_rows=max_model_rows)
            _synchronize(device)
            return dict(normalized_x=torch.cat((anchor, private), 1), per_target=diagnostic,
                        generation_seconds=time.perf_counter() - began)
        part = _cached(cache, f"targets_{start:06d}_{stop:06d}", compute)
        outputs.append(part["normalized_x"])
        per_target.extend(part["per_target"])
        stage_seconds += part["generation_seconds"]
        reused += int(part["cache_reused"])
    candidates = torch.tensor([d["candidate_count"] for d in per_target], dtype=torch.float64)
    ess = torch.tensor([d["ess"] for d in per_target], dtype=torch.float64)
    return torch.cat(outputs), dict(target_count=len(ids), per_target=per_target,
        candidate_count_min=int(candidates.min()), candidate_count_max=int(candidates.max()),
        candidate_count_mean=float(candidates.mean()), ess_min=float(ess.min()), ess_mean=float(ess.mean()),
        low_ess_target_fraction=sum(d["low_ess"] for d in per_target) / len(ids),
        max_candidates_target_fraction=sum(d["budget_exhausted"] for d in per_target) / len(ids),
        sampling_seconds=stage_seconds, reused_chunks=reused, sampler="per-target adaptive SIR",
        initial_candidates=INITIAL_3X_CANDIDATES, maximum_candidates=MAXIMUM_3X_CANDIDATES,
        target_ess=TARGET_3X_ESS, resampling_method="systematic")


def _nine_roles(anchor: int) -> tuple[str, list[str], list[str], list[int]]:
    if anchor not in (1, 2):
        raise ValueError("Only the approved A1 and A2 anchors are defined")
    others = [i for i in range(1, 5) if i != anchor]
    return f"a{anchor}", [f"l{i}" for i in others], [f"p{i}" for i in others], [anchor - 1, 2, 3, 4]


@torch.no_grad()
def _fixed_9x_anchor(models: dict, normalized_z: torch.Tensor, ids: list,
                      *, anchor: int, seed: int, split: str, z_std: torch.Tensor,
                      candidate_count: int, output_count: int,
                      max_model_rows: int) -> dict:
    role, density_roles, _, _ = _nine_roles(anchor)
    rows = len(ids)
    noise_ids, draws = _proposal_identity(ids, candidate_count)
    candidates = _draw_ddpm(models[role], normalized_z[:, anchor - 1:anchor].repeat_interleave(candidate_count, 0),
        cohort="cov_all", seed=seed, split=split, role=role, target_ids=noise_ids,
        draw_indices=draws, max_model_rows=max_model_rows).reshape(rows, candidate_count, 4)
    separator = candidates[:, :, 1:4].reshape(-1, 3)
    log_weights = torch.zeros((rows, candidate_count), dtype=torch.float64)
    for density_role in density_roles:
        j = int(density_role[1:]) - 1
        log_weights += _density_log_prob(models[density_role], normalized_z[:, j:j+1].repeat_interleave(candidate_count, 0),
            separator, raw_z_std=z_std[j:j+1], max_model_rows=max_model_rows).reshape(rows, candidate_count)
    weights, ess = normalized_row_weights(log_weights)
    uniforms = torch.tensor([_uniform("cov_all", seed, split, f"neural21_A{anchor}", identity) for identity in ids], dtype=torch.float64)
    ancestors = systematic_rows(weights, uniforms, output_count)
    selected = candidates[torch.arange(rows)[:, None], ancestors]
    diagnostics = []
    for i, identity in enumerate(ids):
        d = dict(target_id=identity, candidate_count=candidate_count, output_count=output_count,
                 ess=float(ess[i]), ess_fraction=float(ess[i] / candidate_count),
                 max_normalized_weight=float(weights[i].max()),
                 unique_ancestors=int(torch.unique(ancestors[i]).numel()),
                 repeated_anchor_slots=output_count - int(torch.unique(ancestors[i]).numel()))
        if output_count == 1:
            d["ancestor_index"] = int(ancestors[i, 0])
        diagnostics.append(d)
    result = dict(selected_anchor=selected, per_target=diagnostics)
    if rows == 1 and output_count > 1:
        result.update(ancestor_indices=ancestors[0], normalized_weights=weights[0])
    return result


@torch.no_grad()
def _assemble_9x(models: dict, anchor_values: torch.Tensor, normalized_z: torch.Tensor,
                 ids: list, *, anchor: int, seed: int, split: str,
                 max_model_rows: int) -> torch.Tensor:
    _, _, private_roles, anchor_columns = _nine_roles(anchor)
    normalized_x = torch.full((len(ids), 9), torch.nan, dtype=torch.float32)
    normalized_x[:, anchor_columns] = anchor_values
    separator = anchor_values[:, 1:4]
    columns = {"p1": [0], "p2": [1], "p3": [5, 6], "p4": [7, 8]}
    for role in private_roles:
        z_index = int(role[1:]) - 1
        context = torch.cat((separator, normalized_z[:, z_index:z_index+1]), 1)
        normalized_x[:, columns[role]] = _draw_ddpm(models[role], context, cohort="cov_all", seed=seed,
            split=split, role=role, target_ids=ids, draw_indices=[0] * len(ids), max_model_rows=max_model_rows)
    if not bool(torch.isfinite(normalized_x).all()):
        raise FloatingPointError("Neural21 A-factor composition failed to produce finite full X")
    return normalized_x


@torch.no_grad()
def sample_9x(models: dict, normalized_z: torch.Tensor, target_ids,
              *, anchor: int, seed: int, split: str, z_std: torch.Tensor,
              fixed_target: bool, max_model_rows: int = MAX_MODEL_ROWS,
              candidate_count: int = FIXED_9X_CANDIDATES,
              cache: ChunkCache | None = None, progress: Callable[[str], None] | None = None) -> tuple[torch.Tensor, dict]:
    """8192 proposals per conditional pool; no reduction for validation rows.

    Final zero-target generation uses one pool and 1000 systematic descendants.
    Validation uses a separate 8192-proposal pool for EACH of its 1000 targets,
    then one systematic descendant per pool. Private draws are independent per
    output slot/row, conditional on its selected complete separator.
    """
    ids = _identities(target_ids)
    if normalized_z.shape != (len(ids), 4) or len(set(ids)) != len(ids) or candidate_count < 1:
        raise ValueError("9X targets must have unique IDs, shape [N,4], and a positive pool size")
    if fixed_target and (not len(ids) or not torch.equal(normalized_z, normalized_z[0:1].expand_as(normalized_z))):
        raise ValueError("A shared fixed-target pool requires exactly identical Z rows")
    device = next(models[f"a{anchor}"].parameters()).device
    selected_parts, per_target, stage_seconds, reused = [], [], 0., 0
    pool_ids = ["fixed-target-pool"] if fixed_target else ids
    pool_z = normalized_z[:1] if fixed_target else normalized_z
    target_batch_size = max(1, max_model_rows // candidate_count)
    pool_details = {}
    for start in range(0, len(pool_ids), target_batch_size):
        stop = min(start + target_batch_size, len(pool_ids))
        def compute():
            _synchronize(device)
            began = time.perf_counter()
            part = _fixed_9x_anchor(models, pool_z[start:stop], pool_ids[start:stop], anchor=anchor,
                seed=seed, split=split, z_std=z_std, candidate_count=candidate_count,
                output_count=len(ids) if fixed_target else 1, max_model_rows=max_model_rows)
            _synchronize(device)
            part["generation_seconds"] = time.perf_counter() - began
            return part
        part = _cached(cache, f"anchor_{start:06d}_{stop:06d}", compute)
        selected_parts.append(part["selected_anchor"].reshape(-1, 4))
        per_target.extend(part["per_target"])
        stage_seconds += part["generation_seconds"]
        reused += int(part["cache_reused"])
        if fixed_target:
            pool_details = {k: part[k] for k in ("ancestor_indices", "normalized_weights") if k in part}
        if progress and (stop % 50 == 0 or stop == len(pool_ids)):
            progress(f"Neural21 A{anchor} {split}: {stop}/{len(pool_ids)} conditional pools; {candidate_count} proposals each")
    anchor_values = torch.cat(selected_parts)
    def private_compute():
        _synchronize(device)
        began = time.perf_counter()
        x = _assemble_9x(models, anchor_values, normalized_z, ids, anchor=anchor, seed=seed,
                         split=split, max_model_rows=max_model_rows)
        _synchronize(device)
        return dict(normalized_x=x, generation_seconds=time.perf_counter() - began)
    private = _cached(cache, "private_all_targets", private_compute)
    stage_seconds += private["generation_seconds"]
    reused += int(private["cache_reused"])
    return private["normalized_x"], dict(target_count=len(ids), conditional_pool_count=len(pool_ids),
        per_target=per_target, candidate_count_per_pool=candidate_count,
        total_anchor_proposals=len(pool_ids) * candidate_count,
        fixed_target=fixed_target, sampling_seconds=stage_seconds, reused_chunks=reused,
        sampler="fixed-budget per-condition SIR", resampling_method="systematic", **pool_details)


