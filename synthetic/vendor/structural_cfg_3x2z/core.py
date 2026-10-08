"""Shared diffusion coordinates, local MLPs, and the sole CFG composition rule."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


GROUPS = {'eps12': [0, 1], 'eps23': [1, 2], 'eps2': [1]}
NULL = 2


def cosine_schedule(steps):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 2:
        raise ValueError('diffusion_steps must be an integer >= 2')
    j = torch.arange(steps + 1, dtype=torch.float64)
    f = torch.cos((j / steps + .008) / 1.008 * math.pi / 2).square()
    beta = torch.cat([torch.zeros(1, dtype=torch.float64), (1 - f[1:] / f[:-1]).clamp(max=.999)])
    alpha = 1 - beta
    abar = torch.cumprod(alpha, 0)
    variance = torch.zeros_like(beta)
    variance[1:] = beta[1:] * (1 - abar[:-1]) / (1 - abar[1:])
    schedule = dict(beta=beta, alpha=alpha, alpha_bar=abar, posterior_variance=variance)
    validate_schedule(schedule)
    return schedule


def linear_schedule(steps=100):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 2:
        raise ValueError('diffusion_steps must be an integer >= 2')
    beta = torch.cat([torch.zeros(1, dtype=torch.float64), torch.linspace(.0001, .02, steps, dtype=torch.float64)])
    alpha = 1 - beta
    abar = alpha.cumprod(0)
    variance = torch.zeros_like(beta)
    variance[1:] = beta[1:] * (1 - abar[:-1]) / (1 - abar[1:])
    schedule = dict(beta=beta, alpha=alpha, alpha_bar=abar, posterior_variance=variance)
    validate_schedule(schedule, require_noise_endpoint=False)
    return schedule


def validate_schedule(schedule, *, require_noise_endpoint=True):
    keys = {'beta', 'alpha', 'alpha_bar', 'posterior_variance'}
    if set(schedule) != keys:
        raise ValueError(f'schedule must contain {sorted(keys)}')
    beta, alpha, abar, var = [schedule[k].double().cpu() for k in ('beta', 'alpha', 'alpha_bar', 'posterior_variance')]
    if beta.ndim != 1 or len(beta) < 3 or any(v.shape != beta.shape or not torch.isfinite(v).all() for v in (beta, alpha, abar, var)):
        raise ValueError('invalid schedule shapes or nonfinite values')
    if not (beta[0] == 0 and abar[0] == 1 and var[0] == 0 and var[1] == 0
            and bool(((beta[1:] > 0) & (beta[1:] < 1)).all())
            and bool((abar[1:] < abar[:-1]).all()) and bool((abar > 0).all())
            and bool((var >= 0).all()) and (not require_noise_endpoint or abar[-1] < 1e-3)):
        raise ValueError('invalid schedule endpoints, variance, beta or monotonicity')
    if not (torch.allclose(alpha, 1-beta, atol=1e-12, rtol=1e-10)
            and torch.allclose(abar, alpha.cumprod(0), atol=1e-12, rtol=1e-10)
            and torch.allclose(var[1:], beta[1:] * (1-abar[:-1]) / (1-abar[1:]), atol=1e-12, rtol=1e-10)):
        raise ValueError('stored schedule does not match clipped-beta diffusion identities')


def q_sample(x0, t, noise, schedule):
    abar = schedule['alpha_bar'][t].to(dtype=x0.dtype).unsqueeze(1)
    return abar.sqrt() * x0 + (1 - abar).sqrt() * noise


class EpsilonMLP(nn.Module):
    def __init__(self, dimension, conditional, config):
        super().__init__()
        td, cd = config['time_embedding_dimension'], config['condition_embedding_dimension']
        width, depth = config['hidden_width'], config['hidden_depth']
        frequencies = torch.exp(-math.log(10000) * torch.arange(td // 2) / (td // 2))
        self.register_buffer('frequencies', frequencies)
        self.time_projection = nn.Sequential(nn.Linear(td, td), nn.SiLU(), nn.Linear(td, td))
        self.continuous = config.get('condition_type', 'binary') == 'continuous'
        self.condition_embedding = (nn.Linear(2, cd) if self.continuous else nn.Embedding(3, cd)) if conditional else None
        layers, features = [], dimension + td + (cd if conditional else 0)
        for _ in range(depth):
            layers.extend([nn.Linear(features, width), nn.SiLU()])
            features = width
        self.network = nn.Sequential(*layers, nn.Linear(features, dimension))

    def forward(self, x, t, condition=None, missing=None):
        phase = t.to(x.dtype).unsqueeze(1) * self.frequencies.unsqueeze(0)
        pieces = [x, self.time_projection(torch.cat([phase.sin(), phase.cos()], dim=1))]
        if self.condition_embedding is not None:
            if self.continuous:
                if condition is None:
                    condition, missing = x.new_zeros(len(x)), x.new_ones(len(x))
                elif missing is None:
                    missing = torch.zeros_like(condition)
                # A missing condition is (0,1); the valid value zero is (0,0).
                condition = torch.stack([condition * (1-missing.to(x.dtype)), missing.to(x.dtype)], dim=1)
            elif condition is None:
                raise ValueError('conditional model requires labels 0, 1 or NULL=2')
            pieces.append(self.condition_embedding(condition))
        return self.network(torch.cat(pieces, dim=1))


def build_models(config):
    return nn.ModuleDict({name: EpsilonMLP(len(group), name != 'eps2', config) for name, group in GROUPS.items()})


def local_training_losses(models, xt, t, noise, labels, drop_probability, generator):
    # Each full observed row and its one 3D noise draw supply all three models.
    mask = torch.rand(labels.shape, generator=generator, device=labels.device) < drop_probability
    if getattr(models['eps12'], 'continuous', False):
        return {
            **{name: F.mse_loss(models[name](xt[:, GROUPS[name]], t, labels[:, j], missing=mask[:, j]),
                               noise[:, GROUPS[name]]) for j, name in enumerate(('eps12', 'eps23'))},
            'eps2': F.mse_loss(models['eps2'](xt[:, [1]], t), noise[:, [1]]),
        }
    dropped = labels.clone()
    dropped[mask] = NULL
    return {
        'eps12': F.mse_loss(models['eps12'](xt[:, GROUPS['eps12']], t, dropped[:, 0]), noise[:, GROUPS['eps12']]),
        'eps23': F.mse_loss(models['eps23'](xt[:, GROUPS['eps23']], t, dropped[:, 1]), noise[:, GROUPS['eps23']]),
        'eps2': F.mse_loss(models['eps2'](xt[:, GROUPS['eps2']], t), noise[:, GROUPS['eps2']]),
    }


def require_finite(name, value, t):
    if not torch.isfinite(value).all():
        finite = value[torch.isfinite(value)]
        bounds = (float(finite.min()), float(finite.max())) if finite.numel() else None
        raise FloatingPointError(f'{name}: t={t}; shape={tuple(value.shape)}; finite_range={bounds}; '
                                 f'nonfinite={int((~torch.isfinite(value)).sum())}')


def composed_epsilon(models, xt, t, z1=1, z2=1, w1=1., w2=1., *, conditions=None):
    continuous = getattr(models['eps12'], 'continuous', False)
    if xt.ndim != 2 or xt.shape[1] != 3:
        raise ValueError('expected full [B,3] xt')
    if continuous:
        if conditions is None or conditions.shape != (len(xt), 2):
            raise ValueError('continuous model requires normalized [B,2] conditions')
        require_finite('continuous conditions', conditions, t)
        z1, z2 = conditions[:, 0], conditions[:, 1]
    elif conditions is not None or z1 not in (0, 1) or z2 not in (0, 1):
        raise ValueError('expected full [B,3] xt and binary requested labels')
    if not math.isfinite(w1) or not math.isfinite(w2):
        raise ValueError('CFG weights must be finite; 0=unconditional, 1=conditional')
    time = torch.full((len(xt),), t, device=xt.device, dtype=torch.long) if isinstance(t, int) else t
    require_finite('shared xt', xt, t)
    def local(name, z, weight):
        def predict(label):
            condition = label if continuous else torch.full_like(time, label)
            result = models[name](xt[:, GROUPS[name]], time, condition)
            require_finite(name, result, t)
            return result
        if weight == 1:
            return predict(z)
        unconditional = predict(None if continuous else NULL)
        return unconditional if weight == 0 else unconditional + weight * (predict(z) - unconditional)
    g12, g23 = local('eps12', z1, w1), local('eps23', z2, w2)
    e2 = models['eps2'](xt[:, GROUPS['eps2']], time)
    require_finite('eps2', e2, t)
    out = torch.zeros_like(xt)
    out[:, 0] = g12[:, 0]
    out[:, 1] = g12[:, 1] + g23[:, 0] - e2[:, 0]
    out[:, 2] = g23[:, 1]
    require_finite('composed epsilon', out, t)
    return out


def ddpm_step(x, epsilon, t, schedule, generator, *, noise=None):
    beta, alpha, abar, variance = [schedule[k][t].to(dtype=x.dtype) for k in ('beta', 'alpha', 'alpha_bar', 'posterior_variance')]
    mean = (x - beta / (1 - abar).sqrt() * epsilon) / alpha.sqrt()
    if t > 1:
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) if noise is None else noise
        mean = mean + variance.sqrt() * noise
    require_finite('DDPM state (eps12+eps23-eps2)', mean, t)
    return mean


@torch.inference_mode()
def sample_ddpm(models, schedule, mean, std, *, n, batch_size, seed, device,
                z1=1, z2=1, w1=1., w2=1., conditions=None, target_ids=None,
                require_noise_endpoint=True):
    if n < 1 or batch_size < 1:
        raise ValueError('n and batch_size must be positive')
    validate_schedule(schedule, require_noise_endpoint=require_noise_endpoint)
    if conditions is not None:
        conditions = torch.as_tensor(conditions, dtype=torch.float32, device=device)
        if conditions.shape != (n, 2) or not torch.isfinite(conditions).all():
            raise ValueError('conditions must be finite normalized [n,2]')
    if target_ids is not None:
        target_ids = torch.as_tensor(target_ids)
        if target_ids.shape != (n,) or target_ids.dtype != torch.int64 or len(target_ids.unique()) != n:
            raise ValueError('target_ids must be distinct int64 [n]')
    models.to(device).eval()
    schedule = {k: v.to(device) for k, v in schedule.items()}
    mean, std = torch.as_tensor(mean, device=device), torch.as_tensor(std, device=device)
    if mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 1e-8).all():
        raise ValueError('normalization must contain three finite means and positive stds > 1e-8')
    generator = torch.Generator(device=device).manual_seed(seed)
    samples = []
    for start in range(0, n, batch_size):
        count, steps = min(batch_size, n-start), len(schedule['beta']) - 1
        noise_bank = None
        if target_ids is not None:
            streams = []
            for target in target_ids[start:start+count].tolist():
                digest = hashlib.sha256(f'{seed}:structural_cfg:{target}'.encode()).digest()
                keyed = torch.Generator(device=device).manual_seed(int.from_bytes(digest[:8], 'big') % (2**63-1))
                streams.append(torch.randn((steps, 3), generator=keyed, device=device))
            noise_bank = torch.stack(streams, dim=1)
        x = noise_bank[0].clone() if noise_bank is not None else torch.randn((count, 3), generator=generator, device=device)
        for t in range(len(schedule['beta']) - 1, 0, -1):
            epsilon = composed_epsilon(models, x, t, z1, z2, w1, w2,
                                       conditions=None if conditions is None else conditions[start:start+count])
            x = ddpm_step(x, epsilon, t, schedule, generator,
                          noise=noise_bank[steps-t+1] if noise_bank is not None and t > 1 else None)
        raw = x.double() * std.double() + mean.double()
        require_finite('inverse normalization', raw, 0)
        samples.append(raw.cpu())
    return torch.cat(samples)
