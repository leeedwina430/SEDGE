"""Full-separator CFG and parameter-matched historical DDPM/MDN components."""
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import torch
from torch import nn
from factorized_sedge.density import ConditionalGaussianMixture
from raw_diffusion_guidance.models import DDPM, Schedule
from structural_cfg_3x2z.core import EpsilonMLP
from .config import TARGET_PARAMETERS
from .data import Normalizer
from .graphs import graph
from .runtime import seed_for


def diffusion_recipe(cfg):
    return dict(timesteps=cfg.timesteps, beta_start=cfg.beta_start, beta_end=cfg.beta_end,
                time_dim=64, depth=3, condition_dropout=.1)


class StructuralCFG(Schedule):
    def __init__(self, g, width, cfg):
        super().__init__(diffusion_recipe(cfg))
        self.g, self.d = g, g.x_dim
        self.scopes = tuple(u+g.separator for u in g.private)
        settings = dict(time_embedding_dimension=64, condition_embedding_dimension=16,
                        hidden_width=width, hidden_depth=4, condition_type="continuous")
        self.local = nn.ModuleList(EpsilonMLP(len(a), True, settings) for a in self.scopes)
        self.central = EpsilonMLP(len(g.separator), False, settings)

    def component_loss(self, xt, t, noise, z, missing=None):
        losses = []
        for j, (net, scope) in enumerate(zip(self.local, self.scopes)):
            if missing is None:
                loss = .5*(net(xt[:, scope], t+1, z[:, j])-noise[:, scope]).square().mean()
                loss = loss+.5*(net(xt[:, scope], t+1)-noise[:, scope]).square().mean()
            else:
                loss = (net(xt[:, scope], t+1, z[:, j], missing[:, j])-noise[:, scope]).square().mean()
            losses.append(loss)
        losses.append((self.central(xt[:, self.g.separator], t+1)-noise[:, self.g.separator]).square().mean())
        return torch.stack(losses).sum()

    def training_loss(self, x, z, generator):
        t = torch.randint(self.steps, (len(x),), generator=generator, device=x.device)
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        missing = torch.rand(z.shape, generator=generator, device=x.device) < .1
        return self.component_loss(self.corrupt(x, t, noise), t, noise, z, missing)

    def predict_noise(self, x, t, z, weights=None):
        if x.shape != (len(z), self.d) or z.shape[1] != self.g.z_dim or not torch.isfinite(z).all():
            raise ValueError("Full shared X and finite normalized Z are required")
        weights = (1.,)*self.g.z_dim if weights is None else tuple(weights)
        if len(weights) != self.g.z_dim or not all(math.isfinite(w) for w in weights):
            raise ValueError("One finite CFG weight per Z is required")
        out = torch.zeros_like(x)
        for j, (net, scope, w) in enumerate(zip(self.local, self.scopes, weights)):
            if w == 1:
                eps = net(x[:, scope], t+1, z[:, j])
            else:
                eps = net(x[:, scope], t+1)
                if w != 0:
                    eps = eps+w*(net(x[:, scope], t+1, z[:, j])-eps)
            out[:, scope] += eps
        out[:, self.g.separator] -= (self.g.z_dim-1)*self.central(x[:, self.g.separator], t+1)
        return out


@dataclass(frozen=True)
class Factor:
    name: str
    family: str
    output_x: tuple = ()
    output_z: tuple = ()
    input_x: tuple = ()
    input_z: tuple = ()

    def to_dict(self):
        return asdict(self)


def factors(g, method):
    if method in ("structural_cfg", "direct_cd"):
        return (Factor(method, "cfg" if method == "structural_cfg" else "ddpm",
                       tuple(range(g.x_dim)), input_z=tuple(range(g.z_dim))),)
    if method != "neural21":
        raise ValueError(method)
    return (Factor("anchor_1", "ddpm", g.separator+g.private[0], input_z=(0,)),
            *(Factor(f"private_{j+1}", "ddpm", g.private[j], input_x=g.separator, input_z=(j,))
              for j in range(1, g.z_dim)),
            *(Factor(f"likelihood_{j+1}", "mdn", output_z=(j,), input_x=g.separator)
              for j in range(1, g.z_dim)))


def construct(g, factor, width, cfg):
    if factor.family == "cfg":
        return StructuralCFG(g, width, cfg)
    d, k = len(factor.output_x)+len(factor.output_z), len(factor.input_x)+len(factor.input_z)
    if factor.family == "mdn":
        return ConditionalGaussianMixture(k, d, 8, width, 3, .001)
    return DDPM(d, k, "direct", width, diffusion_recipe(cfg))


@lru_cache(maxsize=None)
def capacity(name, method, cfg):
    g = graph(name)
    def counts(width):
        with torch.device("meta"):
            return {f.name: sum(p.numel() for p in construct(g, f, width, cfg).parameters())
                    for f in factors(g, method)}
    low, high = 8, 512
    while low+1 < high:
        mid = (low+high)//2
        if sum(counts(mid).values()) < TARGET_PARAMETERS:
            low = mid
        else:
            high = mid
    width = min((low, high), key=lambda w: (abs(sum(counts(w).values())-TARGET_PARAMETERS), w))
    components = counts(width)
    total = sum(components.values())
    if abs(total/TARGET_PARAMETERS-1) > .01:
        raise ValueError("Parameter budget could not be matched")
    return dict(width=width, components=components, total=total, target=TARGET_PARAMETERS)


def make_model(name, method, factor, cfg, seed, device="cpu"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_for(name, seed, method, factor.to_dict(), "initialization"))
        model = construct(graph(name), factor, capacity(name, method, cfg)["width"], cfg)
    return model.to(device)


def training_tensors(cohort, arm, factor):
    norm = Normalizer(**cohort["normalizers"][arm])
    tensors = []
    for split in ("train", "validation"):
        rows = cohort["arms"][arm][split]
        x, z = norm.normalize_x(rows["x"]).float(), norm.normalize_z(rows["z"]).float()
        target = x[:, factor.output_x] if factor.output_x else z[:, factor.output_z]
        context = torch.cat((x[:, factor.input_x], z[:, factor.input_z]), 1)
        tensors.extend((target, context))
    return tensors
