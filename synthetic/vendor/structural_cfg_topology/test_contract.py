"""Run with python -s -m structural_cfg_topology.test_contract."""
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from .config import NAMES, METHODS, Config
from .data import draw, structural_calibration
from .evaluation import mmd_squared
from .graphs import graph
from .io import completed, seal
from .models import StructuralCFG, capacity, factors, make_model
from .runtime import configure_torch
from .sampling import formula_samples, noise_arrays
from .training import optimize


class Identity(nn.Module):
    def __init__(self, multiplier=1.):
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x, t, condition=None, missing=None):
        return self.multiplier*x


def main():
    configure_torch()
    cfg = replace(Config(), diffusion_updates=4, density_updates=4, validate_every=2,
                  density_validate_every=2)
    for name in NAMES:
        g = graph(name)
        x, z = draw(g, 10000, 42, "contract-check", structural_calibration(g))
        assert ((x >= 0) & (x <= 1)).all()
        residual = (z-(np.column_stack([(x[:, p]**2).sum(1) for p in g.parents])
                       -np.array([len(p)/3 for p in g.parents]))
                    /np.sqrt(np.array([4*len(p)/45 for p in g.parents])))/.2
        assert abs(residual.mean()) < .03 and abs(residual.std()-1) < .03
        model = StructuralCFG(g, 8, cfg)
        assert model.scopes == tuple(u+g.separator for u in g.private)
        model.local = nn.ModuleList(Identity() for _ in g.private)
        model.central = Identity()
        full, t, condition = torch.randn(9, g.x_dim), torch.ones(9, dtype=torch.long), torch.zeros(9, g.z_dim)
        # Independent standard Gaussians are counted exactly once on all coordinates.
        assert torch.allclose(model.predict_noise(full, t, condition), full, atol=2e-6)
        model.local = nn.ModuleList(Identity(j+1) for j in range(g.z_dim))
        model.central = Identity(.5)
        expected = full.clone()
        for j, private in enumerate(g.private):
            expected[:, private] *= j+1
        expected[:, g.separator] *= sum(range(1, g.z_dim+1))-.5*(g.z_dim-1)
        assert torch.allclose(model.predict_noise(full, t, condition), expected, atol=2e-6)
        assert all(abs(capacity(name, method, cfg)["total"]/321435-1) < .01 for method in METHODS)
        neural = factors(g, "neural21")
        assert len(neural) == 2*g.z_dim-1
        assert neural[0].output_x == g.separator+g.private[0]
        assert all(f.input_x == g.separator for f in neural[1:])
    cohort = dict(graph="S7-HUB", seed=42)
    a = noise_arrays(cohort, [11, 12], [0, 0], "test", 3, steps=3, stop=4)
    b = noise_arrays(cohort, [12, 11], [0, 0], "test", 3, steps=3, start=2, stop=4)
    assert torch.equal(a.reshape(3, 2, 4, 3)[:, [1, 0], 2:], b.reshape(3, 2, 2, 3))
    assert not torch.equal(a[:, :1], noise_arrays(cohort, [11], [1], "test", 3, steps=3))
    g = graph("S7-HUB")
    norm = dict(x_mean=torch.zeros(7), x_std=torch.ones(7), z_mean=torch.zeros(3), z_std=torch.ones(3))
    cohort["normalizers"] = {"iid": norm}
    rows = dict(ids=torch.tensor([11, 12]), z=torch.zeros(2, 3))
    def proposals(cohort, models, method, z, ids, draws, active, start, stop, cfg):
        for i in active:
            yield i, torch.ones(stop-start, 3), torch.zeros(stop-start), {}, None
    sir_cfg = replace(cfg, candidate_initial=4, candidate_max=8, target_ess=32)
    with patch("structural_cfg_topology.sampling.isolated_proposals", proposals):
        result = formula_samples(cohort, "iid", "neural21", {}, rows, [0, 0], sir_cfg)
    assert not result["valid"].any() and torch.isnan(result["x"]).all()
    assert all(d["failure"] == "ess_budget_exhausted" and d["proposals"] == 8 for d in result["diagnostics"])
    points = torch.randn(8, 5, dtype=torch.float64)
    assert mmd_squared(points, points, dict(base_bandwidth=2., multipliers=[.5, 1., 2.]))["mmd2_v"] == 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root/"report.json").write_text("{}")
        seal(root/"evaluation", {}, ["../report.json"])
        assert completed(root/"evaluation", {})
        (root/"report.json").write_text('{"changed": true}')
        try:
            completed(root/"evaluation", {})
        except RuntimeError:
            pass
        else:
            raise AssertionError("Changed final report escaped seal verification")
        for method in METHODS:
            # Include MDN continuation as well as both diffusion architectures.
            factor = factors(g, method)[-1]
            model = make_model(g.name, method, factor, cfg, 42)
            tx = torch.randn(16, len(factor.output_x)+len(factor.output_z))
            tz = torch.randn(16, len(factor.input_x)+len(factor.input_z))
            tensors = (tx, tz, tx[:8], tz[:8])
            contract = dict(cell=[g.name, "iid", method, 42], factor=factor.to_dict(), config=cfg.to_dict())
            optimize(model, tensors, factor, cfg, 777, Path(tmp)/method/"full", contract)
            full = {k: v.clone() for k, v in model.state_dict().items()}
            model = make_model(g.name, method, factor, cfg, 42)
            optimize(model, tensors, factor, cfg, 777, Path(tmp)/method/"resume", contract, stop_after=1)
            model = make_model(g.name, method, factor, cfg, 42)
            optimize(model, tensors, factor, cfg, 777, Path(tmp)/method/"resume", contract)
            assert all(torch.equal(v, model.state_dict()[k]) for k, v in full.items())
    print("PASS: graph/DGP, shared-coordinate algebra, capacity, Neural21 scope, keyed noise, SIR cap failures, MMD identity, exact training continuation", flush=True)


if __name__ == "__main__":
    main()
