"""Independent calibration, paired graph innovations and split-safe cohorts."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri, roots_hermitenorm
import torch

from .config import ARMS, PANELS, VERSION
from .graphs import GRAPHS, NAMES, graph
from .io import (code_hashes, completed, fingerprint, load_tensor, log, read_json,
                 save_tensor, seal, tensor_hash, write_json)
from .runtime import seed_for


def structural_calibration(g):
    k = np.asarray([len(p) for p in g.parents], dtype=np.float64)
    variance, convergence = 4*k/45, []
    if g.name == "L10-WITHIN":
        previous = None
        for order in (64, 128, 256):
            a, weights = roots_hermitenorm(order)
            weights = weights/np.sqrt(2*np.pi)
            cross = np.sum(weights[:, None]*weights[None, :]*ndtr(a[:, None])**2*
                           ndtr(.5*a[:, None]+np.sqrt(.75)*a[None, :])**2)
            value = 8/45+2*(cross-1/9)
            convergence.append({"nodes_per_axis": order, "variance": float(value)})
            if previous is not None and abs(value-previous) <= 1e-8:
                variance[:] = value
                break
            previous = value
        else:
            raise RuntimeError("L10 calibration quadrature did not converge")
    return {"mean": (k/3).tolist(), "scale": np.sqrt(variance).tolist(),
            "quadrature": convergence}


def draw(g, count, replicate, role, calibration, *, block=0):
    x = np.column_stack([np.random.default_rng(seed_for(g.pair, replicate, role, block, "x", i)).random(count)
                         for i in range(g.x_dim)])
    for a, b in g.edges:
        parent = np.clip(x[:, a], np.nextafter(0., 1.), np.nextafter(1., 0.))
        innovation = np.clip(x[:, b], np.nextafter(0., 1.), np.nextafter(1., 0.))
        x[:, b] = ndtr(.5*ndtri(parent)+np.sqrt(.75)*ndtri(innovation))
    signal = np.column_stack([np.square(x[:, p]).sum(1) for p in g.parents])
    noise = np.column_stack([np.random.default_rng(seed_for(g.pair, replicate, role, block, "z", j)).standard_normal(count)
                             for j in range(g.z_dim)])
    z = (signal-np.asarray(calibration["mean"]))/np.asarray(calibration["scale"])+.2*noise
    return x, z


def heldout(z, thresholds):
    cut = np.asarray(thresholds)
    selected = z[:, :len(cut)]
    return np.where(np.arange(len(cut)) % 2 == 0, selected < cut, selected > cut).all(1)


def calibrate(output, config):
    root = Path(output) / "calibration"
    contract = {"version": VERSION, "design_count": config.design_count, "audit_count": config.audit_count,
                "source": code_hashes("graphs.py", "data.py", "runtime.py"), "target_mass": .1}
    if completed(root, contract):
        return read_json(root / "calibration.json")
    results = {}
    for g in GRAPHS:
        if g.name not in ("S7-HUB", "L11-CHAIN"):
            continue
        structural = structural_calibration(g)
        if g.parent_graph:
            parent = results[g.parent_graph]
            n = min(len(parent["mean"]), g.z_dim)
            if structural["mean"][:n] != parent["mean"][:n] or structural["scale"][:n] != parent["scale"][:n]:
                raise AssertionError("Paired structural constants differ")
            q, thresholds = parent["q"], parent["thresholds"]
            design_hash = parent["design_hash"]
        else:
            x, z = draw(g, config.design_count, 0, "design", structural)
            lo, hi = 0., 1.
            for _ in range(60):
                q = (lo+hi)/2
                thresholds = [float(np.quantile(z[:, j], q if j % 2 == 0 else 1-q, method="linear"))
                              for j in range(g.z_dim)]
                if heldout(z, thresholds).mean() >= .1:
                    hi = q
                else:
                    lo = q
            q = hi
            thresholds = [float(np.quantile(z[:, j], q if j % 2 == 0 else 1-q, method="linear"))
                          for j in range(g.z_dim)]
            design_hash = {"x": tensor_hash(x), "z": tensor_hash(z)}
        ax, az = draw(g, config.audit_count, 0, "audit", structural)
        audit_mass = float(heldout(az, thresholds).mean())
        if not g.parent_graph and not .095 <= audit_mass <= .105:
            raise RuntimeError(f"Independent H audit failed: {g.name}: {audit_mass}")
        entry = {**structural, "q": q, "thresholds": list(thresholds),
                 "selection_z": list(range(len(thresholds))), "design_hash": design_hash,
                 "audit_hash": {"x": tensor_hash(ax), "z": tensor_hash(az)},
                 "audit_h_mass": audit_mass, "audit_x_mean": ax.mean(0).tolist(),
                 "audit_x_std": ax.std(0).tolist(), "audit_x_correlation": np.corrcoef(ax.T).tolist(),
                 "audit_z_mean": az.mean(0).tolist(), "audit_z_std": az.std(0).tolist(),
                 "graph": g.to_dict()}
        if g.name == "E12-NO-PRIVATE" and audit_mass != results[g.parent_graph]["audit_h_mass"]:
            raise AssertionError("E12 paired H audit must be identical")
        results[g.name] = entry
        log(f"calibrated {g.name}: H mass={audit_mass:.6f}, q={q:.6f}")
    write_json(root / "calibration.json", results)
    seal(root, contract, ["calibration.json"])
    return results


@dataclass
class Normalizer:
    x_mean: torch.Tensor
    x_std: torch.Tensor
    z_mean: torch.Tensor
    z_std: torch.Tensor

    @classmethod
    def fit(cls, x, z, correction=1):
        return cls(x.double().mean(0), x.double().std(0, correction=correction).clamp_min(1e-6),
                   z.double().mean(0), z.double().std(0, correction=correction).clamp_min(1e-6))

    def normalize_x(self, value):
        return (value-self.x_mean.to(value))/self.x_std.to(value)

    def normalize_z(self, value):
        return (value-self.z_mean.to(value))/self.z_std.to(value)

    def inverse_x(self, value):
        return value*self.x_std.to(value)+self.x_mean.to(value)

    def state_dict(self):
        return vars(self)


def rows_from(g, count, seed, role, calibration, in_h=None):
    roles = {"population": 0, "replace_train": 1, "replace_validation": 2, "panel_h": 3, "panel_hc": 4}
    prefix = ((NAMES.index(g.pair)+1)*1000+seed)*10+roles[role]
    if count == 0:
        return {"x": torch.empty(0, g.x_dim, dtype=torch.float64),
                "z": torch.empty(0, g.z_dim, dtype=torch.float64), "ids": torch.empty(0, dtype=torch.long)}
    batches, accepted, proposed = [], 0, 0
    for block in range(1000):
        size = count if in_h is None else 65536
        x, z = draw(g, size, seed, role, calibration, block=block)
        ids = prefix*100_000_000+proposed+np.arange(size, dtype=np.int64)
        mask = np.ones(size, dtype=bool) if in_h is None else heldout(z, calibration["thresholds"]) == in_h
        keep = np.flatnonzero(mask)[:count-accepted]
        batches.append({"x": torch.from_numpy(x[keep].copy()), "z": torch.from_numpy(z[keep].copy()),
                        "ids": torch.from_numpy(ids[keep].copy())})
        proposed += size
        accepted += len(keep)
        if accepted == count:
            return {key: torch.cat([batch[key] for batch in batches]) for key in ("x", "z", "ids")}
    raise RuntimeError(f"Region draw exceeded its fixed safety cap: {g.name}/{role}")


def subset(rows, index):
    return {key: value[index].clone() for key, value in rows.items()}


def build_cohort(output, name, seed, config, calibration):
    g = graph(name)
    destination = Path(output) / "data" / name / f"seed_{seed}"
    contract = {"version": VERSION, "graph": g.to_dict(), "seed": seed, "mode": config.mode,
                "counts": [config.train_count, config.validation_count, config.target_count],
                "calibration": calibration, "source": code_hashes("data.py", "graphs.py", "runtime.py")}
    if completed(destination, contract):
        return load_tensor(destination / "cohort.pt")
    n = config.train_count+config.validation_count+config.target_count
    data_seed = seed if config.mode == "formal" else seed+10000
    population = rows_from(g, n, data_seed, "population", calibration)
    order = torch.from_numpy(np.random.default_rng(seed_for(g.pair, data_seed, "partition")).permutation(n))
    a, b = config.train_count, config.train_count+config.validation_count
    arms = {"iid": {"train": subset(population, order[:a]), "validation": subset(population, order[a:b])}, "hc": {}}
    replacements = {}
    for split in ("train", "validation"):
        original = arms["iid"][split]
        mask = torch.from_numpy(heldout(original["z"].numpy(), calibration["thresholds"]))
        new = subset(original, slice(None))
        replacement = rows_from(g, int(mask.sum()), data_seed, f"replace_{split}", calibration, False)
        for key in new:
            new[key][mask] = replacement[key]
        arms["hc"][split] = new
        replacements[split] = {"mask": mask, "original_ids": original["ids"], "replacement_ids": replacement["ids"]}
    panels = {"population": subset(population, order[b:])}
    for panel in ("h", "hc"):
        panels[panel] = rows_from(g, config.target_count, data_seed, f"panel_{panel}", calibration, panel == "h")
    normalizers = {arm: Normalizer.fit(arms[arm]["train"]["x"], arms[arm]["train"]["z"]).state_dict() for arm in ARMS}
    evaluator = Normalizer.fit(arms["hc"]["train"]["x"], arms["hc"]["train"]["z"], correction=0)
    index = torch.from_numpy(np.random.default_rng(seed_for(name, seed, "kernel_rows")).permutation(a)[:min(1000, a)])
    training = arms["hc"]["train"]
    matrix = torch.cat((evaluator.normalize_x(training["x"][index]), evaluator.normalize_z(training["z"][index])), 1)
    def median_distance(value):
        distances = torch.pdist(value.double())
        answer = float(torch.quantile(distances, .5, interpolation="linear"))
        if not np.isfinite(answer) or answer <= 0:
            raise ValueError("Invalid TRAIN-fixed kernel median")
        return answer
    kernel = {"base_bandwidth": median_distance(matrix), "multipliers": [.25, .5, 1., 2., 4.],
              "row_ids": training["ids"][index], "quantile": "type7_midpoint", "subspaces": {}}
    for label, dims in [("x", tuple(range(g.x_dim))), ("c", g.separator), *[(f"u{j+1}", p) for j, p in enumerate(g.private)]]:
        if dims:
            kernel["subspaces"][label] = {"dims": dims, "base_bandwidth": median_distance(matrix[:, dims])}
    value = {"contract": contract, "fingerprint": fingerprint(contract), "graph": name, "seed": seed,
             "arms": arms, "panels": panels, "normalizers": normalizers,
             "metric_state": evaluator.state_dict(), "kernel": kernel, "replacements": replacements}
    audit = audit_cohort(value)
    save_tensor(destination / "cohort.pt", value)
    write_json(destination / "audit.json", audit)
    seal(destination, contract, ["cohort.pt", "audit.json"])
    log(f"data complete {name}/seed_{seed}")
    return value


def audit_cohort(cohort):
    g = graph(cohort["graph"])
    thresholds = cohort["contract"]["calibration"]["thresholds"]
    owned = {name: set() for name in ("train", "validation", "test")}
    hashes = {}
    for arm, splits in cohort["arms"].items():
        for split, rows in splits.items():
            owned[split].update(rows["ids"].tolist())
            if arm == "hc" and heldout(rows["z"].numpy(), thresholds).any():
                raise AssertionError("H leaked into Hc fitting")
            hashes[f"{arm}_{split}"] = {key: tensor_hash(v) for key, v in rows.items()}
            if rows["x"].shape != (len(rows["ids"]), g.x_dim) or rows["z"].shape[1] != g.z_dim:
                raise AssertionError("Wrong DGP dimensions")
    for panel, rows in cohort["panels"].items():
        identifiers = set(rows["ids"].tolist())
        if len(identifiers) != len(rows["ids"]) or identifiers & owned["test"]:
            raise AssertionError("Duplicate test rows")
        owned["test"].update(identifiers)
        event = heldout(rows["z"].numpy(), thresholds)
        if (panel == "h" and not event.all()) or (panel == "hc" and event.any()):
            raise AssertionError("Wrong regional test law")
        hashes[f"test_{panel}"] = {key: tensor_hash(v) for key, v in rows.items()}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if owned[first] & owned[second]:
            raise AssertionError("Cross-arm split leakage")
    for split in ("train", "validation"):
        mask = cohort["replacements"][split]["mask"]
        for key in ("x", "z", "ids"):
            if not torch.equal(cohort["arms"]["iid"][split][key][~mask], cohort["arms"]["hc"][split][key][~mask]):
                raise AssertionError("Paired retained Hc rows changed")
    return {"passed": True, "rows": hashes, "disjoint_roles": {k: len(v) for k, v in owned.items()}}


def load_cohort(output, name, seed):
    root = Path(output) / "data" / name / f"seed_{seed}"
    if not completed(root):
        raise RuntimeError(f"Unsealed cohort: {root}")
    return load_tensor(root / "cohort.pt")
