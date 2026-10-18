"""Frozen fully observed OOD plans and the 128-condition ID diagnostic panel.

ID conditions are selected by the lexicographic ordering of condition IDs, with
seed 0 and ten draws per condition. Request IDs and serialization retain
the condition/seed/draw inventory contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .data import Split

ID_UNIQUE_CONDITIONS = 128
ID_SEED = 0
ID_DRAWS = 10
ID_MASKS = ((True, True),)

EXPECTED_COUNTS = {"val_ood": 15480, "repeated_test_ood": 15600, "id_diagnostic": 1280}


@dataclass(frozen=True)
class Request:
    request_id: str
    panel: str                  # val_ood | repeated_test_ood | id_diagnostic
    condition_id: str
    mol_logp: float
    tpsa: float
    observed: tuple             # (bool, bool)
    sampling_seed: int
    draw_index: int


def load_official_panel(plan_dir: Path, panel: str):
    """Load an official OOD request plan with both properties observed."""
    plan = json.loads((plan_dir / "plan.json").read_text())
    reqs = []
    with (plan_dir / "requests.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            reqs.append(Request(
                request_id=r["request_id"], panel=panel,
                condition_id=r["condition_id"],
                mol_logp=float(r["targets"]["mol_logp"]),
                tpsa=float(r["targets"]["tpsa"]),
                observed=(True, True),
                sampling_seed=int(r["seed"]), draw_index=int(r["sample_index"])))
    if len(reqs) != plan["request_count"]:
        raise RuntimeError(f"{panel}: requests.jsonl has {len(reqs)} rows; plan expects "
                           f"{plan['request_count']}")
    return reqs, plan


def unique_conditions(split: Split):
    """Keep the first property pair for each condition ID."""
    seen, out = set(), []
    for cid, props in zip(split.condition_ids, split.properties):
        if cid in seen:
            continue
        seen.add(cid)
        out.append((cid, float(props[0]), float(props[1])))
    return out


def build_id_diagnostic(val_id: Split):
    conds = unique_conditions(val_id)
    conds.sort(key=lambda c: c[0])
    chosen = conds[:ID_UNIQUE_CONDITIONS]
    reqs = []
    for cid, logp, tpsa in chosen:
        for mask in ID_MASKS:
            tag = f"{int(mask[0])}{int(mask[1])}"
            for k in range(ID_DRAWS):
                reqs.append(Request(
                    request_id=f"idreq|{cid}|m{tag}|s{ID_SEED}|d{k}",
                    panel="id_diagnostic", condition_id=cid,
                    mol_logp=logp, tpsa=tpsa, observed=mask,
                    sampling_seed=ID_SEED, draw_index=k))
    return reqs, {"unique_conditions": len(chosen), "masks": [list(m) for m in ID_MASKS],
                  "draws_per_condition_per_mask": ID_DRAWS, "seed": ID_SEED,
                  "selection": "condition_id_sorted_ascending"}


def build_inventory(requests_root: Path, val_id: Split):
    """Combine official OOD plans with fully observed ID diagnostics."""
    vo, vo_plan = load_official_panel(requests_root / "val_ood", "val_ood")
    te, te_plan = load_official_panel(requests_root / "test_ood", "repeated_test_ood")
    idr, id_meta = build_id_diagnostic(val_id)
    reqs = vo + te + idr
    counts = {}
    for r in reqs:
        counts[r.panel] = counts.get(r.panel, 0) + 1
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Request inventory counts differ from the frozen protocol: {counts} != {EXPECTED_COUNTS}")
    meta = {
        "counts": counts,
        "official_plans": {
            "val_ood": {"condition_count": vo_plan["condition_count"]},
            "repeated_test_ood": {"condition_count": te_plan["condition_count"]},
        },
        "id_diagnostic": id_meta,
    }
    return reqs, meta


