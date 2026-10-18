"""Freeze generation requests before sampling; preserve every planned attempt."""
import argparse
import csv
import json
import math
from pathlib import Path


def read_requests(path):
    rows, seen = [], set()
    for line in Path(path).read_text().splitlines():
        raw = json.loads(line)
        row = {key: raw[key] for key in ("request_id", "panel", "condition_id", "target_mol_logp",
                                         "target_tpsa", "observed", "sampling_seed", "draw_index")}
        for key in ("request_id", "panel", "condition_id"):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError(f"Invalid {key}")
        if row["panel"] not in ("id_diagnostic", "val_ood", "test_ood", "repeated_test_ood"):
            raise ValueError("Unknown evaluation panel")
        if row["request_id"] in seen:
            raise ValueError("Duplicate request_id")
        seen.add(row["request_id"])
        if row["observed"] != [True, True] or any(type(x) is not bool for x in row["observed"]):
            raise ValueError("Only fully observed property requests are supported")
        for key in ("sampling_seed", "draw_index"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f"Invalid {key}")
        if row["sampling_seed"] >= 2**32:
            raise ValueError("sampling_seed must be below 2**32")
        for key in ("target_mol_logp", "target_tpsa"):
            if type(row[key]) not in (int, float) or not math.isfinite(row[key]):
                raise ValueError(f"Invalid {key}")
            row[key] = float(row[key])
        rows.append(row)
    if not rows:
        raise ValueError("Request inventory is empty")
    ordered = sorted(rows, key=lambda r: (r["panel"], r["sampling_seed"], r["condition_id"], r["draw_index"], r["request_id"]))
    if rows != ordered:
        raise ValueError("Request file must use canonical panel/seed/condition/draw/id order; regenerate with sedge-requests")
    conditions, positions, draws = {}, set(), {}
    for row in rows:
        key = row["panel"], row["condition_id"]
        targets = row["target_mol_logp"], row["target_tpsa"]
        if key in conditions and conditions[key] != targets:
            raise ValueError("One condition_id has conflicting targets")
        conditions[key] = targets
        position = (*key, row["sampling_seed"], row["draw_index"])
        if position in positions:
            raise ValueError("Duplicate condition/seed/draw")
        positions.add(position)
        draws.setdefault((*key, row["sampling_seed"]), set()).add(row["draw_index"])
    if any(indices != set(range(len(indices))) for indices in draws.values()):
        raise ValueError("Draw indices must start at zero without gaps in each condition/seed")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--conditions-csv", type=Path, help="condition_id,mol_logp,tpsa; new/custom requests")
    group.add_argument("--official-plans", type=Path, help="Frozen evaluator directory containing val_ood/ and test_ood/")
    parser.add_argument("--val-id-csv", type=Path, help="Required with official plans for the historical 128-condition ID panel")
    parser.add_argument("--panel", choices=("id_diagnostic", "val_ood", "test_ood", "repeated_test_ood"),
                        help="Required for custom conditions; id_diagnostic does not claim OOD success")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--samples-per-condition", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.samples_per_condition < 1 or any(not 0 <= s < 2**32 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Samples must be positive; seeds must be unique nonnegative integers")
    from .io import atomic_json
    if args.output.exists() or args.output.with_suffix(".meta.json").exists():
        raise ValueError("Refusing to overwrite request inventory")
    rows = []
    if args.official_plans:
        if args.val_id_csv is None:
            parser.error("--official-plans requires --val-id-csv")
        from sedge_molecule.data import load_split
        from sedge_molecule._request_inventory import build_inventory
        if args.val_id_csv.name != "val_id.csv":
            raise ValueError("Historical ID split must be named val_id.csv")
        if args.seeds != [0, 1, 2] or args.samples_per_condition != 10:
            raise ValueError("Official plans retain their frozen seeds and counts")
        inventory, _ = build_inventory(args.official_plans, load_split("val_id", args.val_id_csv.parent))
        inventory = sorted((r for r in inventory if all(r.observed)),
                           key=lambda r: (r.panel, r.sampling_seed, r.condition_id, r.draw_index, r.request_id))
        for r in inventory:
            rows.append(dict(request_id=r.request_id, panel=r.panel, condition_id=r.condition_id,
                             target_mol_logp=r.mol_logp, target_tpsa=r.tpsa, observed=list(r.observed),
                             sampling_seed=r.sampling_seed, draw_index=r.draw_index))
        metadata = dict(mode="official_ood_with_id_diagnostic")
    else:
        if args.panel is None:
            parser.error("--conditions-csv requires an explicit --panel")
        seen = set()
        with args.conditions_csv.open(newline="") as handle:
            for r in csv.DictReader(handle):
                cid = r["condition_id"]
                if not cid or cid in seen:
                    raise ValueError("Conditions must have unique, nonempty condition_id values")
                seen.add(cid)
                for seed in args.seeds:
                    for draw in range(args.samples_per_condition):
                        rid = "custom:" + json.dumps([args.panel, cid, seed, draw], separators=(",", ":"))
                        rows.append(dict(request_id=rid, panel=args.panel, condition_id=cid,
                                         target_mol_logp=float(r["mol_logp"]), target_tpsa=float(r["tpsa"]),
                                         observed=[True, True], sampling_seed=seed, draw_index=draw))
        metadata = dict(mode="custom_inventory")
    rows.sort(key=lambda r: (r["panel"], r["sampling_seed"], r["condition_id"], r["draw_index"], r["request_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        temporary.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows))
        read_requests(temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    metadata.update(count=len(rows))
    atomic_json(args.output.with_suffix(".meta.json"), metadata)
    print(f"Frozen {len(rows)} attempts in {args.output}")


if __name__ == "__main__":
    main()
