"""Portable file adapter around the continuous_eval 0.2.1 chemistry core."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import platform
import shutil
import tempfile

from .core import Evaluator, _parse_molecule, finite_number

METADATA = ("panel", "condition_id", "sampling_seed", "draw_index",
            "target_mol_logp", "target_tpsa", "observed")
PROPERTIES = ("mol_logp", "tpsa")
PANELS = {"id_diagnostic", "val_ood", "test_ood", "repeated_test_ood"}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _json(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_bad_constant)


def _rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = _json(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected JSON object")
            yield row


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _targets(row):
    return {name: finite_number(row.get("target_" + name), "target_" + name)
            for name in PROPERTIES}


def _requests(path, evaluator):
    requests, targets_by_condition, cells = {}, {}, set()
    groups = defaultdict(set)
    for row in _rows(path):
        rid = row.get("request_id")
        if not isinstance(rid, str) or not rid.strip() or rid in requests:
            raise ValueError(f"Invalid or duplicate request_id: {rid!r}")
        if row.get("panel") not in PANELS:
            raise ValueError(f"Unknown panel for {rid}; choose {sorted(PANELS)}")
        cid = row.get("condition_id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError("condition_id must be a nonempty string")
        if (type(row.get("sampling_seed")) is not int or
                not 0 <= row["sampling_seed"] < 2**32):
            raise ValueError("sampling_seed must be an integer in [0,2^32)")
        if type(row.get("draw_index")) is not int or row["draw_index"] < 0:
            raise ValueError("draw_index must be a nonnegative integer")
        if "observed" in row and (row["observed"] != [True, True] or
                                  any(type(value) is not bool for value in row["observed"])):
            raise ValueError("S_STRUCT_CCOND requires both properties observed")
        target = _targets(row)
        if row["panel"] != "id_diagnostic" and not evaluator.region_contains(target):
            raise ValueError(f"OOD target outside profile region: {rid}")
        key = (row["panel"], cid)
        if key in targets_by_condition and targets_by_condition[key] != target:
            raise ValueError(f"Inconsistent target for condition {key}")
        targets_by_condition[key] = target
        group = (row["panel"], cid, row["sampling_seed"])
        cell = (*group, row["draw_index"])
        if cell in cells:
            raise ValueError(f"Duplicate condition/seed/draw cell: {cell}")
        cells.add(cell)
        groups[group].add(row["draw_index"])
        requests[rid] = row
    if not requests:
        raise ValueError("Request manifest is empty")
    for group, indices in groups.items():
        if indices != set(range(len(indices))):
            raise ValueError(f"Manifest draw indices must start at zero without gaps: {group}")
    return requests


def _generations(path, requests):
    received = {}
    for row in _rows(path):
        rid = row.get("request_id")
        if not isinstance(rid, str) or rid not in requests or rid in received:
            raise ValueError(f"Unknown or duplicate generation request_id: {rid!r}")
        if "smiles" not in row:
            raise ValueError(f"Generation is missing smiles: {rid}")
        for name in METADATA:
            if name in row and row[name] != requests[rid].get(name, [True, True] if name == "observed" else None):
                raise ValueError(f"Generation metadata mismatch: {rid}.{name}")
        if "ended" in row and type(row["ended"]) is not bool:
            raise ValueError(f"Generation ended must be boolean: {rid}")
        failed = row.get("ended") is False or row.get("error") is not None
        if failed and row["smiles"] not in (None, ""):
            raise ValueError(f"Failed/incomplete attempt has nonempty smiles: {rid}")
        received[rid] = row
    return received


def _training(path, evaluator):
    structures, bounds, count = set(), {name: [math.inf, -math.inf] for name in PROPERTIES}, 0
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or "smiles" not in reader.fieldnames:
            raise ValueError("Train CSV needs unique headers and a smiles column")
        for count, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Ragged training row: {count}")
            molecule, status = _parse_molecule(row["smiles"])
            if molecule is None:
                raise ValueError(f"Invalid training SMILES row {count}: {status['failure_reason']}")
            values = {name: finite_number(float(evaluator.properties[name].calculator(molecule)), name)
                      for name in PROPERTIES}
            if evaluator.region_contains(values):
                raise ValueError(f"Training row {count} belongs to held-out OOD region")
            structures.add(status["canonical_smiles"])
            for name, value in values.items():
                bounds[name][0] = min(bounds[name][0], value)
                bounds[name][1] = max(bounds[name][1], value)
    if not count:
        raise ValueError("Training CSV is empty")
    return structures, bounds, count


def _aggregate(rows, evaluator, *, ood):
    n = len(rows)
    counts = Counter({name: 0 for name in ("requests", "received", "missing", "molecule_valid",
                                          "ldmol_compatible_valid", "properties_valid", "region_hit", "axis_covered")})
    reasons, groups = Counter(), defaultdict(list)
    valid_structures, novel_structures = set(), set()
    novel_draws = 0
    for row in rows:
        result = row["evaluation"]
        counts["requests"] += 1
        counts["received"] += row["output_present"]
        counts["missing"] += not row["output_present"]
        for name in ("molecule_valid", "ldmol_compatible_valid", "properties_valid", "region_hit", "axis_covered"):
            counts[name] += bool(result[name])
        if result["failure_reason"]:
            reasons[result["failure_reason"]] += 1
        if result["molecule_valid"]:
            valid_structures.add(result["canonical_smiles"])
            if row["novel_vs_train"]:
                novel_draws += 1
                novel_structures.add(result["canonical_smiles"])
        groups[(row["condition_id"], row["sampling_seed"])].append(row)
    valid = counts["molecule_valid"]
    counts["novel_valid_draws"] = novel_draws
    summary = {
        "counts": dict(counts), "failure_reasons": dict(reasons),
        "rates_over_all_requests": {key: counts[key] / n for key in
                                    ("molecule_valid", "ldmol_compatible_valid", "properties_valid")},
        "ldmol_compatible_validity": counts["ldmol_compatible_valid"] / n,
        "ldmol_compatible_validity_over_received_outputs": (
            counts["ldmol_compatible_valid"] / counts["received"] if counts["received"] else None),
        "structure_diagnostics": {
            "unique_valid_structures": len(valid_structures),
            "duplicate_fraction_of_valid_draws": 1 - len(valid_structures) / valid if valid else None,
            "novel_fraction_of_valid_draws": novel_draws / valid if valid else None,
            "unique_novel_valid_structures": len(novel_structures),
            "novelty_reference": "canonical isomeric SMILES in supplied source train CSV"},
        "property_errors": {}, "profiles": {}, "ood_metrics_applicable": ood,
    }
    summary["rates_over_all_requests"].update(
        region_hit=counts["region_hit"] / n if ood else None,
        axis_covered=counts["axis_covered"] / n)
    for name in PROPERTIES:
        errors = [row["evaluation"]["errors"][name] for row in rows if row["evaluation"]["properties_valid"]]
        count = len(errors)
        summary["property_errors"][name] = {
            "n": count, "denominator": "property-valid outputs only",
            "mae": sum(error["absolute"] for error in errors) / count if count else None,
            "mean_signed_error": sum(error["signed"] for error in errors) / count if count else None,
            "rmse": math.sqrt(sum(error["signed"] ** 2 for error in errors) / count) if count else None,
        }
    conditions = []
    ks = [k for k in (1, 5, 10) if k <= min(map(len, groups.values()))]
    for label in evaluator.tolerances:
        outcomes = [row["evaluation"]["profiles"][label] for row in rows]
        matches = sum(outcome["target_match"] for outcome in outcomes)
        successes = sum(bool(outcome["extrapolation_success"]) for outcome in outcomes)
        group_rates, match_rates, by_seed, first_indices = [], [], defaultdict(list), []
        for (cid, seed), members in sorted(groups.items()):
            group_successes = sum(bool(row["evaluation"]["profiles"][label]["extrapolation_success"]) for row in members)
            group_matches = sum(row["evaluation"]["profiles"][label]["target_match"] for row in members)
            rate = group_successes / len(members)
            group_rates.append(rate)
            match_rates.append(group_matches / len(members))
            by_seed[str(seed)].append(rate)
            successful_indices = [row["draw_index"] for row in members
                                  if row["evaluation"]["profiles"][label]["extrapolation_success"]]
            first = min(successful_indices) if successful_indices else None
            first_indices.append(first)
            conditions.append({
                "panel": members[0]["panel"], "condition_id": cid, "sampling_seed": seed,
                "tolerance": label, "requests": len(members), "target_matches": group_matches,
                "target_match_rate": group_matches / len(members),
                "extrapolation_successes": group_successes if ood else None,
                "extrapolation_success_rate": rate if ood else None,
                "ldmol_compatible_validity": sum(row["evaluation"]["ldmol_compatible_valid"] for row in members) / len(members),
                "molecule_validity": sum(row["evaluation"]["molecule_valid"] for row in members) / len(members),
                "properties_validity": sum(row["evaluation"]["properties_valid"] for row in members) / len(members),
                **{f"success_at_{k}": (first is not None and first < k) if ood else None for k in ks},
            })
        unique_success = {row["evaluation"]["canonical_smiles"] for row in rows
                          if row["evaluation"]["profiles"][label]["extrapolation_success"]}
        summary["profiles"][label] = {
            "absolute_tolerances": evaluator.tolerances[label],
            "target_match_count": matches, "target_match_rate": matches / n,
            "condition_macro_target_match_rate": sum(match_rates) / len(match_rates),
            "property_match_rates": {name: sum(outcome["property_match"][name] for outcome in outcomes) / n for name in PROPERTIES},
            "extrapolation_success_count": successes if ood else None,
            "micro_extrapolation_success_rate": successes / n if ood else None,
            "condition_macro_extrapolation_success_rate": sum(group_rates) / len(group_rates) if ood else None,
            "condition_macro_rate_by_seed": {seed: sum(rates) / len(rates) for seed, rates in sorted(by_seed.items())} if ood else None,
            "axis_covered_extrapolation_success_rate": sum(bool(outcome["axis_covered_extrapolation_success"]) for outcome in outcomes) / n if ood else None,
            "success_at_k": {str(k): sum(first is not None and first < k for first in first_indices) / len(first_indices) for k in ks} if ood else None,
            "unique_successful_structures": len(unique_success) if ood else None,
            "duplicate_fraction_of_successful_draws": (1 - len(unique_success) / successes if successes else None) if ood else None,
        }
    summary["primary_metric"] = (f"profiles.{evaluator.main_tolerance}.condition_macro_extrapolation_success_rate"
                                 if ood else f"profiles.{evaluator.main_tolerance}.condition_macro_target_match_rate")
    return summary, conditions


def evaluate_files(requests, generations, train_csv, output, *, profile="v0-2"):
    """Score once into a new directory. Input errors never publish a summary."""
    from rdkit import rdBase

    requests, generations, train_csv, output = map(Path, (requests, generations, train_csv, output))
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing output: {output}")
    profile_path = Path(__file__).parent / "profiles/v0-2.json" if str(profile) == "v0-2" else Path(profile)
    profile_data = _json(profile_path.read_text(encoding="utf-8"))
    evaluator = Evaluator(profile_data)
    if set(evaluator.names) != set(PROPERTIES):
        raise ValueError("S_STRUCT_CCOND evaluation requires mol_logp and tpsa exactly")
    inventory = _requests(requests, evaluator)
    received = _generations(generations, inventory)
    train_structures, bounds, train_rows = _training(train_csv, evaluator)
    diagnostic_profile = evaluator.profile
    diagnostic_profile.update(profile_id="sedge_ccond_id_diagnostic_v1", region={"eq": [0.0, 0.0]})
    diagnostic = Evaluator(diagnostic_profile)
    panels, conditions = {}, []
    provenance = {
        "evaluation_kind": "local_review_only", "formal_model_evaluation": False,
        "model_input_audit": "not_performed", "historical_reproduction": "not_established",
        "chemistry_core_source_version": "continuous_eval 0.2.1",
        "python_version": platform.python_version(), "rdkit_version": rdBase.rdkitVersion,
        "oracle": evaluator.metadata(), "train_rows": train_rows,
        "unique_train_structures": len(train_structures), "train_bounds": bounds,
        "attempt_denominator": "all rows in requests JSONL, including missing/failed generations",
        "macro_grouping": "equal weight per (panel, condition_id, sampling_seed) cell",
        "request_manifest_is_budget": True,
        "randomness": "none; generation seed and draw_index are carried from requests",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evaluation_", dir=output.parent) as temporary:
        staging = Path(temporary) / "results"
        staging.mkdir()
        panel_rows = defaultdict(list)
        with (staging / "results.jsonl").open("w", encoding="utf-8") as handle:
            for rid, request in inventory.items():
                generation = received.get(rid)
                present = generation is not None
                raw = generation["smiles"] if present else None
                ood = request["panel"] != "id_diagnostic"
                result = (evaluator if ood else diagnostic).evaluate(raw, _targets(request), train_bounds=bounds)
                if not present:
                    result["failure_reason"] = "missing_output"
                elif generation.get("error") is not None:
                    result["failure_reason"] = "generation_error"
                elif generation.get("ended") is False:
                    result["failure_reason"] = "generation_incomplete"
                if not ood:
                    result["region_hit"] = None
                    for outcome in [result, *result["profiles"].values()]:
                        outcome["extrapolation_success"] = None
                        outcome["axis_covered_extrapolation_success"] = None
                row = {**request, "output_present": present, "raw_smiles": raw,
                       "generation_ended": generation.get("ended") if present else None,
                       "generation_error": generation.get("error") if present else None,
                       "novel_vs_train": (result["canonical_smiles"] not in train_structures) if result["molecule_valid"] else None,
                       "evaluation": result}
                panel_rows[request["panel"]].append(row)
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        for panel, rows in sorted(panel_rows.items()):
            panels[panel], group_rows = _aggregate(rows, evaluator, ood=panel != "id_diagnostic")
            conditions.extend(group_rows)
        summary = {"schema_version": "sedge_local_evaluation_v1", "primary_tolerance": evaluator.main_tolerance,
                   "panels": panels, "provenance": provenance}
        fields = list(dict.fromkeys(key for row in conditions for key in row))
        with (staging / "condition_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(conditions)
        _write_json(staging / "summary.json", summary)
        _write_json(staging / "provenance.json", provenance)
        _write_json(staging / "profile.json", evaluator.profile)
        _write_json(staging / "id_diagnostic_profile.json", diagnostic.profile)
        if output.exists():
            raise ValueError(f"Refusing to overwrite existing output: {output}")
        shutil.move(str(staging), str(output))
    return summary
