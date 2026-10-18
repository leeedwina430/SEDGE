"""Bounded chemistry, accounting, and publication regression checks."""
import json

import pytest

from sedge_molecule.evaluation.adapter import _aggregate, evaluate_files
from sedge_molecule.evaluation.core import Evaluator, PropertyDefinition, _parse_molecule, load_profile


def _request(rid="r0", index=0, condition="c", **kwargs):
    return {"request_id": rid, "panel": "id_diagnostic", "condition_id": condition,
            "sampling_seed": 11, "draw_index": index, "target_mol_logp": -0.0014,
            "target_tpsa": 20.23, "observed": [True, True], **kwargs}


def _files(tmp_path, requests, generations):
    req, gen, train = (tmp_path / name for name in ("requests.jsonl", "generations.jsonl", "train.csv"))
    for path, rows in ((req, requests), (gen, generations)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    train.write_text("smiles\nCCO\n")
    return req, gen, train, tmp_path / "results"




def test_every_planned_attempt_counts_and_id_is_not_ood_success(tmp_path):
    files = _files(tmp_path, [_request("r0", 0), _request("r1", 1), _request("r2", 2), _request("r3", 3)], [
        {"request_id": "r0", "smiles": "CCO", "ended": True, "error": None},
        {"request_id": "r1", "smiles": None, "ended": False, "error": "decode failed"},
        {"request_id": "r2", "smiles": "bad_smiles", "ended": True, "error": None},
    ])
    result = evaluate_files(*files)
    panel = result["panels"]["id_diagnostic"]
    assert panel["counts"]["requests"] == 4
    assert panel["counts"]["received"] == 3
    assert panel["counts"]["missing"] == 1
    assert panel["profiles"]["main"]["target_match_rate"] == 0.25
    assert panel["profiles"]["main"]["micro_extrapolation_success_rate"] is None
    assert panel["failure_reasons"]["generation_error"] == 1
    assert panel["failure_reasons"]["missing_output"] == 1
    assert panel["structure_diagnostics"]["novel_fraction_of_valid_draws"] == 0
    assert result["provenance"]["formal_model_evaluation"] is False
    assert {"summary.json", "results.jsonl", "condition_metrics.csv"} <= {p.name for p in files[-1].iterdir()}
    with pytest.raises(ValueError, match="overwrite"):
        evaluate_files(*files)


@pytest.mark.parametrize("generations,match", [
    ([{"request_id": "unknown", "smiles": None}], "Unknown or duplicate"),
    ([{"request_id": "r0", "smiles": None}] * 2, "Unknown or duplicate"),
    ([{"request_id": "r0", "smiles": "CCO", "ended": False}], "nonempty smiles"),
    ([{"request_id": "r0", "smiles": "CCO", "target_tpsa": 999}], "metadata mismatch"),
])
def test_malformed_generation_never_publishes(tmp_path, generations, match):
    files = _files(tmp_path, [_request()], generations)
    with pytest.raises(ValueError, match=match):
        evaluate_files(*files)
    assert not files[-1].exists()


@pytest.mark.parametrize("requests,match", [
    ([_request(), _request()], "duplicate request_id"),
    ([_request(target_tpsa=float("nan"))], "Nonfinite JSON"),
    ([_request(index=1)], "without gaps"),
    ([_request(observed=[1, 1])], "both properties observed"),
])
def test_malformed_manifest_never_publishes(tmp_path, requests, match):
    files = _files(tmp_path, requests, [])
    with pytest.raises(ValueError, match=match):
        evaluate_files(*files)
    assert not files[-1].exists()


def test_missing_entire_panel_retains_denominator(tmp_path):
    files = _files(tmp_path, [_request()], [])
    panel = evaluate_files(*files)["panels"]["id_diagnostic"]
    assert panel["counts"]["missing"] == 1
    assert panel["ldmol_compatible_validity"] == 0
    assert panel["profiles"]["main"]["target_match_rate"] == 0


def test_ood_file_evaluation_preserves_success_novelty_and_missing_attempts(tmp_path):
    evaluator = Evaluator(load_profile("v0-2"))
    smiles = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(O)C(O)C(O)C(O)CO"
    molecule, _ = _parse_molecule(smiles)
    measured = {name: float(evaluator.properties[name].calculator(molecule)) for name in evaluator.names}
    assert evaluator.region_contains(measured)
    targets = {"target_" + name: value for name, value in measured.items()}
    requests = [_request("r0", 0, panel="val_ood", **targets),
                _request("r1", 1, panel="val_ood", **targets)]
    files = _files(tmp_path, requests, [{"request_id": "r0", "smiles": smiles}])
    panel = evaluate_files(*files)["panels"]["val_ood"]
    assert panel["profiles"]["main"]["target_match_rate"] == 0.5
    assert panel["profiles"]["main"]["micro_extrapolation_success_rate"] == 0.5
    assert panel["profiles"]["main"]["condition_macro_extrapolation_success_rate"] == 0.5
    assert panel["profiles"]["main"]["unique_successful_structures"] == 1
    assert panel["structure_diagnostics"]["novel_fraction_of_valid_draws"] == 1


def test_target_match_requires_region_hit_for_ood_success():
    properties = {
        "mol_logp": PropertyDefinition("dimensionless", "test constant", lambda mol: 4.9),
        "tpsa": PropertyDefinition("angstrom^2", "test constant", lambda mol: 100.0),
    }
    evaluator = Evaluator(load_profile("v0-2"), properties)
    result = evaluator.evaluate("CCO", {"mol_logp": 5.0, "tpsa": 100.0})
    assert result["target_match"] is True
    assert result["region_hit"] is False
    assert result["extrapolation_success"] is False


def test_official_ulp_guard_and_strict_chemistry():
    properties = {
        "mol_logp": PropertyDefinition("dimensionless", "test boundary", lambda mol: 5.2),
        "tpsa": PropertyDefinition("angstrom^2", "test boundary", lambda mol: 105.0),
    }
    evaluator = Evaluator(load_profile("v0-2"), properties)
    result = evaluator.evaluate("CCO", {"mol_logp": 5.0, "tpsa": 100.0})
    assert result["profiles"]["main"]["extrapolation_success"] is True
    assert result["profiles"]["strict"]["target_match"] is False
    for smiles in ("CCO.C", "[Na+]", "O", "CCO trailing", ""):
        assert evaluator.evaluate(smiles, {"mol_logp": 5.0, "tpsa": 100.0})["molecule_valid"] is False
    # LDMol validity is an independent, intentionally looser metric.
    assert evaluator.evaluate("CCO.C", {"mol_logp": 5.0, "tpsa": 100.0})["ldmol_compatible_valid"] is True


def test_macro_weights_condition_seed_cells_not_attempts():
    evaluator = Evaluator(load_profile("v0-2"))
    rows = []
    for cid, successes in (("a", [True]), ("b", [False, False, False])):
        for index, success in enumerate(successes):
            result = evaluator.evaluate(None, {"mol_logp": 5.0, "tpsa": 100.0})
            for outcome in result["profiles"].values():
                outcome.update(target_match=success, extrapolation_success=success)
            rows.append({**_request(f"{cid}{index}", index, cid, panel="val_ood"),
                         "output_present": True, "novel_vs_train": None, "evaluation": result})
    summary, conditions = _aggregate(rows, evaluator, ood=True)
    assert summary["profiles"]["main"]["micro_extrapolation_success_rate"] == 0.25
    assert summary["profiles"]["main"]["condition_macro_extrapolation_success_rate"] == 0.5
    assert len(conditions) == 6
