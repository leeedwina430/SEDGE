"""A small CPU lifecycle check; it does not test or reproduce paper scores."""
import csv
import gc
import json

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors

from sedge_molecule import prepare, train, requests, generate
from sedge_molecule.evaluation.adapter import evaluate_files
from sedge_molecule.io import read_json


def test_cpu_training_resume_generation_and_evaluation(tmp_path, monkeypatch):
    training_csv = tmp_path / "train.csv"
    with training_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "mol_logp", "tpsa"])
        for smiles in ("CCO", "CCN", "CCCO", "CCCCO"):
            molecule = Chem.MolFromSmiles(smiles)
            writer.writerow([smiles, Crippen.MolLogP(molecule), rdMolDescriptors.CalcTPSA(molecule)])
    config = tmp_path / "train.json"
    config.write_text(json.dumps({"epochs": 1, "batch_size": 2, "precision": "float32"}))
    assets = tmp_path / "assets"
    prepare.main(["--train-csv", str(training_csv), "--custom-data", "--output", str(assets)])
    shared = ["--train-csv", str(training_csv), "--assets", str(assets), "--config", str(config),
              "--device", "cpu", "--threads", "2"]
    split_run = tmp_path / "split_run"
    train.main(shared + ["--output", str(split_run), "--max-updates", "1"])
    assert read_json(split_run / "completion.json")["status"] == "stopped"
    assert not (split_run / "checkpoints/final.pt").exists()
    last_bytes = (split_run / "checkpoints/last.pt").read_bytes()
    # Changing a numerical execution setting must not silently resume the run.
    with pytest.raises(ValueError, match="Resume inputs"):
        train.main(shared + ["--output", str(split_run), "--resume", "--threads", "1"])
    assert (split_run / "checkpoints/last.pt").read_bytes() == last_bytes
    # Earlier SeedSequence checkpoints have no training RNG contract. Reject
    # them before performing an update, rather than mixing training streams.
    original_load = torch.load
    def legacy_load(path, *args, **kwargs):
        value = original_load(path, *args, **kwargs)
        if path == split_run / "checkpoints/last.pt":
            value["binding"].pop("training_rng_contract")
        return value
    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", legacy_load)
        with pytest.raises(ValueError, match="training_rng_contract"):
            train.main(shared + ["--output", str(split_run), "--resume"])
    assert (split_run / "checkpoints/last.pt").read_bytes() == last_bytes
    train.main(shared + ["--output", str(split_run), "--resume"])
    uninterrupted = tmp_path / "uninterrupted"
    train.main(shared + ["--output", str(uninterrupted)])
    gc.collect()
    left = torch.load(split_run / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    right = torch.load(uninterrupted / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    assert left["complete_verified"] and right["complete_verified"]
    assert left["progress"] == right["progress"] == {"update": 2, "epoch": 1, "batch_in_epoch": 0}
    assert left["mask_counts"] == right["mask_counts"] == {"11": 4}
    for name in left["model"]:
        assert torch.equal(left["model"][name], right["model"][name]), name
    # An exhausted but unverified checkpoint must not pass through the smoke override.
    left["complete_verified"] = False
    unverified = tmp_path / "unverified.pt"
    torch.save(left, unverified)
    del left, right
    gc.collect()
    conditions = tmp_path / "conditions.csv"
    conditions.write_text("condition_id,mol_logp,tpsa\ntoy,1.2,20.23\n")
    inventory = tmp_path / "requests.jsonl"
    requests.main(["--conditions-csv", str(conditions), "--panel", "id_diagnostic",
                   "--seeds", "0", "--samples-per-condition", "2", "--output", str(inventory)])
    output = tmp_path / "generation"
    gen_args = ["--assets", str(assets), "--checkpoint", str(split_run / "checkpoints/final.pt"),
                "--requests", str(inventory), "--output", str(output),
                "--device", "cpu", "--threads", "2", "--batch-size", "2"]
    with pytest.raises(ValueError, match="Final training validation"):
        generate.main(gen_args + ["--checkpoint", str(unverified), "--allow-partial-checkpoint"])
    generate.main(gen_args)
    first_bytes = (output / "generations.jsonl").read_bytes()
    assert {json.loads(line)["batch_rng_seed"] for line in first_bytes.splitlines()} == {756774520843587948}
    binding_path = output / "binding.json"
    saved_binding = binding_path.read_bytes()
    binding = json.loads(saved_binding)
    binding["sampling_rng_contract"] = "numpy_seed_sequence_v1(seed,3,panel_index,batch_index)"
    binding_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="Generation settings"):
        generate.main(gen_args)
    assert (output / "generations.jsonl").read_bytes() == first_bytes
    binding_path.write_bytes(saved_binding)
    generate.main(gen_args)
    assert (output / "generations.jsonl").read_bytes() == first_bytes
    # Replay an interrupted batch and require the exact full output bytes.
    rows = (output / "generations.jsonl").read_text().splitlines(keepends=True)
    (output / "generations.jsonl").write_text(rows[0])
    (output / "completion.json").unlink()
    generate.main(gen_args)
    assert (output / "generations.jsonl").read_bytes() == first_bytes
    with pytest.raises(ValueError, match="batch/device/environment"):
        generate.main(gen_args + ["--threads", "1"])
    result = evaluate_files(inventory, output / "generations.jsonl", training_csv, tmp_path / "evaluation")
    panel = result["panels"]["id_diagnostic"]
    assert panel["counts"]["requests"] == panel["counts"]["received"] == 2
    assert panel["profiles"]["main"]["extrapolation_success_count"] is None
