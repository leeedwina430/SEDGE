# SEDGE: Property-to-Molecule Generation

## Install

Requires Python 3.10–3.12 and a CUDA GPU with BF16 support for the default training configuration. Install a compatible PyTorch build, then run from this directory:

```bash
python -m pip install -e '.[test]'
```

## Run

```bash
# Prepare vocabulary, scaler and initial weights.
sedge-prepare --train-csv data/train.csv --output runs/assets

# Train: seed 17, 50 epochs, batch size 512, BF16.
sedge-train --train-csv data/train.csv --assets runs/assets \
  --config configs/train.json --device cuda --num-workers 12 \
  --output runs/train

# Create generation requests.
sedge-requests --conditions-csv data/conditions.csv --panel test_ood \
  --seeds 0 1 2 --samples-per-condition 10 --output runs/requests.jsonl

# Generate molecules.
sedge-generate --assets runs/assets \
  --checkpoint runs/train/checkpoints/final.pt \
  --requests runs/requests.jsonl --device cuda --batch-size 1024 \
  --output runs/generation

# Evaluate against the requested targets.
sedge-evaluate --requests runs/requests.jsonl \
  --generations runs/generation/generations.jsonl \
  --train-csv data/train.csv --output runs/evaluation
```

Use `--panel id_diagnostic` for targets outside the held-out joint region. Results are saved to `runs/generation/generations.jsonl` and `runs/evaluation/summary.json`; per-condition scores are in `condition_metrics.csv`.

To resume training, repeat its command with `--resume`. Repeating generation resumes its existing output. Keep inputs, code and execution settings unchanged; runs with incompatible RNG settings cannot be resumed together.

Run tests with `python -m pytest -q`. See [third-party notices](THIRD_PARTY_NOTICES.md) for vendored code and licenses.
