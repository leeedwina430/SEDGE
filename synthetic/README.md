# Reported synthetic experiments: SEDGE-LWR/LGD

Standalone collection of the reported five-DGP tables, including all 13 underlying
benchmark settings, three seeds per setting, all 26 reported methods/controls,
data generators, complete training and sampling code, and saved outputs. The
historical result key `SEDGE_LWS` is the algorithm called LWR (likelihood
reweighting) in the method draft. It is retained to preserve result provenance.

## Layout

- `datagen/progressive.py`: seven original DGPs, selection audits, train/validation
  splits and exact posterior reference generation.
- `datagen/multimodal.py`: six multimodal discrete extensions.
- `ours/engine.py`: complete fitting, LWR proposal/weight/resampling/private
  completion, LGD noisy-likelihood distillation/guidance/private completion,
  baseline orchestration, evaluation and reporting. Nothing is a pseudocode stub.
- `ours/density.py`, `ours/spline.py`, `ours/noisy.py`: Bernoulli and conditional
  spline densities, learned support envelopes, fitting and noisy likelihoods.
- `ours/multimodal.py`: extension runner and disclosed ridge stabilization for
  the Rep4Ex-inspired adaptation.
- `baselines/adaptations.py`: additive, fixed composition and Rep4Ex-inspired
  architectures, objectives and training; both mean and residual outputs.
- `vendor/`: complete local historical DDPM, sampling, training, normalization,
  graph and metric dependencies. CD, compositional CD and classifier guidance
  are instantiated and sampled in `ours/engine.py` using these modules.
- `results/reported/`: 39 completed per-seed records, original generated outputs,
  original data/reference tensors, training metadata and selection audits.
- `results/published_tables/`: archived CSV/JSON/LaTeX tables and PDF previews. PDF document metadata
  has been removed without changing extracted page text.
- `tables/`: scripts that regenerate tables, including sample SDs, guidance-scale
  choices and bridge diagnostics. Generated tables are separate from snapshots.
- `provenance/`: source mapping/hashes, original source snapshots and validation.

No dependency on the original checkout, external datasets, model downloads,
symlinks, absolute user paths or scheduler installation is needed. Full trained
weights and optimizer checkpoints are not bundled: regenerate them with the
training commands. Saved outputs support results/table reproduction immediately.

## Rebuild reported tables (no GPU; Python standard library only)

From this directory:

```bash
python verify_files.py
python reproduce_tables.py
```

This rebuilds the original and selected five-DGP tables, per-seed JSON, CSVs,
sample-standard-deviation appendix and post-hoc best-scale tables, and compares
them against the archived files. Missing seeds are an error. Saved-only layout
variants with no historical builder remain labeled archived snapshots.

## Verify data and recompute bridge diagnostics

Use Python 3.10 and the packages in `requirements.txt` (the checked environment).
A CUDA build of PyTorch is only needed for training/sampling.

```bash
python check_reproduction.py
python reproduce_tables.py --bridge
python evaluate_saved.py --suite progressive --case linear --seed 0
```

The checked host uses Python 3.10, NumPy 1.22.3, SciPy 1.11.3 and PyTorch
2.0.1+cu117. These are verification versions, not a recovered historical lockfile.

The first command regenerates the full train/validation/reference data for seed 0
of every DGP and checks exact train/validation/target tensor equality against saved data
(reference draws allow absolute error 1e-14 for platform math roundoff), as well as spline
inversion, likelihood gradients and a DDPM forward pass. The second recomputes
bridge diagnostics from saved generated samples and compares the resulting tables.
`evaluate_saved.py` recomputes all 26 methods’ metrics for a chosen saved run,
compares them to the per-seed record with tolerance 1e-10, and writes separately
to `results/recomputed/`. It also supports `--suite multimodal`, `--results-root`
and `--output`.

Only load these trusted bundled `.pt` files: historical PyTorch serialization uses
pickle. PDF snapshots are included; to compile regenerated LaTeX yourself, use a
TeX installation with booktabs, graphicx, rotating and xcolor.

## Rerun training and generation (CUDA required)

```bash
python run_experiment.py --suite progressive --index 6 --smoke
python run_experiment.py --suite multimodal --index 0 --smoke
# Full experiments: case index * 3 + seed, seeds 0, 1, 2.
for i in $(seq 0 20); do python run_experiment.py --suite progressive --index "$i"; done
for i in $(seq 0 17); do python run_experiment.py --suite multimodal --index "$i"; done
```

Outputs go to `runs/<suite>/cells/<case>/<seed>` (smoke uses `smoke` instead of
`cells`). `--output /path/to/new-suite-output` changes the suite root. Existing
completed runs are skipped; use a new output path for a fresh repeat. Archived
reported results are protected from runner writes. The full suite preserves the
historical CUDA requirement and numerical algorithms; a CPU smoke run is not
claimed. Fresh training results need not be bitwise identical across GPU/software
versions; displayed LaTeX reproduction is exact; CSV comparisons allow 1e-12 numerical
roundoff across Python versions. Training provenance
records per-seed runtime, parameter counts, validation histories and configuration.

To build tables from freshly completed runs (both suites required):

```bash
python reproduce_tables.py --results-root runs --bridge
```

This uses new per-seed results and skips comparison to historical values. It also
writes `tables/all_results_per_seed.json` and `tables/all_results_mean_std.csv`
for all 13 cases, with unscaled fractions and sample SDs. The table-specific CSVs
retain their original percentage conventions.

`datagen/*.py` defines the exact ordered case lists. The original five-DGP table
uses Discrete Reference, Linear, Cubic, Interaction and 2D Bridge. The selected
table uses Weighted Discrete Reference, Linear, Cubic, Smoothstep and 2D Bridge.
All other settings are included, rather than discarded after selection.

## Reported protocol and interpretation

20,000 training and 2,000 validation rows; 20,000 diffusion updates with
validation-selected EMA; 10,000 density updates; 3,000 adaptation updates.
DDPMs use 100 steps and width 192. LWR uses 1,024 proposals per output. LGD and CG
report scales 0.5, 1, 2, with gradient norm cap 50. Continuous evaluation uses
64 targets × 32 draws; discrete evaluation uses 2,000 draws. Validity includes
root support and both specification constraints (continuous tolerance 0.005).
Metrics include joint/X/conditional MMD, coverage, bridge overlap and likelihood
fit diagnostics. See the included original README files for definitions.

The selected DGPs are exploratory. Best-scale tables select by test-set mean
validity, not validation performance; all metrics use the same selected scale.
The original discrete reference has 5% conditional exclusion and is explicitly an
exception to the 1% criterion. Multimodal reference controls use relaxed thresholds.
Rep4Ex and fixed-composition methods are adaptations, not exact theorem replications.
Dense LGD differs in supervision as documented. No claim of matched compute is made.
Empty LWR banks use the original uniform fallback and report its frequency. Full
validation NLL is undefined/infinite for support failures, represented by null
plus the outside-support count; supported validation NLL selects the checkpoint.

The collection changes import paths and output locations, removes unrelated
historical launch code, and exposes one portable entry point. Numerical algorithms
and result values are preserved. Source mappings record every collected file and
whether it was adapted. `provenance/validation.json` reports checks actually run;
full CUDA retraining was not performed on the CPU-only collection host.

## Anonymous distribution audit

The distribution excludes personal names, email addresses and personal filesystem
paths. Published commit author and committer identities use anonymous placeholders.
PDF document metadata has been removed; numerical results and extracted page text
are preserved. The audit covers text, binary files and serialized tensor archives.

```bash
python -m pip install -r requirements-audit.txt
python check_anonymity.py --pdf --git --forbidden '<identifying substring>'
```

Pass each private name as a separate `--forbidden` argument. Arguments are not
written into reports. Repository hosting accounts, old cached commits and external
copies are outside the scope of the file audit; use an anonymous distribution
channel for double-blind review.
