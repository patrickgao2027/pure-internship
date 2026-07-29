# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Research workspace for the **PURE Summer Program (Sabancı University)**. The scientific
question: *can a VAE trained on a representative subsample of a huge genomic variant dataset
(52M+ reads × ~95 samples ≈ billions of rows) reproduce the latent space of a VAE trained on
the full data?* Everything here exists to train tabular VAEs on parquet "feature maps" of
single-read SNVs, encode them to a latent space, and cluster that space (UMAP → HDBSCAN →
SigProfiler mutational-signature assignment) — then measure how stable the latent geometry is
across data sizes, seeds, epochs, batch sizes, KL weights, and early-stopping rules.

The top-level `README.md` is a **dated research journal**, not usage docs — read it for intent,
decisions, and rejected ideas (e.g. KL-weight feature selection was explicitly rejected).

## Repository layout

- **`uv_vae/`** — the actual installable Python package and the only code meant to be reused.
  Has its own `pyproject.toml`, `tests/`, and a subfolder-scoped `uv_vae/CLAUDE.md`. The
  importable package is `uv_vae/uv_vae/`.
- **`*_Testing/` and `Early_Stopping_Tests/`** — self-contained experiment folders, each with
  a `Python Files/` dir (entry-point scripts) and a `scripts/` dir (SLURM/bash runners):
  `VAE_Stability_Testing/`, `Batch_Size_Learning_Rate_Testing/`, `KL_Weight_Testing/`,
  `Early_Stopping_Tests/`. These **do not modify the core package** — they import it (see
  "How the experiment folders find the package" below) and add analysis on top.
- **`learning_and_testing/`** — scratch notebooks (VAE/UMAP/HDBSCAN/TensorFlow learning). Not
  part of the pipeline.
- **`train_then_cluster_.../`, `comparison_results/`** — saved run outputs kept in-tree.
- `important links.md` — HPC/SLURM/conda command cheatsheet and cluster URLs.

Note: the git repo root is this `PURE Files/` folder; `uv_vae/` is a subdirectory of it, **not
a separate repo**. On the HPC the folders are instead deployed as **siblings under `$HOME`**
(`$HOME/uv_vae`, `$HOME/Early_Stopping_Tests`, …) — the scripts assume that layout.

## Core package architecture (`uv_vae/uv_vae/`)

Data flow: **parquet → DuckDB SQL filter → feature specs → VAE → latent (mu) → UMAP 2D →
HDBSCAN → SigProfiler**. The default row filter everywhere is
`st = 'MIXED' AND et = 'MIXED' AND FILT = 1`.

- `features.py` — `FeatureSpec` + `load_feature_specs()`. Feature definitions live in
  `uv_vae/ml_features.json` (`type` is `"c"` categorical, `"int"`/`"float"` numeric; categorical
  specs carry a `values` string→code map).
- `data.py` — all DuckDB access: `sample_frame()` (reservoir `USING SAMPLE … REPEATABLE(seed)`),
  `stream_parquet_batches()` (Arrow record-batch reader), `get_row_count`, `get_non_null_counts`,
  `split_specs()`, `quote_ident()`.
- `preprocess.py` — `prepare_tensors()` (in-RAM path) and `transform_frame()`; numeric
  standardisation + categorical encoding + train/val split.
- `model.py` — `TabularVAE(nn.Module)` + `VAEConfig`: per-feature embeddings for categoricals,
  concatenated with numerics, symmetric encoder/decoder MLP, separate numeric + per-categorical
  reconstruction heads.
- `training.py` — the **stock** training loop (`train()`), `seed_everything()`, `build_model()`,
  `run_epoch()`, `compute_loss()`, `TrainingConfig`, and the run-artifact writers. **This file
  is the source of truth other trainers reuse — do not fork its math.**
- `early_stopping.py` — additive trainer `train_with_early_stopping()` (in-RAM). Adds
  per-dimension KL, posterior-mean variance, and **active-unit count** (Burda et al. 2016);
  stops only when val ELBO stagnates **AND** active units stop moving.
- `streaming.py` — additive trainer `train_with_early_stopping_streaming()` for full-dataset
  training with flat memory. `StreamingParquetDataset` (IterableDataset, windowed shuffle,
  deterministic `row_index % val_denominator` split), `TabularVAEWithDropout`, AMP, LR warmup,
  optional cuDF/cuPy GPU encode path.
- `multi_parquet.py` — additive reader for the **95-file** case: one reader per parquet,
  every batch drawn from all of them in proportion to each file's post-filter row count
  (largest-remainder allocation), each file read in **shuffled row-group order** to destroy
  the genomic clustering (the files are sorted by `CHROM,POS` with ~88% of adjacent rows at
  the same locus). Supports epoch sharding. No torch import, so it is testable standalone.
- `splitting.py` — content-hash train/val predicates (SplitMix64, constants pinned locally).
  Replaces the positional split, which cannot survive interleaved reading. Default
  `per_sample_site_hash` keeps every read at a locus on one side.
- `stats_cache.py` — per-file row counts + normalisation statistics in one scan, cached on
  disk, combined across files with Chan–Golub–LeVeque as a balanced tree.
- `multi_streaming.py` — the trainer for the interleaved reader. Imports `_run_training_epoch`,
  `run_val_epoch_with_diagnostics` and `_build_model` from `streaming.py`; adds no maths.
- `convergence.py` — `ConvergenceTracker`: encodes a fixed test set each epoch and compares
  consecutive epochs via Procrustes / linear CKA / trustworthiness.
- `latent_metrics.py` — post-hoc `evaluate_checkpoint()`: activation quality (active units,
  mean KL, collapsed-dim %, participation ratio, off-diagonal covariance), structure
  (trustworthiness), reconstruction MSE.
- `evaluation.py` — Procrustes/CKA/trustworthiness/collapse diagnostics used by sweeps.
- `inference.py` — `LatentInference.from_checkpoint()` encodes new data through a trained model.
- `pipeline_vae.py` / `cli.py` / `train_cli.py` — pipeline orchestration and argparse entry points.

**Run-artifact contract (important):** all three trainers — `training.train`,
`early_stopping.train_with_early_stopping`, `streaming.train_with_early_stopping_streaming` —
write the **same** files to a timestamped `run_YYYYMMDDTHHMMSSZ/` dir: `model.pt`,
`feature_report.json`, `preprocess_report.json`, `training_report.json`, `summary.json` (early
stopping / streaming also add `diagnostics_report.json`, streaming adds `convergence_report.json`
when a test parquet is given). Keeping these identical is what lets `inference.py` and the
clustering pipeline consume any run unchanged. **Preserve this contract when editing any trainer.**

### Three training modes (same `TrainingConfig`, different I/O)

| Mode | Function | Data | Use |
|------|----------|------|-----|
| Sampled | `training.train` | reservoir-sample N rows into RAM | stability sweeps, quick runs |
| All-in-RAM | `early_stopping.train_with_early_stopping` | all filtered rows into RAM | mid-size, wants diagnostics |
| Streaming | `streaming.train_with_early_stopping_streaming` | streamed from parquet, flat memory | full dataset / OOM-safe |
| Interleaved | `multi_streaming.train_interleaved` | many per-sample parquets at once | the 95-file cohort |

`Early_Stopping_Tests/Python Files/train_with_early_stopping.py` is the unified CLI:
`--parquet-paths` (globs accepted) selects the interleaved trainer; otherwise `--parquet-path`
plus `--streaming`, `--use-all` or `--sample-rows` picks one of the other three.
`--parquet-paths` is *refused* alongside those three rather than silently ignoring them.

**Multi-file loading** is documented in `MULTI_PARQUET_LOADING.md` (what was built, what was
measured, what is still unverified); `SAMPLING_STRATEGY.md` is the decision record for why the
other six strategies were rejected and holds the still-open advisor questions. Run the
statistics preflight before the first cohort run — it is cheap and its cache is reused:

```bash
PARQUET_GLOB='/cta/users/patrickgao765/parquet_files/*.featuremap.parquet' \
    STATS_ONLY=1 bash Early_Stopping_Tests/scripts/tmux_train_multi.sh
```

### How the experiment folders find the core package

Scripts under a `Python Files/` dir inject the package onto `sys.path` with
`Path(__file__).resolve().parents[2] / "uv_vae"`. This hard-codes the depth: the script **must**
sit exactly at `<ExperimentFolder>/Python Files/<script>.py` (two levels under the root that
contains `uv_vae/`). Moving such a script to a different depth silently breaks the import — keep
the `Python Files/` layout, or update `parents[N]` to match.

## Determinism (this project cares a lot about it)

- `seed_everything()` (in `training.py`) pins python/numpy/torch RNGs, CUBLAS workspace, disables
  TF32, and enables deterministic algorithms.
- **DuckDB reservoir sampling is only reproducible with `threads=1`** — the sampling connection is
  always opened single-threaded even when the count query uses many threads.
- `TrainingConfig.data_seed` controls *row sampling* independently from `seed` (model init /
  shuffling / split), so you can hold the model seed fixed while sweeping which rows are drawn.

## Commands

Run package-level commands from **`uv_vae/`** (that's where `pyproject.toml` lives; deps are
managed with `uv`, torch pinned to the CUDA-12.8 index on Linux).

```bash
cd uv_vae
uv sync                       # install/resolve the environment
uv run pytest tests/          # run all tests
uv run pytest tests/test_evaluation.py::test_run_subsample_experiment_and_aggregate_results  # single test
```

Training / pipeline (see each script's `--help`; paths in the SLURM scripts point at the HPC and
usually need editing). Feature spec is always passed explicitly as `--feature-spec-path`:

```bash
# Train with early stopping (streaming, full dataset)
python "Early_Stopping_Tests/Python Files/train_with_early_stopping.py" \
    --parquet-path <parquet> --feature-spec-path uv_vae/ml_features.json \
    --output-dir artifacts --streaming --epochs 100 --patience 8

# End-to-end SLURM job: combine parquets → train (early stopping) → cluster → SigProfiler
sbatch uv_vae/scripts/run_full_pipeline.sh      # edit SAMPLE_1/SAMPLE_2 first

# Clustering pipeline on an existing checkpoint
python uv_vae/scripts/run_variant_cluster_pipeline.py --checkpoint-path <model.pt> \
    --parquet-path <parquet> --output-root <dir> --use-all

# Sweeps (each experiment folder has its own SLURM runner)
sbatch Batch_Size_Learning_Rate_Testing/scripts/run_batch_lr_sweep_gpu.sh
sbatch KL_Weight_Testing/scripts/run_kl_sweep.sh
```

Most runs are configured through **environment variables** consumed by the shell scripts
(`PARQUET_PATH`/`COMBINED`, `ROW_FILTER`, `SEED`, `EPOCH_CEILING`, `PATIENCE`, `INPUT_DROPOUT`,
`HIDDEN_DROPOUT`, `UV_VAE_DIR`, `EARLY_STOPPING_DIR`, `TEST_PARQUET`, …) plus per-script CLI flags.

## HPC / SLURM environment

- **GPU node `miletus.sabanciuniv.edu`** — one RTX PRO 5000 Blackwell (48 GB), **no SLURM**.
  Environment is a **micromamba env named `uv_vae`** (`MAMBA_ENV`, the default). Use the
  tmux runners here, not the `sbatch` scripts — see `TMUX_RUNNERS.md`.
- Older CPU cluster `tosun.sabanciuniv.edu`, user `patrickgao765`; conda env `patrickg`.
- SLURM (tosun only): `--account=adelab --partition=genomics --qos=adelab`.
- Env activation in every `.sh` is manager-agnostic: `CONDA_ENV` wins if set, else
  micromamba `${MAMBA_ENV:-uv_vae}` if the binary exists, else conda `patrickg`. One
  script therefore runs on either cluster unedited.
- Parquet data under `/cta/users/patrickgao765/parquet_files/`.
- Scripts are edited on Windows, so strip CRLF before submitting: `sed -i 's/\r$//' <script>.sh`.
- Set `export TQDM_DISABLE=1` for non-interactive jobs.
- `import matplotlib.pyplot` is deferred inside plotting functions (matplotlib can crash on HPC
  when style files are missing).
- cuML/cuDF give zero-code-change GPU acceleration when CUDA is present; every path falls back to
  CPU, so jobs run either way (a CPU-only timing run over-estimates a GPU run's cost).

## Gotchas

- **Windows line endings**: shell scripts fail on the HPC until `sed -i 's/\r$//'` is run; the
  repo has `* text=auto`-style churn (git will warn about LF↔CRLF on nearly every `.sh`/`.py`).
- The subfolder `uv_vae/CLAUDE.md` predates the script reorganisation: its example commands
  (`scripts/run.sh`, `scripts/run_train_then_cluster.sh`) no longer exist — prefer
  `scripts/run_full_pipeline.sh` and the `Early_Stopping_Tests` CLI. Treat that file's *findings*
  section as reliable but verify its *command* examples against the current tree.
