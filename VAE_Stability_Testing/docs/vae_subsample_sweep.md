# VAE Subsample Stability Sweep

Test how a VAE's latent space changes when trained on progressively smaller subsamples of the input data. The script trains one VAE per subsample size, encodes a fixed test set through each, and computes stability metrics comparing each subsample's latent geometry to the full-data baseline.

## Prerequisites

- A source feature map parquet file (the same one used for regular VAE training)
- A pre-drawn test set parquet containing the ML feature columns from `ml_features.json`
- The `uv_vae` project environment (conda or `uv sync`)
- Set `UV_VAE_ROOT` to the path of the `uv_vae` repo (defaults to `~/uv_vae`)

### Creating the test set

Draw a fixed test set once in a notebook or script. Use a different seed from training (default training seed is 42):

```python
import os, sys
from pathlib import Path

sys.path.insert(0, os.environ.get("UV_VAE_ROOT", str(Path.home() / "uv_vae")))

from uv_vae.data import connect_duckdb, sample_frame
from uv_vae.features import load_feature_specs

feature_specs = load_feature_specs("/path/to/uv_vae/ml_features.json")
feature_names = [spec.name for spec in feature_specs]

with connect_duckdb(threads=1) as conn:
    test_df = sample_frame(
        conn=conn,
        parquet_path="/path/to/featuremap.parquet",
        feature_names=feature_names,
        sample_rows=100_000,
        seed=99,
        where="st = 'MIXED' AND et = 'MIXED' AND FILT = 1 AND REF != 'N'",
    )

test_df.write_parquet("/path/to/test_set.parquet")
```

## Usage

```bash
export UV_VAE_ROOT=~/uv_vae
cd ~/VAE_Stability_Testing

python scripts/vae_subsample_sweep.py \
    --parquet-path /path/to/featuremap.parquet \
    --test-set-path /path/to/test_set.parquet \
    --output-dir sweep_results \
    --epochs 10 \
    --threads 8
```

For unattended SLURM/batch jobs, add `--non-interactive` to skip the continue/stop prompt after each training run.

## What it does

1. **Loads the fixed test set** from `--test-set-path`
2. **For each subsample fraction** (default: 100%, 75%, 50%, 25%, 10%, 5%):
   - Trains a VAE on `max_sample_rows x fraction` rows using the existing `training.train()` pipeline
   - Saves the checkpoint to `<output-dir>/run_<N>pct/<timestamp>/model.pt`
   - Encodes the test set through the trained VAE -> `(N_test, latent_dim)` matrix
   - The first (largest) fraction becomes the reference embedding
   - Computes all stability metrics comparing each subset embedding to the reference
   - Saves cumulative results to `sweep_results.json` after each fraction
   - In interactive mode, asks whether to continue to the next fraction
3. **Generates a plot** (`sweep_results.png`) with all metrics vs. sample size

## Checkpoint and resume

The sweep checkpoints after every completed fraction. If the job dies, gets preempted, or you choose to stop:

- `sweep_results.json` contains results for all completed fractions
- Each `run_<N>pct/` directory contains the full training artifacts including `model.pt`

**To resume:** re-run the exact same command. The script will:

1. Read `sweep_results.json` from `--output-dir`
2. Verify each completed fraction's `model.pt` still exists on disk
3. Re-encode the test set through the reference (largest-fraction) VAE
4. Skip all completed fractions
5. Continue training from the next incomplete fraction

## Metrics

All metrics measure VAE latent space quality only — no downstream clustering or classification.

| Metric | What it measures | Ideal |
|---|---|---|
| **Procrustes disparity** | Global shape difference between full and subset latent spaces after optimal rotation/scaling alignment | Lower -> 0 |
| **Linear CKA** | Representational similarity between two embedding matrices, invariant to rotation and isotropic scaling | Higher -> 1.0 |
| **Trustworthiness** | Whether k-nearest neighbours in the input feature space remain neighbours in the latent space | Higher -> 1.0 |
| **Latent collapse score** | Mean standard deviation across latent dimensions — detects whether the VAE is using all latent dims or collapsing some to zero variance | Higher = more active dims |
| **Val total loss** | Reconstruction + KL divergence loss on the training validation split | Lower |

### Interpreting the results

Look for the **elbow point** in the procrustes / CKA curves: the smallest subsample size where the metrics plateau. Below that point, reducing data further causes the latent space geometry to diverge from the full-data baseline. Above it, you can train with less data and get an equivalent latent representation.

- If procrustes stays flat from 100% down to 25% then jumps at 10%, the VAE is stable at 25% of the data.
- If CKA drops steadily with no plateau, the VAE is sensitive to data volume across the range tested — consider testing intermediate fractions.
- If latent collapse drops at small fractions, the VAE doesn't have enough data to learn a distributed representation and some latent dimensions go unused.
- Val loss naturally rises at smaller fractions (less data = worse fit) — what matters is whether the *relative* latent geometry (procrustes/CKA) degrades at the same rate or stays stable despite higher loss.

## Arguments

### Required

| Argument | Description |
|---|---|
| `--parquet-path` | Path to the source feature map parquet (the file the VAE trains on) |
| `--test-set-path` | Path to the fixed test set parquet (drawn once, used for all VAEs) |

### Optional

| Argument | Default | Description |
|---|---|---|
| `--output-dir` | `sweep_results` | Directory for all outputs: checkpoints, results JSON, plot |
| `--subsample-fractions` | `1.0,0.75,0.5,0.25,0.1,0.05` | Comma-separated fractions of `--max-sample-rows` to train at |
| `--max-sample-rows` | `1000000` | Full-data sample size (100% fraction). Fractions are computed from this value |
| `--row-filter` | `st = 'MIXED' AND et = 'MIXED' AND FILT = 1` | DuckDB SQL filter applied to the parquet during training |
| `--epochs` | `10` | Training epochs per VAE |
| `--batch-size` | `4096` | PyTorch batch size for training and inference |
| `--latent-dim` | `16` | VAE latent dimensionality |
| `--hidden-dims` | `256,128` | Comma-separated encoder/decoder hidden layer widths |
| `--learning-rate` | `1e-3` | AdamW learning rate |
| `--kl-weight` | `0.05` | KL divergence weight in the VAE loss |
| `--train-fraction` | `0.9` | Train/val split ratio |
| `--seed` | `42` | Random seed for training (sampling, splitting, PyTorch) |
| `--threads` | `8` | DuckDB thread count for parquet scanning during training |
| `--feature-spec-path` | `$UV_VAE_ROOT/ml_features.json` | Path to the ML feature spec JSON |
| `--non-interactive` | `false` | Skip the continue/stop prompt after each training run |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `UV_VAE_ROOT` | `~/uv_vae` | Path to the `uv_vae` repository (needed for importing the library) |

## Output structure

```
sweep_results/
├── sweep_results.json        # cumulative metrics for all completed fractions
├── sweep_results.png         # 5-panel plot of metrics vs. sample size
├── run_100pct/
│   └── run_<timestamp>/
│       ├── model.pt
│       ├── feature_report.json
│       ├── preprocess_report.json
│       ├── training_report.json
│       └── summary.json
├── run_75pct/
│   └── run_<timestamp>/
│       └── ...
├── run_50pct/
│   └── ...
└── ...
```

## Example: SLURM batch job

```bash
#!/bin/bash
#SBATCH --job-name=vae-sweep
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=sweep_%j.log

source ~/.bashrc
conda activate patrickg

export UV_VAE_ROOT="$HOME/uv_vae"
cd ~/VAE_Stability_Testing

python scripts/vae_subsample_sweep.py \
    --parquet-path data/featuremap.parquet \
    --test-set-path data/test_set.parquet \
    --output-dir sweep_results \
    --max-sample-rows 1000000 \
    --epochs 10 \
    --threads 8 \
    --non-interactive
```

If the job gets preempted, resubmit the same script — it resumes from the last completed fraction.
