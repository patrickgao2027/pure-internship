#!/bin/bash
#
# Epoch sweep — SMALL jobs: row counts 750K–5M (4 counts × 4 epochs = 16 jobs)
# Safe under 4h. Submit large jobs separately via run_epoch_sweep_large.sh.
#
#SBATCH --job-name=vae-epoch-sm
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/epoch_sweep_sm_%A_%a.log

# Tests whether more training epochs stabilize latent geometry at smaller data sizes.
# All models use seed 42 for comparability with existing sweep results.

set -euo pipefail

# Environment: micromamba 'uv_vae' on miletus, conda 'patrickg' on tosun.
# Guarded so one script works on either cluster -- `conda shell.bash hook` is not
# a micromamba subcommand, and on a node without conda it expands to nothing and
# the following `conda activate` aborts the job under `set -e`.
# Override with MAMBA_ENV or CONDA_ENV.
if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif command -v micromamba >/dev/null 2>&1; then
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "${MAMBA_ENV:-uv_vae}"
else
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-patrickg}"
fi

export UV_VAE_ROOT="$HOME/uv_vae"
export TQDM_DISABLE=1
cd ~/uv_vae

mkdir -p /cta/users/patrickgao765/uv_vae/logs

# ── Configuration ──────────────────────────────────────────────
PARQUET_PATH="/cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet"
TEST_SET_PATH="/cta/users/patrickgao765/uv_vae/test_set.parquet"
BASE_OUTPUT="/cta/users/patrickgao765/uv_vae/VAE_Stability_Testing/epoch_sweep"
SCRIPT="VAE_Stability_Testing/scripts/vae_subsample_sweep.py"
SEED=42

ROW_COUNTS=(750000 1000000 2500000 5000000)
EPOCH_VALUES=(10 20 30 50)

# ── Map array index to (row_count, epochs) ─────────────────────
NUM_ROWS=${#ROW_COUNTS[@]}
NUM_EPOCHS=${#EPOCH_VALUES[@]}

ROW_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_EPOCHS ))
EPOCH_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_EPOCHS ))

ROWS=${ROW_COUNTS[$ROW_IDX]}
EPOCHS=${EPOCH_VALUES[$EPOCH_IDX]}

OUTPUT_DIR="${BASE_OUTPUT}/rows_${ROWS}_epochs_${EPOCHS}"

echo "=== Epoch sweep (small) task ${SLURM_ARRAY_TASK_ID} ==="
echo "  Rows:   ${ROWS}"
echo "  Epochs: ${EPOCHS}"
echo "  Seed:   ${SEED}"
echo "  Output: ${OUTPUT_DIR}"

python "$SCRIPT" \
    --parquet-path "$PARQUET_PATH" \
    --test-set-path "$TEST_SET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --max-sample-rows "$ROWS" \
    --subsample-fractions "1.0" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --batch-size 4096 \
    --latent-dim 16 \
    --hidden-dims "256,128" \
    --learning-rate 1e-3 \
    --kl-weight 0.05 \
    --threads 8 \
    --non-interactive

echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
