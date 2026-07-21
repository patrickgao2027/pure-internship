#!/bin/bash
#
# Epoch sweep — LARGE jobs: row counts 10M–26M (3 counts × 4 epochs = 12 jobs)
# Worst case: 26M rows × 50 epochs ≈ 13 hours. Time limit set to 16h.
#
#SBATCH --job-name=vae-epoch-lg
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-11
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/epoch_sweep_lg_%A_%a.log

# Tests whether more training epochs stabilize latent geometry at larger data sizes.
# All models use seed 42 for comparability with existing sweep results.

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate patrickg

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

ROW_COUNTS=(10000000 13000000 26000000)
EPOCH_VALUES=(10 20 30 50)

# ── Map array index to (row_count, epochs) ─────────────────────
NUM_ROWS=${#ROW_COUNTS[@]}
NUM_EPOCHS=${#EPOCH_VALUES[@]}

ROW_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_EPOCHS ))
EPOCH_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_EPOCHS ))

ROWS=${ROW_COUNTS[$ROW_IDX]}
EPOCHS=${EPOCH_VALUES[$EPOCH_IDX]}

OUTPUT_DIR="${BASE_OUTPUT}/rows_${ROWS}_epochs_${EPOCHS}"

echo "=== Epoch sweep (large) task ${SLURM_ARRAY_TASK_ID} ==="
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
