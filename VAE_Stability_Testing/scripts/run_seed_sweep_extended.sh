#!/bin/bash
#SBATCH --job-name=vae-seed-sweep
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-34
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/seed_sweep_%A_%a.log

# Seed sweep: 7 row counts × 5 data seeds = 35 jobs
# Extends previous seed tests (750K, 1M only) to the full 750K–26M range.
# Measures whether DATA SAMPLING variability washes out at larger data sizes.
#
# Only the DuckDB row sampling seed varies (--data-seed).
# Training seed is fixed at 42 (weight init, train/val split, batch order)
# so geometry differences are attributable to which rows were sampled.

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
BASE_OUTPUT="/cta/users/patrickgao765/uv_vae/VAE_Stability_Testing/seed_sweep_extended"
SCRIPT="VAE_Stability_Testing/scripts/vae_subsample_sweep.py"

ROW_COUNTS=(750000 1000000 2500000 5000000 10000000 13000000 26000000)
DATA_SEEDS=(7 13 42 67 99)

# ── Map array index to (row_count, data_seed) ─────────────────
NUM_ROWS=${#ROW_COUNTS[@]}
NUM_SEEDS=${#DATA_SEEDS[@]}

ROW_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))

ROWS=${ROW_COUNTS[$ROW_IDX]}
DATA_SEED=${DATA_SEEDS[$SEED_IDX]}
TRAIN_SEED=42
EPOCHS=10

OUTPUT_DIR="${BASE_OUTPUT}/dataseed_${DATA_SEED}/rows_${ROWS}"

echo "=== Seed sweep task ${SLURM_ARRAY_TASK_ID} ==="
echo "  Rows:       ${ROWS}"
echo "  Data seed:  ${DATA_SEED}  (DuckDB row sampling only)"
echo "  Train seed: ${TRAIN_SEED} (weight init, split, batch order)"
echo "  Epochs:     ${EPOCHS}"
echo "  Output:     ${OUTPUT_DIR}"

python "$SCRIPT" \
    --parquet-path "$PARQUET_PATH" \
    --test-set-path "$TEST_SET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --max-sample-rows "$ROWS" \
    --subsample-fractions "1.0" \
    --epochs "$EPOCHS" \
    --seed "$TRAIN_SEED" \
    --data-seed "$DATA_SEED" \
    --batch-size 4096 \
    --latent-dim 16 \
    --hidden-dims "256,128" \
    --learning-rate 1e-3 \
    --kl-weight 0.05 \
    --threads 8 \
    --non-interactive

echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
