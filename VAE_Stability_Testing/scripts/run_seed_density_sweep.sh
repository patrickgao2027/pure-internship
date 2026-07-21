#!/bin/bash
#SBATCH --job-name=vae-seed-density
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-59
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/seed_density_%A_%a.log

# Dense seed sweep: 3 row counts × 20 data seeds = 60 jobs
# Purpose: validate that the 5M stability threshold found with 5 seeds holds
# with a larger seed population. Uses a fresh set of 20 seeds (none overlap
# with the original 5: 7, 13, 42, 67, 99).
#
# Row counts chosen at the key stability transition: 2.5M (below threshold),
# 5M (at threshold), 10M (above threshold).
#
# Only the DuckDB row sampling seed varies (--data-seed).
# Training seed is fixed at 42 to isolate data sampling variability.

set -euo pipefail

export UV_VAE_ROOT="$HOME/uv_vae"
export TQDM_DISABLE=1
cd ~/uv_vae

source .venv/bin/activate

mkdir -p /cta/users/patrickgao765/uv_vae/logs

# ── Configuration ──────────────────────────────────────────────
PARQUET_PATH="/cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet"
TEST_SET_PATH="/cta/users/patrickgao765/uv_vae/test_set.parquet"
BASE_OUTPUT="/cta/users/patrickgao765/uv_vae/VAE_Stability_Testing/seed_density_sweep"
SCRIPT="VAE_Stability_Testing/scripts/vae_subsample_sweep.py"

ROW_COUNTS=(2500000 5000000 10000000)

# 20 fresh seeds (none of the original 5: 7, 13, 42, 67, 99)
DATA_SEEDS=(1 2 3 4 5 6 8 9 10 11 14 15 20 25 30 50 100 123 200 314)

# ── Map array index to (row_count, data_seed) ─────────────────
NUM_ROWS=${#ROW_COUNTS[@]}      # 3
NUM_SEEDS=${#DATA_SEEDS[@]}     # 20

ROW_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))

ROWS=${ROW_COUNTS[$ROW_IDX]}
DATA_SEED=${DATA_SEEDS[$SEED_IDX]}
TRAIN_SEED=42
EPOCHS=10

OUTPUT_DIR="${BASE_OUTPUT}/dataseed_${DATA_SEED}/rows_${ROWS}"

echo "=== Seed density sweep task ${SLURM_ARRAY_TASK_ID} ==="
echo "  Rows:       ${ROWS}"
echo "  Data seed:  ${DATA_SEED}"
echo "  Train seed: ${TRAIN_SEED}"
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
