#!/bin/bash
#SBATCH --job-name=vae-rq-sweep
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/rq_sweep_%A_%a.log

# rq quality threshold sweep: 5 thresholds × 1 run = 5 jobs
#
# Tests how the read-quality filter (rq) affects VAE latent geometry.
# Base filter: st='MIXED' AND et='MIXED' AND FILT=1
# Sweep: add AND rq < {threshold} for each threshold below.
#
# Uses --subsample-fractions 1.0 so ALL rows passing the filter are used
# (rq threshold already determines the effective dataset size).
# Compare vs 52M reference (no rq filter) via run_rq_vs_52M.sh.

set -euo pipefail

export UV_VAE_ROOT="$HOME/uv_vae"
export TQDM_DISABLE=1
cd ~/uv_vae

source .venv/bin/activate

mkdir -p /cta/users/patrickgao765/uv_vae/logs

# ── Configuration ──────────────────────────────────────────────
PARQUET_PATH="/cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet"
TEST_SET_PATH="/cta/users/patrickgao765/uv_vae/test_set.parquet"
BASE_OUTPUT="/cta/users/patrickgao765/uv_vae/VAE_Stability_Testing/rq_sweep"
SCRIPT="VAE_Stability_Testing/scripts/vae_subsample_sweep.py"

RQ_THRESHOLDS=(0.025 0.05 0.075 0.1 0.15)

RQ=${RQ_THRESHOLDS[$SLURM_ARRAY_TASK_ID]}
# Format threshold as a filesystem-safe string: 0.025 -> rq0025, 0.1 -> rq010
RQ_LABEL="rq$(echo "$RQ" | tr -d '.')"
ROW_FILTER="st = 'MIXED' AND et = 'MIXED' AND FILT = 1 AND rq < ${RQ}"
OUTPUT_DIR="${BASE_OUTPUT}/${RQ_LABEL}"

echo "=== rq sweep task ${SLURM_ARRAY_TASK_ID} ==="
echo "  rq threshold: < ${RQ}"
echo "  Row filter:   ${ROW_FILTER}"
echo "  Output:       ${OUTPUT_DIR}"

# Count rows passing this filter first so we know the effective dataset size
python - <<EOF
import duckdb, os
conn = duckdb.connect()
conn.execute("SET threads = 8")
parquet_path = "${PARQUET_PATH}"
row_filter = """${ROW_FILTER}"""
result = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_path}') WHERE {row_filter}").fetchone()
print(f"[row count] rq < ${RQ}: {result[0]:,} rows", flush=True)
EOF

python "$SCRIPT" \
    --parquet-path "$PARQUET_PATH" \
    --test-set-path "$TEST_SET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --row-filter "$ROW_FILTER" \
    --subsample-fractions "1.0" \
    --epochs 10 \
    --seed 42 \
    --data-seed 42 \
    --batch-size 4096 \
    --latent-dim 16 \
    --hidden-dims "256,128" \
    --learning-rate 1e-3 \
    --kl-weight 0.05 \
    --threads 8 \
    --non-interactive

echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
