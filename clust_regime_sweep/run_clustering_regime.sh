#!/bin/bash
#SBATCH --job-name=clust-regime
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-15
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/clust_regime_%A_%a.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/logs/clust_regime_%A_%a.err
#
# Clustering testing regime -- FOUR SWEEPS, CPU-only.
# See VAE_Stability_Testing/docs/clustering_regime_plan.md for the full design.
#
#   Sweep 1+3 (tasks 0-6): SWEEP THE VAE CHECKPOINT. Each pretrained checkpoint in
#       sweep_results_full was trained on a different row fraction (520K .. 52M) --
#       that IS the subsample ladder. ALL deduplicated rows are embedded through it
#       and ALL of them flow into UMAP/HDBSCAN.
#         variant A = latent -> UMAP -> HDBSCAN   (full pipeline)
#         variant B = latent -> HDBSCAN           (UMAP dropped)
#       A and B share one embedding pass, so they run in the SAME task.
#
#   Sweep 2 (tasks 7-11): SWEEP THE rq FILTER, with the VAE RETRAINED per threshold.
#       Filtering on rq changes the input distribution, so each threshold gets its own
#       VAE fit on the rows that pass it -- trained beforehand by run_rq_vae_train.sh.
#         variant D = latent -> UMAP -> HDBSCAN on rows passing rq < threshold
#
#   Sweep 4 (tasks 12-15): NO VAE, NO UMAP. Nothing is trained, so the subsample IS
#       the clustered data. Rows are reservoir-sampled, then clustered directly.
#         variant C1 = numeric + one-hot      -> HDBSCAN
#         variant C2 = that matrix -> PCA(16) -> HDBSCAN
#
# SigProfiler runs in EVERY sweep. Rows are always FILTERED first, then deduplicated,
# then sampled. No VAE is trained here -- sweep 2's are trained by run_rq_vae_train.sh.
# Cross-cell ARI/AMI/NMI are computed by the separate aggregation job (see the tail).
#
# Windows edit note:
#   sed -i 's/\r$//' scripts/run_clustering_regime.sh
# Submit (sweep 2's VAEs must exist first):
#   TRAIN_JOB=$(sbatch --parsable scripts/run_rq_vae_train.sh)
#   sbatch --dependency=afterany:${TRAIN_JOB} scripts/run_clustering_regime.sh

set -euo pipefail

# ── Environment ────────────────────────────────────────────────
export UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
export TQDM_DISABLE=1

# CPU-only: hide any GPU so torch + (guarded) cuml stay on CPU.
export CUDA_VISIBLE_DEVICES=""

cd "$UV_VAE_DIR"

# Activate the environment that has torch + hdbscan + umap-learn + SigProfilerAssignment.
# Default is the .venv the successful 1841936 array ran under; set CONDA_ENV to switch.
if [ -n "${CONDA_ENV:-}" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

mkdir -p /cta/users/patrickgao765/uv_vae/logs

# ── Configuration ──────────────────────────────────────────────
# The 7M-row test dataset from scripts/make_test_dataset.py (stage 0). Point this at the
# full featuremap parquet instead to run the regime at full scale.
SOURCE_PARQUET="${SOURCE_PARQUET:-/cta/users/patrickgao765/parquet_files/test7M.featuremap.parquet}"
FEATURE_SPEC="${FEATURE_SPEC:-$UV_VAE_DIR/ml_features.json}"
BASE_OUTPUT="${BASE_OUTPUT:-/cta/users/patrickgao765/uv_vae/clustering_regime/run1}"
SCRIPT="${SCRIPT:-scripts/clustering_regime_sweep.py}"

BASE_FILTER="st = 'MIXED' AND et = 'MIXED' AND FILT = 1"
SEED="${SEED:-42}"
THREADS="${SLURM_CPUS_PER_TASK:-16}"
DUCKDB_MEM="${DUCKDB_MEM:-32GB}"

# Root of the pretrained checkpoint ladder (nothing is retrained).
SWEEP_ROOT="${SWEEP_ROOT:-/cta/users/patrickgao765/uv_vae/VAE_Stability_Testing/sweep_results_full}"

# Sweep 1+3: the VAE subsample ladder, largest -> smallest training set.
#   100pct=52M  50pct=26M  25pct=13M  10pct=5.2M  5pct=2.6M  2pct=1.3M  1pct=520K
CHECKPOINT_PCTS=(100 50 25 10 5 2 1)

# Sweep 2: rq thresholds. Each has its OWN VAE, retrained on the rows that pass it by
# run_rq_vae_train.sh. Keep this list identical to the one in that script.
RQ_THRESHOLDS=(0.025 0.05 0.075 0.1 0.15)
RQ_VAE_ROOT="${RQ_VAE_ROOT:-/cta/users/patrickgao765/uv_vae/clustering_regime/rq_vaes}"

# Sweep 4: row subsamples clustered directly (no VAE, no UMAP). Capped at the
# deduplicated population, so points above it collapse onto 'all' -- keep them distinct.
# NOTE: C1 (150-D one-hot space) is the dominant cost -- ~24 h at ~1.1M rows in the
# 1841936 trial. Raise SUBSAMPLE_POINTS only if you have the wall-clock for it.
SUBSAMPLE_POINTS=(250000 500000 1000000 all)

# HDBSCAN min_cluster_size grid + fixed min_samples (plan 5.1).
MIN_CLUSTER_SIZES="${MIN_CLUSTER_SIZES:-100,250,500,1000}"
MIN_SAMPLES="${MIN_SAMPLES:-25}"

# Metric-evaluation subsample for O(N^2) internal metrics (silhouette/DBI/CH). The
# clustering itself still uses all rows; this only bounds the SCORING (plan 9.6).
SIL_EVAL_ROWS="${SIL_EVAL_ROWS:-50000}"

# ── Helpers ────────────────────────────────────────────────────
resolve_checkpoint() {
    # $1 = percentage label (e.g. 25) -> path to that pretrained run's model.pt
    local pct="$1"
    local found
    found="$(find "$SWEEP_ROOT/run_${pct}pct" -name model.pt 2>/dev/null | head -1)"
    if [ -z "$found" ]; then
        echo "ERROR: no model.pt under $SWEEP_ROOT/run_${pct}pct" >&2
        exit 1
    fi
    echo "$found"
}

resolve_rq_checkpoint() {
    # $1 = rq threshold -> the VAE retrained on rows passing it (stage 1 output)
    local rq="$1"
    local found
    found="$(find "$RQ_VAE_ROOT/rq_${rq}" -name model.pt 2>/dev/null | sort | tail -1)"
    if [ -z "$found" ]; then
        echo "ERROR: no retrained VAE for rq < ${rq} under $RQ_VAE_ROOT/rq_${rq}" >&2
        echo "       Run stage 1 first: sbatch scripts/run_rq_vae_train.sh" >&2
        exit 1
    fi
    echo "$found"
}

# ── Preflight ──────────────────────────────────────────────────
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: harness $SCRIPT not found." >&2
    exit 1
fi
if [ ! -d "$SWEEP_ROOT" ]; then
    echo "ERROR: checkpoint ladder not found at $SWEEP_ROOT" >&2
    exit 1
fi
if [ ! -f "$SOURCE_PARQUET" ]; then
    echo "ERROR: source parquet not found: $SOURCE_PARQUET" >&2
    exit 1
fi
mkdir -p "$BASE_OUTPUT"

# ── Resolve this array task ────────────────────────────────────
TASK="${SLURM_ARRAY_TASK_ID:-0}"

COMMON_ARGS=(
    --parquet-path "$SOURCE_PARQUET"
    --feature-spec-path "$FEATURE_SPEC"
    --output-root "$BASE_OUTPUT"
    --row-filter "$BASE_FILTER"
    --min-cluster-sizes "$MIN_CLUSTER_SIZES"
    --min-samples "$MIN_SAMPLES"
    --sil-eval-rows "$SIL_EVAL_ROWS"
    --seed "$SEED"
    --threads "$THREADS"
    --device cpu
    --duckdb-memory-limit "$DUCKDB_MEM"
    --sigprofiler-cpu "$THREADS"
    --cosmic-version 3.5
    --genome-build GRCh38
)

N_CKPT=${#CHECKPOINT_PCTS[@]}                       # 7  -> tasks 0..6
N_RQ=${#RQ_THRESHOLDS[@]}                           # 5  -> tasks 7..11
RQ_START=$N_CKPT
SUB_START=$((N_CKPT + N_RQ))                        # 12 -> tasks 12..15

if [ "$TASK" -lt "$N_CKPT" ]; then
    # ---- Sweep 1+3: VAE checkpoint ladder, ALL data downstream ----
    PCT="${CHECKPOINT_PCTS[$TASK]}"
    CKPT="$(resolve_checkpoint "$PCT")"
    echo "=== task $TASK : VAE ${PCT}pct, ALL rows -> variants A,B ==="
    echo "    checkpoint: $CKPT"
    python "$SCRIPT" \
        --mode checkpoint \
        --checkpoint-path "$CKPT" \
        --variants "A,B" \
        "${COMMON_ARGS[@]}"

elif [ "$TASK" -lt "$SUB_START" ]; then
    # ---- Sweep 2: rq filter ladder, VAE retrained per threshold ----
    RQ="${RQ_THRESHOLDS[$((TASK - RQ_START))]}"
    CKPT="$(resolve_rq_checkpoint "$RQ")"
    echo "=== task $TASK : rq < $RQ (VAE retrained on rq < $RQ) -> variant D ==="
    echo "    checkpoint: $CKPT"
    python "$SCRIPT" \
        --mode rq \
        --rq-threshold "$RQ" \
        --checkpoint-path "$CKPT" \
        --variants "D" \
        "${COMMON_ARGS[@]}"

else
    # ---- Sweep 4: no VAE, no UMAP -- subsample IS the clustered data ----
    POINT="${SUBSAMPLE_POINTS[$((TASK - SUB_START))]}"
    # Checkpoint supplies standardization stats + category vocabulary only.
    CKPT="$(resolve_checkpoint 100)"
    echo "=== task $TASK : no VAE/UMAP, rows=$POINT -> variants C1,C2 ==="
    python "$SCRIPT" \
        --mode subsample \
        --sample-rows "$POINT" \
        --checkpoint-path "$CKPT" \
        --variants "C1,C2" \
        "${COMMON_ARGS[@]}"
fi

echo "=== task $TASK complete ==="

# ─────────────────────────────────────────────────────────────────────────────
# Full submission -- build the test dataset, train sweep 2's VAEs, cluster, aggregate:
#
#   TEST_JOB=$(sbatch --parsable scripts/run_make_test_dataset.sh)
#   TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${TEST_JOB} scripts/run_rq_vae_train.sh)
#   ARRAY_JOB=$(sbatch --parsable --dependency=afterany:${TRAIN_JOB} scripts/run_clustering_regime.sh)
#   sbatch --dependency=afterany:${ARRAY_JOB} scripts/run_clustering_regime_aggregate.sh
#
# afterok on the first hop only -- there is nothing to run if the test dataset fails.
#
# NOTE: afterany (not afterok) -- in the 1841936 trial one task hit the time limit and
# afterok silently prevented aggregation from ever running. afterany still aggregates
# whatever cells did finish.
# ─────────────────────────────────────────────────────────────────────────────
