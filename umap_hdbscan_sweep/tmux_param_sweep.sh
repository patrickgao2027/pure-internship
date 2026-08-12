#!/usr/bin/env bash
# HDBSCAN parameter sweep with full-cohort SigProfiler on every cell.
#
# Runs the grid from hdbscan_param_sweep.py: fit sizes x min_cluster_size x min_samples,
# each cell labelled across all 157.5M loci via fast_predict's RBC backend (5.3 min instead
# of the 4.8 h approximate_predict needs at 25M), then SigProfiler.
#
#   bash umap_hdbscan_sweep/tmux_param_sweep.sh                 # run it
#   DRY_RUN=1 bash umap_hdbscan_sweep/tmux_param_sweep.sh       # grid + projected cost only
#   BUILD_INDEX_ONLY=1 bash umap_hdbscan_sweep/tmux_param_sweep.sh   # one-time SBS96 cache
#   AGGREGATE=1 bash umap_hdbscan_sweep/tmux_param_sweep.sh     # rank what has finished
#
# Env overrides:
#   FIT_SIZES=500000,1000000,5000000,25000000
#   MIN_CLUSTER_SIZES=250,500,1000,2500     held FIXED across fit sizes (the whole point)
#   MIN_SAMPLES=5,15                        15 is skipped at 25M -- it OOMed on the 47 GB card
#   SIGNATURE_SET=uv_only|full              see Phase 0 note below
#   GPU_TOTAL_GB=40
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EMBED_DIR="${EMBED_DIR:-$REPO_ROOT/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed}"
SCALING_DIR="${SCALING_DIR:-$REPO_ROOT/umap_hdbscan_sweep/umap_tests/hdbscan_scaling}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/umap_hdbscan_sweep/umap_tests/param_sweep}"
SCRIPT="$REPO_ROOT/umap_hdbscan_sweep/hdbscan_param_sweep.py"

COORDS="${COORDS:-$SCALING_DIR/coords.npy}"
CONTEXT="${CONTEXT:-$EMBED_DIR/context.parquet}"

FIT_SIZES="${FIT_SIZES:-500000,1000000,5000000,25000000}"
MIN_CLUSTER_SIZES="${MIN_CLUSTER_SIZES:-250,500,1000,2500}"
MIN_SAMPLES="${MIN_SAMPLES:-5,15}"
METHODS="${METHODS:-eom}"
EPSILONS="${EPSILONS:-0.0}"
SIGNATURE_SET="${SIGNATURE_SET:-uv_only}"
GPU_TOTAL_GB="${GPU_TOTAL_GB:-40}"
SEED="${SEED:-42}"

if command -v micromamba &>/dev/null; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate "${MAMBA_ENV:-uv_vae}"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate patrickg
fi

export TQDM_DISABLE=1
mkdir -p "$OUTPUT_DIR"

if [[ "${AGGREGATE:-0}" == "1" ]]; then
    exec python "$SCRIPT" --aggregate --output-dir "$OUTPUT_DIR"
fi

if [[ "${BUILD_INDEX_ONLY:-0}" == "1" ]]; then
    exec python "$SCRIPT" --build-index-only \
        --context "$CONTEXT" --output-dir "$OUTPUT_DIR" \
        --signature-set "$SIGNATURE_SET"
fi

EXTRA=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then EXTRA+=(--dry-run); fi

echo "[$(date +%H:%M:%S)] HDBSCAN parameter sweep"
echo "  coords        : $COORDS"
echo "  context       : $CONTEXT"
echo "  output        : $OUTPUT_DIR"
echo "  fit sizes     : $FIT_SIZES"
echo "  mcs (fixed)   : $MIN_CLUSTER_SIZES"
echo "  min_samples   : $MIN_SAMPLES"
echo "  signature set : $SIGNATURE_SET"
echo
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
echo

python "$SCRIPT" \
    --coords "$COORDS" \
    --context "$CONTEXT" \
    --output-dir "$OUTPUT_DIR" \
    --fit-sizes "$FIT_SIZES" \
    --min-cluster-sizes "$MIN_CLUSTER_SIZES" \
    --min-samples "$MIN_SAMPLES" \
    --methods "$METHODS" \
    --epsilons "$EPSILONS" \
    --signature-set "$SIGNATURE_SET" \
    --gpu-budget-gb "$GPU_TOTAL_GB" \
    --seed "$SEED" \
    "${EXTRA[@]}" 2>&1 | tee -a "$OUTPUT_DIR/param_sweep.log"
