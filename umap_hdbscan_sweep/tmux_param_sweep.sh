#!/usr/bin/env bash
# HDBSCAN parameter sweep: fit, label all 157.5M loci, save the labels.
#
# Labelling goes through fast_predict's RBC backend (5.3 min instead of the 4.8 h
# approximate_predict needs at 25M). Signature assignment is NOT part of this -- run
# uv_vae/scripts/run_variant_cluster_pipeline.py against a saved cohort_labels.npy for that.
#
#   bash umap_hdbscan_sweep/tmux_param_sweep.sh                 # run it
#   DRY_RUN=1 bash umap_hdbscan_sweep/tmux_param_sweep.sh       # grid + projected cost only
#   BUILD_INDEX_ONLY=1 bash umap_hdbscan_sweep/tmux_param_sweep.sh   # one-off SBS96 cache
#   AGGREGATE=1 bash umap_hdbscan_sweep/tmux_param_sweep.sh     # summarise what has finished
#
# Env overrides:
#   FIT_SIZES=500000,1000000,5000000,25000000
#   MIN_CLUSTER_SIZES=250,500,1000,2500     held FIXED across fit sizes (the whole point)
#   MIN_SAMPLES=5,15                        15 is skipped at 25M -- it OOMed on the 47 GB card
#   GPU_TOTAL_GB=40
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCALING_DIR="${SCALING_DIR:-$REPO_ROOT/umap_hdbscan_sweep/umap_tests/hdbscan_scaling}"
EMBED_DIR="${EMBED_DIR:-$REPO_ROOT/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/umap_hdbscan_sweep/umap_tests/param_sweep}"
SCRIPT="$REPO_ROOT/umap_hdbscan_sweep/hdbscan_param_sweep.py"

COORDS="${COORDS:-$SCALING_DIR/coords.npy}"
CONTEXT="${CONTEXT:-$EMBED_DIR/context.parquet}"
SIGPROFILER_CPU="${SIGPROFILER_CPU:-16}"

FIT_SIZES="${FIT_SIZES:-500000,1000000,5000000,25000000}"
MIN_CLUSTER_SIZES="${MIN_CLUSTER_SIZES:-250,500,1000,2500}"
MIN_SAMPLES="${MIN_SAMPLES:-5,15}"
METHODS="${METHODS:-eom}"
EPSILONS="${EPSILONS:-0.0}"
GPU_TOTAL_GB="${GPU_TOTAL_GB:-40}"
SEED="${SEED:-42}"
# Points drawn from EVERY cluster for DBCV. Fixing this rather than a total budget is what
# makes the score comparable between cells -- a fixed total scores a 176-cluster cell and a
# 1,330-cluster cell at different resolutions, and sampled DBCV's bias moves with resolution.
# 400 because lower values cannot see contamination: measured at 30 clusters where the true
# scores were clean +0.861 / contaminated -0.033, N=50 recovered only 19% of that gap, N=400
# recovered 97%. Was previously declared here but never forwarded to python -- the script ran
# on its own default (also 400) regardless of this variable. Now it actually takes effect.
DBCV_PER_CLUSTER="${DBCV_PER_CLUSTER:-400}"
# 'auto' prefers k-DBCV when PYTHONPATH exposes it (up to 42x faster at k=400, see
# SWEEP_METRICS.md), else falls back to hdbscan's validity_index.
DBCV_BACKEND="${DBCV_BACKEND:-auto}"
# cuML's HDBSCAN does not implement outlier_scores_ or exemplars_ at all, and its
# cluster_persistence_ is degenerate (measured: exactly 1.0 for every cluster on a real
# cell, vs 0.10-0.15 median from an equivalent CPU fit). 'cpu' forces the hdbscan package
# for the FIT itself so those three artefacts, plus relative_validity_, are real. Costs
# real wall time -- CPU HDBSCAN fit at 500K rows measured 6.7s on miletus, but this has
# not been benchmarked past that size on this project's embedding; time a probe cell
# before setting this for large FIT_SIZES.
# Explicit, not "auto". auto prefers cuML whenever cuML imports, so the value recorded
# in the run reflects a resolution rather than an intent -- which is how a sweep written
# to a directory named param_sweep_refit_cpu turned out to be cuML, and how "backends
# identical, ARI 1.0000" got published from a cuML-vs-cuML comparison. cuml is the
# deployed backend; set CLUSTER_BACKEND=cpu to fit with the hdbscan package instead.
CLUSTER_BACKEND="${CLUSTER_BACKEND:-cuml}"

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
    exec python "$SCRIPT" --build-index-only --context "$CONTEXT" \
        --output-dir "$OUTPUT_DIR" --genome-build GRCh38 --cosmic-version 3.5
fi

EXTRA=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then EXTRA+=(--dry-run); fi
if [[ -n "$CONTEXT" ]]; then EXTRA+=(--context "$CONTEXT" --sigprofiler-cpu "$SIGPROFILER_CPU"); fi

if [[ ! -f "$CONTEXT" ]]; then
    echo "WARNING: no context.parquet at $CONTEXT" >&2
    echo "         cells will be labelled and saved but NOT scored." >&2
    CONTEXT=""
fi

echo "[$(date +%H:%M:%S)] HDBSCAN parameter sweep"
echo "  coords        : $COORDS"
echo "  context       : ${CONTEXT:-<none, scoring disabled>}"
echo "  output        : $OUTPUT_DIR"
echo "  fit sizes     : $FIT_SIZES"
echo "  mcs (fixed)   : $MIN_CLUSTER_SIZES"
echo "  min_samples   : $MIN_SAMPLES"
echo "  cluster fit   : $CLUSTER_BACKEND   (cpu = real persistence/exemplars/outliers, else cuML degenerate)"
echo "  dbcv backend  : $DBCV_BACKEND, $DBCV_PER_CLUSTER pts/cluster"
echo
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
echo

python "$SCRIPT" \
    --coords "$COORDS" \
    --output-dir "$OUTPUT_DIR" \
    --fit-sizes "$FIT_SIZES" \
    --min-cluster-sizes "$MIN_CLUSTER_SIZES" \
    --min-samples "$MIN_SAMPLES" \
    --methods "$METHODS" \
    --epsilons "$EPSILONS" \
    --gpu-budget-gb "$GPU_TOTAL_GB" \
    --seed "$SEED" \
    --dbcv-per-cluster "$DBCV_PER_CLUSTER" \
    --dbcv-backend "$DBCV_BACKEND" \
    --cluster-backend "$CLUSTER_BACKEND" \
    "${EXTRA[@]}" 2>&1 | tee -a "$OUTPUT_DIR/param_sweep.log"
