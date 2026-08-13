#!/usr/bin/env bash
# Clustering stability: refit each cell on independent subsamples, compare the labellings.
#
# Pure clustering quality -- no SigProfiler, no signature reference, nothing biological.
# Answers "does this parameter setting recover reproducible structure?" rather than "are the
# clusters compact in the embedding", which is the question the internal indices cannot
# answer honestly here because UMAP built the embedding they grade.
#
#   bash umap_hdbscan_sweep/tmux_stability_sweep.sh              # run it
#   DRY_RUN=1 bash umap_hdbscan_sweep/tmux_stability_sweep.sh    # grid + projected cost only
#
# Env overrides:
#   FIT_SIZES=500000,1000000,5000000     25M costs ~9.7 h per cell at 3 replicates -- add it
#                                        deliberately, on one cell, not across the block
#   MIN_CLUSTER_SIZES=250,500,1000,2500
#   MIN_SAMPLES=5                        the 2026-08-12 sweep measured ms as inert (p=0.44)
#   REPLICATES=3                         pairs scored = R*(R-1)/2
#   PROBE_ROWS=5000000                   held out of every fit set
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCALING_DIR="${SCALING_DIR:-$REPO_ROOT/umap_hdbscan_sweep/umap_tests/hdbscan_scaling}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/umap_hdbscan_sweep/umap_tests/stability_sweep}"
SCRIPT="$REPO_ROOT/umap_hdbscan_sweep/stability_sweep.py"

COORDS="${COORDS:-$SCALING_DIR/coords.npy}"
FIT_SIZES="${FIT_SIZES:-500000,1000000,5000000}"
MIN_CLUSTER_SIZES="${MIN_CLUSTER_SIZES:-250,500,1000,2500}"
MIN_SAMPLES="${MIN_SAMPLES:-5}"
METHODS="${METHODS:-eom}"
REPLICATES="${REPLICATES:-3}"
PROBE_ROWS="${PROBE_ROWS:-5000000}"
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

EXTRA=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then EXTRA+=(--dry-run); fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then EXTRA+=(--overwrite); fi

echo "[$(date +%H:%M:%S)] HDBSCAN stability sweep"
echo "  coords        : $COORDS"
echo "  output        : $OUTPUT_DIR"
echo "  fit sizes     : $FIT_SIZES"
echo "  mcs           : $MIN_CLUSTER_SIZES"
echo "  min_samples   : $MIN_SAMPLES"
echo "  replicates    : $REPLICATES  (probe $PROBE_ROWS rows, held out of every fit)"
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
    --replicates "$REPLICATES" \
    --probe-rows "$PROBE_ROWS" \
    --seed "$SEED" \
    "${EXTRA[@]}" 2>&1 | tee -a "$OUTPUT_DIR/stability_sweep.log"
