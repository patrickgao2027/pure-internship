#!/usr/bin/env bash
# Cohort-level low-noise HDBSCAN — miletus runner
# Fits on 1M rows from coords.npy, labels all 157.5M reads, runs SigProfiler (uv_only),
# generates 10 analysis plots.
#
# Usage:
#   tmux new-session -d -s low_noise 'bash umap_hdbscan_sweep/tmux_low_noise_hdbscan.sh'
#
# Adjust COORDS / CONTEXT / OUTPUT_DIR if your layout differs from the defaults.
set -euo pipefail
export TQDM_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── detect cluster and activate environment ─────────────────────────────────────
if command -v micromamba &>/dev/null; then
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "${MAMBA_ENV:-uv_vae}"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-patrickg}"
fi

# ── default paths (miletus) ─────────────────────────────────────────────────────
COORDS="${COORDS:-$HOME/pure-internship/umap_hdbscan_sweep/umap_tests/hdbscan_scaling/coords.npy}"
CONTEXT="${CONTEXT:-$HOME/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed/context.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/pure-internship/umap_hdbscan_sweep/low_noise_hdbscan}"

# Existing sbs96_index from the param_sweep run (saves ~1h rebuild)
SBS96_INDEX="${SBS96_INDEX:-$HOME/pure-internship/umap_hdbscan_sweep/umap_tests/param_sweep_refit_cpu/sbs96_index.npy}"

MCS="${MCS:-2500}"
MS="${MS:-1}"
EPSILON="${EPSILON:-0.05}"
FIT_ROWS="${FIT_ROWS:-1000000}"
SAMPLE_ROWS="${SAMPLE_ROWS:-2000000}"   # rows to scatter-plot
SIGPROFILER_CPU="${SIGPROFILER_CPU:-16}"
SEED="${SEED:-42}"
GENOME_BUILD="${GENOME_BUILD:-GRCh38}"
COSMIC_VERSION="${COSMIC_VERSION:-3.5}"
CLUSTER_BACKEND="${CLUSTER_BACKEND:-auto}"
PREDICT_BACKEND="${PREDICT_BACKEND:-rbc}"    # rbc=cuML GPU (fast), sklearn=CPU KDTree

# ── log setup ───────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/run_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "===== low_noise_hdbscan  $(date -u) =====" | tee -a "$LOG"
echo "  coords:         $COORDS"                   | tee -a "$LOG"
echo "  context:        $CONTEXT"                  | tee -a "$LOG"
echo "  output_dir:     $OUTPUT_DIR"               | tee -a "$LOG"
echo "  mcs=$MCS  ms=$MS  eps=$EPSILON"            | tee -a "$LOG"

# strip Windows CRLF if needed (safe no-op on Linux)
sed -i 's/\r$//' "$SCRIPT_DIR/low_noise_hdbscan.py" 2>/dev/null || true

SBS96_ARG=""
if [ -f "$SBS96_INDEX" ]; then
    echo "  using existing sbs96_index: $SBS96_INDEX" | tee -a "$LOG"
    SBS96_ARG="--sbs96-index $SBS96_INDEX"
fi

python "$SCRIPT_DIR/low_noise_hdbscan.py" \
    --coords         "$COORDS" \
    --context        "$CONTEXT" \
    --output-dir     "$OUTPUT_DIR" \
    --mcs            "$MCS" \
    --ms             "$MS" \
    --epsilon        "$EPSILON" \
    --fit-rows       "$FIT_ROWS" \
    --sample-rows    "$SAMPLE_ROWS" \
    --sigprofiler-cpu "$SIGPROFILER_CPU" \
    --seed           "$SEED" \
    --genome-build   "$GENOME_BUILD" \
    --cosmic-version "$COSMIC_VERSION" \
    --cluster-backend "$CLUSTER_BACKEND" \
    --predict-backend "$PREDICT_BACKEND" \
    $SBS96_ARG \
    2>&1 | tee -a "$LOG"

echo "===== DONE  $(date -u) =====" | tee -a "$LOG"
