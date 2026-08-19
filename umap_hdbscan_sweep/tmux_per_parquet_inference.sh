#!/usr/bin/env bash
# Per-parquet inference — miletus runner
# For each of 95 parquet files: VAE encode -> parametric UMAP -> HDBSCAN -> SigProfiler (uv_only) -> 4 plots
# SigProfiler + plots run in parallel across --n-workers processes.
#
# Usage:
#   tmux new-session -d -s per_parquet 'bash umap_hdbscan_sweep/tmux_per_parquet_inference.sh'
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
PARQUET_GLOB="${PARQUET_GLOB:-/data/lab/ppmseq_parquets/*.parquet}"
CHECKPOINT="${CHECKPOINT:-$HOME/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/training/run_20260802T192814Z/model.pt}"
FEATURE_SPEC="${FEATURE_SPEC:-$HOME/pure-internship/uv_vae/ml_features.json}"
UMAP_MODEL="${UMAP_MODEL:-$HOME/pure-internship/umap_hdbscan_sweep/umap/results/final_models/13_BEST_25M_nn15_md0.1_umap.pt}"
COORDS="${COORDS:-$HOME/pure-internship/umap_hdbscan_sweep/hdbscan/results/hdbscan_scaling/coords.npy}"
CONTEXT="${CONTEXT:-$HOME/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed/context.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/pure-internship/umap_hdbscan_sweep/per_parquet_inference}"

# Reuse the cohort HDBSCAN from low_noise_hdbscan.py rather than refitting here, so
# per-sample labels are directly comparable to the cohort run. Set HDBSCAN_MODEL="" to
# make this script fit its own model with MCS/MS/EPSILON below.
HDBSCAN_MODEL="${HDBSCAN_MODEL:-$HOME/pure-internship/umap_hdbscan_sweep/low_noise_hdbscan/hdbscan_model.pkl}"
FIT_INDICES="${FIT_INDICES:-$HOME/pure-internship/umap_hdbscan_sweep/low_noise_hdbscan/fit_indices.npy}"

MCS="${MCS:-2500}"
MS="${MS:-1}"          # low-noise config; ignored when HDBSCAN_MODEL is set
EPSILON="${EPSILON:-0.05}"
FIT_ROWS="${FIT_ROWS:-1000000}"
SEED="${SEED:-42}"
GENOME_BUILD="${GENOME_BUILD:-GRCh38}"
COSMIC_VERSION="${COSMIC_VERSION:-3.5}"
# 4 workers × 4 CPUs each = 16 total, matches the Blackwell's CPU allocation
N_WORKERS="${N_WORKERS:-4}"
SIGPROFILER_CPU="${SIGPROFILER_CPU:-4}"
DEVICE="${DEVICE:-auto}"

# ── log setup ───────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/run_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "===== per_parquet_inference  $(date -u) =====" | tee -a "$LOG"
echo "  parquet_glob:   $PARQUET_GLOB"               | tee -a "$LOG"
echo "  checkpoint:     $CHECKPOINT"                 | tee -a "$LOG"
echo "  umap_model:     $UMAP_MODEL"                 | tee -a "$LOG"
echo "  output_dir:     $OUTPUT_DIR"                 | tee -a "$LOG"
echo "  mcs=$MCS  ms=$MS  eps=$EPSILON"              | tee -a "$LOG"
echo "  n_workers=$N_WORKERS  sigprofiler_cpu=$SIGPROFILER_CPU"  | tee -a "$LOG"

sed -i 's/\r$//' "$SCRIPT_DIR/per_parquet_inference.py" 2>/dev/null || true

SKIP_DONE_FLAG=""
if [ "${SKIP_DONE:-0}" = "1" ]; then
    SKIP_DONE_FLAG="--skip-done"
fi

MODEL_ARGS=""
if [ -n "$HDBSCAN_MODEL" ]; then
    if [ ! -f "$HDBSCAN_MODEL" ]; then
        echo "ERROR: HDBSCAN_MODEL not found: $HDBSCAN_MODEL" | tee -a "$LOG"
        echo "  Run tmux_low_noise_hdbscan.sh first, or set HDBSCAN_MODEL=\"\" to fit here." | tee -a "$LOG"
        exit 1
    fi
    MODEL_ARGS="--hdbscan-model $HDBSCAN_MODEL"
    [ -f "$FIT_INDICES" ] && MODEL_ARGS="$MODEL_ARGS --fit-indices $FIT_INDICES"
    echo "  hdbscan_model:  $HDBSCAN_MODEL" | tee -a "$LOG"
fi

python "$SCRIPT_DIR/per_parquet_inference.py" \
    --parquet-glob    "$PARQUET_GLOB" \
    --checkpoint      "$CHECKPOINT" \
    --feature-spec    "$FEATURE_SPEC" \
    --umap-model      "$UMAP_MODEL" \
    --coords          "$COORDS" \
    --context         "$CONTEXT" \
    --output-dir      "$OUTPUT_DIR" \
    --mcs             "$MCS" \
    --ms              "$MS" \
    --epsilon         "$EPSILON" \
    --fit-rows        "$FIT_ROWS" \
    --seed            "$SEED" \
    --genome-build    "$GENOME_BUILD" \
    --cosmic-version  "$COSMIC_VERSION" \
    --n-workers       "$N_WORKERS" \
    --sigprofiler-cpu "$SIGPROFILER_CPU" \
    --device          "$DEVICE" \
    $MODEL_ARGS \
    $SKIP_DONE_FLAG \
    2>&1 | tee -a "$LOG"

echo "===== DONE  $(date -u) =====" | tee -a "$LOG"
