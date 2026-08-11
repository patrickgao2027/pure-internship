#!/usr/bin/env bash
# Decompose the HDBSCAN cell timer using model-13 (the chosen parametric encoder).
#
# Phase A for UMAP established:  fit=119s (2%), transform=3291s (98%)
# This script answers the parallel question for HDBSCAN:
#   what fraction of the 3827-4155s cell is HDBSCAN fit / approximate_predict / parquet write?
#
# Run on miletus (needs CUDA + torch + cuml).  Takes ~30 min.
#
# Usage:
#   bash umap_hdbscan_sweep/tmux_hdbscan_timing.sh
#
# Override defaults via env vars:
#   GPU_TOTAL_GB=40     # RMM pool size (default 40, no trainer sharing)
#   MIN_SAMPLES=5       # single min_samples value (default 5, matching the prior sweep)
#   PROBE_SIZES=50000,200000,500000

set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EMBED_DIR="$REPO_ROOT/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed"
ENCODER_MODEL="$REPO_ROOT/umap_hdbscan_sweep/umap_tests/final_models/13_BEST_25M_nn15_md0.1_umap.pt"
OUTPUT_DIR="$REPO_ROOT/umap_hdbscan_sweep/umap_tests"
SCRIPT="$REPO_ROOT/umap_hdbscan_sweep/phase_a_hdbscan_timing.py"

# ── parameters ────────────────────────────────────────────────────────────────
GPU_TOTAL_GB="${GPU_TOTAL_GB:-40}"
MIN_SAMPLES="${MIN_SAMPLES:-5}"
PROBE_SIZES="${PROBE_SIZES:-50000,200000,500000}"

# ── env ───────────────────────────────────────────────────────────────────────
if command -v micromamba &>/dev/null; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate "${MAMBA_ENV:-uv_vae}"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate patrickg
fi

export TQDM_DISABLE=1
export UV_VAE_DISABLE_CUML=0
export GPU_TOTAL_GB="$GPU_TOTAL_GB"
export TRAINER_GPU_GB=0

echo "[$(date +%H:%M:%S)] HDBSCAN timing decomposition"
echo "  encoder : $ENCODER_MODEL"
echo "  embed   : $EMBED_DIR"
echo "  output  : $OUTPUT_DIR"
echo "  min_samples=$MIN_SAMPLES  probe_sizes=$PROBE_SIZES  gpu_total=${GPU_TOTAL_GB}GB"
echo

python "$SCRIPT" \
    --embed-dir "$EMBED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --encoder-model "$ENCODER_MODEL" \
    --min-cluster-sizes 100,500,2500 \
    --min-samples "$MIN_SAMPLES" \
    --probe-sizes "$PROBE_SIZES" \
    --gpu-budget-gb "$GPU_TOTAL_GB" \
    2>&1 | tee "$OUTPUT_DIR/phase_a_hdbscan_timing.log"

echo
echo "[$(date +%H:%M:%S)] done -- results in $OUTPUT_DIR/phase_a_hdbscan_timing.json"
