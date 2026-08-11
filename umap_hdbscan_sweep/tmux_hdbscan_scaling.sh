#!/usr/bin/env bash
# HDBSCAN fit-size scaling sweep: 500K -> 50M rows, on the model-13 embedding.
#
# Two processes on purpose. The embed step needs torch; the sweep needs cuML. Running
# them together is what capped the previous timing run's RMM pool at 20 GB of a 40 GB
# budget (gpu_budget split it torch 20 / rmm 20). Split into two invocations, the sweep
# process never imports torch and STAGE_RMM_SHARE["sweep"]=0.9 gives RMM ~36 GB.
#
#   bash umap_hdbscan_sweep/tmux_hdbscan_scaling.sh
#
# Env overrides:
#   GPU_TOTAL_GB=40            budget for this process
#   FIT_SIZES=500000,...       comma list of fit-set sizes
#   MIN_SAMPLES=5              held fixed across sizes
#   EXTRA_MIN_SAMPLES=15       memory-delta probe at the largest successful size (0 = off)
#   SKIP_EMBED=1               coords.npy already built
#   NO_SAVE_MODELS=1           labels only (models can be GB at the large end)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EMBED_DIR="$REPO_ROOT/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed"
ENCODER_MODEL="$REPO_ROOT/umap_hdbscan_sweep/umap_tests/final_models/13_BEST_25M_nn15_md0.1_umap.pt"
OUTPUT_DIR="$REPO_ROOT/umap_hdbscan_sweep/umap_tests/hdbscan_scaling"
SCRIPT="$REPO_ROOT/umap_hdbscan_sweep/hdbscan_scaling_sweep.py"

GPU_TOTAL_GB="${GPU_TOTAL_GB:-40}"
FIT_SIZES="${FIT_SIZES:-500000,1000000,5000000,10000000,15000000,25000000,50000000}"
MIN_SAMPLES="${MIN_SAMPLES:-5}"
EXTRA_MIN_SAMPLES="${EXTRA_MIN_SAMPLES:-15}"
PROBE_SIZES="${PROBE_SIZES:-50000,200000,500000}"

if command -v micromamba &>/dev/null; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate "${MAMBA_ENV:-uv_vae}"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate patrickg
fi

export TQDM_DISABLE=1
mkdir -p "$OUTPUT_DIR"

MODEL_FLAG="--save-models"
if [[ "${NO_SAVE_MODELS:-0}" == "1" ]]; then MODEL_FLAG="--no-save-models"; fi

echo "[$(date +%H:%M:%S)] HDBSCAN fit-size scaling sweep"
echo "  encoder     : $(basename "$ENCODER_MODEL")"
echo "  output      : $OUTPUT_DIR"
echo "  fit sizes   : $FIT_SIZES"
echo "  min_samples : $MIN_SAMPLES  (extra probe at ${EXTRA_MIN_SAMPLES})"
echo "  mcs policy  : proportional, max(50, round(N * 1e-4))"
echo "  gpu budget  : ${GPU_TOTAL_GB} GB"
echo
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
echo

# ── step 1: coordinates (torch only) ──────────────────────────────────────────
if [[ "${SKIP_EMBED:-0}" != "1" ]]; then
    echo "[$(date +%H:%M:%S)] step 1/2 -- embedding the cohort through the encoder"
    python "$SCRIPT" \
        --embed-dir "$EMBED_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --encoder-model "$ENCODER_MODEL" \
        --embed-only \
        2>&1 | tee "$OUTPUT_DIR/embed.log"
    echo
else
    echo "[$(date +%H:%M:%S)] step 1/2 -- skipped (SKIP_EMBED=1)"
fi

# ── step 2: the sweep (cuML only, no torch in this process) ───────────────────
echo "[$(date +%H:%M:%S)] step 2/2 -- scaling sweep"
python "$SCRIPT" \
    --embed-dir "$EMBED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --fit-sizes "$FIT_SIZES" \
    --min-samples "$MIN_SAMPLES" \
    --extra-min-samples "$EXTRA_MIN_SAMPLES" \
    --probe-sizes "$PROBE_SIZES" \
    --gpu-budget-gb "$GPU_TOTAL_GB" \
    $MODEL_FLAG \
    2>&1 | tee "$OUTPUT_DIR/scaling.log"

echo
echo "[$(date +%H:%M:%S)] done"
echo "  results : $OUTPUT_DIR/scaling_results.json"
echo "  labels  : $OUTPUT_DIR/labels/"
echo "  models  : $OUTPUT_DIR/models/"
