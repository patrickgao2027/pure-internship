#!/bin/bash
#SBATCH --job-name=vae-sweep-gpu
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/sweep_gpu_%A_%a.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/sweep_gpu_%A_%a.err
#SBATCH --array=0-9
#
# Uncomment for GPU. Required for cuDF encoding + float16 AMP.
##SBATCH --gres=gpu:1
#
# Hyperparameter sweep: batch size × learning rate × warmup
# Each SLURM array task runs ONE config, so configs run in parallel.
#
# Streams ALL rows from the combined parquet (no --sample-rows), then runs
# evaluate_sweep.py on the checkpoint to compute full latent-space metrics:
#   - Active units, effective dimensionality (Participation Ratio)
#   - KL balance, posterior collapse fraction
#   - Off-diagonal covariance (entanglement)
#   - Trustworthiness, continuity (k=10, k=50)
#   - kNN Jaccard (input vs latent, k=10, k=30)
#   - Reconstruction MSE + categorical accuracy
#
# After all array tasks finish, run evaluate_sweep.py in --sweep-root mode
# to collect all configs into a single comparison CSV:
#
#   python "$BATCH_LR_DIR/Python Files/evaluate_sweep.py" \
#       --sweep-root /cta/users/patrickgao765/uv_vae/sweep_gpu_<JOBID> \
#       --parquet-path <combined.parquet> \
#       --feature-spec-path ml_features.json
#
# Usage:
#   sed -i 's/\r$//' Batch_Size_Learning_Rate_Testing/scripts/run_batch_lr_sweep_gpu.sh
#   COMBINED=/path/to/combined.parquet sbatch Batch_Size_Learning_Rate_Testing/scripts/run_batch_lr_sweep_gpu.sh
#
# To test a single config without SLURM array:
#   SLURM_ARRAY_TASK_ID=0 bash Batch_Size_Learning_Rate_Testing/scripts/run_batch_lr_sweep_gpu.sh

set -euo pipefail

UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
BATCH_LR_DIR="${BATCH_LR_DIR:-$HOME/Batch_Size_Learning_Rate_Testing}"
COMBINED="${COMBINED:-/cta/users/patrickgao765/uv_vae/train_earlystop_1839691/combined.featuremap.parquet}"
SWEEP_ROOT="${SWEEP_ROOT:-/cta/users/patrickgao765/uv_vae/sweep_gpu_${SLURM_ARRAY_JOB_ID:-manual}}"
ROW_FILTER="${ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
THREADS="${SLURM_CPUS_PER_TASK:-32}"
SEED=42

# Training ceiling — early stopping decides when to actually stop
EPOCH_CEILING=10
PATIENCE=2
MIN_DELTA=0.001
AU_THRESHOLD=0.01
EVAL_ROWS=10000

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

###############################################################################
# Config grid — 10 configs (array 0-9)
#
# Design: one axis varies at a time from a shared baseline, plus 2 "best
# combo" candidates that combine the winning values from each axis.
#
# What each config tests:
#   0  baseline: batch=131K, lr=1e-3, warmup=0, dropout=0.2
#   1  batch effect: batch=524K, lr=1e-3, warmup=0
#   2  batch effect: batch=1M,   lr=1e-3, warmup=0
#   3  lr scaling:   batch=131K, lr=2e-3, warmup=0
#   4  lr scaling:   batch=131K, lr=3e-3, warmup=0
#   5  warmup only:  batch=131K, lr=1e-3, warmup=200
#   6  warmup only:  batch=131K, lr=1e-3, warmup=500
#   7  best combo A: batch=524K, lr=2e-3, warmup=200
#   8  best combo B: batch=1M,   lr=2.83e-3, warmup=500
#   9  no dropout:   batch=131K, lr=1e-3, warmup=200, dropout=0.0
###############################################################################

# defaults shared across configs
BATCH=131072; LR="1e-3"; WARMUP=0
INPUT_DROPOUT="0.1"; HIDDEN_DROPOUT="0.2"
LATENT_DIM=16; HIDDEN_DIMS="256,128"; KL_WEIGHT="0.05"; TRAIN_FRAC="0.8"

case $TASK_ID in
    0) NAME="baseline_b131k_lr1e3_w0"
       ;;
    1) NAME="batch_b524k_lr1e3_w0"
       BATCH=524288
       ;;
    2) NAME="batch_b1M_lr1e3_w0"
       BATCH=1048576
       ;;
    3) NAME="lr_b131k_lr2e3_w0"
       LR="2e-3"
       ;;
    4) NAME="lr_b131k_lr3e3_w0"
       LR="3e-3"
       ;;
    5) NAME="warmup_b131k_lr1e3_w200"
       WARMUP=200
       ;;
    6) NAME="warmup_b131k_lr1e3_w500"
       WARMUP=500
       ;;
    7) NAME="combo_b524k_lr2e3_w200"
       BATCH=524288; LR="2e-3"; WARMUP=200
       ;;
    8) NAME="combo_b1M_lr2p83e3_w500"
       BATCH=1048576; LR="2.83e-3"; WARMUP=500
       ;;
    9) NAME="nodrop_b131k_lr1e3_w200"
       WARMUP=200; INPUT_DROPOUT="0.0"; HIDDEN_DROPOUT="0.0"
       ;;
    *) echo "ERROR: unknown TASK_ID=$TASK_ID"; exit 1 ;;
esac

CONFIG_DIR="$SWEEP_ROOT/$NAME"
mkdir -p "$CONFIG_DIR"

eval "$(conda shell.bash hook)"
conda activate patrickg
cd "$UV_VAE_DIR"

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

fmt_seconds() { printf '%dh %02dm %02ds' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }

echo "====================================================================="
echo "Sweep config  : $NAME (task $TASK_ID of 0-9)"
echo "Parquet       : $COMBINED"
echo "Batch         : $BATCH"
echo "LR            : $LR"
echo "Warmup        : $WARMUP steps"
echo "Input dropout : $INPUT_DROPOUT"
echo "Hidden dropout: $HIDDEN_DROPOUT"
echo "Epochs        : $EPOCH_CEILING (patience=$PATIENCE)"
echo "Output        : $CONFIG_DIR"
echo "====================================================================="
echo ""

###############################################################################
# Phase 1: Train
###############################################################################
echo "[$(date '+%F %T')] ===== BEGIN: train ====="
TRAIN_START=$SECONDS

python scripts/train_with_early_stopping.py \
    --parquet-path "$COMBINED" \
    --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
    --output-dir "$CONFIG_DIR" \
    --row-filter "$ROW_FILTER" \
    --streaming \
    --epochs "$EPOCH_CEILING" \
    --patience "$PATIENCE" \
    --min-delta "$MIN_DELTA" \
    --active-unit-threshold "$AU_THRESHOLD" \
    --batch-size "$BATCH" \
    --latent-dim "$LATENT_DIM" \
    --hidden-dims "$HIDDEN_DIMS" \
    --learning-rate "$LR" \
    --kl-weight "$KL_WEIGHT" \
    --train-fraction "$TRAIN_FRAC" \
    --input-dropout "$INPUT_DROPOUT" \
    --hidden-dropout "$HIDDEN_DROPOUT" \
    --warmup-steps "$WARMUP" \
    --seed "$SEED" \
    --threads "$THREADS" \
    | tee "$CONFIG_DIR/train_result.json"

TRAIN_SECONDS=$((SECONDS - TRAIN_START))
echo "[$(date '+%F %T')] ===== END: train ($(fmt_seconds $TRAIN_SECONDS)) ====="

###############################################################################
# Phase 2: Evaluate checkpoint with full latent metrics
###############################################################################
CHECKPOINT="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint_path'])" "$CONFIG_DIR/train_result.json")"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo ""
echo "[$(date '+%F %T')] ===== BEGIN: evaluate latent metrics ====="
EVAL_START=$SECONDS

python "$BATCH_LR_DIR/Python Files/evaluate_sweep.py" \
    --checkpoint "$CHECKPOINT" \
    --parquet-path "$COMBINED" \
    --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
    --row-filter "$ROW_FILTER" \
    --eval-rows "$EVAL_ROWS" \
    --seed "$SEED" \
    --threads "$THREADS" \
    --output-dir "$CONFIG_DIR"

EVAL_SECONDS=$((SECONDS - EVAL_START))
echo "[$(date '+%F %T')] ===== END: evaluate ($(fmt_seconds $EVAL_SECONDS)) ====="

###############################################################################
# Summary
###############################################################################
echo ""
echo "====================================================================="
echo "Config          : $NAME"
echo "Train time      : $(fmt_seconds $TRAIN_SECONDS)"
echo "Eval time       : $(fmt_seconds $EVAL_SECONDS)"
echo "Total           : $(fmt_seconds $SECONDS)"
echo "Artifacts       : $CONFIG_DIR"
echo ""
echo "Per-epoch history: $CONFIG_DIR/*/training_report.json"
echo "Latent metrics  : $CONFIG_DIR/*_eval.json"
echo ""
echo "After ALL array tasks finish, combine results:"
echo "  python \"$BATCH_LR_DIR/Python Files/evaluate_sweep.py\" \\"
echo "      --sweep-root $SWEEP_ROOT \\"
echo "      --parquet-path $COMBINED \\"
echo "      --feature-spec-path $UV_VAE_DIR/ml_features.json"
echo "====================================================================="
