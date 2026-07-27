#!/bin/bash
# Batch size x learning rate x warmup sweep -- tmux replacement for the SLURM
# array in run_batch_lr_sweep_gpu.sh. Same 10 configs, same hyperparameters.
#
# What changed versus the SLURM version, and why:
#   * The array became a work queue with CONCURRENCY workers. SLURM ran all 10
#     configs at once because the partition had 10 slots; one node with one GPU
#     does not, so concurrency is now an explicit choice (see TMUX_RUNNERS.md).
#   * GPU_TOTAL_GB is split across workers. Both the torch cap and the RMM pool
#     are per process, so N workers each claiming 16 GB would be 16N GB.
#   * SWEEP_ROOT no longer keys off $SLURM_ARRAY_JOB_ID (which collapsed to
#     "manual" without a scheduler and silently merged re-runs into one dir).
#   * conda 'patrickg' is no longer hardcoded -- micromamba 'uv_vae' (MAMBA_ENV) is
#     the miletus default, then .venv; CONDA_ENV overrides both.
#   * Completed configs are skipped on re-launch, so a crash resumes.
#
# Usage -- launches detached and returns immediately:
#     COMBINED=/path/to/combined.parquet \
#     CONCURRENCY=1 GPU_TOTAL_GB=16 \
#     bash Batch_Size_Learning_Rate_Testing/scripts/tmux_batch_lr_sweep.sh
#
#     TASKS="0 3 4" ...     # run a subset
#     FOREGROUND=1 ...      # run in this shell instead of a tmux session
#     DRY_RUN=1 ...         # print the plan and exit
#     FORCE=1 ...           # re-run configs that already finished
#
# Windows edit note:  sed -i 's/\r$//' Batch_Size_Learning_Rate_Testing/scripts/tmux_batch_lr_sweep.sh

set -euo pipefail

UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
BATCH_LR_DIR="${BATCH_LR_DIR:-$HOME/Batch_Size_Learning_Rate_Testing}"
EARLY_STOPPING_DIR="${EARLY_STOPPING_DIR:-$HOME/Early_Stopping_Tests}"

# shellcheck source=/dev/null
source "$UV_VAE_DIR/scripts/tmux_lib.sh"

# ── Configuration ───────────────────────────────────────────────────────────
COMBINED="${COMBINED:-/cta/users/patrickgao765/uv_vae/train_earlystop_1839691/combined.featuremap.parquet}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_ROOT="${SWEEP_ROOT:-$UV_VAE_DIR/sweeps/batch_lr_$RUN_ID}"
ROW_FILTER="${ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
SEED="${SEED:-42}"
CONCURRENCY="${CONCURRENCY:-1}"
SESSION="${SESSION:-batch_lr}"

EPOCH_CEILING="${EPOCH_CEILING:-10}"
PATIENCE="${PATIENCE:-2}"
MIN_DELTA="${MIN_DELTA:-0.001}"
AU_THRESHOLD="${AU_THRESHOLD:-0.01}"
EVAL_ROWS="${EVAL_ROWS:-10000}"

LOG_DIR="$SWEEP_ROOT/logs"
STATE_DIR="$SWEEP_ROOT/state"

# ── Config grid (identical to run_batch_lr_sweep_gpu.sh) ────────────────────
#   0  baseline:     batch=131K, lr=1e-3,    warmup=0,   dropout=0.1/0.2
#   1  batch effect: batch=524K, lr=1e-3,    warmup=0
#   2  batch effect: batch=1M,   lr=1e-3,    warmup=0
#   3  lr scaling:   batch=131K, lr=2e-3,    warmup=0
#   4  lr scaling:   batch=131K, lr=3e-3,    warmup=0
#   5  warmup only:  batch=131K, lr=1e-3,    warmup=200
#   6  warmup only:  batch=131K, lr=1e-3,    warmup=500
#   7  best combo A: batch=524K, lr=2e-3,    warmup=200
#   8  best combo B: batch=1M,   lr=2.83e-3, warmup=500
#   9  no dropout:   batch=131K, lr=1e-3,    warmup=200, dropout=0.0
ALL_TASKS=(0 1 2 3 4 5 6 7 8 9)
ALL_BATCHES=(131072 524288 1048576 131072 131072 131072 131072 524288 1048576 131072)

config_for_task() {
    # Sets NAME BATCH LR WARMUP INPUT_DROPOUT HIDDEN_DROPOUT for $1.
    BATCH=131072; LR="1e-3"; WARMUP=0
    INPUT_DROPOUT="0.1"; HIDDEN_DROPOUT="0.2"
    case "$1" in
        0) NAME="baseline_b131k_lr1e3_w0" ;;
        1) NAME="batch_b524k_lr1e3_w0";     BATCH=524288 ;;
        2) NAME="batch_b1M_lr1e3_w0";       BATCH=1048576 ;;
        3) NAME="lr_b131k_lr2e3_w0";        LR="2e-3" ;;
        4) NAME="lr_b131k_lr3e3_w0";        LR="3e-3" ;;
        5) NAME="warmup_b131k_lr1e3_w200";  WARMUP=200 ;;
        6) NAME="warmup_b131k_lr1e3_w500";  WARMUP=500 ;;
        7) NAME="combo_b524k_lr2e3_w200";   BATCH=524288;  LR="2e-3";    WARMUP=200 ;;
        8) NAME="combo_b1M_lr2p83e3_w500";  BATCH=1048576; LR="2.83e-3"; WARMUP=500 ;;
        9) NAME="nodrop_b131k_lr1e3_w200";  WARMUP=200; INPUT_DROPOUT="0.0"; HIDDEN_DROPOUT="0.0" ;;
        *) echo "ERROR: unknown task id '$1'" >&2; return 1 ;;
    esac
    LATENT_DIM=16; HIDDEN_DIMS="256,128"; KL_WEIGHT="0.05"; TRAIN_FRAC="0.8"
}

# ── One task: train, then evaluate the checkpoint ───────────────────────────
run_task() {
    local task="$1"
    config_for_task "$task"

    local config_dir="$SWEEP_ROOT/$NAME"
    mkdir -p "$config_dir"

    # Per-worker slice of the sweep-wide budget. Read by uv_vae.gpu_budget.
    export UV_VAE_GPU_MEM_GB="$UVV_GPU_PER_WORKER"
    local threads="$UVV_THREADS_PER_WORKER"

    uvv_rule
    echo "Config        : $NAME (task $task)"
    echo "Parquet       : $COMBINED"
    echo "Batch         : $BATCH"
    echo "LR            : $LR"
    echo "Warmup        : $WARMUP steps"
    echo "Dropout       : $INPUT_DROPOUT / $HIDDEN_DROPOUT"
    echo "Epochs        : $EPOCH_CEILING (patience=$PATIENCE)"
    echo "GPU budget    : ${UV_VAE_GPU_MEM_GB} GB    Threads: $threads"
    echo "Output        : $config_dir"
    uvv_rule

    local train_start=$SECONDS
    uvv_log "===== BEGIN: train ====="
    python "$EARLY_STOPPING_DIR/Python Files/train_with_early_stopping.py" \
        --parquet-path "$COMBINED" \
        --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
        --output-dir "$config_dir" \
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
        --threads "$threads" \
        > "$config_dir/train_result.json"
    cat "$config_dir/train_result.json"
    local train_seconds=$((SECONDS - train_start))
    uvv_log "===== END: train ($(uvv_fmt_seconds $train_seconds)) ====="

    local checkpoint
    checkpoint="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint_path'])" \
        "$config_dir/train_result.json")"
    if [ ! -f "$checkpoint" ]; then
        echo "ERROR: checkpoint not found at $checkpoint" >&2
        return 1
    fi

    local eval_start=$SECONDS
    uvv_log "===== BEGIN: evaluate latent metrics ====="
    python "$BATCH_LR_DIR/Python Files/evaluate_sweep.py" \
        --checkpoint "$checkpoint" \
        --parquet-path "$COMBINED" \
        --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
        --row-filter "$ROW_FILTER" \
        --eval-rows "$EVAL_ROWS" \
        --seed "$SEED" \
        --threads "$threads" \
        --output-dir "$config_dir"
    local eval_seconds=$((SECONDS - eval_start))
    uvv_log "===== END: evaluate ($(uvv_fmt_seconds $eval_seconds)) ====="

    uvv_log "task $task ($NAME): train $(uvv_fmt_seconds $train_seconds), eval $(uvv_fmt_seconds $eval_seconds)"
}

# ── Driver ──────────────────────────────────────────────────────────────────
main() {
    uvv_activate_env
    uvv_export_determinism "$SEED"
    uvv_plan_resources "$CONCURRENCY"

    read -ra TASK_LIST <<< "${TASKS:-${ALL_TASKS[*]}}"

    mkdir -p "$SWEEP_ROOT" "$LOG_DIR" "$STATE_DIR"

    uvv_rule
    echo "Batch size / learning rate sweep"
    echo "  sweep root  : $SWEEP_ROOT"
    echo "  parquet     : $COMBINED"
    echo "  row filter  : $ROW_FILTER"
    echo "  tasks       : ${TASK_LIST[*]}"
    echo "  concurrency : $CONCURRENCY"
    echo "  seed        : $SEED"
    uvv_rule

    if [ ! -f "$COMBINED" ] && ! compgen -G "$COMBINED" > /dev/null; then
        echo "ERROR: parquet not found: $COMBINED" >&2
        exit 1
    fi

    # Show the GPU headroom for the batch sizes actually queued, before committing.
    local queued_batches=()
    local task
    for task in "${TASK_LIST[@]}"; do
        queued_batches+=("${ALL_BATCHES[$task]}")
    done
    echo "GPU budget check:"
    uvv_report_batch_ceiling "$UVV_GPU_PER_WORKER" "${queued_batches[@]}"
    echo

    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        uvv_log "running GPU preflight"
        python "$UV_VAE_DIR/scripts/gpu_preflight.py" \
            --batch-size "$(printf '%s\n' "${queued_batches[@]}" | sort -n | tail -1)" \
            --budget-gb "$UVV_GPU_PER_WORKER" \
            --json-out "$SWEEP_ROOT/gpu_preflight.json" \
            || { echo "Preflight failed. Fix the FAIL lines above, or set SKIP_PREFLIGHT=1 to override." >&2; exit 1; }
        echo
    fi

    local overall=$SECONDS
    uvv_run_queue "$CONCURRENCY" "$STATE_DIR" "$LOG_DIR" run_task "${TASK_LIST[@]}"

    uvv_log "collecting sweep results"
    python "$BATCH_LR_DIR/Python Files/evaluate_sweep.py" \
        --sweep-root "$SWEEP_ROOT" \
        --parquet-path "$COMBINED" \
        --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
        || uvv_log "WARNING: sweep collection failed; per-config *_eval.json files are still on disk"

    uvv_rule
    echo "Sweep complete in $(uvv_fmt_seconds $((SECONDS - overall)))"
    echo "  artifacts     : $SWEEP_ROOT"
    echo "  per-task logs : $LOG_DIR"
    echo "  epoch history : $SWEEP_ROOT/*/*/training_report.json"
    echo "  latent metrics: $SWEEP_ROOT/*/*_eval.json"
    uvv_rule
}

if [ "${UVV_CHILD:-0}" = "1" ] || [ "${FOREGROUND:-0}" = "1" ] || [ "${DRY_RUN:-0}" = "1" ]; then
    main
else
    uvv_strip_crlf "$UV_VAE_DIR/scripts/tmux_lib.sh" "$BATCH_LR_DIR/scripts/tmux_batch_lr_sweep.sh"
    mkdir -p "$LOG_DIR"
    uvv_export_child_env \
        UV_VAE_DIR BATCH_LR_DIR EARLY_STOPPING_DIR CONDA_ENV MAMBA_ENV MAMBA_ROOT_PREFIX \
        RUN_ID SWEEP_ROOT COMBINED ROW_FILTER SEED \
        CONCURRENCY GPU_TOTAL_GB THREADS_TOTAL UV_VAE_GPU_OOM_POLICY \
        EPOCH_CEILING PATIENCE MIN_DELTA AU_THRESHOLD EVAL_ROWS \
        TASKS FORCE SKIP_PREFLIGHT
    uvv_launch_tmux "$SESSION" "$LOG_DIR" \
        "UVV_CHILD=1 bash '$BATCH_LR_DIR/scripts/tmux_batch_lr_sweep.sh'"
fi
