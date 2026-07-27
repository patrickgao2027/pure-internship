#!/bin/bash
# Single early-stopping training run -- tmux replacement for run_train_only.sh
# (and run_train_only_dropout0.2.sh, which is this with DROPOUT overrides).
#
# Use it to tune PATIENCE / MIN_DELTA without paying for the clustering stage,
# and as the smoke test that the node is configured correctly before launching a
# sweep: it exercises the same streaming trainer, the same GPU cap and the same
# determinism settings, on whatever parquet you point it at.
#
# What changed versus the SLURM version:
#   * RUN_ROOT no longer keys off $SLURM_JOB_ID (collapsed to "manual" and made
#     consecutive runs overwrite each other); it is timestamped.
#   * THREADS is detected from the machine rather than read from
#     $SLURM_CPUS_PER_TASK, which does not exist here and silently fell back to 32.
#   * conda 'patrickg' is no longer hardcoded -- .venv is preferred, CONDA_ENV wins.
#   * The GPU ceiling is applied and recorded in the run's training_report.json.
#
# Usage:
#     COMBINED=/path/to/combined.parquet \
#     bash Early_Stopping_Tests/scripts/tmux_train_only.sh
#
#     BATCH_SIZE=131072 EPOCH_CEILING=10 PATIENCE=2 ...
#     INPUT_DROPOUT=0.2 HIDDEN_DROPOUT=0.2 ...   # the dropout0.2 variant
#     TEST_PARQUET=/path/to/test.parquet ...     # per-epoch convergence tracking
#     FOREGROUND=1 ...                           # run here instead of in tmux
#
# Windows edit note:  sed -i 's/\r$//' Early_Stopping_Tests/scripts/tmux_train_only.sh

set -euo pipefail

UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
EARLY_STOPPING_DIR="${EARLY_STOPPING_DIR:-$HOME/Early_Stopping_Tests}"

# shellcheck source=/dev/null
source "$UV_VAE_DIR/scripts/tmux_lib.sh"

# ── Configuration ───────────────────────────────────────────────────────────
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$UV_VAE_DIR/runs/train_earlystop_$RUN_ID}"
ROW_FILTER="${ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
SEED="${SEED:-42}"
SESSION="${SESSION:-train_only}"

# Supply COMBINED to reuse a merged parquet, or SAMPLE_1/SAMPLE_2 to build one.
COMBINED="${COMBINED:-$RUN_ROOT/combined.featuremap.parquet}"
SAMPLE_1="${SAMPLE_1:-}"
SAMPLE_2="${SAMPLE_2:-}"
DUCKDB_MEM="${DUCKDB_MEM:-8GB}"

EPOCH_CEILING="${EPOCH_CEILING:-100}"
PATIENCE="${PATIENCE:-2}"
MIN_DELTA="${MIN_DELTA:-0.001}"
AU_THRESHOLD="${AU_THRESHOLD:-0.01}"
INPUT_DROPOUT="${INPUT_DROPOUT:-0.1}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.4}"
BATCH_SIZE="${BATCH_SIZE:-32768}"
LATENT_DIM="${LATENT_DIM:-16}"
HIDDEN_DIMS="${HIDDEN_DIMS:-256,128}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
KL_WEIGHT="${KL_WEIGHT:-0.05}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
TEST_PARQUET="${TEST_PARQUET:-}"
CONVERGENCE_ROWS="${CONVERGENCE_ROWS:-5000}"

LOG_DIR="$RUN_ROOT/logs"

main() {
    uvv_activate_env
    uvv_export_determinism "$SEED"
    # One run, so the whole budget goes to it.
    uvv_plan_resources 1
    export UV_VAE_GPU_MEM_GB="$UVV_GPU_PER_WORKER"
    local threads="$UVV_THREADS_PER_WORKER"

    mkdir -p "$RUN_ROOT" "$LOG_DIR"

    uvv_rule
    echo "Early-stopping training run"
    echo "  run root    : $RUN_ROOT"
    echo "  parquet     : $COMBINED"
    echo "  row filter  : $ROW_FILTER"
    echo "  batch       : $BATCH_SIZE   lr: $LEARNING_RATE   beta: $KL_WEIGHT"
    echo "  dropout     : $INPUT_DROPOUT / $HIDDEN_DROPOUT"
    echo "  epochs      : $EPOCH_CEILING (patience=$PATIENCE, min_delta=$MIN_DELTA)"
    echo "  GPU budget  : ${UV_VAE_GPU_MEM_GB} GB    threads: $threads"
    echo "  seed        : $SEED"
    uvv_rule

    echo "GPU budget check:"
    uvv_report_batch_ceiling "$UVV_GPU_PER_WORKER" "$BATCH_SIZE"
    echo

    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        python "$UV_VAE_DIR/scripts/gpu_preflight.py" \
            --batch-size "$BATCH_SIZE" \
            --budget-gb "$UVV_GPU_PER_WORKER" \
            --json-out "$RUN_ROOT/gpu_preflight.json" \
            || { echo "Preflight failed. Fix the FAIL lines above, or set SKIP_PREFLIGHT=1 to override." >&2; exit 1; }
        echo
    fi

    local combine_seconds=0
    if [ -f "$COMBINED" ] || compgen -G "$COMBINED" > /dev/null; then
        uvv_log "reusing existing parquet: $COMBINED"
    elif [ -n "$SAMPLE_1" ] && [ -n "$SAMPLE_2" ]; then
        uvv_log "===== BEGIN: combine parquets ====="
        local stage_start=$SECONDS
        python "$UV_VAE_DIR/scripts/combine_parquets.py" \
            --inputs "$SAMPLE_1" "$SAMPLE_2" \
            --output "$COMBINED" \
            --threads "$threads" \
            --memory-limit "$DUCKDB_MEM" \
            --temp-directory "$RUN_ROOT/duckdb_tmp"
        combine_seconds=$((SECONDS - stage_start))
        uvv_log "===== END: combine ($(uvv_fmt_seconds $combine_seconds)) ====="
    else
        echo "ERROR: $COMBINED does not exist and SAMPLE_1/SAMPLE_2 were not supplied." >&2
        echo "       Pass COMBINED=<existing parquet>, or SAMPLE_1=... SAMPLE_2=... to build one." >&2
        exit 1
    fi

    uvv_log "===== BEGIN: train VAE (early stopping) ====="
    local stage_start=$SECONDS
    # shellcheck disable=SC2086
    python "$EARLY_STOPPING_DIR/Python Files/train_with_early_stopping.py" \
        --parquet-path "$COMBINED" \
        --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
        --output-dir "$RUN_ROOT/training" \
        --row-filter "$ROW_FILTER" \
        --streaming \
        --epochs "$EPOCH_CEILING" \
        --patience "$PATIENCE" \
        --min-delta "$MIN_DELTA" \
        --active-unit-threshold "$AU_THRESHOLD" \
        --input-dropout "$INPUT_DROPOUT" \
        --hidden-dropout "$HIDDEN_DROPOUT" \
        --batch-size "$BATCH_SIZE" \
        --latent-dim "$LATENT_DIM" \
        --hidden-dims "$HIDDEN_DIMS" \
        --learning-rate "$LEARNING_RATE" \
        --kl-weight "$KL_WEIGHT" \
        --train-fraction "$TRAIN_FRACTION" \
        --warmup-steps "$WARMUP_STEPS" \
        --seed "$SEED" \
        --threads "$threads" \
        ${TEST_PARQUET:+--test-parquet-path "$TEST_PARQUET" --convergence-rows "$CONVERGENCE_ROWS"} \
        > "$RUN_ROOT/train_result.json"
    cat "$RUN_ROOT/train_result.json"
    local train_seconds=$((SECONDS - stage_start))
    uvv_log "===== END: train ($(uvv_fmt_seconds $train_seconds)) ====="

    uvv_rule
    printf '  %-24s %8ss   %s\n' "combine" "$combine_seconds" "$(uvv_fmt_seconds "$combine_seconds")"
    printf '  %-24s %8ss   %s\n' "train"   "$train_seconds"   "$(uvv_fmt_seconds "$train_seconds")"
    printf '  %-24s %8ss   %s\n' "TOTAL"   "$SECONDS"         "$(uvv_fmt_seconds "$SECONDS")"
    python - "$RUN_ROOT/train_result.json" <<'PY' || true
import json, sys
early = (json.load(open(sys.argv[1])).get("early_stopping") or {})
print("  epochs_run      : {} of {}".format(
    early.get("epochs_run", "?"), early.get("epochs_requested", "?")))
print("  stopped_early   : {}".format(early.get("stopped_early", "?")))
print("  best_epoch      : {}".format(early.get("best_epoch", "?")))
print("  final AU count  : {}".format(early.get("final_active_units", "?")))
print("  stop_reason     : {}".format(early.get("stop_reason")))
PY
    echo "  artifacts       : $RUN_ROOT"
    echo "  epoch diagnostics: $RUN_ROOT/training/*/diagnostics_report.json"
    echo "  GPU cap recorded : $RUN_ROOT/training/*/training_report.json  (key: gpu_budget)"
    uvv_rule
}

if [ "${UVV_CHILD:-0}" = "1" ] || [ "${FOREGROUND:-0}" = "1" ] || [ "${DRY_RUN:-0}" = "1" ]; then
    main
else
    uvv_strip_crlf "$UV_VAE_DIR/scripts/tmux_lib.sh" "$EARLY_STOPPING_DIR/scripts/tmux_train_only.sh"
    mkdir -p "$LOG_DIR"
    uvv_export_child_env \
        UV_VAE_DIR EARLY_STOPPING_DIR CONDA_ENV \
        RUN_ID RUN_ROOT COMBINED SAMPLE_1 SAMPLE_2 ROW_FILTER SEED \
        GPU_TOTAL_GB THREADS_TOTAL UV_VAE_GPU_OOM_POLICY DUCKDB_MEM \
        BATCH_SIZE EPOCH_CEILING PATIENCE MIN_DELTA AU_THRESHOLD \
        INPUT_DROPOUT HIDDEN_DROPOUT LATENT_DIM HIDDEN_DIMS LEARNING_RATE \
        KL_WEIGHT TRAIN_FRACTION WARMUP_STEPS TEST_PARQUET CONVERGENCE_ROWS \
        SKIP_PREFLIGHT
    uvv_launch_tmux "$SESSION" "$LOG_DIR" \
        "UVV_CHILD=1 bash '$EARLY_STOPPING_DIR/scripts/tmux_train_only.sh'"
fi
