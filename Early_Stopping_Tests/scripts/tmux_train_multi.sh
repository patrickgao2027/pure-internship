#!/bin/bash
# Train the VAE across ALL per-sample featuremap parquets, proportionally interleaved.
#
# Every batch contains every sample in proportion to its share of the total
# filtered rows, and each file is read in shuffled row-group order so a batch is
# a spread across the genome rather than one contiguous window. See
# uv_vae/uv_vae/multi_parquet.py for why both halves are necessary.
#
# Two stages:
#
#   1. statistics -- one scan per file, cached. Prints each sample's interleave
#      weight and per-batch draw. STATS_ONLY=1 stops here.
#   2. training   -- the interleaved trainer.
#
# Run stage 1 on its own first. It is the cheapest way to confirm all 95 files
# parse, the row filter behaves, and no sample is starved -- and the cache it
# writes is reused by every later run, so it is not throwaway work:
#
#     PARQUET_GLOB='/cta/users/patrickgao765/parquet_files/*.featuremap.parquet' \
#     STATS_ONLY=1 bash Early_Stopping_Tests/scripts/tmux_train_multi.sh
#
# Then drop STATS_ONLY to train. EPOCH_SHARDS=20 makes one full pass over every
# row cost 20 epochs instead of one, so a run that early-stops around epoch 15-20
# costs roughly what a single naive epoch would.
#
#     PARQUET_GLOB='...' EPOCH_SHARDS=20 EPOCH_CEILING=40 PATIENCE=8 \
#     bash Early_Stopping_Tests/scripts/tmux_train_multi.sh
#
# Windows edit note:  sed -i 's/\r$//' Early_Stopping_Tests/scripts/tmux_train_multi.sh

set -euo pipefail

UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
EARLY_STOPPING_DIR="${EARLY_STOPPING_DIR:-$HOME/Early_Stopping_Tests}"

# shellcheck source=/dev/null
source "$UV_VAE_DIR/scripts/tmux_lib.sh"

# ── Configuration ───────────────────────────────────────────────────────────
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$UV_VAE_DIR/runs/train_multi_$RUN_ID}"
ROW_FILTER="${ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
SEED="${SEED:-42}"
SESSION="${SESSION:-train_multi}"

PARQUET_GLOB="${PARQUET_GLOB:-/cta/users/patrickgao765/parquet_files/*.featuremap.parquet}"
# Survives runs, so a re-run and a later 96th sample both cost one scan, not 95.
STATS_CACHE="${STATS_CACHE:-$UV_VAE_DIR/stats_cache.json}"
STATS_ONLY="${STATS_ONLY:-0}"

EPOCH_CEILING="${EPOCH_CEILING:-40}"
EPOCH_SHARDS="${EPOCH_SHARDS:-20}"
PATIENCE="${PATIENCE:-8}"
MIN_DELTA="${MIN_DELTA:-0.001}"
AU_THRESHOLD="${AU_THRESHOLD:-0.01}"
INPUT_DROPOUT="${INPUT_DROPOUT:-0.1}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.4}"
BATCH_SIZE="${BATCH_SIZE:-32768}"
LATENT_DIM="${LATENT_DIM:-16}"
HIDDEN_DIMS="${HIDDEN_DIMS:-256,128}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
KL_WEIGHT="${KL_WEIGHT:-0.05}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.9}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-global_site_hash}"
SHUFFLE_BUFFER_ROWS="${SHUFFLE_BUFFER_ROWS:-32768}"
VAL_MAX_ROWS="${VAL_MAX_ROWS:-5000000}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
TEST_PARQUET="${TEST_PARQUET:-}"
CONVERGENCE_ROWS="${CONVERGENCE_ROWS:-5000}"

# Derived ONCE and passed to BOTH stages. Deriving it separately per stage is a
# real bug, not a style point: in float, `1 - 0.9` is 0.09999999999999998, which
# gives a different 64-bit split threshold than a literal 0.1 and therefore a
# genuinely different partition of the data. The statistics stage would describe
# one split and the trainer would use another.
VAL_FRACTION="${VAL_FRACTION:-$(awk -v t="$TRAIN_FRACTION" 'BEGIN { printf "%.10f", 1 - t }')}"

LOG_DIR="$RUN_ROOT/logs"

main() {
    uvv_activate_env
    uvv_export_determinism "$SEED"
    uvv_plan_resources 1
    export UV_VAE_GPU_MEM_GB="$UVV_GPU_PER_WORKER"
    local threads="$UVV_THREADS_PER_WORKER"

    # The interleaved reader decodes with pyarrow + polars on the CPU, so nothing
    # in this process allocates from RMM and torch should get the whole budget.
    # train_interleaved already defaults to this on its own; setting it here as
    # well is what makes the batch ceiling printed below agree with what the run
    # actually gets, instead of quoting a ceiling 25% low. cuML in-process is the
    # one case that still needs a bounded pool -- unbounded, it grows toward all
    # free memory and the torch cap buys nothing.
    if [ "${UV_VAE_ENABLE_CUML:-0}" = "1" ]; then
        UV_VAE_RMM_SHARE="${UV_VAE_RMM_SHARE:-0.25}"
    else
        UV_VAE_RMM_SHARE="${UV_VAE_RMM_SHARE:-0}"
    fi
    export UV_VAE_RMM_SHARE

    mkdir -p "$RUN_ROOT" "$LOG_DIR"

    local n_files
    n_files=$(compgen -G "$PARQUET_GLOB" | wc -l || echo 0)

    uvv_rule
    echo "Interleaved multi-parquet training"
    echo "  run root    : $RUN_ROOT"
    echo "  parquets    : $PARQUET_GLOB  ($n_files files)"
    echo "  stats cache : $STATS_CACHE"
    echo "  row filter  : $ROW_FILTER"
    echo "  split       : $SPLIT_STRATEGY  val_fraction=$VAL_FRACTION"
    echo "  batch       : $BATCH_SIZE   lr: $LEARNING_RATE   beta: $KL_WEIGHT"
    echo "  epochs      : $EPOCH_CEILING (patience=$PATIENCE, shards=$EPOCH_SHARDS)"
    echo "  GPU budget  : ${UV_VAE_GPU_MEM_GB} GB    threads: $threads"
    echo "  seed        : $SEED"
    uvv_rule

    if [ "$n_files" -eq 0 ]; then
        echo "ERROR: no files matched PARQUET_GLOB=$PARQUET_GLOB" >&2
        exit 1
    fi

    # ── Stage 1: statistics ─────────────────────────────────────────────────
    uvv_log "===== BEGIN: per-file statistics ====="
    local stage_start=$SECONDS
    python "$UV_VAE_DIR/scripts/multi_parquet_stats.py" \
        --parquet-paths "$PARQUET_GLOB" \
        --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
        --row-filter "$ROW_FILTER" \
        --stats-cache-path "$STATS_CACHE" \
        --batch-size "$BATCH_SIZE" \
        --epoch-shards "$EPOCH_SHARDS" \
        --val-fraction "$VAL_FRACTION" \
        --split-strategy "$SPLIT_STRATEGY" \
        --threads "$threads" \
        --json-out "$RUN_ROOT/sampling_plan.json" \
        || { echo "Statistics stage reported problems (see WARNING lines above)." >&2; exit 1; }
    local stats_seconds=$((SECONDS - stage_start))
    uvv_log "===== END: statistics ($(uvv_fmt_seconds $stats_seconds)) ====="

    if [ "$STATS_ONLY" = "1" ]; then
        uvv_rule
        echo "STATS_ONLY=1 -- stopping before training."
        echo "  sampling plan : $RUN_ROOT/sampling_plan.json"
        echo "  stats cache   : $STATS_CACHE  (reused by every later run)"
        uvv_rule
        return 0
    fi

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

    # ── Stage 2: training ───────────────────────────────────────────────────
    uvv_log "===== BEGIN: train VAE (interleaved, early stopping) ====="
    stage_start=$SECONDS
    # shellcheck disable=SC2086
    python "$EARLY_STOPPING_DIR/Python Files/train_with_early_stopping.py" \
        --parquet-paths "$PARQUET_GLOB" \
        --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
        --output-dir "$RUN_ROOT/training" \
        --row-filter "$ROW_FILTER" \
        --split-strategy "$SPLIT_STRATEGY" \
        --val-fraction "$VAL_FRACTION" \
        --epoch-shards "$EPOCH_SHARDS" \
        --stats-cache-path "$STATS_CACHE" \
        --shuffle-buffer-rows "$SHUFFLE_BUFFER_ROWS" \
        --val-max-rows "$VAL_MAX_ROWS" \
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
    printf '  %-24s %8ss   %s\n' "statistics" "$stats_seconds" "$(uvv_fmt_seconds "$stats_seconds")"
    printf '  %-24s %8ss   %s\n' "train"      "$train_seconds" "$(uvv_fmt_seconds "$train_seconds")"
    printf '  %-24s %8ss   %s\n' "TOTAL"      "$SECONDS"       "$(uvv_fmt_seconds "$SECONDS")"
    python - "$RUN_ROOT/train_result.json" <<'PY' || true
import json, sys
payload = json.load(open(sys.argv[1]))
early = payload.get("early_stopping") or {}
sampling = payload.get("sampling") or {}
interleave = sampling.get("interleave") or {}
print("  samples         : {}".format(sampling.get("sample_count", "?")))
print("  filtered rows   : {:,}".format(sampling.get("total_filtered_rows", 0)))
print("  epoch shards    : {}".format(sampling.get("epoch_shards", "?")))
print("  split           : {} @ {}".format(
    (sampling.get("split") or {}).get("strategy"),
    (sampling.get("split") or {}).get("val_fraction")))
ragged = sampling.get("train_ragged_row_fraction_last_epoch")
if ragged is not None:
    print("  ragged tail     : {:.3f}% of the last epoch".format(ragged * 100))
weights = interleave.get("weights") or {}
if weights:
    lo = min(weights.items(), key=lambda kv: kv[1])
    hi = max(weights.items(), key=lambda kv: kv[1])
    print("  weight range    : {} {:.5f} .. {} {:.5f}".format(lo[0], lo[1], hi[0], hi[1]))
print("  epochs_run      : {} of {}".format(
    early.get("epochs_run", "?"), early.get("epochs_requested", "?")))
print("  stopped_early   : {}".format(early.get("stopped_early", "?")))
print("  best_epoch      : {}".format(early.get("best_epoch", "?")))
print("  final AU count  : {}".format(early.get("final_active_units", "?")))
print("  stop_reason     : {}".format(early.get("stop_reason")))
PY
    echo "  artifacts        : $RUN_ROOT"
    echo "  sampling plan    : $RUN_ROOT/sampling_plan.json"
    echo "  per-sample detail: $RUN_ROOT/training/*/training_report.json  (key: sampling)"
    uvv_rule
}

if [ "${UVV_CHILD:-0}" = "1" ] || [ "${FOREGROUND:-0}" = "1" ] || [ "${DRY_RUN:-0}" = "1" ]; then
    main
else
    uvv_strip_crlf "$UV_VAE_DIR/scripts/tmux_lib.sh" "$EARLY_STOPPING_DIR/scripts/tmux_train_multi.sh"
    mkdir -p "$LOG_DIR"
    uvv_export_child_env \
        UV_VAE_DIR EARLY_STOPPING_DIR CONDA_ENV MAMBA_ENV MAMBA_ROOT_PREFIX \
        RUN_ID RUN_ROOT PARQUET_GLOB STATS_CACHE STATS_ONLY ROW_FILTER SEED \
        GPU_TOTAL_GB THREADS_TOTAL UV_VAE_GPU_OOM_POLICY \
        UV_VAE_RMM_SHARE UV_VAE_ENABLE_CUML \
        BATCH_SIZE EPOCH_CEILING EPOCH_SHARDS PATIENCE MIN_DELTA AU_THRESHOLD \
        INPUT_DROPOUT HIDDEN_DROPOUT LATENT_DIM HIDDEN_DIMS LEARNING_RATE \
        KL_WEIGHT TRAIN_FRACTION VAL_FRACTION SPLIT_STRATEGY \
        SHUFFLE_BUFFER_ROWS VAL_MAX_ROWS WARMUP_STEPS TEST_PARQUET \
        CONVERGENCE_ROWS SKIP_PREFLIGHT
    uvv_launch_tmux "$SESSION" "$LOG_DIR" \
        "UVV_CHILD=1 bash '$EARLY_STOPPING_DIR/scripts/tmux_train_multi.sh'"
fi
