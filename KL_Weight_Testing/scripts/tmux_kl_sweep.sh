#!/bin/bash
# KL weight (beta) sweep -- tmux replacement for run_kl_sweep.sh.
#
# What changed versus the SLURM version, and why:
#   * The original ran the beta values in a serial for-loop inside ONE job, because
#     it was one sbatch submission. There is no such constraint now, so the betas
#     go through the same work queue as every other sweep and CONCURRENCY decides
#     whether they overlap.
#   * GPU_TOTAL_GB is split across workers (both the torch cap and the RMM pool
#     are per process).
#   * RUN_ROOT no longer keys off $SLURM_JOB_ID, which collapsed to "manual".
#   * conda 'patrickg' is no longer hardcoded -- micromamba 'uv_vae' (MAMBA_ENV) is
#     the miletus default, then .venv; CONDA_ENV overrides both.
#   * Finished betas are skipped on re-launch, so a crash resumes.
#   * The trailing sacct hint is gone; per-task timings live in the state markers.
#
# The optional clustering stage (CLUSTER=1) is the one place cuML is wanted, so
# that stage -- and only that stage -- gets UV_VAE_ENABLE_CUML=1.
#
# Usage:
#     PARQUET=/path/to/featuremap.parquet \
#     CONCURRENCY=1 GPU_TOTAL_GB=16 \
#     bash KL_Weight_Testing/scripts/tmux_kl_sweep.sh
#
#     KL_WEIGHTS="0.01,0.05,0.1" ...   # narrow the sweep
#     CLUSTER=1 ...                    # add UMAP -> HDBSCAN -> SigProfiler per beta
#     FOREGROUND=1 / DRY_RUN=1 / FORCE=1 / TASKS="0 2 4"
#
# Windows edit note:  sed -i 's/\r$//' KL_Weight_Testing/scripts/tmux_kl_sweep.sh

set -euo pipefail

UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
KL_TESTING_DIR="${KL_TESTING_DIR:-$HOME/KL_Weight_Testing}"
EARLY_STOPPING_DIR="${EARLY_STOPPING_DIR:-$HOME/Early_Stopping_Tests}"

# shellcheck source=/dev/null
source "$UV_VAE_DIR/scripts/tmux_lib.sh"

# ── Configuration ───────────────────────────────────────────────────────────
PARQUET="${PARQUET:-/cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$UV_VAE_DIR/sweeps/kl_$RUN_ID}"
ROW_FILTER="${ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
SEED="${SEED:-42}"
CONCURRENCY="${CONCURRENCY:-1}"
SESSION="${SESSION:-kl_sweep}"

# Log-spaced from reconstruction-dominated to full-VAE. 0.05 is the pipeline baseline.
KL_WEIGHTS="${KL_WEIGHTS:-0.001,0.005,0.01,0.05,0.1,0.5,1.0}"

EPOCH_CEILING="${EPOCH_CEILING:-100}"
PATIENCE="${PATIENCE:-8}"
MIN_DELTA="${MIN_DELTA:-0.001}"
AU_THRESHOLD="${AU_THRESHOLD:-0.01}"
INPUT_DROPOUT="${INPUT_DROPOUT:-0.1}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.4}"
BATCH_SIZE="${BATCH_SIZE:-32768}"
LATENT_DIM="${LATENT_DIM:-16}"
HIDDEN_DIMS="${HIDDEN_DIMS:-256,128}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"

CLUSTER="${CLUSTER:-0}"
DUCKDB_MEM="${DUCKDB_MEM:-8GB}"
UMAP_FIT_ROWS="${UMAP_FIT_ROWS:-100000}"
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-30}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.05}"
UMAP_METRIC="${UMAP_METRIC:-euclidean}"
HDBSCAN_MIN_CLUSTER_SIZE="${HDBSCAN_MIN_CLUSTER_SIZE:-250}"
HDBSCAN_MIN_SAMPLES="${HDBSCAN_MIN_SAMPLES:-25}"

REFERENCE_KL="${REFERENCE_KL:-0.05}"
REFERENCE_ROWS="${REFERENCE_ROWS:-4000}"

LOG_DIR="$RUN_ROOT/logs"
STATE_DIR="$RUN_ROOT/state"

IFS=',' read -ra KL_ARRAY <<< "$KL_WEIGHTS"

# ── One task: train one beta, optionally cluster it ─────────────────────────
run_task() {
    local task="$1"
    local kl="${KL_ARRAY[$task]}"
    kl="${kl// /}"

    local kl_dir="$RUN_ROOT/kl_${kl}"
    mkdir -p "$kl_dir"

    export UV_VAE_GPU_MEM_GB="$UVV_GPU_PER_WORKER"
    local threads="$UVV_THREADS_PER_WORKER"

    uvv_rule
    echo "KL weight     : $kl (task $task)"
    echo "Parquet       : $PARQUET"
    echo "Batch         : $BATCH_SIZE"
    echo "Epochs        : $EPOCH_CEILING (patience=$PATIENCE)"
    echo "GPU budget    : ${UV_VAE_GPU_MEM_GB} GB    Threads: $threads"
    echo "Cluster stage : $CLUSTER"
    echo "Output        : $kl_dir"
    uvv_rule

    local train_start=$SECONDS
    uvv_log "===== BEGIN: train beta=$kl ====="
    python "$EARLY_STOPPING_DIR/Python Files/train_with_early_stopping.py" \
        --parquet-path "$PARQUET" \
        --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
        --output-dir "$kl_dir/training" \
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
        --kl-weight "$kl" \
        --train-fraction "$TRAIN_FRACTION" \
        --seed "$SEED" \
        --threads "$threads" \
        > "$kl_dir/train_result.json"
    cat "$kl_dir/train_result.json"
    uvv_log "===== END: train beta=$kl ($(uvv_fmt_seconds $((SECONDS - train_start)))) ====="

    if [ "$CLUSTER" = "1" ]; then
        local checkpoint
        checkpoint="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint_path'])" \
            "$kl_dir/train_result.json")"

        local cluster_start=$SECONDS
        uvv_log "===== BEGIN: cluster beta=$kl ====="
        # The clustering pipeline is the one stage that genuinely wants cuML.
        UV_VAE_ENABLE_CUML=1 python "$UV_VAE_DIR/scripts/run_variant_cluster_pipeline.py" \
            --checkpoint-path "$checkpoint" \
            --parquet-path "$PARQUET" \
            --row-filter "$ROW_FILTER" \
            --output-root "$kl_dir/clustering" \
            --use-all \
            --umap-fit-rows "$UMAP_FIT_ROWS" \
            --umap-n-neighbors "$UMAP_N_NEIGHBORS" \
            --umap-min-dist "$UMAP_MIN_DIST" \
            --umap-metric "$UMAP_METRIC" \
            --hdbscan-min-cluster-size "$HDBSCAN_MIN_CLUSTER_SIZE" \
            --hdbscan-min-samples "$HDBSCAN_MIN_SAMPLES" \
            --seed "$SEED" \
            --threads "$threads" \
            --sigprofiler-cpu "$threads" \
            --duckdb-memory-limit "$DUCKDB_MEM" \
            > "$kl_dir/cluster_result.json"
        uvv_log "===== END: cluster beta=$kl ($(uvv_fmt_seconds $((SECONDS - cluster_start)))) ====="
    fi
}

# ── Driver ──────────────────────────────────────────────────────────────────
main() {
    uvv_activate_env
    uvv_export_determinism "$SEED"
    uvv_plan_resources "$CONCURRENCY"

    local default_tasks=()
    local i
    for i in "${!KL_ARRAY[@]}"; do default_tasks+=("$i"); done
    read -ra TASK_LIST <<< "${TASKS:-${default_tasks[*]}}"

    mkdir -p "$RUN_ROOT" "$LOG_DIR" "$STATE_DIR"

    uvv_rule
    echo "KL weight sweep"
    echo "  run root    : $RUN_ROOT"
    echo "  parquet     : $PARQUET"
    echo "  row filter  : $ROW_FILTER"
    echo "  kl weights  : ${KL_ARRAY[*]}"
    echo "  tasks       : ${TASK_LIST[*]}"
    echo "  concurrency : $CONCURRENCY"
    echo "  cluster     : $CLUSTER"
    echo "  seed        : $SEED"
    uvv_rule

    if [ ! -f "$PARQUET" ] && ! compgen -G "$PARQUET" > /dev/null; then
        echo "ERROR: parquet not found: $PARQUET" >&2
        exit 1
    fi

    echo "GPU budget check:"
    uvv_report_batch_ceiling "$UVV_GPU_PER_WORKER" "$BATCH_SIZE"
    echo

    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        uvv_log "running GPU preflight"
        python "$UV_VAE_DIR/scripts/gpu_preflight.py" \
            --batch-size "$BATCH_SIZE" \
            --budget-gb "$UVV_GPU_PER_WORKER" \
            --json-out "$RUN_ROOT/gpu_preflight.json" \
            || { echo "Preflight failed. Fix the FAIL lines above, or set SKIP_PREFLIGHT=1 to override." >&2; exit 1; }
        echo
    fi

    local overall=$SECONDS
    uvv_run_queue "$CONCURRENCY" "$STATE_DIR" "$LOG_DIR" run_task "${TASK_LIST[@]}"

    uvv_log "comparing all beta values against beta=$REFERENCE_KL"
    python "$KL_TESTING_DIR/Python Files/compare_kl_sweep.py" \
        --sweep-dir "$RUN_ROOT" \
        --parquet-path "$PARQUET" \
        --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
        --row-filter "$ROW_FILTER" \
        --reference-kl "$REFERENCE_KL" \
        --reference-rows "$REFERENCE_ROWS" \
        --seed "$SEED" \
        --threads "${UVV_THREADS_PER_WORKER}" \
        || uvv_log "WARNING: comparison failed; per-beta checkpoints are still on disk"

    uvv_rule
    echo "KL sweep complete in $(uvv_fmt_seconds $((SECONDS - overall)))"
    echo "  artifacts     : $RUN_ROOT"
    echo "  per-task logs : $LOG_DIR"
    echo "  comparison    : $RUN_ROOT/kl_sweep_comparison.json"
    echo "  plot          : $RUN_ROOT/kl_sweep_comparison.png"
    uvv_rule
}

if [ "${UVV_CHILD:-0}" = "1" ] || [ "${FOREGROUND:-0}" = "1" ] || [ "${DRY_RUN:-0}" = "1" ]; then
    main
else
    uvv_strip_crlf "$UV_VAE_DIR/scripts/tmux_lib.sh" "$KL_TESTING_DIR/scripts/tmux_kl_sweep.sh"
    mkdir -p "$LOG_DIR"
    uvv_export_child_env \
        UV_VAE_DIR KL_TESTING_DIR EARLY_STOPPING_DIR CONDA_ENV MAMBA_ENV MAMBA_ROOT_PREFIX \
        RUN_ID RUN_ROOT PARQUET ROW_FILTER SEED \
        CONCURRENCY GPU_TOTAL_GB THREADS_TOTAL UV_VAE_GPU_OOM_POLICY \
        KL_WEIGHTS CLUSTER BATCH_SIZE EPOCH_CEILING PATIENCE MIN_DELTA \
        AU_THRESHOLD INPUT_DROPOUT HIDDEN_DROPOUT LATENT_DIM HIDDEN_DIMS \
        LEARNING_RATE TRAIN_FRACTION REFERENCE_KL REFERENCE_ROWS \
        DUCKDB_MEM UMAP_FIT_ROWS UMAP_N_NEIGHBORS UMAP_MIN_DIST UMAP_METRIC \
        HDBSCAN_MIN_CLUSTER_SIZE HDBSCAN_MIN_SAMPLES \
        TASKS FORCE SKIP_PREFLIGHT
    uvv_launch_tmux "$SESSION" "$LOG_DIR" \
        "UVV_CHILD=1 bash '$KL_TESTING_DIR/scripts/tmux_kl_sweep.sh'"
fi
