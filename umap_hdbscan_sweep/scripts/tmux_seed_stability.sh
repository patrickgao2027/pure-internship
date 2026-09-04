#!/bin/bash
# Is seed noise the reason cells score differently? Three runs that answer it.
#
# The sweep's family E redraws the fit rows AND changes every seed per replicate,
# so a low adjusted_rand_min cannot be attributed to either. These three tasks
# hold one axis fixed at a time:
#
#   1. 5M_encoder   UMAP fit once, 5 encoders varying only the torch seed.
#                   Whatever spread survives is pure optimiser noise.   ~9 min
#   2. 5M_full      fit rows redrawn and UMAP refit per seed, as a sweep
#                   replicate does. Minus task 1 = the subsample share. ~17 min
#   3. 25M_encoder  the same optimiser-noise floor for cell #13.        ~54 min
#
# Deliberately cheapest-first, NOT the order they were asked for: tasks 1 and 2
# together are the whole decomposition and land in ~26 min, while task 3 only
# confirms it at the other fit size. If the first two show encoder noise is
# negligible, task 3 can be killed without losing the answer.
#
# --axis full on the 25M cell is NOT included. It is 5 x 2797s of UMAP fitting,
# about 4 hours. Add it by hand only if tasks 1-3 leave the question open:
#
#     CELLS_25M_FULL=1 bash umap_hdbscan_sweep/tmux_seed_stability.sh
#
# The queue writes a .done marker per task, so a crash or a killed session
# resumes where it stopped. FORCE=1 re-runs completed tasks.
#
#     bash umap_hdbscan_sweep/tmux_seed_stability.sh          # launch in tmux
#     DRY_RUN=1 bash umap_hdbscan_sweep/tmux_seed_stability.sh  # print, run nothing
#     FOREGROUND=1 bash umap_hdbscan_sweep/tmux_seed_stability.sh
#
# Windows edit note:  sed -i 's/\r$//' umap_hdbscan_sweep/tmux_seed_stability.sh

set -euo pipefail

# Repo layout differs by cluster: on miletus the branch is a git clone at
# ~/pure-internship, on tosun the folders are siblings in $HOME. Probe rather
# than hardcode; an explicit export still wins.
if [ -z "${UV_VAE_DIR:-}" ] && [ -d "$HOME/pure-internship/uv_vae" ]; then
    UV_VAE_DIR="$HOME/pure-internship/uv_vae"
fi
if [ -z "${SWEEP_DIR:-}" ] && [ -d "$HOME/pure-internship/umap_hdbscan_sweep" ]; then
    SWEEP_DIR="$HOME/pure-internship/umap_hdbscan_sweep"
fi
UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
SWEEP_DIR="${SWEEP_DIR:-$HOME/umap_hdbscan_sweep}"
REPO_ROOT="$(dirname "$SWEEP_DIR")"

# shellcheck source=/dev/null
source "$UV_VAE_DIR/scripts/tmux_lib.sh"

# ── Configuration ───────────────────────────────────────────────────────────
SESSION="${SESSION:-seed_stability}"
SEEDS="${SEEDS:-5}"
SEED_BASE="${SEED_BASE:-1000}"
OUT_DIR="${OUT_DIR:-$SWEEP_DIR/umap_tests}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/seed_stability_logs}"
STATE_DIR="${STATE_DIR:-$LOG_DIR/state}"
EMBED_DIR="${EMBED_DIR:-$UV_VAE_DIR/runs/train_multi_20260802T192756Z/stage1_embed}"
RUNC_JSON="${RUNC_JSON:-$SWEEP_DIR/umap_tests/parametric_sweep/runC/parametric_sweep.json}"
RUND_JSON="${RUND_JSON:-$SWEEP_DIR/umap_tests/parametric_sweep/runD/parametric_sweep.json}"
CELL_5M="${CELL_5M:-5000000|nn30_md0.25_nc2|umap}"
CELL_25M="${CELL_25M:-25000000|nn15_md0.1_nc2|umap}"
CELLS_25M_FULL="${CELLS_25M_FULL:-0}"

# One GPU, and every task wants all of it. Concurrency above 1 would just make
# them fight over VRAM and the 25M UMAP fit is the one that would lose.
CONCURRENCY=1

run_one() {
    local task_id="$1"
    local sweep_json cell axis
    case "$task_id" in
        5M_encoder)  sweep_json="$RUNC_JSON"; cell="$CELL_5M";  axis="encoder" ;;
        5M_full)     sweep_json="$RUNC_JSON"; cell="$CELL_5M";  axis="full" ;;
        25M_encoder) sweep_json="$RUND_JSON"; cell="$CELL_25M"; axis="encoder" ;;
        25M_full)    sweep_json="$RUND_JSON"; cell="$CELL_25M"; axis="full" ;;
        *) echo "unknown task '$task_id'" >&2; return 1 ;;
    esac

    local output="$OUT_DIR/seed_stability_${task_id}.json"
    uvv_log "task $task_id: axis=$axis cell=$cell"
    uvv_log "  -> $output"

    if [ ! -f "$sweep_json" ]; then
        echo "ERROR: no sweep JSON at $sweep_json" >&2
        return 1
    fi
    if [ ! -f "$EMBED_DIR/latent.npy" ]; then
        echo "ERROR: no latent.npy under EMBED_DIR=$EMBED_DIR" >&2
        echo "       the sweep's probe draw depends on this exact latent; set EMBED_DIR" >&2
        return 1
    fi

    if [ "${DRY_RUN:-0}" = "1" ]; then
        uvv_log "  DRY_RUN: would run seed_stability.py --axis $axis --seeds $SEEDS"
        return 0
    fi

    python "$SWEEP_DIR/seed_stability.py" \
        --sweep-json "$sweep_json" \
        --embed-dir "$EMBED_DIR" \
        --cells "$cell" \
        --seeds "$SEEDS" \
        --seed-base "$SEED_BASE" \
        --axis "$axis" \
        --output "$output"
}

main() {
    uvv_activate_env
    uvv_log_provenance
    export TQDM_DISABLE=1

    local tasks=(5M_encoder 5M_full 25M_encoder)
    if [ "$CELLS_25M_FULL" = "1" ]; then
        tasks+=(25M_full)
        uvv_log "NOTE: 25M_full added -- that task alone is ~4 hours of UMAP fitting"
    fi

    uvv_rule
    echo "seed stability: ${#tasks[@]} task(s), $SEEDS seeds each"
    echo "  embed dir : $EMBED_DIR"
    echo "  output    : $OUT_DIR/seed_stability_<task>.json"
    echo "  logs      : $LOG_DIR/task_<task>.log"
    echo "  order     : ${tasks[*]}  (cheapest first -- see the header)"
    uvv_rule

    uvv_run_queue "$CONCURRENCY" "$STATE_DIR" "$LOG_DIR" run_one "${tasks[@]}"

    uvv_rule
    echo "done. compare each cell's pairwise block against the sweep_reference"
    echo "printed beside it: if the encoder-axis spread matches what the sweep's"
    echo "replicates showed, seed noise explains the cell differences; if it is"
    echo "much tighter, the variation comes from the row draw."
    for task in "${tasks[@]}"; do
        local out="$OUT_DIR/seed_stability_${task}.json"
        [ -f "$out" ] && echo "  $out"
    done
    uvv_rule
}

if [ "${UVV_CHILD:-0}" = "1" ] || [ "${FOREGROUND:-0}" = "1" ] || [ "${DRY_RUN:-0}" = "1" ]; then
    main
else
    uvv_strip_crlf "$UV_VAE_DIR/scripts/tmux_lib.sh" "$SWEEP_DIR/tmux_seed_stability.sh"
    mkdir -p "$LOG_DIR"
    uvv_export_child_env \
        UV_VAE_DIR SWEEP_DIR CONDA_ENV MAMBA_ENV MAMBA_ROOT_PREFIX \
        SESSION SEEDS SEED_BASE OUT_DIR LOG_DIR STATE_DIR EMBED_DIR \
        RUNC_JSON RUND_JSON CELL_5M CELL_25M CELLS_25M_FULL \
        GPU_TOTAL_GB THREADS_TOTAL FORCE
    uvv_launch_tmux "$SESSION" "$LOG_DIR" \
        "UVV_CHILD=1 bash '$SWEEP_DIR/tmux_seed_stability.sh'"
fi
