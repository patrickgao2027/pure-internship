#!/usr/bin/env bash
# Measure peak VRAM + host RAM for the deployed parametric-UMAP cell, on miletus.
#
# miletus has no SLURM, so this runs under tmux like the other runners here. It does a
# cheap SMOKE pass first (2M rows, ~2 min) purely to prove the harness works, then the
# REAL pass at the deployed 25M. The smoke pass is the point of the script -- the 25M fit
# costs ~47 minutes and you do not want to discover a typo at the end of it.
#
#   SMOKE_ONLY=1     stop after the 2M pass
#   FIT_ROWS         rows for the real pass (default 25000000)
#   GPU_GB           per-process cap; unset means measure uncapped demand (the default)
#   RMM_SHARE        0 (default) = no pool, device reading tracks real demand
#                    0.85        = reproduce the sweep's pooled config instead
#
# Windows edit note: sed -i 's/\r$//' umap_hdbscan_sweep/*.sh
#
# Launch detached:
#   sed -i 's/\r$//' ~/pure-internship/umap_hdbscan_sweep/tmux_measure_umap_memory.sh
#   tmux new-session -d -s memprobe 'bash ~/pure-internship/umap_hdbscan_sweep/tmux_measure_umap_memory.sh'
#   tmux attach -t memprobe
set -uo pipefail

REPO="${REPO:-$HOME/pure-internship}"
SWEEP_DIR="$REPO/umap_hdbscan_sweep"
EMBED_DIR="${EMBED_DIR:-$REPO/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed}"
OUT_DIR="${OUT_DIR:-$SWEEP_DIR/umap_tests/memory_profile}"

FIT_ROWS="${FIT_ROWS:-25000000}"
SMOKE_ROWS="${SMOKE_ROWS:-2000000}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
RMM_SHARE="${RMM_SHARE:-0}"
GPU_GB="${GPU_GB:-}"
NN="${NN:-15}"
MIN_DIST="${MIN_DIST:-0.1}"

export TQDM_DISABLE=1

MAMBA_ENV="${MAMBA_ENV:-uv_vae}"
RUN=(micromamba run -n "$MAMBA_ENV" python)
command -v micromamba >/dev/null 2>&1 || RUN=(python)

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p "$OUT_DIR"

if [[ ! -f "$EMBED_DIR/latent.npy" ]]; then
    say "ERROR: no latent.npy under $EMBED_DIR"
    say "       set EMBED_DIR to the stage1_embed dir holding it."
    exit 1
fi

# The measurement is only meaningful if we are alone on the card -- a co-tenant's
# allocations land in the device-wide reading and cannot be separated out of it.
read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
    --format=csv,noheader,nounits | head -1 | tr -d ',')
say "card: ${used} MiB used of ${total} MiB"
if (( used > 2000 )); then
    say "WARNING: ${used} MiB is already held by another process."
    say "         The device-wide peak will include it. Per-process (NVML) stays honest."
    say "         Ctrl-C now if you would rather wait for the card."
    sleep 10
fi

run_one() {
    local label="$1" rows="$2"
    local out="$OUT_DIR/${label}.json"
    local args=(
        "$SWEEP_DIR/measure_umap_memory.py"
        --embed-dir "$EMBED_DIR"
        --fit-rows "$rows"
        --n-neighbors "$NN"
        --min-dist "$MIN_DIST"
        --rmm-share "$RMM_SHARE"
        --out "$out"
    )
    [[ -n "$GPU_GB" ]] && args+=(--gpu-budget-gb "$GPU_GB")

    say "===== BEGIN $label ($rows rows) ====="
    "${RUN[@]}" "${args[@]}" 2>&1 | tee "$OUT_DIR/${label}.log"
    local status=${PIPESTATUS[0]}
    say "===== END $label (exit $status) ====="
    return "$status"
}

# ── 1. smoke ──────────────────────────────────────────────────────────────────
if ! run_one "smoke_${SMOKE_ROWS}" "$SMOKE_ROWS"; then
    say "smoke pass failed -- not spending 47 minutes on the real one. See the log above."
    exit 1
fi

if [[ "$SMOKE_ONLY" == "1" ]]; then
    say "SMOKE_ONLY=1, stopping. Results in $OUT_DIR"
    exit 0
fi

# ── 2. the deployed cell ──────────────────────────────────────────────────────
say "smoke pass clean. Running the deployed cell at ${FIT_ROWS} rows (~47 min for the fit)."
run_one "deployed_${FIT_ROWS}" "$FIT_ROWS"
status=$?

say "results in $OUT_DIR"
say "  peak numbers:  grep -A6 '\"peaks\"' $OUT_DIR/deployed_${FIT_ROWS}.json"
exit "$status"
