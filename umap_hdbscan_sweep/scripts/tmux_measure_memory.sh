#!/usr/bin/env bash
# Measure peak VRAM + host RAM for the HDBSCAN and UMAP stages, on miletus.
#
# miletus has no SLURM, so this runs under tmux like the other runners here. Each pass is
# its OWN process: RMM and cupy do not hand pool memory back to the driver, so two stages
# in one process would leave the second reading the first one's residue.
#
# The passes, in order (smoke first -- you do not want to find a typo after a 3 h fit):
#
#   1. hdbscan_smoke     200K fit rows, labels 5M    ~1 min   proves the harness
#   2. hdbscan_deployed  1M fit rows (mcs2500 ms1 eps0.05), labels the full cohort
#                                                    ~3 min   THE deployed model
#   3. umap_smoke        2M fit rows                 ~2 min   proves the umap path
#   4. umap_deployed     25M fit rows                ~50 min  the deployed pUMAP cell
#   5. hdbscan_sweepmax  25M fit rows (opt-in)       ~3.5 h   the sweep's largest cell
#
# Pass 5 is the answer to "what did the sweep's worst cell need" and is OFF by default
# because the fit is O(N^1.9) -- 11,684 s at 25M per hdbscan_param_sweep's cost model.
# 50M OOMed on this card at a 47 GB cap, so 25M is the largest cell that ever completed.
#
#   STAGES="hdbscan_smoke hdbscan_deployed"   run only these passes
#   SWEEP_MAX=1                               add pass 5
#   SMOKE_ONLY=1                              run only passes 1 and 3
#   GPU_GB                                    per-process cap; unset = measure real demand
#   RMM_SHARE                                 0 (default) = no pool, device tracks demand
#                                             0.85        = reproduce the sweep's config
#
# Windows edit note: sed -i 's/\r$//' umap_hdbscan_sweep/*.sh
#
# Launch detached:
#   sed -i 's/\r$//' ~/pure-internship/umap_hdbscan_sweep/tmux_measure_memory.sh
#   tmux new-session -d -s memprobe 'bash ~/pure-internship/umap_hdbscan_sweep/tmux_measure_memory.sh'
#   tmux attach -t memprobe
set -uo pipefail

REPO="${REPO:-$HOME/pure-internship}"
SWEEP_DIR="$REPO/umap_hdbscan_sweep"
SCRIPT="$SWEEP_DIR/measure_pipeline_memory.py"
EMBED_DIR="${EMBED_DIR:-$REPO/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed}"
OUT_DIR="${OUT_DIR:-$SWEEP_DIR/umap_tests/memory_profile}"

# The 2-D coordinates HDBSCAN clusters. The deployed model was fit on the coords the
# scaling run wrote; fall back to the embed dir's copy if that path is not present.
COORDS="${COORDS:-$SWEEP_DIR/hdbscan/results/hdbscan_scaling/coords.npy}"
[[ -f "$COORDS" ]] || COORDS="$EMBED_DIR/umap_coords_2d.npy"

RMM_SHARE="${RMM_SHARE:-0}"
GPU_GB="${GPU_GB:-}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SWEEP_MAX="${SWEEP_MAX:-0}"

# Deployed HDBSCAN, from UV_VAE_Deployment/models/hdbscan/metrics.json
MCS="${MCS:-2500}"
MS="${MS:-1}"
EPS="${EPS:-0.05}"
HDB_FIT_ROWS="${HDB_FIT_ROWS:-1000000}"

# Deployed pUMAP cell: 25M|nn15_md0.1_nc2|umap
UMAP_FIT_ROWS="${UMAP_FIT_ROWS:-25000000}"
NN="${NN:-15}"
MIN_DIST="${MIN_DIST:-0.1}"

export TQDM_DISABLE=1

MAMBA_ENV="${MAMBA_ENV:-uv_vae}"
RUN=(micromamba run -n "$MAMBA_ENV" python)
command -v micromamba >/dev/null 2>&1 || RUN=(python)

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p "$OUT_DIR"

[[ -f "$SCRIPT" ]] || { say "ERROR: $SCRIPT missing -- did the pull land?"; exit 1; }
[[ -f "$EMBED_DIR/latent.npy" ]] || {
    say "ERROR: no latent.npy under $EMBED_DIR (set EMBED_DIR)"; exit 1; }
[[ -f "$COORDS" ]] || {
    say "ERROR: no coords .npy found (set COORDS). Tried:"
    say "       $SWEEP_DIR/hdbscan/results/hdbscan_scaling/coords.npy"
    say "       $EMBED_DIR/umap_coords_2d.npy"
    exit 1; }

say "latent : $EMBED_DIR/latent.npy"
say "coords : $COORDS"

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

common_args() {
    printf '%s\n' --rmm-share "$RMM_SHARE"
    [[ -n "$GPU_GB" ]] && printf '%s\n' --gpu-budget-gb "$GPU_GB"
}

run_pass() {
    local label="$1"; shift
    local out="$OUT_DIR/${label}.json"
    local args=("$SCRIPT" "$@" --out "$out")
    mapfile -t -O "${#args[@]}" args < <(common_args)

    say "===== BEGIN $label ====="
    "${RUN[@]}" "${args[@]}" 2>&1 | tee "$OUT_DIR/${label}.log"
    local status=${PIPESTATUS[0]}
    say "===== END $label (exit $status) ====="
    return "$status"
}

hdbscan_pass() {   # label, fit rows, label rows
    run_pass "$1" --stage hdbscan --coords "$COORDS" \
        --hdbscan-fit-rows "$2" --label-rows "$3" \
        --min-cluster-size "$MCS" --min-samples "$MS" --cluster-selection-epsilon "$EPS"
}

umap_pass() {      # label, fit rows
    run_pass "$1" --stage umap --embed-dir "$EMBED_DIR" \
        --fit-rows "$2" --n-neighbors "$NN" --min-dist "$MIN_DIST"
}

DEFAULT_STAGES="hdbscan_smoke hdbscan_deployed umap_smoke umap_deployed"
[[ "$SMOKE_ONLY" == "1" ]] && DEFAULT_STAGES="hdbscan_smoke umap_smoke"
[[ "$SWEEP_MAX" == "1" && "$SMOKE_ONLY" != "1" ]] && DEFAULT_STAGES="$DEFAULT_STAGES hdbscan_sweepmax"
STAGES="${STAGES:-$DEFAULT_STAGES}"

say "passes: $STAGES"
failed=()

for stage in $STAGES; do
    case "$stage" in
        hdbscan_smoke)    hdbscan_pass hdbscan_smoke 200000 5000000 ;;
        hdbscan_deployed) hdbscan_pass hdbscan_deployed "$HDB_FIT_ROWS" -1 ;;
        hdbscan_sweepmax) hdbscan_pass hdbscan_sweepmax 25000000 -1 ;;
        umap_smoke)       umap_pass umap_smoke 2000000 ;;
        umap_deployed)    umap_pass umap_deployed "$UMAP_FIT_ROWS" ;;
        *) say "unknown pass '$stage', skipping"; continue ;;
    esac
    if [[ $? -ne 0 ]]; then
        failed+=("$stage")
        # A smoke failure means the harness is broken; nothing after it is worth the wait.
        case "$stage" in
            *_smoke) say "$stage failed -- the harness is broken, stopping."; break ;;
            *) say "$stage failed; continuing with the rest." ;;
        esac
    fi
done

say "results in $OUT_DIR"
say "  summary:  grep -A4 '\"peaks\"' $OUT_DIR/*.json"
if ((${#failed[@]})); then
    say "FAILED: ${failed[*]}"
    exit 1
fi
say "all passes clean"
