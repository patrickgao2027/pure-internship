#!/bin/bash
# UMAP x HDBSCAN sweep over the 95-file cohort, with SigProfiler at every grid cell.
#
# Four stages. Each writes a manifest the next one reads, so they can be run
# separately and resumed independently:
#
#   0  dedup    5.08B reads -> one row per (CHROM,POS,REF,ALT), chromosome by chromosome
#   1  embed    every deduplicated locus -> VAE latent.npy + context.parquet
#   2  sweep    UMAP grid x HDBSCAN grid, metrics + SigProfiler per cell   <- the sweep
#   3  apply    all 5.08B reads through ONE chosen cell -> read-weighted SigProfiler
#
# Stage 2 is the deliverable. Stage 3 is a separate, expensive question (what do the
# signatures look like when every read votes, not every locus) -- run it for the cells
# worth the hours, not for all 480.
#
# RUN STAGE 0 FIRST AND READ ITS OUTPUT. It prints the unique-locus count, which decides
# whether stage 2 can fit on everything or needs FIT_ROWS. Nothing downstream is sized
# correctly until that number is known:
#
#     STAGE=0 bash umap_hdbscan_sweep/run_sweep.sh
#
# Then, with a checkpoint from the interleaved trainer:
#
#     STAGE=1 CHECKPOINT=~/pure-internship/uv_vae/runs/train_multi_.../training/run_.../model.pt \
#     bash umap_hdbscan_sweep/run_sweep.sh
#
#     STAGE=2 bash umap_hdbscan_sweep/run_sweep.sh              # the full 480-cell grid
#     STAGE=2 DRY_RUN=1 bash umap_hdbscan_sweep/run_sweep.sh    # print the grid, fit nothing
#     STAGE=3 CELL=nn30_md0.05_nc2/mcs500_ms25 bash umap_hdbscan_sweep/run_sweep.sh
#
# Stage 2 resumes by default: cells whose metrics.json exists are skipped, so a killed
# sweep restarts where it stopped. NO_RESUME=1 forces a clean re-run.
#
# Windows edit note:  sed -i 's/\r$//' umap_hdbscan_sweep/run_sweep.sh

set -euo pipefail

# Repo layout differs by cluster: miletus has a git clone at ~/pure-internship, tosun
# deploys the folders as siblings in $HOME. Probe rather than hardcode.
if [ -z "${UV_VAE_DIR:-}" ] && [ -d "$HOME/pure-internship/uv_vae" ]; then
    UV_VAE_DIR="$HOME/pure-internship/uv_vae"
fi
if [ -z "${SWEEP_DIR:-}" ] && [ -d "$HOME/pure-internship/umap_hdbscan_sweep" ]; then
    SWEEP_DIR="$HOME/pure-internship/umap_hdbscan_sweep"
fi
UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
SWEEP_DIR="${SWEEP_DIR:-$HOME/umap_hdbscan_sweep}"

# shellcheck source=/dev/null
source "$UV_VAE_DIR/scripts/tmux_lib.sh"

# ── Configuration ───────────────────────────────────────────────────────────
STAGE="${STAGE:-2}"
RUN_ID="${RUN_ID:-cohort}"
RUN_ROOT="${RUN_ROOT:-$UV_VAE_DIR/runs/umap_sweep_$RUN_ID}"
ROW_FILTER="${ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
SEED="${SEED:-42}"
SESSION="${SESSION:-umap_sweep}"
NO_TMUX="${NO_TMUX:-0}"

# Same cohort-location probe the trainer uses; the miletus folder is shared lab data and
# is only ever read.
if [ -z "${PARQUET_GLOB:-}" ]; then
    for _dir in /data/lab/ppmseq_parquets /cta/users/patrickgao765/parquet_files; do
        if [ -d "$_dir" ]; then
            PARQUET_GLOB="$_dir/*.featuremap.parquet"
            break
        fi
    done
fi
PARQUET_GLOB="${PARQUET_GLOB:-/data/lab/ppmseq_parquets/*.featuremap.parquet}"

CHECKPOINT="${CHECKPOINT:-}"
FEATURE_SPEC="${FEATURE_SPEC:-$UV_VAE_DIR/ml_features.json}"
DEDUP_DIR="${DEDUP_DIR:-$RUN_ROOT/stage0_dedup}"
EMBED_DIR="${EMBED_DIR:-$RUN_ROOT/stage1_embed}"
SWEEP_OUT="${SWEEP_OUT:-$RUN_ROOT/stage2_sweep}"
APPLY_OUT="${APPLY_OUT:-$RUN_ROOT/stage3_apply}"

# Stage 0 spills heavily: the per-chromosome window sorts far more than fits in RAM.
DUCKDB_MEMORY="${DUCKDB_MEMORY:-200GB}"
TEMP_DIR="${TEMP_DIR:-$RUN_ROOT/duckdb_tmp}"
CHROMOSOMES="${CHROMOSOMES:-}"

# The grid the sweep covers -- 4 x 4 x 2 = 32 UMAP fits x (5 x 3) = 15 cuts = 480 cells.
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-15,30,50,100}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.0,0.05,0.1,0.25}"
UMAP_N_COMPONENTS="${UMAP_N_COMPONENTS:-2,16}"
UMAP_N_EPOCHS="${UMAP_N_EPOCHS:-200}"
MIN_CLUSTER_SIZES="${MIN_CLUSTER_SIZES:-100,250,500,1000,2500}"
MIN_SAMPLES="${MIN_SAMPLES:-5,25,50}"

# 'all' fits UMAP/HDBSCAN on every deduplicated locus. Set a number if stage 2 OOMs --
# the rest of the population is then labelled by transform + approximate_predict, so
# every locus still reaches SigProfiler either way.
FIT_ROWS="${FIT_ROWS:-all}"
# NOT a free performance knob: UMAP.transform is not batch-invariant, so this is part of
# the result. Keep it fixed across runs being compared. See sweep_core.FittedUmap.transform.
TRANSFORM_BATCH="${TRANSFORM_BATCH:-5000000}"
EMBED_BATCH="${EMBED_BATCH:-262144}"
APPLY_BATCH="${APPLY_BATCH:-1048576}"
SIL_EVAL_ROWS="${SIL_EVAL_ROWS:-50000}"
AGREEMENT_ROWS="${AGREEMENT_ROWS:-200000}"

# ── GPU budget when sharing the card with a running trainer ─────────────────
# The caps gpu_budget installs are PER PROCESS, so two capped processes sum. To hold
# the total at GPU_TOTAL_GB, the trainer's cap and this sweep's cap must add up to it:
#
#     GPU_TOTAL_GB (16)  =  TRAINER_GPU_GB (10)  +  SWEEP_GPU_GB (6)
#
# The trainer's half is only honoured if the trainer was STARTED with it -- its cap is
# fixed at launch and cannot be changed from here. A trainer launched with the default
# 16 GB will still be permitted 16 GB, and the total can reach 16 + SWEEP_GPU_GB. To
# actually enforce the sum, relaunch the trainer with GPU_TOTAL_GB=10.
#
# Note the asymmetry in how the two caps behave: torch's is a ceiling that is only
# reached if training allocates that much, whereas RMM allocates a quarter of its pool
# immediately -- so the sweep's footprint shows up on the card as soon as it starts.
GPU_TOTAL_GB="${GPU_TOTAL_GB:-16}"
TRAINER_GPU_GB="${TRAINER_GPU_GB:-10}"
SWEEP_GPU_GB="${SWEEP_GPU_GB:-$(awk -v t="$GPU_TOTAL_GB" -v r="$TRAINER_GPU_GB" \
    'BEGIN{ v = t - r; if (v < 1) v = 1; printf "%.2f", v }')}"

COSMIC_VERSION="${COSMIC_VERSION:-3.5}"
GENOME_BUILD="${GENOME_BUILD:-GRCh38}"
NO_SIGPROFILER="${NO_SIGPROFILER:-0}"
DRY_RUN="${DRY_RUN:-0}"
NO_RESUME="${NO_RESUME:-0}"
FORCE_CPU="${FORCE_CPU:-0}"
CELL="${CELL:-}"

LOG_DIR="$RUN_ROOT/logs"

# Who else is on the card right now. The sweep's own check (sweep_core.apply_gpu_budget)
# refuses to start if the slice will not fit, but seeing it here -- before the stage even
# launches -- is what makes a bad split obvious rather than a surprise 40 minutes in.
report_gpu_tenancy() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local used total
    read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ',' | head -1)
    [ -z "${used:-}" ] && return 0
    awk -v u="$used" -v t="$total" -v want="$SWEEP_GPU_GB" 'BEGIN{
        free = (t - u) / 1024
        printf "  GPU now     : %.1f GB used / %.1f GB total -> %.1f GB free", u/1024, t/1024, free
        if (want > free - 1) printf "   <-- %.2f GB REQUESTED WILL NOT FIT", want
        printf "\n"
    }'
    echo "  processes   :"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
        | sed 's/^/                  /' || true
}

require_checkpoint() {
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: CHECKPOINT is required for stage $STAGE." >&2
        echo "  e.g. CHECKPOINT=\$HOME/pure-internship/uv_vae/runs/train_multi_*/training/run_*/model.pt" >&2
        exit 1
    fi
    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
        exit 1
    fi
}

stage0_dedup() {
    require_checkpoint
    uvv_log "===== BEGIN stage 0: deduplicate the cohort ====="
    python "$SWEEP_DIR/stage0_dedup.py" \
        --parquet-paths "$PARQUET_GLOB" \
        --checkpoint-path "$CHECKPOINT" \
        --output-dir "$DEDUP_DIR" \
        --row-filter "$ROW_FILTER" \
        --threads "$UVV_THREADS_PER_WORKER" \
        --memory-limit "$DUCKDB_MEMORY" \
        --temp-directory "$TEMP_DIR" \
        ${CHROMOSOMES:+--chromosomes "$CHROMOSOMES"}
    uvv_log "===== END stage 0 ====="
}

stage1_embed() {
    require_checkpoint
    if [ ! -f "$DEDUP_DIR/dedup_manifest.json" ]; then
        echo "ERROR: no dedup manifest at $DEDUP_DIR -- run STAGE=0 first." >&2
        exit 1
    fi
    uvv_log "===== BEGIN stage 1: embed deduplicated loci ====="
    python "$SWEEP_DIR/stage1_embed.py" \
        --dedup-manifest "$DEDUP_DIR/dedup_manifest.json" \
        --checkpoint-path "$CHECKPOINT" \
        --feature-spec-path "$FEATURE_SPEC" \
        --output-dir "$EMBED_DIR" \
        --batch-size "$EMBED_BATCH" \
        --gpu-budget-gb "$SWEEP_GPU_GB" \
        --device cuda
    uvv_log "===== END stage 1 ====="
}

stage2_sweep() {
    if [ ! -f "$EMBED_DIR/embed_summary.json" ]; then
        echo "ERROR: no embedding at $EMBED_DIR -- run STAGE=1 first." >&2
        exit 1
    fi
    # Built as an array so an unset flag contributes nothing at all -- inline
    # ${VAR:+...} expansions of optional flags are easy to get subtly wrong.
    local flags=()
    [ "$NO_SIGPROFILER" = "1" ] && flags+=(--no-sigprofiler)
    [ "$DRY_RUN"        = "1" ] && flags+=(--dry-run)
    [ "$NO_RESUME"      = "1" ] && flags+=(--no-resume)
    [ "$FORCE_CPU"      = "1" ] && flags+=(--force-cpu)

    uvv_log "===== BEGIN stage 2: UMAP x HDBSCAN sweep ====="
    python "$SWEEP_DIR/stage2_sweep.py" \
        --embed-dir "$EMBED_DIR" \
        --output-root "$SWEEP_OUT" \
        --umap-n-neighbors "$UMAP_N_NEIGHBORS" \
        --umap-min-dist "$UMAP_MIN_DIST" \
        --umap-n-components "$UMAP_N_COMPONENTS" \
        --umap-n-epochs "$UMAP_N_EPOCHS" \
        --min-cluster-sizes "$MIN_CLUSTER_SIZES" \
        --min-samples "$MIN_SAMPLES" \
        --fit-rows "$FIT_ROWS" \
        --transform-batch-size "$TRANSFORM_BATCH" \
        --sil-eval-rows "$SIL_EVAL_ROWS" \
        --agreement-rows "$AGREEMENT_ROWS" \
        --seed "$SEED" \
        --threads "$UVV_THREADS_PER_WORKER" \
        --gpu-budget-gb "$SWEEP_GPU_GB" \
        --cosmic-version "$COSMIC_VERSION" \
        --genome-build "$GENOME_BUILD" \
        "${flags[@]}"
    uvv_log "===== END stage 2 ====="
}

stage3_apply() {
    require_checkpoint
    if [ -z "$CELL" ]; then
        echo "ERROR: CELL is required for stage 3 (e.g. CELL=nn30_md0.05_nc2/mcs500_ms25)." >&2
        echo "  The DBCV-best cell is named in $SWEEP_OUT/sweep_summary.json" >&2
        exit 1
    fi
    uvv_log "===== BEGIN stage 3: project all reads through $CELL ====="
    python "$SWEEP_DIR/stage3_apply_full.py" \
        --parquet-paths "$PARQUET_GLOB" \
        --sweep-root "$SWEEP_OUT" \
        --cell "$CELL" \
        --embed-dir "$EMBED_DIR" \
        --checkpoint-path "$CHECKPOINT" \
        --feature-spec-path "$FEATURE_SPEC" \
        --output-dir "$APPLY_OUT/$(echo "$CELL" | tr '/' '_')" \
        --row-filter "$ROW_FILTER" \
        --batch-size "$APPLY_BATCH" \
        --threads "$UVV_THREADS_PER_WORKER" \
        --gpu-budget-gb "$SWEEP_GPU_GB" \
        --cosmic-version "$COSMIC_VERSION" \
        --genome-build "$GENOME_BUILD"
    uvv_log "===== END stage 3 ====="
}

main() {
    uvv_activate_env
    uvv_export_determinism "$SEED"
    # Threads only -- the GPU slice comes from SWEEP_GPU_GB, not from dividing the card by
    # a concurrency count, because the other tenant is a separate process this script did
    # not launch and whose cap it cannot change.
    uvv_plan_resources 1
    export UV_VAE_GPU_MEM_GB="$SWEEP_GPU_GB"
    # Deliberately NOT set here. Each stage passes its own share to gpu_budget.apply
    # (STAGE_RMM_SHARE in sweep_core): embed is torch-only, sweep is cuML-only, apply
    # splits. One global value would be wrong for two of the three.
    unset UV_VAE_RMM_SHARE
    export TQDM_DISABLE=1

    mkdir -p "$RUN_ROOT" "$LOG_DIR"

    local n_files
    n_files=$(compgen -G "$PARQUET_GLOB" | wc -l || echo 0)

    uvv_rule
    echo "UMAP x HDBSCAN sweep -- stage $STAGE"
    echo "  run root    : $RUN_ROOT"
    echo "  parquets    : $PARQUET_GLOB  ($n_files files)"
    echo "  checkpoint  : ${CHECKPOINT:-<unset>}"
    echo "  row filter  : $ROW_FILTER"
    case "$STAGE" in
      2)
        echo "  UMAP grid   : nn=$UMAP_N_NEIGHBORS  min_dist=$UMAP_MIN_DIST  n_comp=$UMAP_N_COMPONENTS"
        echo "  HDBSCAN grid: mcs=$MIN_CLUSTER_SIZES  min_samples=$MIN_SAMPLES"
        echo "  fit rows    : $FIT_ROWS   transform batch: $TRANSFORM_BATCH"
        echo "  SigProfiler : $([ "$NO_SIGPROFILER" = "1" ] && echo disabled || echo "every cell (COSMIC v$COSMIC_VERSION, $GENOME_BUILD)")"
        ;;
      3)
        echo "  cell        : $CELL"
        echo "  batch       : $APPLY_BATCH"
        ;;
    esac
    echo "  GPU split   : ${GPU_TOTAL_GB} GB total = ${TRAINER_GPU_GB} trainer + ${SWEEP_GPU_GB} this sweep"
    echo "  threads     : $UVV_THREADS_PER_WORKER"
    echo "  seed        : $SEED"
    report_gpu_tenancy
    uvv_rule

    if [ "$n_files" -eq 0 ] && [ "$STAGE" != "2" ]; then
        echo "ERROR: no files matched PARQUET_GLOB=$PARQUET_GLOB" >&2
        exit 1
    fi

    case "$STAGE" in
        0) stage0_dedup ;;
        1) stage1_embed ;;
        2) stage2_sweep ;;
        3) stage3_apply ;;
        all)
            stage0_dedup
            stage1_embed
            stage2_sweep
            echo "Stages 0-2 done. Pick a cell from $SWEEP_OUT/sweep_summary.json, then:"
            echo "    STAGE=3 CELL=<cell> bash $SWEEP_DIR/run_sweep.sh"
            ;;
        *) echo "ERROR: unknown STAGE=$STAGE (expected 0, 1, 2, 3 or all)" >&2; exit 1 ;;
    esac

    uvv_rule
    printf '  %-24s %8ss   %s\n' "TOTAL" "$SECONDS" "$(uvv_fmt_seconds "$SECONDS")"
    uvv_rule
}

if [ "$NO_TMUX" = "1" ] || [ -n "${TMUX:-}" ]; then
    uvv_run_main "$LOG_DIR" main
else
    uvv_launch_tmux "$SESSION" "$LOG_DIR" \
        "STAGE='$STAGE' RUN_ID='$RUN_ID' CHECKPOINT='$CHECKPOINT' CELL='$CELL' \
         PARQUET_GLOB='$PARQUET_GLOB' FIT_ROWS='$FIT_ROWS' DRY_RUN='$DRY_RUN' \
         NO_RESUME='$NO_RESUME' NO_SIGPROFILER='$NO_SIGPROFILER' FORCE_CPU='$FORCE_CPU' \
         NO_TMUX=1 bash '${BASH_SOURCE[0]}'"
fi
