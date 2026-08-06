#!/bin/bash
# End-to-end test: VAE with a 2-D latent -> HDBSCAN directly -> SigProfiler. No UMAP.
#
# The normal pipeline is  VAE(16-D) -> UMAP(2-D) -> HDBSCAN.  This asks whether the VAE
# can be made to produce the 2-D clustering space itself, so UMAP can be deleted from the
# pipeline rather than tuned.
#
# Four stages. Each writes something the next reads, so they run and resume independently:
#
#   train  train the cohort VAE with LATENT_DIM=2   (the only thing that differs from the
#          baseline trainer -- same data, same filter, same interleaved reader)
#   0      dedup   5.08B reads -> one row per (CHROM,POS,REF,ALT)
#   1      embed   every deduplicated locus -> latent.npy (N x 2) + context.parquet
#   2      cluster HDBSCAN grid on the raw latent, metrics + SigProfiler per cell
#
# STAGE 0 IS REUSED, NOT RE-RUN. Deduplication depends on the checkpoint only through its
# feature_report -- which columns to keep -- and a 2-D-latent VAE trained from the same
# ml_features.json has the identical feature list. So this points DEDUP_DIR at the 16-D
# sweep's stage-0 output when it exists, which saves the single most expensive stage AND
# guarantees the two runs cluster exactly the same population (a prerequisite for the ARI
# comparison in stage 2).
#
#     STAGE=train bash latent2d_test/run_latent2d.sh
#     STAGE=1 CHECKPOINT=<.../model.pt> bash latent2d_test/run_latent2d.sh
#     STAGE=2 bash latent2d_test/run_latent2d.sh
#     STAGE=2 DRY_RUN=1 bash latent2d_test/run_latent2d.sh     # print the grid, fit nothing
#     STAGE=all bash latent2d_test/run_latent2d.sh             # train -> 0 -> 1 -> 2
#
# Windows edit note:  sed -i 's/\r$//' latent2d_test/run_latent2d.sh

set -euo pipefail

# Repo layout differs by cluster: miletus has a git clone at ~/pure-internship, tosun
# deploys the folders as siblings in $HOME. Probe rather than hardcode.
if [ -z "${UV_VAE_DIR:-}" ] && [ -d "$HOME/pure-internship/uv_vae" ]; then
    UV_VAE_DIR="$HOME/pure-internship/uv_vae"
fi
if [ -z "${SWEEP_DIR:-}" ] && [ -d "$HOME/pure-internship/umap_hdbscan_sweep" ]; then
    SWEEP_DIR="$HOME/pure-internship/umap_hdbscan_sweep"
fi
if [ -z "${LATENT2D_DIR:-}" ] && [ -d "$HOME/pure-internship/latent2d_test" ]; then
    LATENT2D_DIR="$HOME/pure-internship/latent2d_test"
fi
if [ -z "${EARLY_STOPPING_DIR:-}" ] && [ -d "$HOME/pure-internship/Early_Stopping_Tests" ]; then
    EARLY_STOPPING_DIR="$HOME/pure-internship/Early_Stopping_Tests"
fi
UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
SWEEP_DIR="${SWEEP_DIR:-$HOME/umap_hdbscan_sweep}"
LATENT2D_DIR="${LATENT2D_DIR:-$HOME/latent2d_test}"
EARLY_STOPPING_DIR="${EARLY_STOPPING_DIR:-$HOME/Early_Stopping_Tests}"

# shellcheck source=/dev/null
source "$UV_VAE_DIR/scripts/tmux_lib.sh"

# ── Configuration ───────────────────────────────────────────────────────────
STAGE="${STAGE:-2}"
RUN_ID="${RUN_ID:-cohort}"
RUN_ROOT="${RUN_ROOT:-$UV_VAE_DIR/runs/latent2d_$RUN_ID}"
ROW_FILTER="${ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
SEED="${SEED:-42}"
SESSION="${SESSION:-latent2d}"
NO_TMUX="${NO_TMUX:-0}"

# Same cohort-location probe the trainer and the 16-D sweep use.
if [ -z "${PARQUET_GLOB:-}" ]; then
    for _dir in /data/lab/ppmseq_parquets /cta/users/patrickgao765/parquet_files; do
        if [ -d "$_dir" ]; then
            PARQUET_GLOB="$_dir/*.featuremap.parquet"
            break
        fi
    done
fi
PARQUET_GLOB="${PARQUET_GLOB:-/data/lab/ppmseq_parquets/*.featuremap.parquet}"

FEATURE_SPEC="${FEATURE_SPEC:-$UV_VAE_DIR/ml_features.json}"
CHECKPOINT="${CHECKPOINT:-}"

# ── The 16-D + UMAP baseline this is measured against ───────────────────────
# Its stage-0 dedup is reused (see the header) and its stage-2 labels are the reference
# the ARI is computed against.
BASELINE_ROOT="${BASELINE_ROOT:-$UV_VAE_DIR/runs/umap_sweep_cohort}"
BASELINE_CELL="${BASELINE_CELL:-nn15_md0.0_nc2/mcs1000_ms5}"
AGREEMENT_ROWS="${AGREEMENT_ROWS:-200000}"
if [ -z "${AGREEMENT_REFERENCE:-}" ] \
   && [ -f "$BASELINE_ROOT/stage2_sweep/$BASELINE_CELL/agreement_labels.npy" ]; then
    AGREEMENT_REFERENCE="$BASELINE_ROOT/stage2_sweep/$BASELINE_CELL/agreement_labels.npy"
fi
AGREEMENT_REFERENCE="${AGREEMENT_REFERENCE:-}"

# Reuse the baseline's dedup when it is there; otherwise this run makes its own.
if [ -z "${DEDUP_DIR:-}" ] && [ -f "$BASELINE_ROOT/stage0_dedup/dedup_manifest.json" ]; then
    DEDUP_DIR="$BASELINE_ROOT/stage0_dedup"
    DEDUP_REUSED=1
fi
DEDUP_DIR="${DEDUP_DIR:-$RUN_ROOT/stage0_dedup}"
DEDUP_REUSED="${DEDUP_REUSED:-0}"

EMBED_DIR="${EMBED_DIR:-$RUN_ROOT/stage1_embed}"
CLUSTER_OUT="${CLUSTER_OUT:-$RUN_ROOT/latent2d_cluster}"
TRAIN_ROOT="${TRAIN_ROOT:-$RUN_ROOT/train}"

# Stage 0 spills heavily: the per-chromosome window sorts far more than fits in RAM.
DUCKDB_MEMORY="${DUCKDB_MEMORY:-200GB}"
TEMP_DIR="${TEMP_DIR:-$RUN_ROOT/duckdb_tmp}"
CHROMOSOMES="${CHROMOSOMES:-}"

# ── Training: everything is the baseline's value except LATENT_DIM and KL_WEIGHT ──
# LATENT_DIM=2 is the experiment and is not configurable here.
#
# KL_WEIGHT is lowered from the baseline 0.05 for a reason specific to this test. beta
# pulls the aggregate posterior toward a single N(0, I) blob; that is harmless when UMAP
# sits downstream (UMAP rebuilds structure from the kNN graph and rescales everything),
# but here the latent IS the density HDBSCAN reads. A latent squeezed into one Gaussian
# has no density gaps, and HDBSCAN's honest answer to that is "one cluster, everything
# else noise". Lower beta buys separation at the cost of a less regular latent.
#
# 0.01 is a starting point, not a settled value -- sweep it (see the README).
KL_WEIGHT="${KL_WEIGHT:-0.01}"
HIDDEN_DIMS="${HIDDEN_DIMS:-256,128}"
BATCH_SIZE="${BATCH_SIZE:-32768}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
EPOCH_CEILING="${EPOCH_CEILING:-40}"
EPOCH_SHARDS="${EPOCH_SHARDS:-20}"
PATIENCE="${PATIENCE:-8}"
INPUT_DROPOUT="${INPUT_DROPOUT:-0.1}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.4}"

# ── Clustering grid ─────────────────────────────────────────────────────────
MIN_CLUSTER_SIZES="${MIN_CLUSTER_SIZES:-100,250,500,1000,2500}"
MIN_SAMPLES="${MIN_SAMPLES:-5,25,50}"
# 'leaf' is worth having in this test specifically: a raw VAE latent tends toward a few
# broad density regions rather than the many tight islands UMAP manufactures, and eom
# answers that with a handful of huge clusters. leaf cuts the condensed tree at its
# finest level instead.
SELECTION_METHODS="${SELECTION_METHODS:-eom,leaf}"
# In LATENT units, not UMAP units. 0 = off. Read the per-dimension extent stage 2 prints
# before setting this to anything else.
SELECTION_EPSILONS="${SELECTION_EPSILONS:-0}"
SCALE="${SCALE:-none}"

# Matches the baseline's fit size so the two clusterings are comparable. The 2-D fit is
# far cheaper than the 16-D one, so 'all' is more realistic here than it was there -- but
# changing it changes what is being compared.
FIT_ROWS="${FIT_ROWS:-5000000}"
PREDICT_BATCH="${PREDICT_BATCH:-5000000}"
EMBED_BATCH="${EMBED_BATCH:-262144}"
SIL_EVAL_ROWS="${SIL_EVAL_ROWS:-50000}"
PLOT_ROWS="${PLOT_ROWS:-200000}"

# ── GPU budget when sharing the card ────────────────────────────────────────
# Per-process caps, so two capped processes sum. See umap_hdbscan_sweep/README.md.
GPU_TOTAL_GB="${GPU_TOTAL_GB:-16}"
TRAINER_GPU_GB="${TRAINER_GPU_GB:-10}"
SWEEP_GPU_GB="${SWEEP_GPU_GB:-$(awk -v t="$GPU_TOTAL_GB" -v r="$TRAINER_GPU_GB" \
    'BEGIN{ v = t - r; if (v < 1) v = 1; printf "%.2f", v }')}"

COSMIC_VERSION="${COSMIC_VERSION:-3.5}"
GENOME_BUILD="${GENOME_BUILD:-GRCh38}"
NO_SIGPROFILER="${NO_SIGPROFILER:-0}"
NO_PLOT="${NO_PLOT:-0}"
DRY_RUN="${DRY_RUN:-0}"
NO_RESUME="${NO_RESUME:-0}"
FORCE_CPU="${FORCE_CPU:-0}"

LOG_DIR="$RUN_ROOT/logs"

resolve_trained_checkpoint() {
    find "$TRAIN_ROOT/training" -name model.pt 2>/dev/null | sort | tail -1
}

require_checkpoint() {
    if [ -z "$CHECKPOINT" ]; then
        CHECKPOINT="$(resolve_trained_checkpoint)"
    fi
    if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: no 2-D-latent checkpoint. Run STAGE=train first, or pass CHECKPOINT=..." >&2
        exit 1
    fi
    # A 16-D checkpoint here would run to completion and produce a totally different
    # experiment, so refuse rather than warn.
    python - "$CHECKPOINT" <<'PY' || exit 1
import sys, torch
# All four trainers write "model_config" = asdict(model.config) -- part of the
# run-artifact contract in CLAUDE.md, so this key is safe to read directly.
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
latent = (payload.get("model_config") or {}).get("latent_dim")
if latent is None:
    print(f"WARNING: could not read latent_dim from {sys.argv[1]}; proceeding", file=sys.stderr)
elif int(latent) != 2:
    print(f"ERROR: checkpoint has latent_dim={latent}, not 2: {sys.argv[1]}", file=sys.stderr)
    print("  This test drops UMAP, so the latent IS the clustering space.", file=sys.stderr)
    print("  For the no-UMAP control on a 16-D latent, call latent2d_cluster.py directly", file=sys.stderr)
    print("  with --allow-any-latent-dim.", file=sys.stderr)
    sys.exit(1)
PY
}

stage_train() {
    uvv_log "===== BEGIN train: cohort VAE with latent_dim=2 ====="
    # FOREGROUND=1 keeps the trainer in THIS tmux session instead of detaching its own.
    # DRY_RUN is forced to 0: in tmux_train_multi.sh it means "run in the foreground",
    # not "plan only", and this script's DRY_RUN means the opposite.
    FOREGROUND=1 DRY_RUN=0 \
    UV_VAE_DIR="$UV_VAE_DIR" EARLY_STOPPING_DIR="$EARLY_STOPPING_DIR" \
    RUN_ROOT="$TRAIN_ROOT" RUN_ID="$RUN_ID" \
    PARQUET_GLOB="$PARQUET_GLOB" ROW_FILTER="$ROW_FILTER" SEED="$SEED" \
    LATENT_DIM=2 KL_WEIGHT="$KL_WEIGHT" HIDDEN_DIMS="$HIDDEN_DIMS" \
    BATCH_SIZE="$BATCH_SIZE" LEARNING_RATE="$LEARNING_RATE" \
    EPOCH_CEILING="$EPOCH_CEILING" EPOCH_SHARDS="$EPOCH_SHARDS" PATIENCE="$PATIENCE" \
    INPUT_DROPOUT="$INPUT_DROPOUT" HIDDEN_DROPOUT="$HIDDEN_DROPOUT" \
    GPU_TOTAL_GB="$GPU_TOTAL_GB" \
        bash "$EARLY_STOPPING_DIR/scripts/tmux_train_multi.sh"
    CHECKPOINT="$(resolve_trained_checkpoint)"
    uvv_log "===== END train -> ${CHECKPOINT:-<none found>} ====="
    # With only 2 latent dimensions the early-stopping rule's active-unit half saturates
    # at 2/2 within a few epochs and stops moving, so stopping is driven by val ELBO
    # alone. Not a bug, but it means "final AU count: 2" is not evidence of convergence.
    echo "NOTE: active-unit count saturates at 2/2 almost immediately at this latent size --"
    echo "      read stop_reason and the val ELBO curve, not the AU count."
}

stage0_dedup() {
    if [ -f "$DEDUP_DIR/dedup_manifest.json" ]; then
        uvv_log "stage 0: reusing existing dedup at $DEDUP_DIR (nothing to do)"
        return 0
    fi
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
    uvv_log "===== BEGIN stage 1: embed deduplicated loci through the 2-D VAE ====="
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

stage2_cluster() {
    if [ ! -f "$EMBED_DIR/embed_summary.json" ]; then
        echo "ERROR: no embedding at $EMBED_DIR -- run STAGE=1 first." >&2
        exit 1
    fi
    local flags=()
    [ "$NO_SIGPROFILER" = "1" ] && flags+=(--no-sigprofiler)
    [ "$NO_PLOT"        = "1" ] && flags+=(--no-plot)
    [ "$DRY_RUN"        = "1" ] && flags+=(--dry-run)
    [ "$NO_RESUME"      = "1" ] && flags+=(--no-resume)
    [ "$FORCE_CPU"      = "1" ] && flags+=(--force-cpu)
    [ -n "$AGREEMENT_REFERENCE" ] && flags+=(--agreement-reference "$AGREEMENT_REFERENCE"
                                             --agreement-reference-label "$BASELINE_CELL")

    uvv_log "===== BEGIN stage 2: HDBSCAN on the 2-D latent (no UMAP) ====="
    python "$LATENT2D_DIR/latent2d_cluster.py" \
        --embed-dir "$EMBED_DIR" \
        --output-root "$CLUSTER_OUT" \
        --min-cluster-sizes "$MIN_CLUSTER_SIZES" \
        --min-samples "$MIN_SAMPLES" \
        --cluster-selection-methods "$SELECTION_METHODS" \
        --cluster-selection-epsilons "$SELECTION_EPSILONS" \
        --scale "$SCALE" \
        --fit-rows "$FIT_ROWS" \
        --predict-batch-size "$PREDICT_BATCH" \
        --sil-eval-rows "$SIL_EVAL_ROWS" \
        --agreement-rows "$AGREEMENT_ROWS" \
        --plot-rows "$PLOT_ROWS" \
        --seed "$SEED" \
        --threads "$UVV_THREADS_PER_WORKER" \
        --gpu-budget-gb "$SWEEP_GPU_GB" \
        --cosmic-version "$COSMIC_VERSION" \
        --genome-build "$GENOME_BUILD" \
        "${flags[@]}"
    uvv_log "===== END stage 2 ====="
}

main() {
    uvv_activate_env
    uvv_export_determinism "$SEED"
    uvv_plan_resources 1
    export UV_VAE_GPU_MEM_GB="$SWEEP_GPU_GB"
    # Each stage passes its own RMM share to gpu_budget.apply (STAGE_RMM_SHARE in
    # sweep_core): embed is torch-only, cluster is cuML-only. One global value is wrong
    # for one of the two.
    unset UV_VAE_RMM_SHARE
    export TQDM_DISABLE=1

    mkdir -p "$RUN_ROOT" "$LOG_DIR"

    local n_files
    n_files=$(compgen -G "$PARQUET_GLOB" | wc -l || echo 0)

    uvv_rule
    echo "2-D latent, no UMAP -- stage $STAGE"
    echo "  run root    : $RUN_ROOT"
    echo "  parquets    : $PARQUET_GLOB  ($n_files files)"
    echo "  checkpoint  : ${CHECKPOINT:-<from STAGE=train>}"
    echo "  dedup       : $DEDUP_DIR $([ "$DEDUP_REUSED" = "1" ] && echo '(reused from the 16-D baseline)')"
    echo "  row filter  : $ROW_FILTER"
    case "$STAGE" in
      train)
        echo "  latent dim  : 2 (fixed -- this is the experiment)"
        echo "  beta (KL)   : $KL_WEIGHT   hidden: $HIDDEN_DIMS   batch: $BATCH_SIZE"
        echo "  epochs      : $EPOCH_CEILING (patience=$PATIENCE, shards=$EPOCH_SHARDS)"
        ;;
      2)
        echo "  HDBSCAN grid: mcs=$MIN_CLUSTER_SIZES  min_samples=$MIN_SAMPLES"
        echo "                selection=$SELECTION_METHODS  epsilon=$SELECTION_EPSILONS  scale=$SCALE"
        echo "  fit rows    : $FIT_ROWS   predict batch: $PREDICT_BATCH"
        echo "  baseline ARI: ${AGREEMENT_REFERENCE:-<none -- no 16-D reference found>}"
        echo "  SigProfiler : $([ "$NO_SIGPROFILER" = "1" ] && echo disabled || echo "every cell (COSMIC v$COSMIC_VERSION, $GENOME_BUILD)")"
        ;;
    esac
    echo "  GPU split   : ${GPU_TOTAL_GB} GB total = ${TRAINER_GPU_GB} trainer + ${SWEEP_GPU_GB} this job"
    echo "  threads     : $UVV_THREADS_PER_WORKER"
    echo "  seed        : $SEED"
    uvv_rule

    if [ "$n_files" -eq 0 ] && [ "$STAGE" != "2" ]; then
        echo "ERROR: no files matched PARQUET_GLOB=$PARQUET_GLOB" >&2
        exit 1
    fi

    case "$STAGE" in
        train) stage_train ;;
        0) stage0_dedup ;;
        1) stage1_embed ;;
        2) stage2_cluster ;;
        all)
            stage_train
            stage0_dedup
            stage1_embed
            stage2_cluster
            ;;
        *) echo "ERROR: unknown STAGE=$STAGE (expected train, 0, 1, 2 or all)" >&2; exit 1 ;;
    esac

    uvv_rule
    printf '  %-24s %8ss   %s\n' "TOTAL" "$SECONDS" "$(uvv_fmt_seconds "$SECONDS")"
    uvv_rule
}

if [ "$NO_TMUX" = "1" ] || [ -n "${TMUX:-}" ]; then
    uvv_run_main "$LOG_DIR" main
else
    uvv_strip_crlf "$UV_VAE_DIR/scripts/tmux_lib.sh" "$LATENT2D_DIR/run_latent2d.sh"
    mkdir -p "$LOG_DIR"
    uvv_launch_tmux "$SESSION" "$LOG_DIR" \
        "STAGE='$STAGE' RUN_ID='$RUN_ID' CHECKPOINT='$CHECKPOINT' \
         PARQUET_GLOB='$PARQUET_GLOB' FIT_ROWS='$FIT_ROWS' DRY_RUN='$DRY_RUN' \
         KL_WEIGHT='$KL_WEIGHT' NO_RESUME='$NO_RESUME' NO_SIGPROFILER='$NO_SIGPROFILER' \
         NO_PLOT='$NO_PLOT' FORCE_CPU='$FORCE_CPU' AGREEMENT_REFERENCE='$AGREEMENT_REFERENCE' \
         NO_TMUX=1 bash '${BASH_SOURCE[0]}'"
fi
