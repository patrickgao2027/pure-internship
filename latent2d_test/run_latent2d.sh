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

# The 16-D latent both pipelines are scored against. Row-aligned with this run only
# because both stage 1s consumed the same stage-0 dedup manifest -- which is exactly
# what reusing DEDUP_DIR below guarantees.
if [ -z "${REFERENCE_LATENT:-}" ] && [ -f "$BASELINE_ROOT/stage1_embed/latent.npy" ]; then
    REFERENCE_LATENT="$BASELINE_ROOT/stage1_embed/latent.npy"
fi
REFERENCE_LATENT="${REFERENCE_LATENT:-}"
if [ -z "${REFERENCE_COORDINATES:-}" ] \
   && [ -f "$BASELINE_ROOT/stage2_sweep/$BASELINE_CELL/analysis.parquet" ]; then
    REFERENCE_COORDINATES="$BASELINE_ROOT/stage2_sweep/$BASELINE_CELL/analysis.parquet"
fi
REFERENCE_COORDINATES="${REFERENCE_COORDINATES:-}"

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

# 256 -> 128 -> 64 -> 16 -> 2. The baseline's 256,128 ends in a 128 -> 2 step, a 64x
# contraction in one layer; this tapers instead, so the largest single step is 16 -> 2.
# The decoder is built symmetrically, so this deepens both halves.
HIDDEN_DIMS="${HIDDEN_DIMS:-256,128,64,16}"

DECODE_WORKERS="${DECODE_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32768}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
EPOCH_CEILING="${EPOCH_CEILING:-40}"
EPOCH_SHARDS="${EPOCH_SHARDS:-20}"
PATIENCE="${PATIENCE:-8}"
INPUT_DROPOUT="${INPUT_DROPOUT:-0.1}"

# Lowered from the baseline 0.4 *because* of the taper above, not as a free choice.
# TabularVAEWithDropout.encode applies hidden_drop after EVERY ReLU, so with four hidden
# layers it now also hits the 16-unit layer feeding mu. At p=0.4 that keeps ~9.6 of 16
# units, with enough variance that some batches reach the 2-D bottleneck through very few
# survivors -- a severe bottleneck stacked on a severe bottleneck. p=0.2 keeps ~12.8 while
# still regularising the wide layers, where most of the parameters are.
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.2}"

# ── Clustering: ONE configuration, not a grid ───────────────────────────────
# Each of these still accepts a comma-separated list if you want a grid back; they are
# single values because 30 cells is not worth the wall clock before there is one result
# to react to.
#
# mcs=1000 is the baseline's best-silhouette cut (0.578 at 1,150 clusters, 29.6% noise),
#   so it is the most directly comparable single choice, and clusters that size hold
#   enough mutations for a stable 96-context SigProfiler catalog.
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-1000}"
# ms=25 is the project default everywhere else (clustering_core, run_variant_cluster_
#   pipeline). The baseline only used 5 because 25 and 50 OOMed in 16-D UMAP space -- a
#   memory accident, not a finding. A raw latent is smoother and noisier than UMAP's
#   manufactured islands, so the more conservative core-distance smoothing is right here.
MIN_SAMPLES="${MIN_SAMPLES:-25}"
# eom, not leaf. If the latent really is one smooth blob, eom answers with two or three
#   enormous clusters -- and that IS the result, the honest answer to the research
#   question. leaf would cut the condensed tree finely enough to manufacture structure
#   that the density does not support. Every prior result in this project used eom too.
SELECTION_METHOD="${SELECTION_METHOD:-eom}"
# In LATENT units, not UMAP units, and the latent's extent is not knowable until the VAE
#   has trained. Off until you have read the extent stage 2 prints.
SELECTION_EPSILON="${SELECTION_EPSILON:-0}"
SCALE="${SCALE:-none}"

# 'all' -- fit on every deduplicated locus, not a 5M subsample.
#
# 5M was a constraint of the 16-D + UMAP path (cuML UMAP's kNN graph, plus a 65-minute
# approximate_predict per cell over 157M rows, times 480 cells). None of that applies
# here: there is no UMAP graph, the space is 2-D, and there is ONE cell.
#
# It also costs real signal. Fitting on 5M of ~157.5M is a 3.2% sample, so a genuine
# cluster of 30,000 loci contributes ~950 rows to the fit -- under min_cluster_size, so
# it vanishes. Anything rarer is invisible. Rare mutational processes are the point.
#
# If it OOMs (the estimate stage 2 prints will say so first), walk it down:
#   FIT_ROWS=50000000  ->  25000000  ->  10000000
# and prefer running when the trainer is idle so the whole card is available:
#   GPU_TOTAL_GB=40 TRAINER_GPU_GB=0 STAGE=2 bash latent2d_test/run_latent2d.sh
FIT_ROWS="${FIT_ROWS:-all}"
PREDICT_BATCH="${PREDICT_BATCH:-5000000}"
EMBED_BATCH="${EMBED_BATCH:-262144}"
# Metric tiers, matching umap_hdbscan_sweep/size_convergence.py exactly so the numbers
# land in the same table as that sweep's. knn = trustworthiness/continuity/RNX,
# pair = silhouette + distance correlations (both need pairwise distances).
KNN_ROWS="${KNN_ROWS:-250000}"
PAIR_ROWS="${PAIR_ROWS:-20000}"
DISTANCE_PAIRS="${DISTANCE_PAIRS:-20000000}"
METRIC_K="${METRIC_K:-15}"
# 400 is measured, not guessed: at k_max=100 real umap-learn output clamped 40% of
# ranks and reported trustworthiness 0.032 too high. Watch clamped_fraction.
METRIC_K_MAX="${METRIC_K_MAX:-400}"
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
    GPU_TOTAL_GB="$GPU_TOTAL_GB" DECODE_WORKERS="$DECODE_WORKERS" \
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
    [ -n "$REFERENCE_LATENT" ]    && flags+=(--reference-latent "$REFERENCE_LATENT")
    [ -n "$REFERENCE_COORDINATES" ] && flags+=(--reference-coordinates "$REFERENCE_COORDINATES")

    uvv_log "===== BEGIN stage 2: HDBSCAN on the 2-D latent (no UMAP) ====="
    python "$LATENT2D_DIR/latent2d_cluster.py" \
        --embed-dir "$EMBED_DIR" \
        --output-root "$CLUSTER_OUT" \
        --min-cluster-sizes "$MIN_CLUSTER_SIZE" \
        --min-samples "$MIN_SAMPLES" \
        --cluster-selection-methods "$SELECTION_METHOD" \
        --cluster-selection-epsilons "$SELECTION_EPSILON" \
        --scale "$SCALE" \
        --fit-rows "$FIT_ROWS" \
        --predict-batch-size "$PREDICT_BATCH" \
        --knn-rows "$KNN_ROWS" \
        --pair-rows "$PAIR_ROWS" \
        --distance-pairs "$DISTANCE_PAIRS" \
        --metric-k "$METRIC_K" \
        --metric-k-max "$METRIC_K_MAX" \
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
        echo "  HDBSCAN     : mcs=$MIN_CLUSTER_SIZE  min_samples=$MIN_SAMPLES"
        echo "                selection=$SELECTION_METHOD  epsilon=$SELECTION_EPSILON  scale=$SCALE"
        echo "  fit rows    : $FIT_ROWS   predict batch: $PREDICT_BATCH"
        echo "  metric tiers: full / knn $KNN_ROWS / pair $PAIR_ROWS   k=$METRIC_K k_max=$METRIC_K_MAX"
        echo "  baseline ARI: ${AGREEMENT_REFERENCE:-<none -- no 16-D reference found>}"
        echo "  16-D ref    : ${REFERENCE_LATENT:-<none -- embedding-quality panel disabled>}"
        echo "  SigProfiler : $([ "$NO_SIGPROFILER" = "1" ] && echo disabled || echo "every cell (COSMIC v$COSMIC_VERSION, $GENOME_BUILD)")"
        ;;
    esac
    echo "  GPU split   : ${GPU_TOTAL_GB} GB total = ${TRAINER_GPU_GB} trainer + ${SWEEP_GPU_GB} this job"
    echo "  decode workers: $DECODE_WORKERS"
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

# ── Re-exec into tmux ───────────────────────────────────────────────────────
# Everything the "Configuration" block above reads from the environment has to be handed
# to the inner invocation, because that invocation is a FRESH shell that re-runs this
# whole file -- anything missing here silently falls back to its default. This used to be
# a hand-written list of ~25 names, and the ones it omitted were not harmless:
# CLUSTER_OUT sent output to the default directory, and SEED changed the fit subsample and
# the metric tiers, i.e. a silent reproducibility break. So the list is generated from the
# variables the configuration block actually defines; keep them in sync when adding one.
#
# SESSION and NO_TMUX are deliberately absent: the first is consumed by the launcher out
# here, and the second is set to 1 below to stop the inner shell re-exec'ing again.
UVV_PASSTHROUGH_VARS="
    UV_VAE_DIR SWEEP_DIR LATENT2D_DIR EARLY_STOPPING_DIR
    STAGE RUN_ID RUN_ROOT ROW_FILTER SEED PARQUET_GLOB FEATURE_SPEC CHECKPOINT
    BASELINE_ROOT BASELINE_CELL AGREEMENT_ROWS AGREEMENT_REFERENCE
    REFERENCE_LATENT REFERENCE_COORDINATES
    DEDUP_DIR DEDUP_REUSED EMBED_DIR CLUSTER_OUT TRAIN_ROOT
    DUCKDB_MEMORY TEMP_DIR CHROMOSOMES
    KL_WEIGHT HIDDEN_DIMS HIDDEN_DROPOUT DECODE_WORKERS BATCH_SIZE LEARNING_RATE
    EPOCH_CEILING EPOCH_SHARDS PATIENCE INPUT_DROPOUT
    MIN_CLUSTER_SIZE MIN_SAMPLES SELECTION_METHOD SELECTION_EPSILON SCALE
    FIT_ROWS PREDICT_BATCH EMBED_BATCH
    KNN_ROWS PAIR_ROWS DISTANCE_PAIRS METRIC_K METRIC_K_MAX PLOT_ROWS
    GPU_TOTAL_GB TRAINER_GPU_GB SWEEP_GPU_GB
    COSMIC_VERSION GENOME_BUILD NO_SIGPROFILER NO_PLOT DRY_RUN NO_RESUME FORCE_CPU
"

# Single-quote a value so it survives the extra shell parse tmux puts it through.
# ROW_FILTER is the reason this cannot be a plain "NAME='$VALUE'": its default
# (st = 'MIXED' AND ...) contains single quotes, which would close the quoting early and
# corrupt the rest of the command line.
#
# The pattern and replacement are held in variables rather than written inline: spelling
# the '\'' idiom directly inside the expansion makes bash process the backslashes twice
# and silently yields '\\ instead, which fails only on values that contain a quote.
uvv_shell_quote() {
    local single="'" escaped="'\\''"
    local quoted=${1//$single/$escaped}
    printf "'%s'" "$quoted"
}

uvv_env_prefix() {
    local name value prefix=""
    for name in $UVV_PASSTHROUGH_VARS; do
        value="${!name-}"
        prefix="$prefix $name=$(uvv_shell_quote "$value")"
    done
    printf '%s' "$prefix"
}

if [ "$NO_TMUX" = "1" ] || [ -n "${TMUX:-}" ]; then
    uvv_run_main "$LOG_DIR" main
else
    uvv_strip_crlf "$UV_VAE_DIR/scripts/tmux_lib.sh" "$LATENT2D_DIR/run_latent2d.sh"
    mkdir -p "$LOG_DIR"
    uvv_launch_tmux "$SESSION" "$LOG_DIR" \
        "$(uvv_env_prefix) NO_TMUX=1 bash $(uvv_shell_quote "${BASH_SOURCE[0]}")"
fi
