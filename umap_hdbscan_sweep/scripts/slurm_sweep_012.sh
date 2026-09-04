#!/bin/bash
#SBATCH --job-name=umap_sweep_012
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Runs stages 0, 1, and 2 of the UMAP/HDBSCAN sweep on tosun (CPU path).
# Stage 3 (project all 5.08B reads through the best cell) is excluded --
# run it separately once you have picked a cell from sweep_summary.json.
#
# miletus note: miletus has no SLURM. Use run_sweep.sh (tmux runner) there.
# This script forces the CPU path (FORCE_CPU=1) for tosun, where cuML is absent.
#
# Usage:
#   # edit CHECKPOINT below, then:
#   sbatch umap_hdbscan_sweep/slurm_sweep_012.sh
#
#   # dry-run (prints the 480-cell grid, fits nothing):
#   DRY_RUN=1 sbatch umap_hdbscan_sweep/slurm_sweep_012.sh
#
#   # resume an interrupted run (cells with metrics.json are skipped automatically):
#   sbatch umap_hdbscan_sweep/slurm_sweep_012.sh
#
# Windows edit note: sed -i 's/\r$//' umap_hdbscan_sweep/slurm_sweep_012.sh

set -euo pipefail
export TQDM_DISABLE=1

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT="$HOME/pure-internship"
UV_VAE_DIR="$REPO_ROOT/uv_vae"
SWEEP_DIR="$REPO_ROOT/umap_hdbscan_sweep"

# Set this to your chosen model checkpoint before submitting.
CHECKPOINT="${CHECKPOINT:-$UV_VAE_DIR/runs/train_multi_20260801T054132Z/training/run_20260801T054132Z/model.pt}"

FEATURE_SPEC="$UV_VAE_DIR/ml_features.json"
PARQUET_GLOB="${PARQUET_GLOB:-/cta/users/patrickgao765/parquet_files/*.featuremap.parquet}"

RUN_ID="${RUN_ID:-cohort_slurm}"
RUN_ROOT="$UV_VAE_DIR/runs/umap_sweep_$RUN_ID"
DEDUP_DIR="$RUN_ROOT/stage0_dedup"
EMBED_DIR="$RUN_ROOT/stage1_embed"
SWEEP_OUT="$RUN_ROOT/stage2_sweep"

LOG_DIR="$RUN_ROOT/logs"
TEMP_DIR="$RUN_ROOT/duckdb_tmp"

SEED="${SEED:-42}"
ROW_FILTER="${ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
DUCKDB_MEMORY="${DUCKDB_MEMORY:-200GB}"
CHROMOSOMES="${CHROMOSOMES:-}"

# ── Sweep grid (480 cells: 4 x 4 x 2 UMAP x 5 x 3 HDBSCAN) ───────────────
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-15,30,50,100}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.0,0.05,0.1,0.25}"
UMAP_N_COMPONENTS="${UMAP_N_COMPONENTS:-2,16}"
UMAP_N_EPOCHS="${UMAP_N_EPOCHS:-200}"
MIN_CLUSTER_SIZES="${MIN_CLUSTER_SIZES:-100,250,500,1000,2500}"
MIN_SAMPLES="${MIN_SAMPLES:-5,25,50}"
FIT_ROWS="${FIT_ROWS:-all}"
TRANSFORM_BATCH="${TRANSFORM_BATCH:-5000000}"
EMBED_BATCH="${EMBED_BATCH:-262144}"
SIL_EVAL_ROWS="${SIL_EVAL_ROWS:-50000}"
AGREEMENT_ROWS="${AGREEMENT_ROWS:-200000}"

DRY_RUN="${DRY_RUN:-0}"
NO_RESUME="${NO_RESUME:-0}"

COSMIC_VERSION="${COSMIC_VERSION:-3.5}"
GENOME_BUILD="${GENOME_BUILD:-GRCh38}"

THREADS="${SLURM_CPUS_PER_TASK:-16}"

# ── Env activation (mirrors the logic in tmux_lib.sh) ─────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

if [ -n "${CONDA_ENV:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
elif command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "${MAMBA_ENV:-uv_vae}"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate patrickg
fi

# Add the package to the path exactly as the Python Files/ scripts do.
export PYTHONPATH="$UV_VAE_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ── GPU budget test (runs even on CPU nodes; reports enabled=False there) ──
# This exercises apply_gpu_budget before any fitting starts, so a misconfigured
# ceiling is caught here rather than mid-run.
log "===== GPU budget preflight ====="
python - <<'PYEOF'
import sys
sys.path.insert(0, "$UV_VAE_DIR")
from uv_vae import gpu_budget

report = gpu_budget.apply(verbose=True)
d = report.as_dict()
print(f"  enabled        : {d['enabled']}")
print(f"  budget_gb      : {d['budget_gb']}")
print(f"  device_total_gb: {d['device_total_gb']}")
print(f"  torch_budget_gb: {d['torch_budget_gb']}")
print(f"  rmm_pool_gb    : {d['rmm_pool_gb']}")
print(f"  notes          : {d['notes']}")
PYEOF
log "===== GPU budget preflight done ====="

# ── Validate inputs ────────────────────────────────────────────────────────
if [ ! -f "$CHECKPOINT" ]; then
    log "ERROR: checkpoint not found: $CHECKPOINT"
    log "Set CHECKPOINT= before submitting."
    exit 1
fi

mkdir -p "$RUN_ROOT" "$LOG_DIR" "$TEMP_DIR"

log "================================================================"
log "UMAP x HDBSCAN sweep -- stages 0, 1, 2   (stage 3 excluded)"
log "  run root    : $RUN_ROOT"
log "  parquets    : $PARQUET_GLOB"
log "  checkpoint  : $CHECKPOINT"
log "  row filter  : $ROW_FILTER"
log "  UMAP grid   : nn=$UMAP_N_NEIGHBORS  min_dist=$UMAP_MIN_DIST  n_comp=$UMAP_N_COMPONENTS"
log "  HDBSCAN grid: mcs=$MIN_CLUSTER_SIZES  min_samples=$MIN_SAMPLES"
log "  fit rows    : $FIT_ROWS   transform batch: $TRANSFORM_BATCH"
log "  SigProfiler : every cell (COSMIC v$COSMIC_VERSION, $GENOME_BUILD)"
log "  threads     : $THREADS"
log "  seed        : $SEED"
log "  dry run     : $DRY_RUN"
log "================================================================"

# ── Stage 0: deduplicate ────────────────────────────────────────────────────
log "===== BEGIN stage 0: deduplicate the cohort ====="
python "$SWEEP_DIR/stage0_dedup.py" \
    --parquet-paths $PARQUET_GLOB \
    --checkpoint-path "$CHECKPOINT" \
    --output-dir "$DEDUP_DIR" \
    --row-filter "$ROW_FILTER" \
    --threads "$THREADS" \
    --memory-limit "$DUCKDB_MEMORY" \
    --temp-directory "$TEMP_DIR" \
    ${CHROMOSOMES:+--chromosomes "$CHROMOSOMES"}
log "===== END stage 0 ====="

# Print the locus count so you know whether stage 2 can fit on everything.
python - <<PYEOF
import json, pathlib
manifest = json.loads(pathlib.Path("$DEDUP_DIR/dedup_manifest.json").read_text())
total = manifest["total_loci"]
ratio = manifest["collapse_ratio"]
print(f"  total unique loci : {total:,}")
print(f"  collapse ratio    : {ratio}x")
print(f"  stage 2 fit size  : {'fits' if total < 5_000_000 else 'LARGE -- consider FIT_ROWS=5000000'}")
PYEOF

# ── Stage 1: embed ──────────────────────────────────────────────────────────
log "===== BEGIN stage 1: embed deduplicated loci ====="
python "$SWEEP_DIR/stage1_embed.py" \
    --dedup-manifest "$DEDUP_DIR/dedup_manifest.json" \
    --checkpoint-path "$CHECKPOINT" \
    --feature-spec-path "$FEATURE_SPEC" \
    --output-dir "$EMBED_DIR" \
    --batch-size "$EMBED_BATCH" \
    --device cpu
log "===== END stage 1 ====="

# ── Stage 2: sweep ──────────────────────────────────────────────────────────
log "===== BEGIN stage 2: UMAP x HDBSCAN sweep ====="
flags=()
[ "$DRY_RUN"   = "1" ] && flags+=(--dry-run)
[ "$NO_RESUME" = "1" ] && flags+=(--no-resume)

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
    --threads "$THREADS" \
    --cosmic-version "$COSMIC_VERSION" \
    --genome-build "$GENOME_BUILD" \
    --force-cpu \
    "${flags[@]}"
log "===== END stage 2 ====="

log "================================================================"
log "Stages 0-2 complete. Results in: $SWEEP_OUT"
log "Pick a cell from: $SWEEP_OUT/sweep_summary.json"
log "Then run stage 3 separately (not part of this job):"
log "  STAGE=3 CELL=<cell> bash $SWEEP_DIR/run_sweep.sh"
log "================================================================"
