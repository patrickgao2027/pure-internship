#!/usr/bin/env bash
# Run plot_feature_atlas.py on every cohort_reports cell.
#
# Reads each cell's analysis.parquet and writes feature_atlas/ alongside plots/.
# Cells are processed in parallel batches (PARALLEL_JOBS at a time); each
# individual run is single-process so you can tune PARALLEL_JOBS to taste
# without hitting RAM limits (each job loads ~3M rows × 27 cols into pandas,
# which is roughly 1–2 GB).
#
# Usage (from repo root or miletus home, adjust paths via env vars):
#   bash clust_regime_sweep/scripts/run_feature_atlas.sh
#
# Env overrides:
#   REPORTS_DIR   -- parent folder of per-cell subdirs   [default: auto-detected]
#   SCRIPT        -- plot_feature_atlas.py path           [default: auto-detected]
#   SAMPLE_ROWS   -- rows to sample per cell              [default: 5000000]
#   PARALLEL_JOBS -- cells to run in parallel             [default: 4]
#   WORKERS       -- matplotlib worker processes per cell [default: 8]
#   DPI           -- output PNG resolution                [default: 150]

set -euo pipefail

# ── environment ────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Detect HPC vs local
if [ -d "$HOME/pure-internship" ]; then
    DEFAULT_REPORTS="$HOME/pure-internship/clust_regime_sweep/cohort_reports_original"
    DEFAULT_SCRIPT="$HOME/pure-internship/umap_hdbscan_sweep/plot_feature_atlas.py"
elif [ -d "$REPO_ROOT/clust_regime_sweep/cohort_reports_original" ]; then
    DEFAULT_REPORTS="$REPO_ROOT/clust_regime_sweep/cohort_reports_original"
    DEFAULT_SCRIPT="$REPO_ROOT/umap_hdbscan_sweep/plot_feature_atlas.py"
else
    echo "ERROR: cannot locate cohort_reports_original. Set REPORTS_DIR." >&2
    exit 1
fi

REPORTS_DIR="${REPORTS_DIR:-$DEFAULT_REPORTS}"
SCRIPT="${SCRIPT:-$DEFAULT_SCRIPT}"
SAMPLE_ROWS="${SAMPLE_ROWS:-5000000}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
WORKERS="${WORKERS:-8}"
DPI="${DPI:-150}"

# Activate environment (manager-agnostic, same pattern as other scripts)
if [ -n "${CONDA_ENV:-}" ]; then
    conda activate "$CONDA_ENV"
elif command -v micromamba &>/dev/null; then
    eval "$(micromamba shell hook -s bash 2>/dev/null || true)"
    micromamba activate "${MAMBA_ENV:-uv_vae}"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null || true)"
    conda activate patrickg
fi

echo "=== feature atlas sweep ==="
echo "  reports : $REPORTS_DIR"
echo "  script  : $SCRIPT"
echo "  rows    : $SAMPLE_ROWS"
echo "  jobs    : $PARALLEL_JOBS  (workers per job: $WORKERS)"
echo "  dpi     : $DPI"
echo ""

# ── collect cells ──────────────────────────────────────────────────────────────
mapfile -t CELLS < <(
    find "$REPORTS_DIR" -maxdepth 2 -name "analysis.parquet" -printf "%h\n" | sort
)

if [ ${#CELLS[@]} -eq 0 ]; then
    echo "No analysis.parquet found under $REPORTS_DIR" >&2
    exit 1
fi
echo "Found ${#CELLS[@]} cells."
echo ""

# ── run ────────────────────────────────────────────────────────────────────────
DONE=0
SKIPPED=0
FAILED=0
PIDS=()
CELL_OF_PID=()

run_cell() {
    local cell_dir="$1"
    local cell_name
    cell_name="$(basename "$cell_dir")"
    local out_dir="$cell_dir/feature_atlas"
    local analysis="$cell_dir/analysis.parquet"
    local log_file="$out_dir/run.log"

    mkdir -p "$out_dir"
    echo "[START] $cell_name"

    python "$SCRIPT" \
        --analysis "$analysis" \
        --output-dir "$out_dir" \
        --sample-rows "$SAMPLE_ROWS" \
        --workers "$WORKERS" \
        --dpi "$DPI" \
        >"$log_file" 2>&1
    echo "[DONE]  $cell_name"
}

export -f run_cell
export SCRIPT SAMPLE_ROWS WORKERS DPI

for cell_dir in "${CELLS[@]}"; do
    cell_name="$(basename "$cell_dir")"
    out_dir="$cell_dir/feature_atlas"

    # Skip if already complete (feature_atlas.csv exists)
    if [ -f "$out_dir/feature_atlas.csv" ]; then
        echo "[SKIP]  $cell_name  (feature_atlas.csv exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    run_cell "$cell_dir" &
    PIDS+=($!)
    CELL_OF_PID+=("$cell_name")

    # Drain when we hit the parallelism limit
    if [ ${#PIDS[@]} -ge "$PARALLEL_JOBS" ]; then
        for i in "${!PIDS[@]}"; do
            if wait "${PIDS[$i]}"; then
                DONE=$((DONE + 1))
            else
                echo "[FAIL]  ${CELL_OF_PID[$i]}" >&2
                FAILED=$((FAILED + 1))
            fi
        done
        PIDS=()
        CELL_OF_PID=()
    fi
done

# Drain remaining
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        DONE=$((DONE + 1))
    else
        echo "[FAIL]  ${CELL_OF_PID[$i]}" >&2
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== done: $DONE completed, $SKIPPED skipped, $FAILED failed ==="
[ "$FAILED" -eq 0 ]
