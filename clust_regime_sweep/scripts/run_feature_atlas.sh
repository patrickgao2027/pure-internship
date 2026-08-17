#!/usr/bin/env bash
# Run plot_feature_atlas.py over every cohort_reports cell.
#
# Each cell's analysis.parquet holds exactly 2,000,000 rows and 39 columns; five of those
# (umap_1, umap_2, cluster_label, cluster_probability, POS) are excluded as coordinates or
# identifiers, leaving 34 panels per cell -- 29 numeric and 5 categorical (CHROM, REF, ALT,
# X_PREV1, X_NEXT1). The categoricals are the whole reason this runs: the existing plots/
# folders come from cohort_cluster_report.py, which renders numerics only, so the
# context columns that score highest against cluster_label currently have no panel at all.
#
# Output goes to <cell>/feature_atlas/ and does NOT touch the existing <cell>/plots/.
#
# PARALLELISM. Two levels, and they multiply -- that is the thing to get right.
# PARALLEL_JOBS is how many cells run at once (each a separate python process holding its own
# 2M-row frame). WORKERS is the matplotlib process pool *inside* each of those. Defaults here
# are 28 x 1: every cell at once, one renderer each, so the box sees 28 processes rather than
# 28 x 8 = 224 fighting over the same cores. If you lower PARALLEL_JOBS, raise WORKERS to
# match, keeping the product near the core count.
#
# COST, measured on one cell at the full 2M rows (not extrapolated):
#     load parquet            ~1 s
#     score 34 features      ~12 s   (eta2 + NMI + the permuted null, all of it)
#     render numeric panel   ~23 s   x 29
#     render categorical      ~11 s   x 5
# so ~13 min per cell single-threaded, and rasterisation is ~98% of it. Scoring is noise.
# At the old --sample-rows 300000 a panel took ~3 s, so a full pass costs about 7x per cell --
# but with every cell running at once the wall clock is one cell, ~15 min for the whole sweep.
#
# MEMORY. Each job holds the 2M-row frame (~1.0 GB as pandas) plus the prepared render
# payloads, which are materialised for all 34 panels before rendering starts (~0.5 GB at 2M
# rows) -- budget ~3 GB per concurrent job. The 28-wide default therefore wants ~84 GB free.
# The script measures and warns before starting. If the box is smaller, trading outer for
# inner parallelism costs nothing in wall time and a third of the RAM:
#     PARALLEL_JOBS=10 WORKERS=3 bash clust_regime_sweep/scripts/run_feature_atlas.sh
#
# LOGS. Every cell writes <cell>/feature_atlas/run.log (full stdout+stderr of that cell's run)
# and <cell>/feature_atlas/.exit_status (its exit code). With 28 cells interleaving, the
# console only carries START/DONE/FAIL lines; the per-cell log is where the feature table,
# the timing, and any traceback actually land. Tail one while the sweep runs:
#
#     tail -f clust_regime_sweep/cohort_reports_original/*/feature_atlas/run.log
#
# Usage:
#     bash clust_regime_sweep/scripts/run_feature_atlas.sh
#
# Env overrides:
#     REPORTS_DIR    parent of the per-cell subdirs      [auto-detected]
#     SCRIPT         plot_feature_atlas.py path          [auto-detected]
#     SAMPLE_ROWS    rows per cell; 0/all = every row    [0 -> full 2M pass]
#     PARALLEL_JOBS  cells running concurrently          [28 = all at once]
#     WORKERS        render processes inside each cell   [1]
#     DPI            panel resolution                    [150]
#     FORCE          1 to redo cells already finished    [0]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if [ -d "$HOME/pure-internship/clust_regime_sweep/cohort_reports_original" ]; then
    DEFAULT_REPORTS="$HOME/pure-internship/clust_regime_sweep/cohort_reports_original"
    DEFAULT_SCRIPT="$HOME/pure-internship/umap_hdbscan_sweep/plot_feature_atlas.py"
else
    DEFAULT_REPORTS="$REPO_ROOT/clust_regime_sweep/cohort_reports_original"
    DEFAULT_SCRIPT="$REPO_ROOT/umap_hdbscan_sweep/plot_feature_atlas.py"
fi

REPORTS_DIR="${REPORTS_DIR:-$DEFAULT_REPORTS}"
SCRIPT="${SCRIPT:-$DEFAULT_SCRIPT}"
SAMPLE_ROWS="${SAMPLE_ROWS:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-28}"
WORKERS="${WORKERS:-1}"
DPI="${DPI:-150}"
FORCE="${FORCE:-0}"

# 0 / "all" means every row. sample_positions() takes min(wanted, total), so any number at or
# above the cell size is a full pass; this is just a readable way to ask for one.
case "$SAMPLE_ROWS" in
    0|all|ALL|full|FULL) SAMPLE_ROWS=1000000000 ;;
esac

[ -d "$REPORTS_DIR" ] || { echo "ERROR: no such REPORTS_DIR: $REPORTS_DIR" >&2; exit 1; }
[ -f "$SCRIPT" ]      || { echo "ERROR: no such SCRIPT: $SCRIPT" >&2; exit 1; }

# Manager-agnostic activation, matching the other scripts in this repo.
if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null || true)"; conda activate "$CONDA_ENV"
elif command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook -s bash 2>/dev/null || true)"
    micromamba activate "${MAMBA_ENV:-uv_vae}" || true
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null || true)"; conda activate patrickg || true
fi

# Each cell already runs its own outer process; letting BLAS spin up a full thread pool inside
# every one of them is how 28 jobs turn into several hundred runnable threads.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TQDM_DISABLE=1
export MPLBACKEND=Agg

mapfile -t CELLS < <(find "$REPORTS_DIR" -mindepth 2 -maxdepth 2 -name analysis.parquet -printf '%h\n' | sort)
[ "${#CELLS[@]}" -gt 0 ] || { echo "ERROR: no analysis.parquet under $REPORTS_DIR" >&2; exit 1; }

CORES="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
RAM_GB="$(awk '/^MemAvailable:/ {printf "%.0f", $2/1048576}' /proc/meminfo 2>/dev/null || true)"

# A probe can come back empty as well as absent -- cygwin has /proc/meminfo but no
# MemAvailable line, so awk succeeds and prints nothing. Anything that is not a run of digits
# is unusable, and the arithmetic guards below must not see it.
[[ "$CORES"  =~ ^[0-9]+$ ]] || CORES='?'
[[ "$RAM_GB" =~ ^[0-9]+$ ]] || RAM_GB='?'

echo "=== feature atlas sweep ==="
echo "  reports : $REPORTS_DIR"
echo "  script  : $SCRIPT"
echo "  cells   : ${#CELLS[@]}"
echo "  rows    : $([ "$SAMPLE_ROWS" -ge 1000000000 ] && echo 'all (full 2M pass per cell)' || echo "$SAMPLE_ROWS")"
echo "  jobs    : $PARALLEL_JOBS concurrent x $WORKERS render worker(s) = $((PARALLEL_JOBS * WORKERS)) processes"
echo "  host    : ${CORES} cores, ${RAM_GB} GB RAM available (? = could not probe)"
echo "  dpi     : $DPI"

if [ "$RAM_GB" != "?" ] && [ "$RAM_GB" -lt $((PARALLEL_JOBS * 3)) ]; then
    SAFE_JOBS=$((RAM_GB / 3)); [ "$SAFE_JOBS" -lt 1 ] && SAFE_JOBS=1
    echo ""
    echo "  WARNING: ~3 GB per concurrent job means $PARALLEL_JOBS jobs want $((PARALLEL_JOBS * 3)) GB,"
    echo "           but only ${RAM_GB} GB is available -- the OOM killer will take some of them."
    echo "           Same wall time for a third of the RAM:"
    echo "             PARALLEL_JOBS=$SAFE_JOBS WORKERS=3 bash $0"
fi
if [ "$CORES" != "?" ] && [ $((PARALLEL_JOBS * WORKERS)) -gt $((CORES * 2)) ]; then
    echo ""
    echo "  WARNING: $((PARALLEL_JOBS * WORKERS)) processes on ${CORES} cores is oversubscribed;"
    echo "           consider PARALLEL_JOBS=$CORES WORKERS=1."
fi
echo ""

run_cell() {
    local cell_dir="$1" name out_dir status
    name="$(basename "$cell_dir")"
    out_dir="$cell_dir/feature_atlas"
    mkdir -p "$out_dir"
    rm -f "$out_dir/.exit_status"

    # `|| status=$?` keeps set -e from killing the background job before the status is
    # recorded -- the tally below reads these files, not the wait status.
    status=0
    python "$SCRIPT" \
        --analysis   "$cell_dir/analysis.parquet" \
        --output-dir "$out_dir" \
        --sample-rows "$SAMPLE_ROWS" \
        --workers     "$WORKERS" \
        --dpi         "$DPI" \
        >"$out_dir/run.log" 2>&1 || status=$?

    echo "$status" >"$out_dir/.exit_status"
    if [ "$status" -eq 0 ]; then
        echo "[DONE] $name"
    else
        echo "[FAIL] $name  (exit $status) -- see $out_dir/run.log"
    fi
}

LAUNCHED=0
SKIPPED=0
for cell_dir in "${CELLS[@]}"; do
    name="$(basename "$cell_dir")"
    # Completion is judged on the recorded exit status, not on feature_atlas.csv: the csv is
    # written before the contact sheets, so a cell that died during sheet assembly would look
    # finished by file presence alone and get skipped on the retry pass.
    if [ "$FORCE" != "1" ] \
       && [ -f "$cell_dir/feature_atlas/feature_atlas.csv" ] \
       && [ "$(cat "$cell_dir/feature_atlas/.exit_status" 2>/dev/null || echo 1)" = "0" ]; then
        echo "[SKIP] $name  (already complete; FORCE=1 to redo)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # jobs -rp is portable back to bash 4.0, unlike `wait -n`.
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL_JOBS" ]; do sleep 2; done

    echo "[START] $name"
    run_cell "$cell_dir" &
    LAUNCHED=$((LAUNCHED + 1))
done

wait

OK=0
BAD=0
FAILED_CELLS=()
for cell_dir in "${CELLS[@]}"; do
    status_file="$cell_dir/feature_atlas/.exit_status"
    [ -f "$status_file" ] || continue
    if [ "$(cat "$status_file")" = "0" ]; then
        OK=$((OK + 1))
    else
        BAD=$((BAD + 1))
        FAILED_CELLS+=("$(basename "$cell_dir")")
    fi
done

echo ""
echo "=== this run: launched $LAUNCHED, skipped $SKIPPED ==="
echo "=== tree now: $OK of ${#CELLS[@]} cells complete, $BAD failed ==="
if [ "$BAD" -gt 0 ]; then
    printf '  failed: %s\n' "${FAILED_CELLS[@]}"
    echo "  re-run just those with FORCE=1 after fixing the cause."
    exit 1
fi
echo "  per-cell output: <cell>/feature_atlas/{feature_atlas.csv,contact_sheet_*.png,*.png,run.log}"
