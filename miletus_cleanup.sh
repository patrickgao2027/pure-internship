#!/usr/bin/env bash
# miletus_cleanup.sh — free ~170+ GB of regenerable sweep artifacts on miletus
# Run with --dry-run first to confirm, then without to execute.
#
# KEEPS: model.pt, latent.npy, context.parquet, umap_model.joblib,
#        all code/scripts/json results, cohort_reports JSON summaries
# DELETES: hdbscan .pkl/.joblib models, per_parquet_inference_cuml,
#          analysis.parquet sweep outputs, loose home-dir junk

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

REPO="$HOME/pure-internship"
FREED=0

RM() {
    local path="$1"
    if [[ ! -e "$path" ]]; then return; fi
    local size
    size=$(du -sb "$path" 2>/dev/null | cut -f1)
    FREED=$(( FREED + size ))
    echo "  [$(du -sh "$path" 2>/dev/null | cut -f1)]  $path"
    $DRY_RUN || rm -rf "$path"
}

RM_PATTERN() {
    local label="$1"; shift
    echo
    echo "── $label"
    find "$@" 2>/dev/null | while read -r f; do RM "$f"; done
}

SEP() { echo; echo "──────────────────────────────────────────────────────"; }

SEP
echo "miletus_cleanup.sh"
$DRY_RUN && echo "  DRY RUN — nothing deleted" || echo "  LIVE RUN — files will be removed"
echo "  repo: $REPO"
SEP

# ── 1. per_parquet_inference_cuml — 98 GB ────────────────────────────────────
echo
echo "── 1. per_parquet_inference_cuml/ (~98 GB)"
echo "   row_assignments, umap_coords, labels — regenerable via inference.py"
RM "$REPO/umap_hdbscan_sweep/per_parquet_inference_cuml"

# ── 2. HDBSCAN model pickles in sweep results (~50 GB) ───────────────────────
RM_PATTERN "2. HDBSCAN .pkl model files in sweep results (~50 GB)" \
    "$REPO/umap_hdbscan_sweep" -name "hdbscan_model.pkl"

# ── 3. HDBSCAN scaling .joblib models ────────────────────────────────────────
RM_PATTERN "3. HDBSCAN scaling .joblib models" \
    "$REPO/umap_hdbscan_sweep/hdbscan/results/hdbscan_scaling/models" \
    -name "*.joblib"

# ── 4. stage2_sweep analysis.parquet files (~10 GB) ──────────────────────────
echo
echo "── 4. stage2_sweep analysis.parquet (~10 GB)"
echo "   regenerable: re-run UMAP+HDBSCAN on latent.npy"
find "$REPO/uv_vae/runs" -path "*/stage2_sweep/*/analysis.parquet" 2>/dev/null \
    | while read -r f; do RM "$f"; done

# ── 5. latent2d_cluster analysis.parquet files (~7.4 GB) ─────────────────────
echo
echo "── 5. latent2d_cluster analysis.parquet (~7.4 GB)"
find "$REPO/uv_vae/runs/latent2d_cohort/latent2d_cluster" -name "analysis.parquet" 2>/dev/null \
    | while read -r f; do RM "$f"; done

# ── 6. hdbscan_sweep zip/tar at HOME root ────────────────────────────────────
echo
echo "── 6. Loose archives at HOME root"
for f in \
    "$HOME/param_sweep_results.tar.gz" \
    "$HOME/param_sweep_pkl.log" \
    "$HOME/run_two.log" \
    "$HOME/vram_log.txt" \
    "$HOME/vram_per_parquet.csv" \
    "$HOME/trinuc96_snvq_onefile.html"
do
    [[ -f "$f" ]] && RM "$f"
done

# ── 7. Old audit log files at HOME root ──────────────────────────────────────
echo
echo "── 7. Old audit logs at HOME root (keep none — they re-run easily)"
find "$HOME" -maxdepth 1 -name "miletus_audit_*.txt" 2>/dev/null \
    | while read -r f; do RM "$f"; done

# ── 8. __pycache__ dirs ───────────────────────────────────────────────────────
echo
echo "── 8. __pycache__ dirs under pure-internship"
find "$REPO" -type d -name "__pycache__" 2>/dev/null \
    | while read -r f; do RM "$f"; done

SEP
FREED_GB=$(echo "scale=1; $FREED / 1073741824" | bc 2>/dev/null || echo "?")
echo "Estimated freed: ~${FREED_GB} GB"
$DRY_RUN && echo "(dry run — rerun without --dry-run to actually delete)"
echo
echo "Remaining irreplaceable files:"
echo "  model.pt        — $(find "$REPO" -name 'model.pt' 2>/dev/null | wc -l) files"
echo "  latent.npy      — $(find "$REPO" -name 'latent.npy' 2>/dev/null | wc -l) files"
echo "  umap_model.joblib — $(find "$REPO" -name 'umap_model.joblib' 2>/dev/null | wc -l) files"
echo "  context.parquet — $(find "$REPO" -name 'context.parquet' 2>/dev/null | wc -l) files"
df -h "$HOME"
