#!/usr/bin/env bash
# miletus_audit.sh — run on miletus to see what lives where and how big it is
# Usage: bash miletus_audit.sh | tee ~/miletus_audit_$(date +%Y%m%dT%H%M%S).txt

set -euo pipefail
SEP="$(printf '%.0s─' {1..70})"
H1() { echo; echo "$SEP"; echo "  $*"; echo "$SEP"; }
H2() { echo; echo "── $* ──"; }

H1 "MILETUS FILE AUDIT — $(date -u '+%Y-%m-%d %H:%M UTC')"

# ── 1. Home directory top-level ─────────────────────────────────────────────
H1 "1. HOME DIRECTORY — top-level contents"
du -sh "$HOME"/*/  2>/dev/null | sort -rh || true
du -sh "$HOME"/.[^.]*/  2>/dev/null | sort -rh | head -20 || true

H2 "Loose files in HOME (not dirs)"
find "$HOME" -maxdepth 1 -type f | sort

# ── 2. pure-internship repo tree ─────────────────────────────────────────────
H1 "2. pure-internship REPO — top-level structure"
du -sh "$HOME/pure-internship"/*/  2>/dev/null | sort -rh || true

H2 "  git status (tracked changes)"
git -C "$HOME/pure-internship" status --short 2>/dev/null || echo "(not a git repo or git unavailable)"

H2 "  uv_vae/ breakdown"
du -sh "$HOME/pure-internship/uv_vae"/*/  2>/dev/null | sort -rh || true

# ── 3. Run artifacts (model checkpoints, reports) ───────────────────────────
H1 "3. RUN ARTIFACTS — model.pt files and their run dirs"
find "$HOME" -name "model.pt" 2>/dev/null | while read -r f; do
    dir=$(dirname "$f")
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    mtime=$(stat -c '%y' "$f" 2>/dev/null | cut -d' ' -f1)
    echo "  [$mtime]  $size  $dir"
done | sort

H2 "  All run_* directories"
find "$HOME" -type d -name 'run_*' 2>/dev/null | while read -r d; do
    size=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "  $size  $d"
done | sort -k2

# ── 4. Large files (>100 MB) ─────────────────────────────────────────────────
H1 "4. LARGE FILES — anything > 100 MB under HOME"
find "$HOME" -not -path '/data/*' -size +100M -type f 2>/dev/null \
    | xargs -I{} du -sh {} 2>/dev/null \
    | sort -rh \
    | head -40

# ── 5. Cluster / sweep output dirs ──────────────────────────────────────────
H1 "5. SWEEP & CLUSTER OUTPUT DIRS"
for d in \
    "$HOME/pure-internship/umap_hdbscan_sweep" \
    "$HOME/pure-internship/hdbscan_sweep" \
    "$HOME/umap_hdbscan_sweep" \
    "$HOME/hdbscan_sweep" \
    "$HOME/pure-internship/comparison_results" \
    "$HOME/comparison_results"
do
    if [[ -d "$d" ]]; then
        size=$(du -sh "$d" 2>/dev/null | cut -f1)
        echo "  FOUND  $size  $d"
        du -sh "$d"/*/  2>/dev/null | sort -rh | sed 's/^/      /' || true
    fi
done

# ── 6. Python / conda envs ────────────────────────────────────────────────────
H1 "6. CONDA / MICROMAMBA ENVIRONMENTS"
if command -v micromamba &>/dev/null; then
    micromamba env list 2>/dev/null || true
fi
if command -v conda &>/dev/null; then
    conda env list 2>/dev/null || true
fi
du -sh "$HOME/micromamba/envs"/*/  2>/dev/null | sort -rh || true
du -sh "$HOME/.conda/envs"/*/      2>/dev/null | sort -rh || true

# ── 7. Parquet / data files (not in /data/lab) ───────────────────────────────
H1 "7. PARQUET FILES UNDER HOME (not /data/lab)"
find "$HOME" -name '*.parquet' -not -path '/data/*' 2>/dev/null \
    | xargs -I{} du -sh {} 2>/dev/null \
    | sort -rh | head -30 || echo "  (none found)"

# ── 8. Disk quota overview ────────────────────────────────────────────────────
H1 "8. DISK USAGE SUMMARY"
df -h "$HOME" 2>/dev/null || true
quota -s 2>/dev/null || true

H1 "AUDIT COMPLETE"
