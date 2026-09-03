#!/usr/bin/env bash
# setup_deployment.sh — create $HOME/uv_vae_deployment/ on miletus
# Usage: bash setup_deployment.sh [--dry-run]
#
# Source: $HOME/pure-internship/UV_VAE_Deployment/ (code + models + results + docs)
#         $HOME/pure-internship/uv_vae/runs/       (all training run artifacts)
# Dest:   $HOME/uv_vae_deployment/                 (sibling of pure-internship)
#
# After this runs:
#   - import uv_vae works from any script depth (editable install)
#   - all run artifacts are under runs/pure_internship/
#   - UV_VAE_Deployment inside pure-internship is untouched

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

RUN()  { echo "  + $*"; $DRY_RUN || "$@"; }
SEP()  { echo; echo "──────────────────────────────────────────────────────"; }

REPO="$HOME/pure-internship"
SRC="$REPO/UV_VAE_Deployment"
RUNS_SRC="$REPO/uv_vae/runs"
DEST="$HOME/uv_vae_deployment"

SEP
echo "uv_vae_deployment setup"
echo "  source (deployment) : $SRC"
echo "  source (runs)       : $RUNS_SRC"
echo "  dest                : $DEST"
$DRY_RUN && echo "  mode                : DRY RUN"
SEP

# ── guards ───────────────────────────────────────────────────────────────────
[[ -d "$REPO" ]]   || { echo "ERROR: $REPO not found"; exit 1; }
[[ -d "$SRC" ]]    || { echo "ERROR: $SRC not found — UV_VAE_Deployment missing"; exit 1; }
[[ -d "$RUNS_SRC" ]] || { echo "ERROR: $RUNS_SRC not found"; exit 1; }

# ── 1. base: rsync UV_VAE_Deployment as the skeleton ─────────────────────────
SEP
echo "1. Rsyncing UV_VAE_Deployment → $DEST"
echo "   (code, models, results, docs, signature_db — excludes __pycache__, *.pyc)"

RUN rsync -av --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    "$SRC/" "$DEST/"

# ── 2. runs: copy all training run artifacts ──────────────────────────────────
SEP
echo "2. Copying training run artifacts → $DEST/runs/pure_internship/"
$DRY_RUN || mkdir -p "$DEST/runs/pure_internship"

find "$RUNS_SRC" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | while read -r d; do
    name=$(basename "$d")
    size=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "  [$size]  $name"
    RUN rsync -a --progress "$d/" "$DEST/runs/pure_internship/$name/"
done

# ── 3. editable install ───────────────────────────────────────────────────────
SEP
echo "3. Editable install — import uv_vae works from any script depth"

if $DRY_RUN; then
    echo "   (dry-run) would run: pip install -e $DEST/uv_vae"
elif command -v micromamba &>/dev/null; then
    echo "   micromamba env: ${MAMBA_ENV:-uv_vae}"
    micromamba run -n "${MAMBA_ENV:-uv_vae}" pip install -e "$DEST/uv_vae" --quiet
    echo "   installed."
else
    echo "   micromamba not found — run manually:"
    echo "   pip install -e $DEST/uv_vae"
fi

# ── 4. CLAUDE.md ──────────────────────────────────────────────────────────────
SEP
echo "4. Writing CLAUDE.md"

CLAUDE_FILE="$DEST/CLAUDE.md"
if [[ -f "$CLAUDE_FILE" ]]; then
    echo "   (exists — skipping)"
else
    $DRY_RUN || cat > "$CLAUDE_FILE" <<'EOF'
# uv_vae_deployment

Post-PURE working deployment. The `uv_vae` package is installed as an editable
install into the `uv_vae` micromamba env — scripts import it with `import uv_vae`
from any directory depth, no `sys.path` manipulation needed.

## Layout

```
uv_vae_deployment/
├── uv_vae/                     core package (pip install -e done)
├── umap_hdbscan_sweep/         pipeline stages + sweep code
├── Early_Stopping_Tests/       training CLI and tmux runners
├── models/                     VAE, UMAP, HDBSCAN checkpoints + coordinates
├── results/                    cohort SigProfiler assignments
├── runs/
│   └── pure_internship/        all training run artifacts from the PURE internship
├── docs/                       design decision records
├── RUNNING_INFERENCE.md
├── REBUILDING_MODELS.md
├── ACCESSING_ASSIGNMENTS.md
├── DEPLOYMENT_MANIFEST.md
└── pipeline_parameters.md
```

## Key paths (miletus)

- Parquets      : /data/lab/ppmseq_parquets/
- PURE archive  : $HOME/pure-internship/   (read-only)
- GPU env       : micromamba env `uv_vae`

## Import

No `parents[N]` path injection needed in new scripts — just `import uv_vae`.
For scripts already in `umap_hdbscan_sweep/` the existing `sys.path` injection
still works since `uv_vae/` is a sibling.

## Best checkpoint

`models/vae/model.pt` — 5.08 B rows, 38 epochs, best epoch 30, val_loss 0.220917.
Run artifact JSONs in `models/vae/run_20260802T192814Z/`.

## Notes

- cuML HDBSCAN safe limit: 5M rows (~10 min, ~1.3 GB). 70M rows OOMs the 47 GB card.
- HDBSCAN model requires cuML to unpickle; CPU machines can read everything else.
- Row arrays in `models/coords/` are aligned by position — do not re-run stage0_dedup
  without regenerating all five arrays together.
EOF
    echo "   written."
fi

# ── summary ───────────────────────────────────────────────────────────────────
SEP
echo "Done."
echo ""
echo "Verify install:"
echo "  micromamba run -n uv_vae python -c 'import uv_vae; print(uv_vae.__file__)'"
echo ""
df -h "$HOME"
