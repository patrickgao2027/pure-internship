#!/usr/bin/env bash
# setup_deployment.sh — run on miletus to create uv_vae_deployment/
# Usage: bash setup_deployment.sh [--dry-run]
#
# What it does:
#   1. Creates $HOME/uv_vae_deployment/ directory tree
#   2. Rsyncs the uv_vae package from pure-internship
#   3. Copies all run_* checkpoint dirs into deployment/runs/pure_internship/
#   4. Does an editable pip install so "import uv_vae" works everywhere
#   5. Creates stub experiment dirs for post-PURE work

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

RUN()  { echo "  + $*"; $DRY_RUN || "$@"; }
ECHO() { echo "$*"; }
SEP()  { echo; echo "──────────────────────────────────────────────────────"; }

SOURCE="$HOME/pure-internship"
DEST="$HOME/uv_vae_deployment"

SEP
ECHO "uv_vae_deployment setup"
ECHO "  source : $SOURCE"
ECHO "  dest   : $DEST"
$DRY_RUN && ECHO "  mode   : DRY RUN — nothing will be written"
SEP

# ── guard ────────────────────────────────────────────────────────────────────
if [[ ! -d "$SOURCE" ]]; then
    echo "ERROR: $SOURCE not found — is this miletus?" >&2
    exit 1
fi
if [[ ! -d "$SOURCE/uv_vae" ]]; then
    echo "ERROR: $SOURCE/uv_vae not found — check SOURCE path" >&2
    exit 1
fi

# ── 1. directory skeleton ─────────────────────────────────────────────────────
SEP
ECHO "1. Creating directory skeleton"

for d in \
    "$DEST" \
    "$DEST/uv_vae" \
    "$DEST/runs/pure_internship" \
    "$DEST/experiments/publication_figures/Python Files" \
    "$DEST/experiments/publication_figures/scripts"
do
    if [[ ! -d "$d" ]]; then
        RUN mkdir -p "$d"
    else
        echo "  (exists) $d"
    fi
done

# ── 2. sync the uv_vae package ───────────────────────────────────────────────
SEP
ECHO "2. Syncing uv_vae package  ($SOURCE/uv_vae → $DEST/uv_vae)"
ECHO "   (excludes: __pycache__, *.pyc, runs/, .git)"

RUN rsync -av --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='runs/' \
    "$SOURCE/uv_vae/" "$DEST/uv_vae/"

# ── 3. copy run artifacts ─────────────────────────────────────────────────────
SEP
ECHO "3. Discovering run_* directories under $SOURCE"

RUN_DIRS=()
while IFS= read -r d; do
    RUN_DIRS+=("$d")
done < <(find "$SOURCE" -type d -name 'run_*' 2>/dev/null | sort)

# also catch named run dirs (e.g. latent2d_cohort, train_multi_*)
while IFS= read -r d; do
    RUN_DIRS+=("$d")
done < <(find "$SOURCE" -maxdepth 5 -type d \( -name 'train_multi_*' -o -name 'latent2d_cohort' \) 2>/dev/null | sort)

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    ECHO "  (no run dirs found — check SOURCE)"
else
    for d in "${RUN_DIRS[@]}"; do
        rel="${d#"$SOURCE/"}"                       # e.g. uv_vae/runs/run_20260802T...
        flat_name="${rel//\//__}"                   # flatten slashes to __ for readability
        dest_d="$DEST/runs/pure_internship/$flat_name"
        size=$(du -sh "$d" 2>/dev/null | cut -f1)
        ECHO "  [$size]  $rel"
        RUN rsync -a --progress "$d/" "$dest_d/"
    done
fi

# ── 4. editable install ───────────────────────────────────────────────────────
SEP
ECHO "4. Editable install of uv_vae package"
ECHO "   This lets scripts 'import uv_vae' from any depth — no parents[N] needed."

if $DRY_RUN; then
    ECHO "   (dry-run) would run: pip install -e $DEST/uv_vae"
else
    # detect env manager
    if command -v micromamba &>/dev/null; then
        ECHO "   using micromamba env: ${MAMBA_ENV:-uv_vae}"
        micromamba run -n "${MAMBA_ENV:-uv_vae}" pip install -e "$DEST/uv_vae" --quiet
    elif command -v conda &>/dev/null; then
        ECHO "   using conda — activate your env first, then run:"
        ECHO "   pip install -e $DEST/uv_vae"
        ECHO "   (skipping automatic install)"
    else
        ECHO "   no conda/micromamba found — run manually:"
        ECHO "   pip install -e $DEST/uv_vae"
    fi
fi

# ── 5. stub CLAUDE.md ─────────────────────────────────────────────────────────
SEP
ECHO "5. Writing stub CLAUDE.md"

CLAUDE_FILE="$DEST/CLAUDE.md"
if [[ -f "$CLAUDE_FILE" ]]; then
    ECHO "   (exists — skipping)"
else
    $DRY_RUN || cat > "$CLAUDE_FILE" <<'EOF'
# uv_vae_deployment

Post-PURE continuation workspace. The `uv_vae` package is installed as an editable
package into the `uv_vae` micromamba env — scripts import it directly with no
sys.path manipulation needed.

## Layout

- `uv_vae/`                   — the core package (`pip install -e .` already done)
- `runs/pure_internship/`     — checkpoints copied from the PURE internship runs
- `experiments/`              — new post-PURE experiment folders
  - `post_pure_analysis/`
  - `publication_figures/`

## Key paths (miletus)

- Parquets  : /data/lab/ppmseq_parquets/
- PURE archive : $HOME/pure-internship/  (read-only reference)
- GPU env   : micromamba env `uv_vae`

## Running scripts

No `parents[N]` path injection needed — just `import uv_vae` directly.
EOF
    ECHO "   written: $CLAUDE_FILE"
fi

# ── summary ───────────────────────────────────────────────────────────────────
SEP
ECHO "Done."
ECHO ""
ECHO "Deployment tree:"
if ! $DRY_RUN; then
    find "$DEST" -maxdepth 4 -not -path '*/\.*' -not -path '*/__pycache__/*' \
        | sort | sed "s|$DEST||" | sed 's|^|  |'
fi
ECHO ""
ECHO "Next: verify with  python -c 'import uv_vae; print(uv_vae.__file__)'"
