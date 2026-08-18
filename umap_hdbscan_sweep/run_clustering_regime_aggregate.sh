#!/bin/bash
#SBATCH --job-name=clust-regime-agg
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/clust_regime_agg_%j.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/logs/clust_regime_agg_%j.err
#
# Aggregation step for the clustering testing regime. Computes the cross-cell
# comparisons (subsample-vs-all, D-vs-A-all), regime_summary.json, and plots.
#
# Submit it to run automatically after the array job:
#   ARRAY_JOB=$(sbatch --parsable scripts/run_clustering_regime.sh)
#   sbatch --dependency=afterany:${ARRAY_JOB} scripts/run_clustering_regime_aggregate.sh
#
# Windows edit note:  sed -i 's/\r$//' scripts/run_clustering_regime_aggregate.sh

set -euo pipefail

export UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
export CUDA_VISIBLE_DEVICES=""
cd "$UV_VAE_DIR"

if [ -n "${CONDA_ENV:-}" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

mkdir -p /cta/users/patrickgao765/uv_vae/logs

BASE_OUTPUT="${BASE_OUTPUT:-/cta/users/patrickgao765/uv_vae/clustering_regime/run1}"

python scripts/aggregate_clustering_regime.py --output-root "$BASE_OUTPUT"

echo "Aggregation complete. Summary: $BASE_OUTPUT/regime_summary.json"
