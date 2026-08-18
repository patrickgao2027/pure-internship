#!/bin/bash
#SBATCH --job-name=rq-vae-train
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/rq_vae_train_%A_%a.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/logs/rq_vae_train_%A_%a.err
#
# STAGE 1 of sweep 2 (rq filter sweep): retrain one VAE per rq threshold.
#
# Sweeps 1/3/4 reuse the pretrained sweep_results_full ladder and train nothing. Sweep 2
# is different: filtering on rq changes the input distribution, so the VAE must be refit
# on the rows that survive the filter. Each array task trains one threshold on ALL rows
# passing "<base filter> AND rq < threshold" (training clamps the row ceiling down to
# whatever actually passes).
#
# Hyperparameters match the ladder (10 epochs, latent 16, hidden 256,128, kl 0.05,
# seed 42) so the rq VAEs stay comparable to the 100pct reference.
#
# Reference cost: the 100pct ladder VAE trained on 50,591,627 rows on CPU in ~4 h.
# rq subsets are strictly smaller, so 24 h is generous headroom.
#
# Windows edit note:
#   sed -i 's/\r$//' scripts/run_rq_vae_train.sh
# Submit (stage 1), then chain the clustering array after it:
#   TRAIN_JOB=$(sbatch --parsable scripts/run_rq_vae_train.sh)
#   ARRAY_JOB=$(sbatch --parsable --dependency=afterany:${TRAIN_JOB} scripts/run_clustering_regime.sh)
#   sbatch --dependency=afterany:${ARRAY_JOB} scripts/run_clustering_regime_aggregate.sh

set -euo pipefail

export UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
export TQDM_DISABLE=1
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

# Must match SOURCE_PARQUET in run_clustering_regime.sh -- the rq VAEs have to be fit on
# the same data the rq cells will cluster.
SOURCE_PARQUET="${SOURCE_PARQUET:-/cta/users/patrickgao765/parquet_files/test7M.featuremap.parquet}"
FEATURE_SPEC="${FEATURE_SPEC:-$UV_VAE_DIR/ml_features.json}"
RQ_VAE_ROOT="${RQ_VAE_ROOT:-/cta/users/patrickgao765/uv_vae/clustering_regime/rq_vaes}"

BASE_FILTER="st = 'MIXED' AND et = 'MIXED' AND FILT = 1"
SEED="${SEED:-42}"
THREADS="${SLURM_CPUS_PER_TASK:-32}"

# Must stay identical to RQ_THRESHOLDS in run_clustering_regime.sh -- that script looks
# up the checkpoint each threshold produced here.
RQ_THRESHOLDS=(0.025 0.05 0.075 0.1 0.15)

TASK="${SLURM_ARRAY_TASK_ID:-0}"
RQ="${RQ_THRESHOLDS[$TASK]}"

if [ ! -f "$SOURCE_PARQUET" ]; then
    echo "ERROR: source parquet not found: $SOURCE_PARQUET" >&2
    exit 1
fi
mkdir -p "$RQ_VAE_ROOT"

echo "=== task $TASK : training VAE on rows with rq < $RQ ==="

python scripts/train_rq_vae.py \
    --parquet-path "$SOURCE_PARQUET" \
    --feature-spec-path "$FEATURE_SPEC" \
    --output-root "$RQ_VAE_ROOT" \
    --rq-threshold "$RQ" \
    --row-filter "$BASE_FILTER" \
    --seed "$SEED" \
    --threads "$THREADS"

echo "=== task $TASK complete -> $RQ_VAE_ROOT/rq_$RQ ==="
