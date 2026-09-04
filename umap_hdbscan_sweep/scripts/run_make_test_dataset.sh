#!/bin/bash
#SBATCH --job-name=make-test-dataset
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/make_test_dataset_%j.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/logs/make_test_dataset_%j.err
#
# STAGE 0: build the 7M-row test dataset the whole regime runs on.
#
# Reservoir row sample of the filtered parquet streamed straight to parquet,
# single-threaded for determinism, on seed 99. No dedup -- the regime dedups per cell.
# The result must exercise the whole pipeline (dedup -> VAE -> UMAP -> HDBSCAN ->
# metrics -> SigProfiler), so every column is carried through and the script fails
# loudly if the identity/context columns are absent.
#
# Windows edit note:
#   sed -i 's/\r$//' scripts/run_make_test_dataset.sh
# Submit:
#   sbatch scripts/run_make_test_dataset.sh

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

SOURCE_PARQUET="${SOURCE_PARQUET:-/cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet}"
TEST_PARQUET="${TEST_PARQUET:-/cta/users/patrickgao765/parquet_files/test7M.featuremap.parquet}"
SAMPLE_ROWS="${SAMPLE_ROWS:-7000000}"
TEST_SEED="${TEST_SEED:-99}"

if [ -s "$TEST_PARQUET" ]; then
    echo "Test dataset already exists: $TEST_PARQUET"
    exit 0
fi

python scripts/make_test_dataset.py \
    --parquet-path "$SOURCE_PARQUET" \
    --output-path "$TEST_PARQUET" \
    --sample-rows "$SAMPLE_ROWS" \
    --seed "$TEST_SEED"

echo "Test dataset ready: $TEST_PARQUET"
