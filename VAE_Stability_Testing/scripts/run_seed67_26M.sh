#!/bin/bash
#SBATCH --job-name=vae-seed-67-26M
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/seed_67_26M_%j.log

set -euo pipefail

# Environment: micromamba 'uv_vae' on miletus, conda 'patrickg' on tosun.
# Guarded so one script works on either cluster -- `conda shell.bash hook` is not
# a micromamba subcommand, and on a node without conda it expands to nothing and
# the following `conda activate` aborts the job under `set -e`.
# Override with MAMBA_ENV or CONDA_ENV.
if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif command -v micromamba >/dev/null 2>&1; then
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "${MAMBA_ENV:-uv_vae}"
else
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-patrickg}"
fi

export UV_VAE_ROOT="$HOME/uv_vae"
export TQDM_DISABLE=1
cd ~/uv_vae

python VAE_Stability_Testing/scripts/vae_subsample_sweep.py \
    --parquet-path /cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet \
    --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
    --output-dir VAE_Stability_Testing/seed_sweep_extended/dataseed_67/rows_26000000 \
    --max-sample-rows 26000000 \
    --subsample-fractions "1.0" \
    --epochs 10 \
    --seed 42 \
    --data-seed 67 \
    --threads 8 \
    --non-interactive
