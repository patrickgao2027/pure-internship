#!/bin/bash
#SBATCH --job-name=vae-compare
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/compare_%j.log

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

cd ~/uv_vae

python VAE_Stability_Testing/scripts/compare_sweeps.py \
    --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
    --sweep-1m-json VAE_Stability_Testing/sweep_results_1M/sweep_results.json \
    --sweep-10m-json VAE_Stability_Testing/sweep_results_10M/sweep_results.json \
    --output-dir VAE_Stability_Testing/comparison_results
