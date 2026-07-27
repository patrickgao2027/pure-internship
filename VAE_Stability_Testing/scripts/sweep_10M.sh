#!/bin/bash
#SBATCH --job-name=vae-sweep-10M
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/sweep_10M_%j.log


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

cd ~/uv_vae

python VAE_Stability_Testing/scripts/vae_subsample_sweep.py \
    --parquet-path /cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet \
    --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
    --output-dir VAE_Stability_Testing/sweep_results_10M \
    --max-sample-rows 10000000 \
    --subsample-fractions "1.0,0.5,0.25,0.1,0.05,0.025,0.005" \
    --epochs 10 \
    --threads 32 \
    --non-interactive