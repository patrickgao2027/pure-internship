#!/bin/bash
#SBATCH --job-name=seed-vs-52M
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/seed_vs_52M_%j.log

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

python VAE_Stability_Testing/scripts/seed_sweep_vs_52M.py \
    --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
    --ref-json VAE_Stability_Testing/sweep_results_full/sweep_results.json \
    --seed-sweep-dir VAE_Stability_Testing/seed_sweep_extended \
    --output-json VAE_Stability_Testing/seed_sweep_extended/vs_52M_results.json
