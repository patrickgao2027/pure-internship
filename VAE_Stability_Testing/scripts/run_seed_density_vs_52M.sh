#!/bin/bash
#SBATCH --job-name=seed-density-52M
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/seed_density_52M_%j.log

# Compute Procrustes + CKA vs 52M reference for the density sweep checkpoints.
# Submit after run_seed_density_sweep.sh completes:
#   SWEEP_JOB=$(sbatch --parsable run_seed_density_sweep.sh)
#   sbatch --dependency=afterok:$SWEEP_JOB run_seed_density_vs_52M.sh

set -euo pipefail

export UV_VAE_ROOT="$HOME/uv_vae"
export TQDM_DISABLE=1
cd ~/uv_vae

source .venv/bin/activate

python VAE_Stability_Testing/scripts/seed_sweep_vs_52M.py \
    --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
    --ref-json VAE_Stability_Testing/sweep_results_full/sweep_results.json \
    --seed-sweep-dir /cta/users/patrickgao765/uv_vae/VAE_Stability_Testing/seed_density_sweep \
    --output-json /cta/users/patrickgao765/uv_vae/VAE_Stability_Testing/seed_density_sweep/vs_52M_results.json
