#!/bin/bash
#SBATCH --job-name=vae-compare
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/compare_%j.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/compare_%j.err

eval "$(conda shell.bash hook)"
conda activate patrickg

export UV_VAE_ROOT="$HOME/uv_vae"

cd ~/uv_vae

python scripts/compare_sweeps.py \
    --test-set-path /cta/users/patrickgao765/test_set.parquet \
    --sweep-1m-json sweep_results/sweep_results.json \
    --sweep-10m-json sweep_results_10M/sweep_results.json \
    --output-dir comparison_results
