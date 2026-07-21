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

eval "$(conda shell.bash hook)"
conda activate patrickg

export UV_VAE_ROOT="$HOME/uv_vae"

cd ~/uv_vae

python VAE_Stability_Testing/scripts/compare_sweeps.py \
    --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
    --sweep-1m-json VAE_Stability_Testing/sweep_results_1M/sweep_results.json \
    --sweep-10m-json VAE_Stability_Testing/sweep_results_10M/sweep_results.json \
    --output-dir VAE_Stability_Testing/comparison_results
