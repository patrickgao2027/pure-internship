#!/bin/bash
#SBATCH --job-name=vae-seed-vs-10M
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/seed_vs_10M_%j.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/seed_vs_10M_%j.err

eval "$(conda shell.bash hook)"
conda activate patrickg

export UV_VAE_ROOT="$HOME/uv_vae"

cd ~/uv_vae

python VAE_Stability_Testing/scripts/seed_vs_10M.py \
    --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
    --ref-10m-json VAE_Stability_Testing/sweep_results_10M/sweep_results.json \
    --seed-dirs VAE_Stability_Testing/sweep_seed_results/sweep_seed7,VAE_Stability_Testing/sweep_seed_results/sweep_seed13,VAE_Stability_Testing/sweep_seed_results/sweep_seed67,VAE_Stability_Testing/sweep_seed_results/sweep_seed99 \
    --seed-42-json VAE_Stability_Testing/sweep_results/sweep_results.json \
    --output-dir VAE_Stability_Testing/seed_comparison
