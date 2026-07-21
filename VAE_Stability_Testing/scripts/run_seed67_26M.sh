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

eval "$(conda shell.bash hook)"
conda activate patrickg

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
