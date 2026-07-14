#!/bin/bash
#SBATCH --job-name=vae-sweep-10M
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/sweep_10M_%j.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/sweep_10M_%j.err


eval "$(conda shell.bash hook)"
conda activate patrickg

cd ~/uv_vae

python scripts/vae_subsample_sweep.py \
    --parquet-path /cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet \
    --test-set-path /cta/users/patrickgao765/test_set.parquet \
    --output-dir sweep_results_10M \
    --max-sample-rows 10000000 \
    --subsample-fractions "1.0,0.5,0.25,0.1,0.05,0.025,0.005" \
    --epochs 10 \
    --threads 32 \
    --non-interactive