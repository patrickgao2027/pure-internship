#!/bin/bash
#SBATCH --job-name=vae-sweep-c1
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/sweep_c1_%j.log

eval "$(conda shell.bash hook)"
conda activate patrickg

export UV_VAE_ROOT="$HOME/uv_vae"
export TQDM_DISABLE=1
cd ~/uv_vae

python VAE_Stability_Testing/scripts/vae_subsample_sweep.py \
    --parquet-path /cta/users/patrickgao765/parquet_files/wtR12-c1-ppm0018.featuremap.parquet \
    --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
    --output-dir VAE_Stability_Testing/sweep_results_c1 \
    --max-sample-rows 48478197 \
    --subsample-fractions "1.0,0.5,0.25,0.1,0.05,0.025,0.01" \
    --epochs 10 \
    --threads 32 \
    --non-interactive
