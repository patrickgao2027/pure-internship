#!/bin/bash
#SBATCH --job-name=rq-vs-52M
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/rq_vs_52M_%j.log

set -euo pipefail

export UV_VAE_ROOT="$HOME/uv_vae"
export TQDM_DISABLE=1
cd ~/uv_vae

source .venv/bin/activate

python VAE_Stability_Testing/scripts/rq_sweep_vs_52M.py \
    --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
    --ref-json VAE_Stability_Testing/sweep_results_full/sweep_results.json \
    --rq-sweep-dir /cta/users/patrickgao765/uv_vae/VAE_Stability_Testing/rq_sweep \
    --parquet-path /cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet \
    --output-json /cta/users/patrickgao765/uv_vae/VAE_Stability_Testing/rq_sweep/vs_52M_results.json
