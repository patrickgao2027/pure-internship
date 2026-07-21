#!/bin/bash
#SBATCH --job-name=epoch-vs-52M
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/epoch_vs_52M_%j.log

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate patrickg

export UV_VAE_ROOT="$HOME/uv_vae"
export TQDM_DISABLE=1

cd "$UV_VAE_ROOT"

python VAE_Stability_Testing/scripts/epoch_sweep_vs_52M.py \
    --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
    --ref-json VAE_Stability_Testing/sweep_results_full/sweep_results.json \
    --epoch-sweep-dir VAE_Stability_Testing/epoch_sweep \
    --output-json VAE_Stability_Testing/epoch_sweep/vs_52M_results.json \
    --batch-size 4096

echo "Done."