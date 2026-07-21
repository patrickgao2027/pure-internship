#!/bin/bash
#SBATCH --job-name=vae-compare-sweeps
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/logs/compare_sweeps_%j.log

# Run this AFTER both epoch and seed sweep array jobs have completed.
# Submit with dependency: sbatch --dependency=afterok:<epoch_jobid>:<seed_jobid> VAE_Stability_Testing/scripts/run_compare_sweeps.sh

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate patrickg

export UV_VAE_ROOT="$HOME/uv_vae"
cd ~/uv_vae

# ── Update this path to point to the 52M reference model.pt ───
# Find it with: find VAE_Stability_Testing/sweep_results_full -name model.pt
REF_CHECKPOINT="VAE_Stability_Testing/sweep_results_full/run_100pct/run_20260713T120724Z/model.pt"

python VAE_Stability_Testing/scripts/compare_sweeps_vs_52M.py \
    --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
    --ref-checkpoint "$REF_CHECKPOINT" \
    --epoch-sweep-dir /cta/users/patrickgao765/uv_vae/VAE_Stability_Testing/epoch_sweep \
    --seed-sweep-dir /cta/users/patrickgao765/uv_vae/VAE_Stability_Testing/seed_sweep_extended \
    --output-dir VAE_Stability_Testing/combined_comparison
