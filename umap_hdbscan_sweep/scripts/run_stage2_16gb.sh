#!/bin/bash
# Run stage 2 (UMAP x HDBSCAN sweep) against the already-completed stage 0/1 outputs.
# Assumes no trainer is sharing the GPU -- full 16 GB allocated to the sweep.
#
# Usage (miletus, from repo root):
#   bash umap_hdbscan_sweep/run_stage2_16gb.sh
#
# Resume an interrupted run (cells with metrics.json are skipped automatically):
#   bash umap_hdbscan_sweep/run_stage2_16gb.sh
#
# Force a clean re-run:
#   NO_RESUME=1 bash umap_hdbscan_sweep/run_stage2_16gb.sh
#
# Dry-run (prints the 480-cell grid, fits nothing):
#   DRY_RUN=1 bash umap_hdbscan_sweep/run_stage2_16gb.sh
#
# Windows edit note: sed -i 's/\r$//' umap_hdbscan_sweep/run_stage2_16gb.sh

STAGE=2 \
RUN_ROOT="$HOME/pure-internship/uv_vae/runs/train_multi_20260802T192756Z" \
GPU_TOTAL_GB=16 \
TRAINER_GPU_GB=0 \
FIT_ROWS=5000000 \
bash "$(dirname "$0")/run_sweep.sh"
