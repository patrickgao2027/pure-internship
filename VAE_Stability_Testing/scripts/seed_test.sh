#!/bin/bash
#SBATCH --job-name=vae-seed-test
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/seed_test_%j.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/seed_test_%j.err

eval "$(conda shell.bash hook)"
conda activate patrickg

export UV_VAE_ROOT="$HOME/uv_vae"

cd ~/uv_vae

PARQUET=/cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet
TEST_SET=/cta/users/patrickgao765/uv_vae/test_set.parquet
FRACTIONS="1.0,0.75"

for SEED in 7 13 67 99; do
    echo "========================================="
    echo "Running seed=$SEED"
    echo "========================================="
    python VAE_Stability_Testing/scripts/vae_subsample_sweep.py \
        --parquet-path $PARQUET \
        --test-set-path $TEST_SET \
        --output-dir VAE_Stability_Testing/sweep_seed${SEED} \
        --max-sample-rows 1000000 \
        --subsample-fractions "$FRACTIONS" \
        --seed $SEED \
        --threads 32 \
        --non-interactive
done

echo "All seeds done"
