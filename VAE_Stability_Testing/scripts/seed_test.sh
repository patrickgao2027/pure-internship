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

# Environment: micromamba 'uv_vae' on miletus, conda 'patrickg' on tosun.
# Guarded so one script works on either cluster -- `conda shell.bash hook` is not
# a micromamba subcommand, and on a node without conda it expands to nothing and
# the following `conda activate` aborts the job under `set -e`.
# Override with MAMBA_ENV or CONDA_ENV.
if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif command -v micromamba >/dev/null 2>&1; then
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "${MAMBA_ENV:-uv_vae}"
else
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-patrickg}"
fi

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
