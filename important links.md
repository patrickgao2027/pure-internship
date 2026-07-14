# Important Links

## DADE (Sabancı University)
- https://dade.sabanciuniv.edu/home

## Project Document
- https://docs.google.com/document/d/1JXTMFXF2g-cpLvOV7lBw6q_55UaUOJmhzX6kcNr8fhk/edit?tab=t.0#heading=h.nsi0jeoui1i

## Sabancı University HPC Tutorials
- https://su-hpc-tutorials.readthedocs.io/en/latest/

# Important Commands for HPC
htop
hostname
pwd
whoami
squeue -u $USER
scancel JOBID
scancel -u $USER
sinfo
srun   --account=adelab   --partition=genomics   --qos=adelab   --nodelist=cn10   --ntasks=1   --cpus-per-task=8   --mem=16G   --time=04:00:00   --pty bash

# Important Commands for UV_VAE
sed -i 's/\r$//' scripts/run_train_then_cluster.sh
fix the linux back spaces

PARQUET_PATH=/cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet \
TRAIN_USE_ALL=0 \
TRAIN_SAMPLE_ROWS=1000 \
TRAIN_SEED=42 \
bash scripts/run_train_then_cluster.sh \
  --cluster-row-filter "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"

# Stability Run
sbatch scripts/sweep_10M.sh # prints: Submitted batch job 12345
exit   # safe to close VS Code / SSH now

squeue -u patrickgao765          # see if job is running/pending/done
sacct -j 12345 --format=JobID,State,Elapsed,MaxRSS   # after it finishes — runtime and memory used

tail -f ~/uv_vae/sweep_10M_12345.log    # live-follow while running
or after it finishes:
cat ~/uv_vae/sweep_10M_12345.log

# Important Commands for Conda
conda activate patrickg
conda list
which python
python --version
conda env list
