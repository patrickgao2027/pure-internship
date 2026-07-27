#!/bin/bash
#SBATCH --job-name=vae-sweep
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/sweep_%j.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/sweep_%j.err
#
# Sweep over batch size, learning rate, and warmup steps.
#
# Two phases run sequentially in a single SLURM job:
#
#   Phase 1 — Batch / LR grid (non-streaming, 5M rows, 8 epochs, fast)
#     Loads a 5M-row subsample into RAM. warmup is 0 (streaming-only).
#     Tests: does a larger batch + scaled lr converge to a lower val loss
#     and/or more active units in fewer wall-clock seconds?
#
#   Phase 2 — Warmup effect on full stream (streaming, all ~99M rows, 3 epochs)
#     Streams the full combined parquet. Tests whether warmup_steps > 0
#     prevents KL collapse at init (active_units after epoch 1 vs epoch 3).
#     batch=32768, lr=1e-3 baseline; one run without warmup, one with warmup=200.
#
# Override the combined parquet path:
#   COMBINED=/path/to/combined.parquet sbatch scripts/run_batch_lr_sweep.sh
#
#   sed -i 's/\r$//' Batch_Size_Learning_Rate_Testing/scripts/run_batch_lr_sweep.sh

set -euo pipefail

UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
EARLY_STOPPING_DIR="${EARLY_STOPPING_DIR:-$HOME/Early_Stopping_Tests}"
COMBINED="${COMBINED:-/cta/users/patrickgao765/uv_vae/train_earlystop_1839691/combined.featuremap.parquet}"
SWEEP_ROOT="${SWEEP_ROOT:-/cta/users/patrickgao765/uv_vae/sweep_${SLURM_JOB_ID:-manual}}"
ROW_FILTER="${ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
THREADS="${SLURM_CPUS_PER_TASK:-32}"
SEED=42

# Phase 1 — non-streaming, loads into RAM
P1_ROWS=5000000
P1_EPOCHS=8

# Phase 2 — streaming full dataset, few epochs only (each epoch is ~10-12 min)
P2_EPOCHS=3

mkdir -p "$SWEEP_ROOT"

eval "$(conda shell.bash hook)"
conda activate patrickg
cd "$UV_VAE_DIR"

export PYTHONHASHSEED="$SEED"

fmt_seconds() { printf '%dh %02dm %02ds' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }

echo "====================================================================="
echo "Sweep root    : $SWEEP_ROOT"
echo "Parquet       : $COMBINED"
echo "Phase 1 rows  : $P1_ROWS  ($P1_EPOCHS epochs, non-streaming)"
echo "Phase 2 epochs: $P2_EPOCHS (streaming all rows, warmup only)"
echo "====================================================================="
echo ""

###############################################################################
# Helper: run one non-streaming config (5M-row RAM load, warmup=0 N/A)
###############################################################################
run_p1() {
    local name="$1" batch="$2" lr="$3"
    local out_dir="$SWEEP_ROOT/$name"
    local t0=$SECONDS
    echo "[$(date '+%T')] P1 START  $name  batch=$batch  lr=$lr"
    python "$EARLY_STOPPING_DIR/Python Files/train_with_early_stopping.py" \
        --parquet-path "$COMBINED" \
        --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
        --output-dir "$out_dir" \
        --row-filter "$ROW_FILTER" \
        --sample-rows "$P1_ROWS" \
        --epochs "$P1_EPOCHS" \
        --patience 0 \
        --batch-size "$batch" \
        --latent-dim 16 \
        --hidden-dims 256,128 \
        --learning-rate "$lr" \
        --kl-weight 0.05 \
        --train-fraction 0.8 \
        --input-dropout 0.1 \
        --hidden-dropout 0.2 \
        --seed "$SEED" \
        --threads "$THREADS" \
        > "$SWEEP_ROOT/${name}_result.json" 2>&1 || true
    echo "[$(date '+%T')] P1 END    $name  ($(fmt_seconds $((SECONDS - t0))))"
}

###############################################################################
# Helper: run one streaming config (warmup test, 3 epochs, full dataset)
###############################################################################
run_p2() {
    local name="$1" warmup="$2"
    local out_dir="$SWEEP_ROOT/$name"
    local t0=$SECONDS
    echo "[$(date '+%T')] P2 START  $name  warmup=$warmup  (streaming, $P2_EPOCHS epochs)"
    python "$EARLY_STOPPING_DIR/Python Files/train_with_early_stopping.py" \
        --parquet-path "$COMBINED" \
        --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
        --output-dir "$out_dir" \
        --row-filter "$ROW_FILTER" \
        --streaming \
        --epochs "$P2_EPOCHS" \
        --patience 0 \
        --batch-size 32768 \
        --latent-dim 16 \
        --hidden-dims 256,128 \
        --learning-rate 1e-3 \
        --kl-weight 0.05 \
        --train-fraction 0.8 \
        --input-dropout 0.1 \
        --hidden-dropout 0.2 \
        --seed "$SEED" \
        --threads "$THREADS" \
        --warmup-steps "$warmup" \
        > "$SWEEP_ROOT/${name}_result.json" 2>&1 || true
    echo "[$(date '+%T')] P2 END    $name  ($(fmt_seconds $((SECONDS - t0))))"
}

###############################################################################
# Phase 1: batch size × learning rate grid (non-streaming, warmup=0)
#
#   Batch rows   LR       Rationale
#   32 768       1e-3     current default / baseline
#   131 072      1e-3     4x batch, same lr (isolate batch effect)
#   131 072      2e-3     4x batch + sqrt-scaled lr (McCandlish recommendation)
#   262 144      1e-3     8x batch, same lr
#   262 144      2.8e-3   8x batch + sqrt-scaled lr  (2.83 ≈ sqrt(8) × 1e-3)
###############################################################################
echo "===== PHASE 1: batch / LR grid (non-streaming, $P1_ROWS rows) ====="
run_p1 "p1_b32k_lr1e3"     32768  1e-3
run_p1 "p1_b131k_lr1e3"   131072  1e-3
run_p1 "p1_b131k_lr2e3"   131072  2e-3
run_p1 "p1_b262k_lr1e3"   262144  1e-3
run_p1 "p1_b262k_lr2p8e3" 262144  2.83e-3

###############################################################################
# Phase 2: warmup effect on full dataset (streaming)
#
#   batch=32768, lr=1e-3, dropout=0.2
#   run_no_warmup — KL may spike at epoch 1 if lr is too aggressive
#   run_warmup200 — LinearLR ramp over 200 steps, should stabilise AU at epoch 1
###############################################################################
echo ""
echo "===== PHASE 2: warmup test (streaming, full dataset, $P2_EPOCHS epochs) ====="
run_p2 "p2_warmup0"   0
run_p2 "p2_warmup200" 200

###############################################################################
# Summary: parse training_report.json from each run dir and write CSV
###############################################################################
echo ""
echo "[$(date '+%T')] Generating summary CSV ..."
python - "$SWEEP_ROOT" <<'PY'
import json, sys, csv
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for result_file in sorted(root.glob("*_result.json")):
    name = result_file.stem.replace("_result", "")
    try:
        text = result_file.read_text()
        result = None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{") and "run_dir" in line:
                result = json.loads(line)
                break
        if result is None:
            rows.append({"config": name, "status": "ERROR: no JSON", "error": text[-400:]})
            continue

        run_dir = Path(result["run_dir"])
        report = json.loads((run_dir / "training_report.json").read_text())
        history = report["history"]
        best_val = min(e["val_total_loss"] for e in history)
        last = history[-1]
        # per-epoch summary: active_units at epoch 1 and last epoch
        au_e1 = history[0]["active_units"] if history else "?"
        rows.append({
            "config": name,
            "status": "ok",
            "epochs_run": len(history),
            "best_val_loss": round(best_val, 6),
            "final_val_loss": round(last["val_total_loss"], 6),
            "final_train_loss": round(last["train_total_loss"], 6),
            "au_epoch1": au_e1,
            "au_final": last["active_units"],
            "kl_total_final": round(last.get("kl_total", 0), 4),
        })
    except Exception as exc:
        rows.append({"config": name, "status": f"ERROR: {exc}"})

if not rows:
    print("No result files found.")
    sys.exit(0)

all_keys = ["config", "status", "epochs_run", "best_val_loss", "final_val_loss",
            "final_train_loss", "au_epoch1", "au_final", "kl_total_final", "error"]
fieldnames = [k for k in all_keys if any(k in r for r in rows)]

out = root / "sweep_summary.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

print(f"\nWrote {out}\n")
header = f"{'config':<26}  {'epochs':>6}  {'best_val':>10}  {'au_e1':>6}  {'au_fin':>6}  {'kl_fin':>8}"
print(header)
print("-" * len(header))
for r in rows:
    if r.get("status") == "ok":
        print(f"{r['config']:<26}  {r['epochs_run']:>6}  {r['best_val_loss']:>10.6f}  "
              f"{r['au_epoch1']:>6}  {r['au_final']:>6}  {r['kl_total_final']:>8.4f}")
    else:
        print(f"{r['config']:<26}  {r.get('status','?')}")
PY

echo ""
echo "Sweep complete. Results: $SWEEP_ROOT/sweep_summary.csv"
echo "sacct -j ${SLURM_JOB_ID:-<jobid>} --format=JobID,State,Elapsed,MaxRSS"
