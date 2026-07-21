"""Compute Procrustes and CKA for all seed_sweep_extended checkpoints vs the 52M reference.

Usage (on HPC):
    export UV_VAE_ROOT=~/uv_vae
    cd ~/uv_vae
    python VAE_Stability_Testing/scripts/seed_sweep_vs_52M.py \
        --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
        --ref-json VAE_Stability_Testing/sweep_results_full/sweep_results.json \
        --seed-sweep-dir VAE_Stability_Testing/seed_sweep_extended \
        --output-json VAE_Stability_Testing/seed_sweep_extended/vs_52M_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

UV_VAE_ROOT = Path(os.environ.get("UV_VAE_ROOT", Path.home() / "uv_vae")).resolve()
if str(UV_VAE_ROOT) not in sys.path:
    sys.path.insert(0, str(UV_VAE_ROOT))

from scipy.spatial import procrustes
from uv_vae.inference import LatentInference
from uv_vae.preprocess import transform_frame


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    dx = np.linalg.norm(X.T @ X, "fro")
    dy = np.linalg.norm(Y.T @ Y, "fro")
    return float(num / (dx * dy)) if dx > 1e-10 and dy > 1e-10 else 0.0


def encode(checkpoint_path, test_df, batch_size=4096):
    inf = LatentInference.from_checkpoint(str(checkpoint_path), device="auto")
    return inf, inf.encode_frame(test_df.select(inf.feature_names), batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set-path", required=True)
    parser.add_argument("--ref-json", required=True,
                        help="sweep_results.json from the full 52M sweep")
    parser.add_argument("--seed-sweep-dir", required=True,
                        help="Root of seed_sweep_extended (contains dataseed_* dirs)")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    log("Loading test set")
    test_df = pl.read_parquet(args.test_set_path)

    ref_json = json.loads(Path(args.ref_json).read_text())
    ref_entry = max(ref_json, key=lambda e: e["sample_rows"])
    ref_dir = Path(ref_entry["run_dir"])
    ref_cp = ref_dir / "model.pt"
    if not ref_cp.exists():
        ref_base = Path(args.ref_json).resolve().parent
        for depth in (2, 3):
            parts = ref_dir.parts
            if len(parts) >= depth:
                candidate = ref_base / Path(*parts[-depth:]) / "model.pt"
                if candidate.exists():
                    ref_cp = candidate
                    break
    if not ref_cp.exists():
        log(f"ERROR: 52M reference checkpoint not found: {ref_dir}")
        return 1

    log(f"Encoding test set through 52M reference ({ref_entry['sample_rows']:,} rows)")
    ref_inf, ref_emb = encode(ref_cp, test_df, args.batch_size)
    log(f"  ref_emb shape: {ref_emb.shape}")

    sweep_root = Path(args.seed_sweep_dir)
    results = []

    for seed_dir in sorted(sweep_root.glob("dataseed_*")):
        data_seed = int(seed_dir.name.split("_")[1])

        for rows_dir in sorted(seed_dir.glob("rows_*")):
            row_count = int(rows_dir.name.split("_")[1])
            sweep_json = rows_dir / "sweep_results.json"
            if not sweep_json.exists():
                log(f"  SKIP dataseed={data_seed} rows={row_count:,} — no sweep_results.json")
                continue

            sweep = json.loads(sweep_json.read_text())
            if not sweep:
                log(f"  SKIP dataseed={data_seed} rows={row_count:,} — empty results")
                continue

            entry = sweep[0]
            run_dir = Path(entry["run_dir"])
            cp = run_dir / "model.pt"
            if not cp.exists():
                cp = rows_dir / "run_100pct" / run_dir.name / "model.pt"
            if not cp.exists():
                for candidate in rows_dir.rglob("model.pt"):
                    cp = candidate
                    break
            if not cp.exists():
                log(f"  SKIP dataseed={data_seed} rows={row_count:,} — checkpoint not found")
                continue

            log(f"  Encoding dataseed={data_seed} rows={row_count:,}")
            _, emb = encode(cp, test_df, args.batch_size)

            _, _, proc = procrustes(ref_emb, emb)
            cka = linear_cka(ref_emb, emb)

            results.append({
                "data_seed": data_seed,
                "rows": row_count,
                "procrustes_disparity": round(float(proc), 6),
                "linear_cka": round(cka, 6),
                "val_total_loss": entry.get("val_total_loss"),
                "trustworthiness": entry.get("trustworthiness"),
                "latent_collapse": entry.get("latent_collapse"),
            })
            log(f"    proc={proc:.4f}  cka={cka:.4f}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(results, indent=2))
    log(f"\nSaved {len(results)} results to {args.output_json}")

    print(f"\n{'═' * 78}")
    print(f"  Seed sweep extended — Procrustes & CKA vs 52M reference")
    print(f"{'═' * 78}")
    print(f"{'Seed':>6}  {'Rows':>12}  {'Procrustes':>11}  {'CKA':>8}  {'Val Loss':>9}")
    print("─" * 78)
    for r in sorted(results, key=lambda x: (x["data_seed"], x["rows"])):
        vl = f"{r['val_total_loss']:.4f}" if r["val_total_loss"] else "  n/a  "
        print(f"{r['data_seed']:>6}  {r['rows']:>12,}  "
              f"{r['procrustes_disparity']:>11.4f}  {r['linear_cka']:>8.4f}  {vl:>9}")
    print("─" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
