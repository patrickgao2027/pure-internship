"""Compute Procrustes, CKA, trustworthiness for rq_sweep checkpoints vs 52M reference.

Usage (on HPC):
    export UV_VAE_ROOT=~/uv_vae
    cd ~/uv_vae
    python VAE_Stability_Testing/scripts/rq_sweep_vs_52M.py \
        --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
        --ref-json VAE_Stability_Testing/sweep_results_full/sweep_results.json \
        --rq-sweep-dir VAE_Stability_Testing/rq_sweep \
        --parquet-path /cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet \
        --output-json VAE_Stability_Testing/rq_sweep/vs_52M_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

UV_VAE_ROOT = Path(os.environ.get("UV_VAE_ROOT", Path.home() / "uv_vae")).resolve()
if str(UV_VAE_ROOT) not in sys.path:
    sys.path.insert(0, str(UV_VAE_ROOT))

from scipy.spatial import procrustes
from sklearn.manifold import trustworthiness

from uv_vae.inference import LatentInference
from uv_vae.preprocess import transform_frame


RQ_THRESHOLDS = [0.025, 0.05, 0.075, 0.1, 0.15]
BASE_FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    dx = np.linalg.norm(X.T @ X, "fro")
    dy = np.linalg.norm(Y.T @ Y, "fro")
    return float(num / (dx * dy)) if dx > 1e-10 and dy > 1e-10 else 0.0


def compute_trust(X: np.ndarray, Z: np.ndarray, n_neighbors: int = 10,
                  max_rows: int = 5000, seed: int = 0) -> float:
    n = X.shape[0]
    if n > max_rows:
        idx = np.random.default_rng(seed).choice(n, max_rows, replace=False)
        X, Z = X[idx], Z[idx]
    return float(trustworthiness(X, Z, n_neighbors=n_neighbors))


def encode(checkpoint_path: Path, test_df: pl.DataFrame,
           batch_size: int = 4096) -> tuple[LatentInference, np.ndarray]:
    inf = LatentInference.from_checkpoint(str(checkpoint_path), device="auto")
    emb = inf.encode_frame(test_df.select(inf.feature_names), batch_size=batch_size)
    return inf, emb


def get_input_matrix(inf: LatentInference, test_df: pl.DataFrame) -> np.ndarray:
    _, t = transform_frame(
        frame=test_df,
        categorical_specs=inf.categorical_specs,
        numeric_specs=inf.numeric_specs,
        numeric_means=inf.numeric_means,
        numeric_stds=inf.numeric_stds,
    )
    return t.numpy()


def count_rows(parquet_path: str, row_filter: str, threads: int = 8) -> int:
    conn = duckdb.connect()
    conn.execute(f"SET threads = {threads}")
    result = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquet_path}') WHERE {row_filter}"
    ).fetchone()
    return result[0]


def find_checkpoint(rq_dir: Path) -> Path | None:
    """Find model.pt inside rq_dir (looks inside run_100pct/run_*/model.pt)."""
    for cp in rq_dir.rglob("model.pt"):
        return cp
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set-path", required=True)
    parser.add_argument("--ref-json", required=True,
                        help="sweep_results.json from the full 52M sweep (no rq filter)")
    parser.add_argument("--rq-sweep-dir", required=True,
                        help="Root of rq_sweep (contains rq0025/, rq005/, etc.)")
    parser.add_argument("--parquet-path", required=True,
                        help="Parquet file (to count rows per threshold)")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    log("Loading test set")
    test_df = pl.read_parquet(args.test_set_path)

    # Load 52M reference checkpoint
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
    input_matrix = get_input_matrix(ref_inf, test_df)
    log(f"  ref_emb shape: {ref_emb.shape}")

    sweep_root = Path(args.rq_sweep_dir)
    results = []

    for rq in RQ_THRESHOLDS:
        rq_label = "rq" + str(rq).replace(".", "")
        rq_dir = sweep_root / rq_label
        if not rq_dir.exists():
            log(f"SKIP rq<{rq} — directory not found: {rq_dir}")
            continue

        cp = find_checkpoint(rq_dir)
        if cp is None:
            log(f"SKIP rq<{rq} — no model.pt found in {rq_dir}")
            continue

        row_filter = f"{BASE_FILTER} AND rq < {rq}"
        log(f"Counting rows for rq < {rq}")
        n_rows = count_rows(args.parquet_path, row_filter, args.threads)
        log(f"  {n_rows:,} rows pass filter")

        log(f"Encoding rq < {rq} ({n_rows:,} rows)")
        _, emb = encode(cp, test_df, args.batch_size)

        _, _, proc = procrustes(ref_emb, emb)
        cka = linear_cka(ref_emb, emb)
        trust = compute_trust(input_matrix, emb)
        collapse = float(np.std(emb, axis=0).mean())

        # pull val loss from sweep_results.json if it exists
        sweep_json = rq_dir / "sweep_results.json"
        val_loss = None
        val_kl = None
        if sweep_json.exists():
            sweep = json.loads(sweep_json.read_text())
            if sweep:
                val_loss = sweep[0].get("val_total_loss")
                val_kl = sweep[0].get("val_kl_loss")

        results.append({
            "rq_threshold": rq,
            "row_filter": row_filter,
            "n_rows": n_rows,
            "procrustes_disparity": round(float(proc), 6),
            "linear_cka": round(cka, 6),
            "trustworthiness": round(trust, 6),
            "latent_collapse": round(collapse, 6),
            "val_total_loss": val_loss,
            "val_kl_loss": val_kl,
        })
        log(f"  proc={proc:.4f}  cka={cka:.4f}  trust={trust:.4f}  collapse={collapse:.4f}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(results, indent=2))
    log(f"\nSaved {len(results)} results to {args.output_json}")

    print(f"\n{'═' * 80}")
    print(f"  rq threshold sweep — vs 52M reference (no rq filter)")
    print(f"{'═' * 80}")
    print(f"{'rq<':>6}  {'N rows':>12}  {'Procrustes':>11}  {'CKA':>8}  "
          f"{'Trust':>8}  {'Collapse':>9}  {'Val Loss':>9}")
    print("─" * 80)
    for r in sorted(results, key=lambda x: x["rq_threshold"]):
        vl = f"{r['val_total_loss']:.4f}" if r["val_total_loss"] else "  n/a  "
        print(f"{r['rq_threshold']:>6.3f}  {r['n_rows']:>12,}  "
              f"{r['procrustes_disparity']:>11.4f}  {r['linear_cka']:>8.4f}  "
              f"{r['trustworthiness']:>8.4f}  {r['latent_collapse']:>9.4f}  {vl:>9}")
    print("─" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
