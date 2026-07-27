"""Re-encode seed test checkpoints against the 10M reference VAE.

Usage:
    export UV_VAE_ROOT=~/uv_vae
    cd ~/uv_vae
    python VAE_Stability_Testing/scripts/seed_vs_10M.py \
        --test-set-path /cta/users/patrickgao765/test_set.parquet \
        --ref-10m-json VAE_Stability_Testing/sweep_results_10M/sweep_results.json \
        --seed-dirs VAE_Stability_Testing/sweep_seed_results/sweep_seed7,VAE_Stability_Testing/sweep_seed_results/sweep_seed13,VAE_Stability_Testing/sweep_seed_results/sweep_seed99 \
        --seed-42-json VAE_Stability_Testing/sweep_results/sweep_results.json \
        --output-dir VAE_Stability_Testing/seed_comparison
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
from sklearn.manifold import trustworthiness

from uv_vae.inference import LatentInference
from uv_vae.preprocess import transform_frame


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    dx = np.linalg.norm(X.T @ X, "fro")
    dy = np.linalg.norm(Y.T @ Y, "fro")
    return float(num / (dx * dy)) if dx > 1e-10 and dy > 1e-10 else 0.0


def compute_trust(X, Z, n_neighbors=10, max_rows=5000, seed=0):
    n = X.shape[0]
    if n > max_rows:
        idx = np.random.default_rng(seed).choice(n, max_rows, replace=False)
        X, Z = X[idx], Z[idx]
    return float(trustworthiness(X, Z, n_neighbors=n_neighbors))


def encode(checkpoint_path, test_df, batch_size=4096):
    inf = LatentInference.from_checkpoint(str(checkpoint_path), device="auto")
    return inf, inf.encode_frame(test_df.select(inf.feature_names), batch_size=batch_size)


def get_input_matrix(inf, test_df):
    _, t = transform_frame(
        frame=test_df,
        categorical_specs=inf.categorical_specs,
        numeric_specs=inf.numeric_specs,
        numeric_means=inf.numeric_means,
        numeric_stds=inf.numeric_stds,
    )
    return t.numpy()


def metrics(ref_emb, emb, input_matrix):
    _, _, proc = procrustes(ref_emb, emb)
    return {
        "procrustes": round(float(proc), 6),
        "cka": round(linear_cka(ref_emb, emb), 6),
        "collapse": round(float(np.std(emb, axis=0).mean()), 6),
        "trustworthiness": round(compute_trust(input_matrix, emb), 6),
    }


def find_checkpoint(run_dir_str):
    """Resolve checkpoint path, fixing stale absolute paths."""
    cp = Path(run_dir_str) / "model.pt"
    if cp.exists():
        return cp
    # Try relative to cwd
    parts = Path(run_dir_str).parts
    for i in range(len(parts)):
        candidate = Path(*parts[i:]) / "model.pt"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set-path", required=True)
    parser.add_argument("--ref-10m-json", required=True,
                        help="sweep_results.json from the 10M sweep")
    parser.add_argument("--seed-dirs", required=True,
                        help="Comma-separated paths to seed sweep output dirs")
    parser.add_argument("--seed-42-json", default=None,
                        help="Original seed 42 sweep_results.json (1M sweep) for comparison")
    parser.add_argument("--output-dir", default="seed_comparison")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log("Loading test set")
    test_df = pl.read_parquet(args.test_set_path)

    # Load 10M reference
    ref_json = json.loads(Path(args.ref_10m_json).read_text())
    ref_entry = ref_json[0]
    ref_cp = find_checkpoint(ref_entry["run_dir"])
    if ref_cp is None:
        log(f"ERROR: 10M reference checkpoint not found: {ref_entry['run_dir']}")
        return 1

    log(f"Encoding test set through 10M reference VAE")
    ref_inf, ref_emb = encode(ref_cp, test_df, args.batch_size)
    input_matrix = get_input_matrix(ref_inf, test_df)
    log(f"  ref_emb shape: {ref_emb.shape}")

    results = []

    # Original seed 42 (1M, 750k)
    if args.seed_42_json:
        s42 = json.loads(Path(args.seed_42_json).read_text())
        for entry in s42:
            if entry["sample_rows"] not in (1000000, 750000):
                continue
            cp = find_checkpoint(entry["run_dir"])
            if cp is None:
                log(f"  SKIP seed=42 {entry['sample_rows']:,} rows — checkpoint missing")
                continue
            log(f"Encoding seed=42 {entry['sample_rows']:,} rows")
            _, emb = encode(cp, test_df, args.batch_size)
            m = metrics(ref_emb, emb, input_matrix)
            results.append({
                "seed": 42,
                "sample_rows": entry["sample_rows"],
                "val_total_loss": entry.get("val_total_loss"),
                "val_kl_loss": entry.get("val_kl_loss"),
                **m,
            })
            log(f"  procrustes={m['procrustes']:.4f}  cka={m['cka']:.4f}  collapse={m['collapse']:.4f}")

    # Seed test dirs
    seed_dirs = [Path(p.strip()) for p in args.seed_dirs.split(",") if p.strip()]
    for sd in seed_dirs:
        results_json = sd / "sweep_results.json"
        if not results_json.exists():
            log(f"SKIP {sd} — sweep_results.json not found")
            continue

        sweep = json.loads(results_json.read_text())
        # Infer seed from the dir name (sweep_seedN)
        seed_label = sd.name.replace("sweep_seed", "")

        for entry in sweep:
            cp = find_checkpoint(entry["run_dir"])
            if cp is None:
                # Try path relative to the seed dir
                run_parts = Path(entry["run_dir"]).parts[-2:]
                cp = sd / Path(*run_parts) / "model.pt"
                if not cp.exists():
                    log(f"  SKIP seed={seed_label} {entry['sample_rows']:,} rows — checkpoint missing")
                    continue
            log(f"Encoding seed={seed_label} {entry['sample_rows']:,} rows")
            _, emb = encode(cp, test_df, args.batch_size)
            m = metrics(ref_emb, emb, input_matrix)
            results.append({
                "seed": int(seed_label) if seed_label.isdigit() else seed_label,
                "sample_rows": entry["sample_rows"],
                "val_total_loss": entry.get("val_total_loss"),
                "val_kl_loss": entry.get("val_kl_loss"),
                **m,
            })
            log(f"  procrustes={m['procrustes']:.4f}  cka={m['cka']:.4f}  collapse={m['collapse']:.4f}")

    # Save
    out_json = out / "seed_comparison.json"
    out_json.write_text(json.dumps(results, indent=2))
    log(f"Saved {len(results)} results to {out_json}")

    # Print summary table
    print("\n── Seed comparison vs 10M reference ──────────────────────────────")
    print(f"{'Seed':>6}  {'Rows':>9}  {'Procrustes':>11}  {'CKA':>8}  {'Collapse':>9}  {'Val Loss':>9}")
    print("─" * 66)
    for r in sorted(results, key=lambda x: (x["sample_rows"], x["seed"])):
        vl = f"{r['val_total_loss']:.4f}" if r["val_total_loss"] else "  n/a "
        print(f"{r['seed']:>6}  {r['sample_rows']:>9,}  {r['procrustes']:>11.4f}  "
              f"{r['cka']:>8.4f}  {r['collapse']:>9.4f}  {vl:>9}")
    print("─" * 66)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
