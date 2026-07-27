"""Re-encode 1M, 10M, and seed-test checkpoints against the 52M full-data VAE reference.

Usage (on HPC):
    export UV_VAE_ROOT=~/uv_vae
    cd ~/uv_vae
    python VAE_Stability_Testing/scripts/vs_52M_reference.py \
        --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
        --ref-full-json VAE_Stability_Testing/sweep_results_full/sweep_results.json \
        --sweep-jsons VAE_Stability_Testing/sweep_results_1M/sweep_results.json,VAE_Stability_Testing/sweep_results_10M/sweep_results.json \
        --sweep-labels 1M,10M \
        --seed-dirs VAE_Stability_Testing/sweep_seed_results/sweep_seed7,VAE_Stability_Testing/sweep_seed_results/sweep_seed13,VAE_Stability_Testing/sweep_seed_results/sweep_seed67,VAE_Stability_Testing/sweep_seed_results/sweep_seed99 \
        --seed-42-json VAE_Stability_Testing/sweep_results_1M/sweep_results.json \
        --output-dir VAE_Stability_Testing/vs_52M_comparison
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


def find_checkpoint(run_dir_str, sweep_base=None):
    """Resolve checkpoint path, handling moved directories.

    Tries: absolute path, relative subpaths, and — if sweep_base is given —
    resolving the run subdirectory (run_XXpct/run_TIMESTAMP) relative to the
    directory that contains the sweep_results.json.
    """
    cp = Path(run_dir_str) / "model.pt"
    if cp.exists():
        return cp
    parts = Path(run_dir_str).parts
    for i in range(len(parts)):
        candidate = Path(*parts[i:]) / "model.pt"
        if candidate.exists():
            return candidate
    if sweep_base is not None:
        run_parts = Path(run_dir_str).parts
        for depth in (2, 3):
            if len(run_parts) >= depth:
                candidate = sweep_base / Path(*run_parts[-depth:]) / "model.pt"
                if candidate.exists():
                    return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Re-encode sweep checkpoints against the 52M full-data reference VAE")
    parser.add_argument("--test-set-path", required=True)
    parser.add_argument("--ref-full-json", required=True,
                        help="sweep_results.json from the full 52M sweep")
    parser.add_argument("--sweep-jsons", required=True,
                        help="Comma-separated paths to sweep_results.json files to compare")
    parser.add_argument("--sweep-labels", required=True,
                        help="Comma-separated labels for each sweep (e.g. 1M,10M)")
    parser.add_argument("--seed-dirs", default=None,
                        help="Comma-separated paths to seed sweep output dirs (e.g. sweep_seed7,sweep_seed13)")
    parser.add_argument("--seed-42-json", default=None,
                        help="sweep_results.json containing seed 42 checkpoints (1M sweep) for seed comparison")
    parser.add_argument("--output-dir", default="vs_52M_comparison")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sweep_paths = [p.strip() for p in args.sweep_jsons.split(",") if p.strip()]
    sweep_labels = [l.strip() for l in args.sweep_labels.split(",") if l.strip()]
    if len(sweep_paths) != len(sweep_labels):
        log("ERROR: --sweep-jsons and --sweep-labels must have the same number of entries")
        return 1

    log("Loading test set")
    test_df = pl.read_parquet(args.test_set_path)

    # Load 52M reference — use the 100% fraction entry
    ref_json_path = Path(args.ref_full_json)
    ref_json = json.loads(ref_json_path.read_text())
    ref_entry = max(ref_json, key=lambda e: e["sample_rows"])
    ref_cp = find_checkpoint(ref_entry["run_dir"], sweep_base=ref_json_path.resolve().parent)
    if ref_cp is None:
        log(f"ERROR: 52M reference checkpoint not found: {ref_entry['run_dir']}")
        return 1

    log(f"Encoding test set through 52M reference VAE ({ref_entry['sample_rows']:,} rows)")
    ref_inf, ref_emb = encode(ref_cp, test_df, args.batch_size)
    input_matrix = get_input_matrix(ref_inf, test_df)
    log(f"  ref_emb shape: {ref_emb.shape}")

    results = []

    for sweep_path, label in zip(sweep_paths, sweep_labels):
        sp = Path(sweep_path)
        sweep = json.loads(sp.read_text())
        sweep_base = sp.resolve().parent
        log(f"\n{'─' * 50}")
        log(f"Processing {label} sweep ({len(sweep)} checkpoints) — base: {sweep_base}")

        for entry in sorted(sweep, key=lambda e: -e["sample_rows"]):
            cp = find_checkpoint(entry["run_dir"], sweep_base=sweep_base)
            if cp is None:
                log(f"  SKIP {label} {entry['sample_rows']:,} rows — checkpoint not found")
                continue

            log(f"  Encoding {label} {entry['sample_rows']:,} rows")
            _, emb = encode(cp, test_df, args.batch_size)
            m = metrics(ref_emb, emb, input_matrix)
            results.append({
                "sweep": label,
                "fraction": entry["fraction"],
                "sample_rows": entry["sample_rows"],
                "val_total_loss": entry.get("val_total_loss"),
                "val_kl_loss": entry.get("val_kl_loss"),
                **m,
            })
            log(f"    proc={m['procrustes']:.4f}  cka={m['cka']:.4f}  "
                f"collapse={m['collapse']:.4f}  trust={m['trustworthiness']:.4f}")

    # ── Seed test checkpoints ──────────────────────────────────────────
    seed_results = []

    if args.seed_42_json:
        s42 = json.loads(Path(args.seed_42_json).read_text())
        s42_base = Path(args.seed_42_json).resolve().parent
        for entry in s42:
            if entry["sample_rows"] not in (1000000, 750000):
                continue
            cp = find_checkpoint(entry["run_dir"], sweep_base=s42_base)
            if cp is None:
                log(f"  SKIP seed=42 {entry['sample_rows']:,} rows — checkpoint not found")
                continue
            log(f"  Encoding seed=42 {entry['sample_rows']:,} rows")
            _, emb = encode(cp, test_df, args.batch_size)
            m = metrics(ref_emb, emb, input_matrix)
            seed_results.append({
                "seed": 42,
                "sample_rows": entry["sample_rows"],
                "val_total_loss": entry.get("val_total_loss"),
                "val_kl_loss": entry.get("val_kl_loss"),
                **m,
            })
            log(f"    proc={m['procrustes']:.4f}  cka={m['cka']:.4f}  "
                f"collapse={m['collapse']:.4f}  trust={m['trustworthiness']:.4f}")

    if args.seed_dirs:
        seed_dir_list = [Path(p.strip()) for p in args.seed_dirs.split(",") if p.strip()]
        for sd in seed_dir_list:
            results_json = sd / "sweep_results.json"
            if not results_json.exists():
                log(f"SKIP {sd} — sweep_results.json not found")
                continue
            sweep = json.loads(results_json.read_text())
            seed_label = sd.name.replace("sweep_seed", "")
            sweep_base = sd.resolve()
            log(f"\n{'─' * 50}")
            log(f"Processing seed={seed_label} ({len(sweep)} checkpoints) — base: {sweep_base}")

            for entry in sorted(sweep, key=lambda e: -e["sample_rows"]):
                cp = find_checkpoint(entry["run_dir"], sweep_base=sweep_base)
                if cp is None:
                    log(f"  SKIP seed={seed_label} {entry['sample_rows']:,} rows — checkpoint not found")
                    continue
                log(f"  Encoding seed={seed_label} {entry['sample_rows']:,} rows")
                _, emb = encode(cp, test_df, args.batch_size)
                m = metrics(ref_emb, emb, input_matrix)
                seed_results.append({
                    "seed": int(seed_label) if seed_label.isdigit() else seed_label,
                    "sample_rows": entry["sample_rows"],
                    "val_total_loss": entry.get("val_total_loss"),
                    "val_kl_loss": entry.get("val_kl_loss"),
                    **m,
                })
                log(f"    proc={m['procrustes']:.4f}  cka={m['cka']:.4f}  "
                    f"collapse={m['collapse']:.4f}  trust={m['trustworthiness']:.4f}")

    # ── Save results ─────────────────────────────────────────────────
    out_json = out / "vs_52M_comparison.json"
    out_json.write_text(json.dumps(results, indent=2))

    if seed_results:
        seed_json = out / "seed_vs_52M.json"
        seed_json.write_text(json.dumps(seed_results, indent=2))
        log(f"Saved {len(seed_results)} seed results to {seed_json}")

    log(f"Saved {len(results)} sweep results to {out_json}")

    # Summary tables
    print(f"\n{'═' * 78}")
    print(f"  Sweep checkpoints re-encoded against 52M full-data reference")
    print(f"{'═' * 78}")
    print(f"{'Sweep':>6}  {'Rows':>10}  {'Procrustes':>11}  {'CKA':>8}  "
          f"{'Collapse':>9}  {'Trust':>8}  {'Val Loss':>9}")
    print("─" * 78)
    for r in sorted(results, key=lambda x: (x["sweep"], -x["sample_rows"])):
        vl = f"{r['val_total_loss']:.4f}" if r["val_total_loss"] else "  n/a  "
        print(f"{r['sweep']:>6}  {r['sample_rows']:>10,}  {r['procrustes']:>11.4f}  "
              f"{r['cka']:>8.4f}  {r['collapse']:>9.4f}  "
              f"{r['trustworthiness']:>8.4f}  {vl:>9}")
    print("─" * 78)

    if seed_results:
        print(f"\n{'═' * 78}")
        print(f"  Seed comparison re-encoded against 52M full-data reference")
        print(f"{'═' * 78}")
        print(f"{'Seed':>6}  {'Rows':>10}  {'Procrustes':>11}  {'CKA':>8}  "
              f"{'Collapse':>9}  {'Trust':>8}  {'Val Loss':>9}")
        print("─" * 78)
        for r in sorted(seed_results, key=lambda x: (x["sample_rows"], x["seed"])):
            vl = f"{r['val_total_loss']:.4f}" if r["val_total_loss"] else "  n/a  "
            print(f"{r['seed']:>6}  {r['sample_rows']:>10,}  {r['procrustes']:>11.4f}  "
                  f"{r['cka']:>8.4f}  {r['collapse']:>9.4f}  "
                  f"{r['trustworthiness']:>8.4f}  {vl:>9}")
        print("─" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
