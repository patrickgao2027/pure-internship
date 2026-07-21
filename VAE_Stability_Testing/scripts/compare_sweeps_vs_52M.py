"""Compare epoch-sweep and seed-sweep models against the 52M full-data reference VAE.

Walks structured output directories, finds trained checkpoints, encodes a shared
test set through each, and computes geometry metrics vs the 52M reference embedding.

Usage:
    python VAE_Stability_Testing/scripts/compare_sweeps_vs_52M.py \
        --test-set-path test_set.parquet \
        --ref-checkpoint VAE_Stability_Testing/sweep_results_full/run_100pct/.../model.pt \
        --epoch-sweep-dir VAE_Stability_Testing/epoch_sweep \
        --seed-sweep-dir VAE_Stability_Testing/seed_sweep_extended \
        --output-dir VAE_Stability_Testing/combined_comparison
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl
from scipy.spatial import procrustes

from uv_vae.inference import LatentInference
from uv_vae.preprocess import transform_frame


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    numerator = np.linalg.norm(X.T @ Y, "fro") ** 2
    denom_x = np.linalg.norm(X.T @ X, "fro")
    denom_y = np.linalg.norm(Y.T @ Y, "fro")
    if denom_x < 1e-10 or denom_y < 1e-10:
        return 0.0
    return float(numerator / (denom_x * denom_y))


def latent_collapse_score(embeddings: np.ndarray) -> float:
    return float(np.std(embeddings, axis=0).mean())


def compute_trustworthiness(
    input_matrix: np.ndarray,
    embeddings: np.ndarray,
    n_neighbors: int = 10,
    max_rows: int = 5_000,
    seed: int = 0,
) -> float:
    from sklearn.manifold import trustworthiness
    n = input_matrix.shape[0]
    if n > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_rows, replace=False)
        embeddings = embeddings[idx]
        input_matrix = input_matrix[idx]
    return float(trustworthiness(input_matrix, embeddings, n_neighbors=n_neighbors))


def encode_test_set(checkpoint_path: Path, test_df: pl.DataFrame, batch_size: int = 4096) -> np.ndarray:
    inference = LatentInference.from_checkpoint(str(checkpoint_path), device="auto")
    features_df = test_df.select(inference.feature_names)
    return inference.encode_frame(features_df, batch_size=batch_size)


def get_input_matrix(checkpoint_path: Path, test_df: pl.DataFrame) -> np.ndarray:
    inference = LatentInference.from_checkpoint(str(checkpoint_path), device="auto")
    _, num_tensor = transform_frame(
        frame=test_df,
        categorical_specs=inference.categorical_specs,
        numeric_specs=inference.numeric_specs,
        numeric_means=inference.numeric_means,
        numeric_stds=inference.numeric_stds,
    )
    return num_tensor.numpy()


def load_val_metrics(run_dir: Path) -> dict:
    report_path = run_dir / "training_report.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text())
    history = report.get("history", [])
    return history[-1] if history else {}


def find_checkpoint(base_dir: Path) -> Path | None:
    """Find model.pt under a sweep output directory (handles run_100pct/run_TIMESTAMP nesting)."""
    direct = base_dir / "model.pt"
    if direct.exists():
        return direct
    for p in sorted(base_dir.rglob("model.pt")):
        return p
    return None


def collect_epoch_sweep(epoch_sweep_dir: Path) -> list[dict]:
    """Discover trained models from the epoch sweep directory structure.
    Expected: epoch_sweep/rows_{N}_epochs_{E}/run_100pct/run_TIMESTAMP/model.pt
    """
    entries = []
    if not epoch_sweep_dir.exists():
        log(f"Epoch sweep dir not found: {epoch_sweep_dir}")
        return entries

    for subdir in sorted(epoch_sweep_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        if not name.startswith("rows_"):
            continue
        parts = name.split("_")
        try:
            rows_idx = parts.index("rows") + 1
            epochs_idx = parts.index("epochs") + 1
            rows = int(parts[rows_idx])
            epochs = int(parts[epochs_idx])
        except (ValueError, IndexError):
            log(f"  SKIP {name}: can't parse rows/epochs")
            continue

        cp = find_checkpoint(subdir)
        if cp is None:
            log(f"  SKIP {name}: no model.pt found")
            continue

        entries.append({
            "type": "epoch_sweep",
            "rows": rows,
            "epochs": epochs,
            "seed": 42,
            "checkpoint": cp,
            "run_dir": cp.parent,
        })
        log(f"  FOUND {name}: {cp}")

    return entries


def collect_seed_sweep(seed_sweep_dir: Path) -> list[dict]:
    """Discover trained models from the extended seed sweep directory structure.
    Expected: seed_sweep_extended/dataseed_{S}/rows_{N}/run_100pct/run_TIMESTAMP/model.pt
    Also accepts legacy seed_{S} directory naming.
    """
    entries = []
    if not seed_sweep_dir.exists():
        log(f"Seed sweep dir not found: {seed_sweep_dir}")
        return entries

    for seed_dir in sorted(seed_sweep_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        name = seed_dir.name
        if name.startswith("dataseed_"):
            seed_str = name.split("_")[1]
        elif name.startswith("seed_"):
            seed_str = name.split("_")[1]
        else:
            continue
        try:
            seed = int(seed_str)
        except (ValueError, IndexError):
            continue

        for rows_dir in sorted(seed_dir.iterdir()):
            if not rows_dir.is_dir() or not rows_dir.name.startswith("rows_"):
                continue
            try:
                rows = int(rows_dir.name.split("_")[1])
            except (ValueError, IndexError):
                continue

            cp = find_checkpoint(rows_dir)
            if cp is None:
                log(f"  SKIP dataseed_{seed}/rows_{rows}: no model.pt found")
                continue

            entries.append({
                "type": "seed_sweep",
                "rows": rows,
                "epochs": 10,
                "data_seed": seed,
                "checkpoint": cp,
                "run_dir": cp.parent,
            })
            log(f"  FOUND dataseed_{seed}/rows_{rows}: {cp}")

    return entries


def compare_entry(
    entry: dict,
    ref_emb: np.ndarray,
    input_matrix: np.ndarray,
    test_df: pl.DataFrame,
) -> dict:
    cp = entry["checkpoint"]
    seed_label = entry.get("data_seed", entry.get("seed", "?"))
    log(f"Encoding: {entry['type']} rows={entry['rows']} epochs={entry['epochs']} seed={seed_label}")
    t0 = perf_counter()
    emb = encode_test_set(cp, test_df)
    encode_time = perf_counter() - t0

    _, _, proc_disp = procrustes(ref_emb, emb)
    cka = linear_cka(ref_emb, emb)
    collapse = latent_collapse_score(emb)
    trust = compute_trustworthiness(input_matrix, emb)
    val_metrics = load_val_metrics(entry["run_dir"])

    result = {
        "type": entry["type"],
        "rows": entry["rows"],
        "epochs": entry["epochs"],
        "run_dir": str(entry["run_dir"]),
        "procrustes_disparity": float(proc_disp),
        "linear_cka": float(cka),
        "latent_collapse": float(collapse),
        "trustworthiness": float(trust),
        "val_total_loss": val_metrics.get("val_total_loss", float("nan")),
        "val_kl_loss": val_metrics.get("val_kl_loss", float("nan")),
        "encode_time_s": round(encode_time, 1),
    }
    if "data_seed" in entry:
        result["data_seed"] = entry["data_seed"]
    else:
        result["seed"] = entry.get("seed", 42)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare epoch/seed sweep models vs 52M reference")
    parser.add_argument("--test-set-path", required=True)
    parser.add_argument("--ref-checkpoint", required=True, help="Path to the 52M reference model.pt")
    parser.add_argument("--epoch-sweep-dir", default="", help="Root dir of epoch sweep results")
    parser.add_argument("--seed-sweep-dir", default="", help="Root dir of seed sweep results")
    parser.add_argument("--output-dir", default="VAE_Stability_Testing/combined_comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = pl.read_parquet(args.test_set_path)
    log(f"Test set: {len(test_df)} rows")

    ref_cp = Path(args.ref_checkpoint)
    if not ref_cp.exists():
        log(f"ERROR: reference checkpoint not found: {ref_cp}")
        return 1

    log("Encoding reference (52M) embeddings")
    ref_emb = encode_test_set(ref_cp, test_df)
    input_matrix = get_input_matrix(ref_cp, test_df)
    log(f"Reference embedding shape: {ref_emb.shape}")

    entries = []
    if args.epoch_sweep_dir:
        log(f"\nCollecting epoch sweep from: {args.epoch_sweep_dir}")
        entries.extend(collect_epoch_sweep(Path(args.epoch_sweep_dir)))
    if args.seed_sweep_dir:
        log(f"\nCollecting seed sweep from: {args.seed_sweep_dir}")
        entries.extend(collect_seed_sweep(Path(args.seed_sweep_dir)))

    if not entries:
        log("No models found to compare.")
        return 1

    log(f"\nComparing {len(entries)} models against 52M reference\n")

    epoch_results = []
    seed_results = []

    for entry in entries:
        result = compare_entry(entry, ref_emb, input_matrix, test_df)
        if result["type"] == "epoch_sweep":
            epoch_results.append(result)
        else:
            seed_results.append(result)

        log(f"  proc={result['procrustes_disparity']:.4f}  "
            f"cka={result['linear_cka']:.4f}  "
            f"collapse={result['latent_collapse']:.4f}  "
            f"trust={result['trustworthiness']:.4f}")

    if epoch_results:
        epoch_path = output_dir / "epoch_sweep_vs_52M.json"
        epoch_path.write_text(json.dumps(epoch_results, indent=2))
        log(f"\nEpoch sweep results: {epoch_path} ({len(epoch_results)} entries)")

    if seed_results:
        seed_path = output_dir / "seed_sweep_extended_vs_52M.json"
        seed_path.write_text(json.dumps(seed_results, indent=2))
        log(f"Seed sweep results: {seed_path} ({len(seed_results)} entries)")

    log("\n=== Summary ===")
    if epoch_results:
        log(f"Epoch sweep: {len(epoch_results)} models compared")
        for ep in sorted(set(r["epochs"] for r in epoch_results)):
            subset = [r for r in epoch_results if r["epochs"] == ep]
            procs = [r["procrustes_disparity"] for r in subset]
            log(f"  {ep} epochs: proc range [{min(procs):.3f}, {max(procs):.3f}]")

    if seed_results:
        log(f"Seed sweep: {len(seed_results)} models compared")
        for rows in sorted(set(r["rows"] for r in seed_results)):
            subset = [r for r in seed_results if r["rows"] == rows]
            procs = [r["procrustes_disparity"] for r in subset]
            log(f"  {rows:>10,} rows: proc range [{min(procs):.3f}, {max(procs):.3f}] "
                f"(spread={max(procs)-min(procs):.3f}, n={len(subset)} seeds)")

    log("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
