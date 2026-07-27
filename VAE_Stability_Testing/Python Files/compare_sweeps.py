"""Compare 1M and 10M sweep models against the 10M reference VAE.

Re-encodes the test set through all 1M sweep checkpoints, computes
metrics against the 10M reference, loads the 10M sweep results, and
plots both series on the same graph using absolute row counts.

Run on the HPC after the 10M sweep completes:

    export UV_VAE_ROOT=~/uv_vae
    cd ~/VAE_Stability_Testing
    python scripts/compare_sweeps.py \
        --test-set-path /cta/users/patrickgao765/test_set.parquet \
        --sweep-1m-json sweep_results_1M/sweep_results.json \
        --sweep-10m-json sweep_results_10M/sweep_results.json \
        --output-dir comparison_results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

UV_VAE_ROOT = Path(os.environ.get("UV_VAE_ROOT", Path.home() / "uv_vae")).resolve()
if str(UV_VAE_ROOT) not in sys.path:
    sys.path.insert(0, str(UV_VAE_ROOT))

from scipy.spatial import procrustes
from sklearn.manifold import trustworthiness

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
    n = input_matrix.shape[0]
    if n > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_rows, replace=False)
        input_matrix = input_matrix[idx]
        embeddings = embeddings[idx]
    return float(trustworthiness(input_matrix, embeddings, n_neighbors=n_neighbors))


def get_input_matrix(inference: LatentInference, test_df: pl.DataFrame) -> np.ndarray:
    _, num_tensor = transform_frame(
        frame=test_df,
        categorical_specs=inference.categorical_specs,
        numeric_specs=inference.numeric_specs,
        numeric_means=inference.numeric_means,
        numeric_stds=inference.numeric_stds,
    )
    return num_tensor.numpy()


def encode_checkpoint(checkpoint_path: str, test_df: pl.DataFrame, batch_size: int) -> np.ndarray:
    inference = LatentInference.from_checkpoint(checkpoint_path, device="auto")
    features_df = test_df.select(inference.feature_names)
    return inference.encode_frame(features_df, batch_size=batch_size)


def compute_metrics(
    ref_emb: np.ndarray,
    emb: np.ndarray,
    input_matrix: np.ndarray,
) -> dict:
    _, _, proc_disp = procrustes(ref_emb, emb)
    return {
        "procrustes_disparity": float(proc_disp),
        "linear_cka": linear_cka(ref_emb, emb),
        "latent_collapse": latent_collapse_score(emb),
        "trustworthiness": compute_trustworthiness(input_matrix, emb),
    }


def plot_comparison(
    results_1m: list[dict],
    results_10m: list[dict],
    output_dir: Path,
) -> None:
    metrics_config = [
        ("procrustes_disparity", "Procrustes Disparity\n(lower = more similar to 10M VAE)"),
        ("linear_cka", "Linear CKA\n(higher = more similar to 10M VAE)"),
        ("trustworthiness", "Trustworthiness\n(higher = better local structure)"),
        ("latent_collapse", "Latent Collapse Score\n(higher = more active latent dims)"),
        ("val_total_loss", "Val Total Loss\n(lower = better reconstruction)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("VAE Stability: 1M vs 10M Sweep (reference = 10M VAE)", fontsize=14, fontweight="bold")

    for ax in axes.flat[len(metrics_config):]:
        ax.set_visible(False)

    for ax, (key, title) in zip(axes.flat, metrics_config):
        rows_1m = [r["sample_rows"] for r in results_1m if key in r and r[key] == r[key]]
        vals_1m = [r[key] for r in results_1m if key in r and r[key] == r[key]]

        rows_10m = [r["sample_rows"] for r in results_10m if key in r and r[key] == r[key]]
        vals_10m = [r[key] for r in results_10m if key in r and r[key] == r[key]]

        if rows_1m:
            ax.plot(rows_1m, vals_1m, marker="s", linewidth=2, markersize=7,
                    color="#dc2626", label="1M sweep", linestyle="--")
        if rows_10m:
            ax.plot(rows_10m, vals_10m, marker="o", linewidth=2, markersize=7,
                    color="#2563eb", label="10M sweep")

        ax.set_xscale("log")
        ax.set_xlabel("Training rows", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Plot saved to {plot_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare 1M and 10M sweeps against 10M reference")
    parser.add_argument("--test-set-path", required=True)
    parser.add_argument("--sweep-1m-json", required=True, help="Path to 1M sweep_results.json")
    parser.add_argument("--sweep-10m-json", required=True, help="Path to 10M sweep_results.json")
    parser.add_argument("--output-dir", default="comparison_results")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_1m = json.loads(Path(args.sweep_1m_json).read_text())
    sweep_10m = json.loads(Path(args.sweep_10m_json).read_text())

    # Find 10M reference checkpoint (first entry = largest fraction)
    ref_entry = sweep_10m[0]
    ref_checkpoint = Path(ref_entry["run_dir"]) / "model.pt"
    if not ref_checkpoint.exists():
        log(f"ERROR: 10M reference checkpoint not found: {ref_checkpoint}")
        return 1

    log(f"Loading test set from {args.test_set_path}")
    test_df = pl.read_parquet(args.test_set_path)

    log(f"Encoding test set through 10M reference VAE ({ref_entry['sample_rows']:,} rows)")
    inference = LatentInference.from_checkpoint(str(ref_checkpoint), device="auto")
    features_df = test_df.select(inference.feature_names)
    ref_emb = inference.encode_frame(features_df, batch_size=args.batch_size)
    input_matrix = get_input_matrix(inference, test_df)

    # Re-compute 1M sweep metrics against 10M reference
    log("Re-computing 1M sweep metrics against 10M reference")
    results_1m_recomputed = []
    for entry in sweep_1m:
        cp = Path(entry["run_dir"]) / "model.pt"
        if not cp.exists():
            log(f"  SKIP {entry['sample_rows']:,} rows — checkpoint missing: {cp}")
            continue

        log(f"  Encoding {entry['sample_rows']:,} rows model")
        emb = encode_checkpoint(str(cp), test_df, batch_size=args.batch_size)
        metrics = compute_metrics(ref_emb, emb, input_matrix)
        results_1m_recomputed.append({
            "sweep": "1M",
            "fraction": entry["fraction"],
            "sample_rows": entry["sample_rows"],
            "run_dir": entry["run_dir"],
            "val_total_loss": entry.get("val_total_loss", float("nan")),
            "val_kl_loss": entry.get("val_kl_loss", float("nan")),
            **metrics,
        })
        log(f"    procrustes={metrics['procrustes_disparity']:.4f}  cka={metrics['linear_cka']:.4f}")

    # Tag 10M sweep results
    results_10m_tagged = []
    for entry in sweep_10m:
        results_10m_tagged.append({
            "sweep": "10M",
            "fraction": entry["fraction"],
            "sample_rows": entry["sample_rows"],
            "run_dir": entry["run_dir"],
            "procrustes_disparity": entry["procrustes_disparity"],
            "linear_cka": entry["linear_cka"],
            "latent_collapse": entry["latent_collapse"],
            "trustworthiness": entry["trustworthiness"],
            "val_total_loss": entry.get("val_total_loss", float("nan")),
            "val_kl_loss": entry.get("val_kl_loss", float("nan")),
        })

    combined = results_1m_recomputed + results_10m_tagged
    combined_path = output_dir / "comparison_results.json"
    combined_path.write_text(json.dumps(combined, indent=2))
    log(f"Combined results saved to {combined_path}")

    log("Generating comparison plot")
    plot_comparison(results_1m_recomputed, results_10m_tagged, output_dir)

    log("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
