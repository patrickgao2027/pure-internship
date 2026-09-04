"""Quick UMAP cluster scatter plot from a completed stage-2 cell's analysis.parquet.

Usage (on miletus):
    micromamba run -n uv_vae python umap_hdbscan_sweep/plot_umap_clusters.py \
        --analysis <path/to/analysis.parquet> \
        --output umap_clusters.png \
        --sample-rows 500000
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True, help="path to analysis.parquet")
    p.add_argument("--output", default="umap_clusters.png")
    p.add_argument("--sample-rows", type=int, default=500_000,
                   help="rows to plot (default 500k — enough to see structure)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--title", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import polars as pl

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    print(f"Loading {args.sample_rows:,} rows from {args.analysis} ...")
    df = (
        pl.scan_parquet(args.analysis)
        .select("umap_1", "umap_2", "cluster_label")
        .collect()
        .sample(n=min(args.sample_rows, pl.scan_parquet(args.analysis).select(pl.len()).collect().item()),
                seed=args.seed)
    )

    umap1 = df["umap_1"].to_numpy()
    umap2 = df["umap_2"].to_numpy()
    labels = df["cluster_label"].to_numpy()

    noise_mask = labels == -1
    cluster_mask = ~noise_mask
    unique_clusters = np.unique(labels[cluster_mask])
    n_clusters = len(unique_clusters)

    print(f"  {n_clusters:,} clusters, {noise_mask.mean():.1%} noise in sample")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Noise points: light gray, small, behind clusters
    if noise_mask.any():
        ax.scatter(umap1[noise_mask], umap2[noise_mask],
                   s=0.3, c="#cccccc", alpha=0.15, linewidths=0, rasterized=True,
                   label=f"noise ({noise_mask.sum():,})")

    # Cluster points: colored by label using a cyclic colormap
    normed = (labels[cluster_mask] % 256) / 255.0
    ax.scatter(umap1[cluster_mask], umap2[cluster_mask],
               s=0.5, c=normed, cmap="hsv", alpha=0.4, linewidths=0,
               rasterized=True, label=f"{n_clusters:,} clusters")

    title = args.title or (
        f"UMAP n_neighbors=15  min_dist=0.0  |  "
        f"HDBSCAN mcs=1000  ms=5\n"
        f"{n_clusters:,} clusters · {noise_mask.mean():.1%} noise · "
        f"{args.sample_rows:,} sampled rows shown"
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=6, framealpha=0.7, fontsize=9)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
