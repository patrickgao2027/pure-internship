"""Comparison plots across completed stage-2 sweep cells.

Produces four figures from whatever cells have finished under one UMAP config dir:

  1. comparison_grid.png   -- side-by-side UMAP scatter per cell (cluster_label colored)
  2. metrics_vs_param.png  -- silhouette & noise_fraction vs min_cluster_size (from metrics.json only, no parquet read)
  3. density_heatmap.png   -- hexbin of raw UMAP density, ignoring cluster labels
  4. cluster_size_hist.png -- per-cell histogram of cluster sizes (log-scaled)

Usage (on miletus):
    micromamba run -n uv_vae python umap_hdbscan_sweep/plot_sweep_comparison.py \
        --config-dir uv_vae/runs/train_multi_20260802T192756Z/stage2_sweep/nn15_md0.0_nc2 \
        --output-dir plots/ \
        --sample-rows 300000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config-dir", required=True, help="one UMAP config dir, e.g. .../nn15_md0.0_nc2")
    p.add_argument("--output-dir", default="plots")
    p.add_argument("--sample-rows", type=int, default=300_000,
                   help="rows sampled per cell for scatter/density plots")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_cells(config_dir: Path) -> list[dict]:
    cells = []
    for metrics_path in sorted(config_dir.glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text())
        if "error" in payload:
            continue
        payload["_cell_dir"] = metrics_path.parent
        payload["_name"] = metrics_path.parent.name
        cells.append(payload)
    # order by min_cluster_size for readable comparisons
    cells.sort(key=lambda c: c.get("hdbscan", {}).get("min_cluster_size", 0))
    return cells


def main() -> None:
    args = parse_args()

    import numpy as np
    import polars as pl
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config_dir = Path(args.config_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = load_cells(config_dir)
    if not cells:
        raise SystemExit(f"no completed cells found under {config_dir}")
    print(f"Found {len(cells)} completed cells: {[c['_name'] for c in cells]}")

    config_name = config_dir.name

    # ---- 1. side-by-side comparison grid --------------------------------
    n = len(cells)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)

    sampled_frames = {}  # cache for reuse in density plot
    for i, cell in enumerate(cells):
        ax = axes[i // ncols][i % ncols]
        analysis_path = cell["_cell_dir"] / "analysis.parquet"
        total = pl.scan_parquet(analysis_path).select(pl.len()).collect().item()
        n_sample = min(args.sample_rows, total)
        df = (
            pl.scan_parquet(analysis_path)
            .select("umap_1", "umap_2", "cluster_label")
            .collect()
            .sample(n=n_sample, seed=args.seed)
        )
        sampled_frames[cell["_name"]] = df

        u1, u2, lab = df["umap_1"].to_numpy(), df["umap_2"].to_numpy(), df["cluster_label"].to_numpy()
        noise = lab == -1
        if noise.any():
            ax.scatter(u1[noise], u2[noise], s=0.3, c="#cccccc", alpha=0.15, linewidths=0, rasterized=True)
        clustered = ~noise
        normed = (lab[clustered] % 256) / 255.0
        ax.scatter(u1[clustered], u2[clustered], s=0.4, c=normed, cmap="hsv", alpha=0.4,
                   linewidths=0, rasterized=True)

        m = cell["metrics"]
        ax.set_title(f"{cell['_name']}\nk={m['n_clusters']:,}  noise={m['noise_fraction']:.2f}  "
                      f"sil={m['silhouette']:.3f}", fontsize=10)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_aspect("equal")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"{config_name} -- cluster comparison across min_cluster_size", fontsize=13)
    plt.tight_layout()
    grid_path = output_dir / "comparison_grid.png"
    plt.savefig(grid_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {grid_path}")

    # ---- 2. metrics vs parameter (cheap -- metrics.json only) ------------
    mcs = [c["hdbscan"]["min_cluster_size"] for c in cells]
    sil = [c["metrics"]["silhouette"] for c in cells]
    noise_frac = [c["metrics"]["noise_fraction"] for c in cells]
    n_clusters = [c["metrics"]["n_clusters"] for c in cells]

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(mcs, sil, "o-", color="#1f77b4", label="silhouette")
    ax1.set_xlabel("min_cluster_size")
    ax1.set_ylabel("silhouette", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(mcs, noise_frac, "s--", color="#d62728", label="noise fraction")
    ax2.set_ylabel("noise fraction", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    for x, k in zip(mcs, n_clusters):
        ax1.annotate(f"k={k:,}", (x, sil[mcs.index(x)]), textcoords="offset points",
                     xytext=(0, 10), fontsize=8, ha="center")

    ax1.set_title(f"{config_name} -- quality vs min_cluster_size (min_samples=5)")
    fig.tight_layout()
    metrics_path_out = output_dir / "metrics_vs_param.png"
    plt.savefig(metrics_path_out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {metrics_path_out}")

    # ---- 3. density heatmap (raw manifold shape, no cluster labels) -----
    # Use the largest available sample (they share the same UMAP coordinates,
    # so any one cell's rows work -- pick the last one for the biggest clusters).
    ref_df = sampled_frames[cells[-1]["_name"]]
    u1, u2 = ref_df["umap_1"].to_numpy(), ref_df["umap_2"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 8))
    hb = ax.hexbin(u1, u2, gridsize=200, cmap="inferno", bins="log", mincnt=1)
    fig.colorbar(hb, ax=ax, label="log10(count)")
    ax.set_title(f"{config_name} -- raw UMAP density ({len(u1):,} sampled reads, cluster-agnostic)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")
    plt.tight_layout()
    density_path = output_dir / "density_heatmap.png"
    plt.savefig(density_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {density_path}")

    # ---- 4. cluster size distribution, one panel per cell ----------------
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for i, cell in enumerate(cells):
        ax = axes[0][i]
        labels = pl.scan_parquet(cell["_cell_dir"] / "analysis.parquet").select("cluster_label").collect()["cluster_label"].to_numpy()
        clustered = labels[labels != -1]
        sizes = np.bincount(clustered) if clustered.size else np.array([])
        sizes = sizes[sizes > 0]
        if sizes.size:
            ax.hist(sizes, bins=np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 40),
                     color="#2a9d8f", edgecolor="none")
            ax.set_xscale("log")
        ax.set_title(f"{cell['_name']}\n{sizes.size:,} clusters", fontsize=10)
        ax.set_xlabel("cluster size (reads, log scale)")
        ax.set_ylabel("count of clusters")

    fig.suptitle(f"{config_name} -- cluster size distributions (full 157.5M-row labels)", fontsize=12)
    plt.tight_layout()
    hist_path = output_dir / "cluster_size_hist.png"
    plt.savefig(hist_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {hist_path}")

    print(f"\nAll plots written to {output_dir}")


if __name__ == "__main__":
    main()
