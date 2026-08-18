#!/usr/bin/env python
"""UMAP colored by cosine similarity to cluster centroid.

For each cell in --results-dir produces two figures:

  cosine_sim_umap.png
      UMAP scatter where each point is coloured by its cosine similarity to its
      cluster's centroid in the numeric feature space.  High values (yellow in
      viridis) mean the point is representative of its cluster; low values
      (purple) are within-cluster outliers.  Noise points (label = -1) are
      drawn in grey behind everything else.

  cluster_cosine_heatmap.png
      Cluster × cluster cosine similarity matrix for the top-N clusters (by
      member count), showing which clusters occupy similar regions of feature
      space.  Diagonal = 1 by definition.  Off-diagonal ≈ 1 means two clusters
      are nearly indistinguishable by their mean feature vectors; ≈ 0 means
      orthogonal in feature space.

Cosine similarity is computed in the numeric feature space after filling nulls
with 0 (null features contribute nothing to the dot product, which is the most
neutral assumption).

All cells share the same UMAP layout and feature values; only the cluster
partition differs, so within-cluster similarity patterns shift per cell.

Usage::

    python umap_hdbscan_sweep/plot_cosine_similarity.py \\
        --results-dir ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/cohort_reports_original \\
        --enriched    ~/pure-internship/umap_hdbscan_sweep/hdbscan/enriched.parquet \\
        --output-dir  ~/pure-internship/umap_hdbscan_sweep/hdbscan/cluster_sweep_plots \\
        --top-clusters 30 --sample-rows 2000000 --dpi 150
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for _c in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


# ── constants ──────────────────────────────────────────────────────────────────

_NOISE_COLOUR   = "#cccccc"
_NUMERIC_DTYPES = (pl.Float32, pl.Float64,
                   pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
_EXCLUDE        = {"umap_1", "umap_2", "cluster_label", "cluster_probability",
                   "POS", "__row", "locus_reads"}


def log(msg: str) -> None:
    print(msg, flush=True)


# ── data loading ───────────────────────────────────────────────────────────────

def parse_cell_name(path: Path) -> str:
    m = re.search(r"fit(\d+)_mcs(\d+)_ms(\d+)", path.name)
    if not m:
        return path.name
    fit = int(m.group(1))
    fit_s = f"{fit // 1_000_000}M" if fit >= 1_000_000 else f"{fit // 1_000}K"
    return f"fit {fit_s}  mcs {m.group(2)}  ms {m.group(3)}"


def find_cells(results_dir: Path) -> list[Path]:
    return sorted(
        (p for p in results_dir.iterdir()
         if p.is_dir() and (p / "analysis.parquet").exists()),
        key=lambda p: p.name,
    )


def load_enriched(path: Path, sample_rows: int, seed: int
                  ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Returns xy (N,2), feat_filled (N,F), feat_names, positions."""
    total = pl.scan_parquet(path).select(pl.len()).collect().item()
    rng = np.random.default_rng(seed)
    if 0 < sample_rows < total:
        positions = np.sort(rng.choice(total, size=sample_rows, replace=False))
    else:
        positions = np.arange(total)

    frame = pl.read_parquet(path)
    if positions.size < frame.height:
        frame = frame[positions.tolist()]

    xy = frame.select(["umap_1", "umap_2"]).to_numpy().astype(np.float32)

    num_cols = [c for c in frame.columns
                if c not in _EXCLUDE
                and frame[c].dtype in _NUMERIC_DTYPES
                and frame[c].null_count() < frame.height]

    # fill NaN with 0: null features contribute nothing to the dot product
    feat_filled = np.nan_to_num(
        frame.select(num_cols).to_numpy().astype(np.float32), nan=0.0)

    return xy, feat_filled, num_cols, positions


def load_labels(cell_path: Path, positions: np.ndarray) -> np.ndarray:
    frame = pl.read_parquet(cell_path / "analysis.parquet",
                            columns=["cluster_label"])
    if positions.size < frame.height:
        frame = frame[positions.tolist()]
    return frame["cluster_label"].to_numpy().astype(np.int32)


# ── cosine similarity ──────────────────────────────────────────────────────────

def _l2(v: np.ndarray) -> np.ndarray:
    """Row-wise L2 norm, safe against zero vectors."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return n


def compute_cluster_centroids(feat: np.ndarray, labels: np.ndarray,
                               top_n: int) -> dict[int, np.ndarray]:
    """Mean feature vector per cluster (not normalised)."""
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = np.argsort(-counts)[:top_n]
    centroids: dict[int, np.ndarray] = {}
    for idx in order:
        lab  = unique[idx]
        mask = labels == lab
        centroids[int(lab)] = feat[mask].mean(axis=0)
    return centroids


def per_point_cosine_sim(feat: np.ndarray, labels: np.ndarray,
                          centroids: dict[int, np.ndarray]) -> np.ndarray:
    """Cosine similarity of each point to its cluster centroid; NaN for noise."""
    sim = np.full(len(labels), np.nan, dtype=np.float32)
    feat_norm = feat / _l2(feat)          # (N, F) row-normalised
    for lab, centroid in centroids.items():
        mask = labels == lab
        if not mask.any():
            continue
        c_norm = centroid / float(np.linalg.norm(centroid) or 1.0)
        sim[mask] = feat_norm[mask] @ c_norm
    return sim


def cluster_cosine_matrix(centroids: dict[int, np.ndarray]
                           ) -> tuple[np.ndarray, list[int]]:
    """Pairwise cosine similarity between cluster centroids."""
    labs = sorted(centroids.keys())
    vecs = np.stack([centroids[l] for l in labs], axis=0)   # (K, F)
    vecs_norm = vecs / _l2(vecs)
    mat = vecs_norm @ vecs_norm.T
    return mat.astype(np.float32), labs


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_cosine_umap(xy: np.ndarray, labels: np.ndarray, sim: np.ndarray,
                     centroids: dict[int, np.ndarray],
                     cell_title: str, out_path: Path,
                     point_size: float = 1.5, dpi: int = 150) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # noise behind everything
    noise = labels == -1
    if noise.any():
        ax.scatter(xy[noise, 0], xy[noise, 1], s=point_size, c=_NOISE_COLOUR,
                   linewidths=0, alpha=0.25, rasterized=True, zorder=1)

    # non-top clusters (no centroid computed) — uniform dim grey
    no_centroid = (labels >= 0) & ~np.isin(labels, list(centroids.keys()))
    if no_centroid.any():
        ax.scatter(xy[no_centroid, 0], xy[no_centroid, 1], s=point_size,
                   c="#bbbbbb", linewidths=0, alpha=0.20, rasterized=True, zorder=1)

    # top-cluster points coloured by cosine similarity
    top_mask = np.isin(labels, list(centroids.keys()))
    if top_mask.any():
        sc = ax.scatter(xy[top_mask, 0], xy[top_mask, 1],
                        s=point_size, c=sim[top_mask],
                        cmap="viridis", vmin=0.0, vmax=1.0,
                        linewidths=0, alpha=0.75, rasterized=True, zorder=2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("cosine similarity to cluster centroid", fontsize=8)

    n_clusters = len(centroids)
    n_noise    = int(noise.sum())
    valid_sim  = sim[top_mask]
    mean_sim   = float(np.nanmean(valid_sim)) if valid_sim.size else float("nan")

    ax.set_title(
        f"{cell_title}  |  cosine similarity to centroid\n"
        f"{n_clusters} clusters shown  ·  mean sim {mean_sim:.3f}  ·  "
        f"{n_noise / len(labels) * 100:.1f}% noise",
        fontsize=9
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_cosine_heatmap(mat: np.ndarray, labs: list[int],
                        counts: dict[int, int],
                        cell_title: str, out_path: Path, dpi: int = 150) -> None:
    n = len(labs)
    fig_size = max(5, n * 0.28 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="cosine similarity", fraction=0.046, pad=0.04)

    tick_labels = [f"C{l}\n(n={counts[l]:,})" for l in labs]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=5)

    ax.set_title(
        f"{cell_title}  |  cluster × cluster cosine similarity\n"
        f"top {n} clusters by size  (centroids in numeric feature space)",
        fontsize=9
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── entry point ────────────────────────────────────────────────────────────────

def make_contact_sheet(umap_paths: list[Path], labels: list[str],
                       out_path: Path, ncols: int = 7, dpi: int = 120) -> None:
    """Tile all per-cell cosine-sim UMAPs into one comparison grid."""
    import math
    nrows = math.ceil(len(umap_paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.2, nrows * 2.8))
    axes = np.array(axes).reshape(-1)
    for i, (path, label) in enumerate(zip(umap_paths, labels)):
        img = plt.imread(path)
        axes[i].imshow(img)
        axes[i].set_title(label, fontsize=6)
        axes[i].axis("off")
    for j in range(len(umap_paths), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Cosine similarity to cluster centroid — all HDBSCAN parameterisations",
                 fontsize=10, y=1.002)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"contact sheet -> {out_path}  ({nrows}×{ncols}, {len(umap_paths)} cells)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--enriched",    type=Path, required=True,
                   help="enriched.parquet with umap_1/umap_2 and all source columns")
    p.add_argument("--output-dir",  type=Path, required=True,
                   help="root for output; figures written per cell subdir")
    p.add_argument("--sample-rows", type=int, default=2_000_000)
    p.add_argument("--top-clusters", type=int, default=30,
                   help="clusters to include (by size); rest shown as grey")
    p.add_argument("--no-heatmap",    action="store_true",
                   help="skip the cluster×cluster heatmap (faster)")
    p.add_argument("--contact-cols", type=int, default=7,
                   help="columns in the contact sheet grid (default 7)")
    p.add_argument("--point-size",   type=float, default=1.5)
    p.add_argument("--dpi",         type=int, default=150)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cells = find_cells(args.results_dir)
    if not cells:
        raise SystemExit(f"no cell subdirs with analysis.parquet under {args.results_dir}")
    log(f"found {len(cells)} cells")

    log(f"loading enriched.parquet ({args.sample_rows:,} rows) ...")
    xy, feat, feat_names, positions = load_enriched(
        args.enriched, args.sample_rows, args.seed)
    log(f"  {xy.shape[0]:,} points, {len(feat_names)} numeric features")

    sheet_paths: list[Path] = []
    sheet_labels: list[str] = []

    for i, cell in enumerate(cells, 1):
        title = parse_cell_name(cell)
        log(f"[{i:>2}/{len(cells)}] {cell.name}")

        labels = load_labels(cell, positions)
        centroids = compute_cluster_centroids(feat, labels, args.top_clusters)
        if not centroids:
            log("  no clusters found, skipping")
            continue

        counts = {lab: int((labels == lab).sum()) for lab in centroids}
        log(f"  {len(centroids)} cluster centroids, computing cosine similarities ...")

        sim = per_point_cosine_sim(feat, labels, centroids)

        cell_out = args.output_dir / cell.name
        umap_out = cell_out / "cosine_sim_umap.png"
        plot_cosine_umap(xy, labels, sim, centroids, title, umap_out,
                         point_size=args.point_size, dpi=args.dpi)
        log("  cosine_sim_umap.png")
        sheet_paths.append(umap_out)
        sheet_labels.append(title)

        if not args.no_heatmap:
            mat, labs = cluster_cosine_matrix(centroids)
            plot_cosine_heatmap(mat, labs, counts, title,
                                cell_out / "cluster_cosine_heatmap.png",
                                dpi=args.dpi)
            log("  cluster_cosine_heatmap.png")

    if sheet_paths:
        make_contact_sheet(sheet_paths, sheet_labels,
                           args.output_dir / "cosine_sim_contact_sheet.png",
                           ncols=args.contact_cols)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
