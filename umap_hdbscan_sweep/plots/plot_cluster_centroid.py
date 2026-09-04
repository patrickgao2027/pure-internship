#!/usr/bin/env python
"""Per-cell UMAP with cluster centroids annotated by their top distinguishing features.

For each cell in --results-dir produces:

  cluster_centroid_annotation.png
      UMAP scatter (2 M sampled rows).  Noise points are grey; the top-N clusters are
      coloured.  A star marker sits at each cluster centroid, labelled with the cluster
      ID and its 2-3 most distinguishing numeric features — measured by z-score of the
      within-cluster mean versus the whole-cohort mean.  ↑/↓ arrows show direction so
      a cluster that is high in VAF_ALT and low in DEPTH reads at a glance.

All cells share the same UMAP layout and feature values (enriched.parquet); only the
cluster partition differs, so the centroid positions shift but the scatter cloud is
identical across cells.

Usage::

    python umap_hdbscan_sweep/plot_cluster_centroid.py \\
        --results-dir ~/pure-internship/umap_hdbscan_sweep/hdbscan/cohort_reports_original \\
        --enriched    ~/pure-internship/umap_hdbscan_sweep/hdbscan/enriched.parquet \\
        --output-dir  ~/pure-internship/umap_hdbscan_sweep/hdbscan/cluster_sweep_plots \\
        --top-clusters 20 --top-features 3
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
import matplotlib.colors as mc
import numpy as np
import polars as pl


# ── constants ──────────────────────────────────────────────────────────────────

_NOISE_COLOUR   = "#cccccc"
_OTHER_COLOUR   = "#bbbbbb"
_NUMERIC_DTYPES = (pl.Float32, pl.Float64,
                   pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
_EXCLUDE        = {"umap_1", "umap_2", "cluster_label", "cluster_probability",
                   "POS", "__row", "locus_reads"}

_TAB_COLOURS = (list(plt.cm.tab20.colors) +
                list(plt.cm.tab20b.colors) +
                list(plt.cm.tab20c.colors))


def log(msg: str) -> None:
    print(msg, flush=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_cell_name(path: Path) -> dict:
    m = re.search(r"fit(\d+)_mcs(\d+)_ms(\d+)", path.name)
    if not m:
        return {"label": path.name}
    fit = int(m.group(1))
    fit_s = f"{fit // 1_000_000}M" if fit >= 1_000_000 else f"{fit // 1_000}K"
    return {"label": f"fit {fit_s}  mcs {m.group(2)}  ms {m.group(3)}"}


def find_cells(results_dir: Path) -> list[Path]:
    return sorted(
        (p for p in results_dir.iterdir()
         if p.is_dir() and (p / "analysis.parquet").exists()),
        key=lambda p: p.name,
    )


def load_enriched(path: Path, sample_rows: int, seed: int
                  ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Returns xy, features_np, feature_names, positions."""
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
    feat_np = frame.select(num_cols).to_numpy().astype(np.float32)

    return xy, feat_np, num_cols, positions


def load_labels(cell_path: Path, positions: np.ndarray) -> np.ndarray:
    frame = pl.read_parquet(cell_path / "analysis.parquet",
                            columns=["cluster_label"])
    if positions.size < frame.height:
        frame = frame[positions.tolist()]
    return frame["cluster_label"].to_numpy().astype(np.int32)


# ── centroid computation ───────────────────────────────────────────────────────

def compute_centroids(
    xy: np.ndarray,
    labels: np.ndarray,
    feat_np: np.ndarray,
    feat_names: list[str],
    top_clusters: int,
    top_features: int,
) -> list[dict]:
    """One dict per top cluster with centroid position, colour, and annotation text."""
    global_mean = np.nanmean(feat_np, axis=0)
    global_std  = np.nanstd(feat_np, axis=0)
    global_std[global_std == 0] = 1.0

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = np.argsort(-counts)[:top_clusters]

    results = []
    for rank, idx in enumerate(order):
        lab   = unique[idx]
        size  = int(counts[idx])
        mask  = labels == lab
        pts   = xy[mask]
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())

        feat_sub = feat_np[mask]
        cluster_mean = np.nanmean(feat_sub, axis=0)
        zscores = (cluster_mean - global_mean) / global_std

        top_fi = np.argsort(-np.abs(zscores))[:top_features]
        parts  = []
        for fi in top_fi:
            arrow = "↑" if zscores[fi] > 0 else "↓"
            parts.append(f"{feat_names[fi]}{arrow}")

        annotation = f"C{lab} (n={size:,})\n" + "\n".join(parts)
        colour = _TAB_COLOURS[rank % len(_TAB_COLOURS)]

        results.append(dict(lab=int(lab), size=size, rank=rank,
                            cx=cx, cy=cy, colour=colour, annotation=annotation))
    return results


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_centroid_annotation(
    xy: np.ndarray,
    labels: np.ndarray,
    centroids: list[dict],
    cell_meta: dict,
    out_path: Path,
    point_size: float = 1.5,
    dpi: int = 150,
) -> None:
    top_labs = {c["lab"] for c in centroids}
    colour_map = {c["lab"]: c["colour"] for c in centroids}

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # noise
    noise = labels == -1
    if noise.any():
        ax.scatter(xy[noise, 0], xy[noise, 1], s=point_size, c=_NOISE_COLOUR,
                   linewidths=0, alpha=0.25, rasterized=True, zorder=1)

    # non-top clusters (uniform dim grey so top clusters pop)
    other = (labels >= 0) & ~np.isin(labels, list(top_labs))
    if other.any():
        ax.scatter(xy[other, 0], xy[other, 1], s=point_size, c=_OTHER_COLOUR,
                   linewidths=0, alpha=0.20, rasterized=True, zorder=1)

    # top clusters — draw lowest-rank (smallest) first so large ones appear on top
    for c in sorted(centroids, key=lambda x: x["rank"], reverse=True):
        mask = labels == c["lab"]
        ax.scatter(xy[mask, 0], xy[mask, 1], s=point_size,
                   c=[c["colour"]], linewidths=0, alpha=0.70,
                   rasterized=True, zorder=2)

    # centroid markers + annotations
    for c in centroids:
        ax.scatter([c["cx"]], [c["cy"]], s=70, c=[c["colour"]], marker="*",
                   linewidths=0.5, edgecolors="black", zorder=8)
        ax.annotate(
            c["annotation"],
            xy=(c["cx"], c["cy"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=5, color=mc.to_hex(c["colour"]),
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.72),
            zorder=9,
        )

    n_noise = int(noise.sum())
    ax.set_title(
        f"{cell_meta['label']}  |  top {len(centroids)} clusters "
        f"annotated by feature z-score  ({n_noise / len(labels) * 100:.1f}% noise)",
        fontsize=9
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True,
                   help="directory containing cell subdirs (each with analysis.parquet)")
    p.add_argument("--enriched", type=Path, required=True,
                   help="enriched.parquet with umap_1/umap_2 and all source columns")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="root for output; cluster_centroid_annotation.png written per cell. "
                        "Pass same path as --results-dir to write into cell dirs.")
    p.add_argument("--sample-rows", type=int, default=2_000_000,
                   help="rows to use from enriched.parquet (0 = all; default 2M)")
    p.add_argument("--top-clusters", type=int, default=20,
                   help="clusters to annotate per figure, ranked by member count")
    p.add_argument("--top-features", type=int, default=3,
                   help="distinguishing features to label per cluster centroid")
    p.add_argument("--point-size", type=float, default=1.5)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cells = find_cells(args.results_dir)
    if not cells:
        raise SystemExit(f"no cell subdirs with analysis.parquet under {args.results_dir}")
    log(f"found {len(cells)} cells")

    log(f"loading enriched.parquet ({args.sample_rows:,} rows) ...")
    xy, feat_np, feat_names, positions = load_enriched(
        args.enriched, args.sample_rows, args.seed)
    log(f"  {xy.shape[0]:,} points, {len(feat_names)} numeric features for z-scoring")

    for i, cell in enumerate(cells, 1):
        meta = parse_cell_name(cell)
        log(f"[{i:>2}/{len(cells)}] {cell.name}")

        labels = load_labels(cell, positions)
        centroids = compute_centroids(
            xy, labels, feat_np, feat_names,
            args.top_clusters, args.top_features)
        if not centroids:
            log("  no clusters found, skipping")
            continue
        log(f"  {len(centroids)} cluster centroids computed")

        out_cell = args.output_dir / cell.name
        out_path = out_cell / "cluster_centroid_annotation.png"
        plot_centroid_annotation(xy, labels, centroids, meta, out_path,
                                 point_size=args.point_size, dpi=args.dpi)
        log(f"  -> {out_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
