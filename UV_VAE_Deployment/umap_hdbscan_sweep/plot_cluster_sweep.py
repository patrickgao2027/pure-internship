#!/usr/bin/env python
"""Cluster-colored UMAPs and per-cluster feature profiles for every sweep cell.

For each cell subdirectory under --results-dir that contains an analysis.parquet:

  cluster_umap.png
      UMAP colored by cluster_label. Noise points (label = -1) are drawn in grey
      behind the clusters so they don't compete for colour. Tab20 cycles for
      > 20 clusters; a small legend shows the top-10 by size.

  cluster_feature_heatmap.png
      Top --top-clusters clusters (by member count) × all numeric features.
      Each cell = z-scored mean value for that feature within that cluster.
      Shows at a glance which features distinguish which clusters.

Plus one composite:

  cluster_contact_sheet.png
      All cells' cluster UMAPs tiled in a grid, labelled by fit / mcs / ms,
      for quick visual comparison of how HDBSCAN parameterisation changes the
      partition.

All cells share the same UMAP layout (same embedding, same 2 M sampled rows),
so the contact sheet is a direct apples-to-apples comparison.

Usage::

    python umap_hdbscan_sweep/plot_cluster_sweep.py \\
        --results-dir ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/cohort_reports_original \\
        --enriched    ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/enriched.parquet \\
        --output-dir  ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/cluster_sweep_plots \\
        --workers 8
"""
from __future__ import annotations

import argparse
import math
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
import matplotlib.colors as mcolors
import numpy as np
import polars as pl


# ── helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)


def parse_cell_name(path: Path) -> dict:
    """Parse fit / mcs / ms from a directory name like fit1000000_mcs250_ms5_eom."""
    m = re.search(r"fit(\d+)_mcs(\d+)_ms(\d+)", path.name)
    if not m:
        return {"fit": "?", "mcs": "?", "ms": "?", "label": path.name}
    fit = int(m.group(1))
    mcs = int(m.group(2))
    ms  = int(m.group(3))
    fit_str = f"{fit // 1_000_000}M" if fit >= 1_000_000 else f"{fit // 1_000}K"
    return {"fit": fit_str, "mcs": mcs, "ms": ms,
            "label": f"fit {fit_str}\nmcs {mcs} ms {ms}",
            "sort_key": (fit, mcs, ms)}


def find_cells(results_dir: Path) -> list[Path]:
    cells = sorted(
        (p for p in results_dir.iterdir()
         if p.is_dir() and (p / "analysis.parquet").exists()),
        key=lambda p: parse_cell_name(p).get("sort_key", (0, 0, 0))
    )
    return cells


def load_enriched(path: Path, sample_rows: int, seed: int
                  ) -> tuple[np.ndarray, pl.DataFrame]:
    """Load UMAP coordinates and numeric features from enriched.parquet."""
    total = pl.scan_parquet(path).select(pl.len()).collect().item()
    rng = np.random.default_rng(seed)
    if sample_rows > 0 and sample_rows < total:
        positions = np.sort(rng.choice(total, size=sample_rows, replace=False))
    else:
        positions = np.arange(total)

    frame = pl.read_parquet(path)
    if positions.size < total:
        frame = frame[positions.tolist()]

    xy = frame.select(["umap_1", "umap_2"]).to_numpy().astype(np.float32)
    non_coord = frame.drop(["umap_1", "umap_2", "cluster_label", "cluster_probability",
                             "POS"])
    numeric_cols = [c for c in non_coord.columns
                    if non_coord[c].dtype in (pl.Float32, pl.Float64,
                                              pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                              pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
                    and non_coord[c].null_count() < non_coord.height]
    features = non_coord.select(numeric_cols)
    return xy, features, positions


def load_labels(cell_path: Path, positions: np.ndarray) -> np.ndarray:
    """Load cluster_label for the same positional sample used in enriched."""
    frame = pl.read_parquet(cell_path / "analysis.parquet",
                            columns=["cluster_label"])
    if positions.size < frame.height:
        frame = frame[positions.tolist()]
    return frame["cluster_label"].to_numpy().astype(np.int32)


# ── per-cell plots ─────────────────────────────────────────────────────────────

_NOISE_COLOUR = "#cccccc"
_BASE_COLOURS = (
    list(plt.cm.tab20.colors) +          # 20
    list(plt.cm.tab20b.colors) +         # 20
    list(plt.cm.tab20c.colors)           # 20
)  # 60 colours before cycling


def _cluster_colours(n_clusters: int) -> list:
    colours = []
    for i in range(n_clusters):
        colours.append(_BASE_COLOURS[i % len(_BASE_COLOURS)])
    return colours


def plot_cluster_umap(xy: np.ndarray, labels: np.ndarray,
                      cell_meta: dict, out_path: Path,
                      point_size: float = 1.5, dpi: int = 150) -> None:
    unique = sorted(l for l in np.unique(labels) if l >= 0)
    colours = _cluster_colours(len(unique))
    colour_map = {lab: colours[i] for i, lab in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    # noise first, behind everything
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(xy[noise_mask, 0], xy[noise_mask, 1],
                   s=point_size, c=_NOISE_COLOUR, linewidths=0, rasterized=True,
                   alpha=0.4, zorder=1)

    # clusters
    for lab in unique:
        mask = labels == lab
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   s=point_size, c=[colour_map[lab]], linewidths=0,
                   rasterized=True, alpha=0.7, zorder=2)

    # legend: top 10 by size
    sizes = {lab: int((labels == lab).sum()) for lab in unique}
    top10 = sorted(unique, key=lambda l: -sizes[l])[:10]
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=colour_map[l], markersize=6,
                          label=f"C{l} ({sizes[l]:,})")
               for l in top10]
    if len(unique) > 10:
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor="#aaa", markersize=6,
                                  label=f"…+{len(unique) - 10} more"))
    ax.legend(handles=handles, fontsize=6, loc="upper right",
              framealpha=0.7, handlelength=0.8)

    n_noise = int(noise_mask.sum())
    noise_pct = 100 * n_noise / len(labels)
    ax.set_title(
        f"{cell_meta['label'].replace(chr(10), '  ')}   "
        f"| {len(unique)} clusters  {noise_pct:.1f}% noise",
        fontsize=9
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_feature_heatmap(xy: np.ndarray, labels: np.ndarray,
                         features: pl.DataFrame, cell_meta: dict,
                         out_path: Path, top_n: int = 30, dpi: int = 150) -> None:
    """Heatmap: top_n clusters × numeric features, z-scored mean per cluster."""
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = unique[np.argsort(-counts)][:top_n]

    feat_np = features.to_numpy().astype(np.float32)
    col_names = features.columns

    # mean per cluster, z-scored across clusters
    matrix = np.zeros((len(order), len(col_names)), dtype=np.float32)
    for i, lab in enumerate(order):
        mask = labels == lab
        with np.errstate(invalid="ignore"):
            matrix[i] = np.nanmean(feat_np[mask], axis=0)

    col_std = np.nanstd(matrix, axis=0)
    col_std[col_std == 0] = 1.0
    zscored = (matrix - np.nanmean(matrix, axis=0)) / col_std

    fig_h = max(5, len(order) * 0.22 + 1.5)
    fig_w = max(8, len(col_names) * 0.18 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(zscored, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="z-score (mean per cluster)", fraction=0.02, pad=0.02)

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=90, fontsize=6)
    y_labels = [f"C{lab} (n={counts[list(unique).index(lab)]:,})" for lab in order]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_title(
        f"{cell_meta['label'].replace(chr(10), '  ')}  "
        f"| top {len(order)} clusters × {len(col_names)} features  (z-scored mean)",
        fontsize=9
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── contact sheet ──────────────────────────────────────────────────────────────

def make_contact_sheet(umap_paths: list[Path], labels: list[str],
                       out_path: Path, ncols: int = 7, dpi: int = 120) -> None:
    from PIL import Image  # lazy import; only needed here

    imgs = [Image.open(p) for p in umap_paths]
    w, h = imgs[0].size
    nrows = math.ceil(len(imgs) / ncols)
    canvas = Image.new("RGB", (w * ncols, h * nrows), color=(255, 255, 255))
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, ncols)
        canvas.paste(img, (c * w, r * h))
    canvas.save(out_path, dpi=(dpi, dpi))
    log(f"contact sheet -> {out_path}  ({nrows}×{ncols}, {len(imgs)} cells)")


def make_contact_sheet_matplotlib(umap_paths: list[Path], labels: list[str],
                                  out_path: Path, ncols: int = 7, dpi: int = 120) -> None:
    """Fallback that doesn't need Pillow."""
    nrows = math.ceil(len(umap_paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.8))
    axes = np.array(axes).reshape(-1)
    for i, (path, label) in enumerate(zip(umap_paths, labels)):
        img = plt.imread(path)
        axes[i].imshow(img)
        axes[i].set_title(label, fontsize=6)
        axes[i].axis("off")
    for j in range(len(umap_paths), len(axes)):
        axes[j].axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"contact sheet -> {out_path}  ({nrows}×{ncols}, {len(umap_paths)} cells)")


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True,
                   help="directory containing cell subdirs (each with analysis.parquet)")
    p.add_argument("--enriched", type=Path, required=True,
                   help="enriched.parquet with umap_1/umap_2 and all 70 source columns")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-rows", type=int, default=2_000_000,
                   help="rows to sample from enriched.parquet (0 = all; default 2M)")
    p.add_argument("--top-clusters", type=int, default=30,
                   help="clusters shown in heatmap rows (by descending size)")
    p.add_argument("--point-size", type=float, default=1.5)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--contact-cols", type=int, default=7,
                   help="columns in the contact sheet grid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-heatmap", action="store_true",
                   help="skip per-cell heatmaps (faster if you only want contact sheet)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cells = find_cells(args.results_dir)
    if not cells:
        raise SystemExit(f"no cell subdirs with analysis.parquet under {args.results_dir}")
    log(f"found {len(cells)} cells")

    log(f"loading enriched.parquet ({args.sample_rows:,} rows) ...")
    xy, features, positions = load_enriched(args.enriched, args.sample_rows, args.seed)
    log(f"  {xy.shape[0]:,} points, {len(features.columns)} numeric features")

    umap_paths: list[Path] = []
    contact_labels: list[str] = []

    for i, cell in enumerate(cells, 1):
        meta = parse_cell_name(cell)
        log(f"[{i:>2}/{len(cells)}] {cell.name}")
        cell_out = args.output_dir / cell.name
        cell_out.mkdir(exist_ok=True)

        labels = load_labels(cell, positions)

        umap_out = cell_out / "cluster_umap.png"
        plot_cluster_umap(xy, labels, meta, umap_out,
                          point_size=args.point_size, dpi=args.dpi)
        log(f"  cluster_umap.png")

        if not args.no_heatmap:
            heatmap_out = cell_out / "cluster_feature_heatmap.png"
            plot_feature_heatmap(xy, labels, features, meta, heatmap_out,
                                 top_n=args.top_clusters, dpi=args.dpi)
            log(f"  cluster_feature_heatmap.png")

        umap_paths.append(umap_out)
        contact_labels.append(meta["label"].replace("\n", " "))

    # contact sheet
    contact_out = args.output_dir / "cluster_contact_sheet.png"
    try:
        make_contact_sheet(umap_paths, contact_labels, contact_out,
                           ncols=args.contact_cols, dpi=120)
    except ImportError:
        log("Pillow not available; using matplotlib fallback for contact sheet")
        make_contact_sheet_matplotlib(umap_paths, contact_labels, contact_out,
                                      ncols=args.contact_cols, dpi=120)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
