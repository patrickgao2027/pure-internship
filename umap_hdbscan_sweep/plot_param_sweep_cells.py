#!/usr/bin/env python
"""UMAP scatter panels for finished param-sweep cells.

``plot_umap_clusters.py`` reads a stage-2 ``analysis.parquet`` with a ``cluster_label``
column. The parameter sweep does not produce that: it keeps one shared ``coords.npy`` and
writes a per-cell ``cohort_labels.npy`` beside ``metrics.json``. This script plots that
layout, several cells at a time, on shared axes so the panels are actually comparable.

Two colourings, and the second is the interesting one:

``--color-by cluster``
    Cyclic colour per cluster id. Shows granularity -- how finely the map has been cut.

``--color-by substitution``
    Every cluster painted by the substitution type of its *dominant* SBS96 channel, in the
    standard COSMIC six-colour palette. This is the picture behind the single-context
    finding: if the map resolves into large blocks of one colour, the clusters are sorting
    reads by trinucleotide context rather than by mutational process. Reads the per-cell
    ``cluster_sbs96_matrix.tsv`` that SigProfiler was already given, so it costs nothing.

Usage (on miletus)::

    python umap_hdbscan_sweep/plot_param_sweep_cells.py \
        --cells fit500000_mcs1000_ms15_eom fit1000000_mcs2500_ms5_eom \
        --color-by substitution --output umap_substitution.png

    # every finished cell, cluster colouring, 1M points each
    python umap_hdbscan_sweep/plot_param_sweep_cells.py --all --sample-rows 1000000

Memory: ``coords.npy`` and each ``cohort_labels.npy`` are opened with ``mmap_mode='r'``, so
only the sampled rows are ever read. Sampling positions are sorted before gathering, which
turns a random gather over 157.5M rows into a mostly-sequential one.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
DEFAULT_COORDS = REPO / "umap_tests" / "hdbscan_scaling" / "coords.npy"
DEFAULT_CELLS = REPO / "umap_tests" / "param_sweep" / "cells"

# COSMIC's standard SBS96 substitution palette.
SUBSTITUTIONS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
SUB_COLOURS = {"C>A": "#03BCEE", "C>G": "#000000", "C>T": "#E32926",
               "T>A": "#CAC9C9", "T>C": "#A1CE63", "T>G": "#EBC6C4"}
NOISE_COLOUR = "#d9d9d9"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coords", type=Path, default=DEFAULT_COORDS)
    p.add_argument("--cells-root", type=Path, default=DEFAULT_CELLS)
    p.add_argument("--cells", nargs="*", default=None,
                   help="cell labels to plot; default is a spread across the grid")
    p.add_argument("--all", action="store_true", help="plot every finished cell")
    p.add_argument("--color-by", choices=["cluster", "substitution"], default="cluster")
    p.add_argument("--sample-rows", type=int, default=750_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--point-size", type=float, default=0.6)
    p.add_argument("--ncols", type=int, default=0, help="0 = choose automatically")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--output", type=Path, default=Path("umap_param_sweep.png"))
    return p.parse_args()


def dominant_substitution(matrix_path: Path) -> dict[int, str]:
    """cluster id -> substitution type of that cluster's largest SBS96 channel.

    The matrix is the one handed to SigProfiler: 96 rows of ``A[C>T]G`` style channels,
    one column per cluster named ``cluster_<id>``.
    """
    import polars as pl

    frame = pl.read_csv(matrix_path, separator="\t")
    channels = frame[:, 0].to_list()
    subs = [c[c.index("[") + 1:c.index("]")] for c in channels]

    out: dict[int, str] = {}
    for name in frame.columns[1:]:
        match = re.search(r"(\d+)$", name)
        if match is None:
            continue
        counts = frame[name].to_numpy()
        if counts.sum() <= 0:
            continue
        out[int(match.group(1))] = subs[int(np.argmax(counts))]
    return out


def sample_positions(total: int, wanted: int, seed: int) -> np.ndarray:
    if wanted >= total:
        return np.arange(total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=wanted, replace=False))


def discover(cells_root: Path) -> list[str]:
    return sorted(d.name for d in cells_root.iterdir()
                  if (d / "cohort_labels.npy").exists())


def default_selection(available: list[str]) -> list[str]:
    """A spread across the grid rather than four near-identical panels."""
    wanted = ["fit500000_mcs2500_ms5_eom", "fit500000_mcs1000_ms15_eom",
              "fit1000000_mcs2500_ms5_eom", "fit5000000_mcs250_ms15_eom"]
    picked = [c for c in wanted if c in available]
    return picked or available[:4]


def panel(ax, xy, labels, title, colour_by, dominant, point_size):
    noise = labels < 0
    clustered = ~noise

    if noise.any():
        ax.scatter(xy[noise, 0], xy[noise, 1], s=point_size * 0.6, c=NOISE_COLOUR,
                   alpha=0.35, linewidths=0, rasterized=True)

    n_clusters = int(np.unique(labels[clustered]).size) if clustered.any() else 0

    if colour_by == "cluster":
        ax.scatter(xy[clustered, 0], xy[clustered, 1], s=point_size,
                   c=(labels[clustered] % 256) / 255.0, cmap="hsv",
                   alpha=0.45, linewidths=0, rasterized=True)
    else:
        # One scatter call per substitution type keeps the legend honest and avoids
        # building a 750k-long RGBA array.
        for sub in SUBSTITUTIONS:
            ids = {cid for cid, s in dominant.items() if s == sub}
            if not ids:
                continue
            mask = clustered & np.isin(labels, np.fromiter(ids, dtype=labels.dtype))
            if mask.any():
                ax.scatter(xy[mask, 0], xy[mask, 1], s=point_size, c=SUB_COLOURS[sub],
                           alpha=0.45, linewidths=0, rasterized=True)

    ax.set_title(f"{title}\n{n_clusters:,} clusters · {100 * noise.mean():.1f}% noise",
                 fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    return n_clusters


def main() -> int:
    args = parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    available = discover(args.cells_root)
    if not available:
        raise SystemExit(f"no cells with cohort_labels.npy under {args.cells_root}")
    cells = available if args.all else (args.cells or default_selection(available))
    missing = [c for c in cells if c not in available]
    if missing:
        raise SystemExit(f"no cohort_labels.npy for: {', '.join(missing)}")

    coords = np.load(args.coords, mmap_mode="r")
    positions = sample_positions(coords.shape[0], args.sample_rows, args.seed)
    xy = np.asarray(coords[positions], dtype=np.float32)
    print(f"coords {coords.shape[0]:,} rows; plotting {positions.size:,} sampled")

    ncols = args.ncols or (1 if len(cells) == 1 else min(3, len(cells)))
    nrows = int(np.ceil(len(cells) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 5.4 * nrows),
                             squeeze=False)

    for i, cell in enumerate(cells):
        cell_dir = args.cells_root / cell
        labels = np.asarray(np.load(cell_dir / "cohort_labels.npy",
                                    mmap_mode="r")[positions])
        dominant: dict[int, str] = {}
        if args.color_by == "substitution":
            hits = sorted(cell_dir.glob("**/cluster_sbs96_matrix.tsv"))
            if not hits:
                print(f"  {cell}: no cluster_sbs96_matrix.tsv, drawing it grey")
            else:
                dominant = dominant_substitution(hits[0])
        n = panel(axes[i // ncols][i % ncols], xy, labels, cell, args.color_by,
                  dominant, args.point_size)
        print(f"  {cell:32} {n:5,} clusters  {100 * (labels < 0).mean():5.2f}% noise")

    for j in range(len(cells), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if args.color_by == "substitution":
        handles = [Line2D([], [], marker="o", linestyle="", markersize=7,
                          color=SUB_COLOURS[s], label=s) for s in SUBSTITUTIONS]
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=7,
                              color=NOISE_COLOUR, label="noise"))
        fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
                   fontsize=9, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle("Clusters coloured by the substitution type of their dominant "
                     "SBS96 channel", fontsize=11)
    else:
        fig.suptitle("HDBSCAN clusters on the shared UMAP embedding", fontsize=11)

    fig.tight_layout(rect=(0, 0.03 if args.color_by == "substitution" else 0, 1, 0.97))
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
