#!/usr/bin/env python
"""Where each sample sits in the UMAP, shown as density ratio rather than scatter.

Why not a scatter
-----------------
Measured on the 9.5 M stratified embed: centroid spread across the 95 samples is
0.95 / 1.79 in x / y, against a mean within-sample spread of 15.4 / 12.8. Every sample
covers essentially the whole embedding, so 100 K coloured points per panel saturate and
all 95 panels look identical whether or not structure exists. The scatter is not wrong,
it is uninformative at this density.

The signal that is actually there is a *systematic shift of a couple of units* -- csb0
and xpa0 sit near y = -16, the ddb series near y = -23. Two units inside a 13-unit cloud
is invisible in a scatter and obvious in a density ratio.

What this draws instead
-----------------------
For each sample, a 2-D histogram of its rows on a shared grid, divided by the pooled
histogram of all samples scaled to the same total. log2 of that ratio, on a diverging
colormap centred at 0:

    +1  =  twice as dense here as the cohort average
     0  =  exactly the cohort average
    -1  =  half as dense

Bins where the pooled count is too low to estimate a ratio are left blank rather than
rendered as extreme values -- a bin holding three cohort rows says nothing about a
two-fold difference, and colouring it would make noise the loudest thing on the page.

Three figures:

  contact_sheet_enrichment.png   95 panels, one per sample, log2 enrichment
  genotype_enrichment.png        one panel per genotype, samples pooled
  genotype_centroids.png         95 centroids with 1-sigma ellipses, coloured by
                                 genotype, sized by timepoint -- the whole
                                 between-sample story in one axes

Sample names are parsed as ``{genotype}{timepoint}-{replicate}-ppm{id}``, e.g.
``ddbR12-b1-ppm0053`` -> genotype ddb, timepoint R12. Names that do not match are
grouped under "other" rather than dropped.

Usage::

    python umap_hdbscan_sweep/plot_sample_enrichment.py \\
        --stratified ~/pure-internship/umap_hdbscan_sweep/hdbscan/stratified_9M.parquet \\
        --output-dir ~/pure-internship/umap_hdbscan_sweep/hdbscan/sample_source_plots
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
import numpy as np
import polars as pl

# Order matters only for consistent colouring across the three figures.
GENOTYPE_COLOURS = {
    "wt":    "#3d6fb4",
    "csb":   "#c4573f",
    "xpa":   "#4f9d69",
    "ddb":   "#8a6bab",
    "other": "#8a8a8a",
}
# Repair timepoints, earliest first; used for marker size in the centroid figure.
TIMEPOINT_ORDER = ["0", "R3", "R6", "R9", "R12"]

NAME_PATTERN = re.compile(r"^([A-Za-z]+?)(0|R\d+)-([^-]+)-ppm(\d+)")


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_sample_name(name: str) -> tuple[str, str]:
    """'ddbR12-b1-ppm0053.featuremap' -> ('ddb', 'R12'). Unmatched -> ('other', '?')."""
    match = NAME_PATTERN.match(name)
    if not match:
        return "other", "?"
    return match.group(1).lower(), match.group(2)


# ── histogram machinery ────────────────────────────────────────────────────────

def build_grid(xy: np.ndarray, bins: int, clip_percentile: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Shared bin edges, trimmed to the central percentile range.

    A handful of far-flung outliers would otherwise stretch the grid so that every
    sample's mass lands in a couple of central bins, which is the histogram equivalent of
    the saturation problem this module exists to avoid.
    """
    low, high = (100.0 - clip_percentile) / 2.0, 100.0 - (100.0 - clip_percentile) / 2.0
    x_low, x_high = np.percentile(xy[:, 0], [low, high])
    y_low, y_high = np.percentile(xy[:, 1], [low, high])
    return (np.linspace(x_low, x_high, bins + 1),
            np.linspace(y_low, y_high, bins + 1))


def histogram(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(x_edges, y_edges))
    return counts


def log2_enrichment(subset: np.ndarray, pooled: np.ndarray,
                    min_pooled: float) -> np.ndarray:
    """log2(observed / expected), NaN where the pooled count cannot support a ratio.

    ``expected`` scales the pooled histogram to the subset's total, so the map answers
    "denser or sparser than the cohort here", not "how many rows are here".
    """
    subset_total, pooled_total = subset.sum(), pooled.sum()
    if subset_total == 0 or pooled_total == 0:
        return np.full(subset.shape, np.nan)

    expected = pooled * (subset_total / pooled_total)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(expected > 0, subset / expected, np.nan)
        out = np.log2(ratio)
    out[pooled < min_pooled] = np.nan
    return out


def draw_enrichment(axis, values: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray,
                    vmax: float, title: str, title_size: float,
                    title_colour: str = "black"):
    # values is indexed [x, y] by histogram2d; imshow wants [row, col] = [y, x].
    image = axis.imshow(
        values.T, origin="lower", aspect="equal", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#bbbbbb")
    axis.set_title(title, fontsize=title_size, pad=2.0, color=title_colour)
    return image


# ── figures ────────────────────────────────────────────────────────────────────

def plot_sample_contact_sheet(xy: np.ndarray, ids: np.ndarray, names: list[str],
                              x_edges, y_edges, pooled, min_pooled: float,
                              vmax: float, out_path: Path, ncols: int, dpi: int) -> None:
    nrows = math.ceil(len(names) / ncols)
    figure, axes = plt.subplots(nrows, ncols,
                                figsize=(ncols * 1.7, nrows * 1.6), squeeze=False)
    image = None

    for index in range(nrows * ncols):
        axis = axes[index // ncols][index % ncols]
        if index >= len(names):
            axis.axis("off")
            continue
        genotype, timepoint = parse_sample_name(names[index])
        values = log2_enrichment(histogram(xy[ids == index], x_edges, y_edges),
                                 pooled, min_pooled)
        image = draw_enrichment(axis, values, x_edges, y_edges, vmax,
                                f"{genotype}{timepoint}", 5.0,
                                GENOTYPE_COLOURS.get(genotype, GENOTYPE_COLOURS["other"]))

    figure.suptitle(
        f"Per-sample UMAP density vs cohort average  —  {len(names)} samples, "
        f"{len(xy):,} rows, equal representation",
        fontsize=9, y=1.001,
    )
    figure.tight_layout(pad=0.3, rect=(0, 0.02, 1, 1))
    if image is not None:
        bar = figure.colorbar(image, ax=axes.ravel().tolist(),
                              orientation="horizontal", fraction=0.012, pad=0.015)
        bar.set_label("log2 (sample density / cohort density)", fontsize=7)
        bar.ax.tick_params(labelsize=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    log(f"  -> {out_path.name}  ({nrows}x{ncols})")


def plot_genotype_enrichment(xy: np.ndarray, ids: np.ndarray, names: list[str],
                             x_edges, y_edges, pooled, min_pooled: float,
                             vmax: float, out_path: Path, dpi: int) -> None:
    """One panel per genotype, its samples pooled -- more rows per bin, less noise."""
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(parse_sample_name(name)[0], []).append(index)
    ordered = [g for g in GENOTYPE_COLOURS if g in groups]

    figure, axes = plt.subplots(1, len(ordered),
                                figsize=(3.1 * len(ordered), 3.4), squeeze=False)
    image = None
    for column, genotype in enumerate(ordered):
        members = groups[genotype]
        values = log2_enrichment(histogram(xy[np.isin(ids, members)], x_edges, y_edges),
                                 pooled, min_pooled)
        image = draw_enrichment(axes[0][column], values, x_edges, y_edges, vmax,
                                f"{genotype}  ({len(members)} samples)", 9.0,
                                GENOTYPE_COLOURS[genotype])

    figure.suptitle("UMAP density vs cohort average, pooled by genotype", fontsize=10)
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    if image is not None:
        bar = figure.colorbar(image, ax=axes.ravel().tolist(),
                              orientation="horizontal", fraction=0.04, pad=0.06)
        bar.set_label("log2 (genotype density / cohort density)", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    log(f"  -> {out_path.name}  ({len(ordered)} genotypes)")


def plot_genotype_centroids(xy: np.ndarray, ids: np.ndarray, names: list[str],
                            out_path: Path, dpi: int) -> None:
    """Every sample as one point: centroid, 1-sigma ellipse, coloured by genotype.

    The between-sample differences are small relative to each sample's own spread, so
    this axes is deliberately zoomed to the centroids. The ellipses are drawn at
    0.05 sigma purely as an orientation cue -- at full scale they would all overlap and
    fill the frame, which is the same saturation problem in a different guise.
    """
    from matplotlib.patches import Ellipse
    from matplotlib.lines import Line2D

    figure, axis = plt.subplots(figsize=(8.0, 6.5))
    seen_genotypes: list[str] = []

    for index, name in enumerate(names):
        genotype, timepoint = parse_sample_name(name)
        points = xy[ids == index]
        if points.shape[0] < 2:
            continue
        cx, cy = float(points[:, 0].mean()), float(points[:, 1].mean())
        sx, sy = float(points[:, 0].std()), float(points[:, 1].std())
        colour = GENOTYPE_COLOURS.get(genotype, GENOTYPE_COLOURS["other"])
        size = 30.0 + 22.0 * (TIMEPOINT_ORDER.index(timepoint)
                              if timepoint in TIMEPOINT_ORDER else 0)

        axis.add_patch(Ellipse((cx, cy), width=0.10 * sx, height=0.10 * sy, fill=False,
                               edgecolor=colour, linewidth=0.7, alpha=0.55, zorder=2))
        axis.scatter([cx], [cy], s=size, c=[colour], linewidths=0.5,
                     edgecolors="white", alpha=0.9, zorder=3)
        axis.annotate(f"{genotype}{timepoint}", (cx, cy), xytext=(3, 3),
                      textcoords="offset points", fontsize=4.5, color=colour, zorder=4)
        if genotype not in seen_genotypes:
            seen_genotypes.append(genotype)

    axis.set_xlabel("UMAP 1  (sample centroid)")
    axis.set_ylabel("UMAP 2  (sample centroid)")
    axis.set_title("Sample centroids in the cohort UMAP\n"
                   "marker size = repair timepoint; ellipse = 0.05 sigma, orientation only",
                   fontsize=10)
    axis.grid(True, linewidth=0.3, alpha=0.3)
    axis.legend(
        handles=[Line2D([], [], marker="o", linestyle="", markersize=6,
                        markerfacecolor=GENOTYPE_COLOURS[g], markeredgecolor="white",
                        label=g)
                 for g in seen_genotypes],
        fontsize=8, frameon=False, loc="best", title="genotype", title_fontsize=8,
    )
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    log(f"  -> {out_path.name}")


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stratified", type=Path, required=True,
                        help="build_stratified_embed.py output (umap_1, umap_2, source_file)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=180,
                        help="grid resolution per axis (default 180)")
    parser.add_argument("--min-pooled", type=float, default=200.0,
                        help="pooled rows a bin needs before a ratio is drawn (default 200)")
    parser.add_argument("--vmax", type=float, default=1.5,
                        help="colour scale limit in log2 units (default 1.5 = 2.8x)")
    parser.add_argument("--clip-percentile", type=float, default=99.5,
                        help="central percentile range the grid spans (default 99.5)")
    parser.add_argument("--contact-cols", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading {args.stratified} ...")
    frame = pl.read_parquet(args.stratified, columns=["umap_1", "umap_2", "source_file"])
    xy = frame.select(["umap_1", "umap_2"]).to_numpy().astype(np.float32)
    names = sorted(frame["source_file"].drop_nulls().unique().to_list())
    lookup = {name: index for index, name in enumerate(names)}
    ids = np.array([lookup.get(value, -1) for value in frame["source_file"].to_list()],
                   dtype=np.int32)
    log(f"  {xy.shape[0]:,} rows, {len(names)} samples")

    x_edges, y_edges = build_grid(xy, args.bins, args.clip_percentile)
    pooled = histogram(xy, x_edges, y_edges)
    occupied = int((pooled >= args.min_pooled).sum())
    log(f"  grid {args.bins}x{args.bins}, {occupied:,} bins above the "
        f"{args.min_pooled:g}-row floor")

    log("\nrendering ...")
    plot_genotype_centroids(xy, ids, names,
                            args.output_dir / "genotype_centroids.png", args.dpi)
    plot_genotype_enrichment(xy, ids, names, x_edges, y_edges, pooled, args.min_pooled,
                             args.vmax, args.output_dir / "genotype_enrichment.png",
                             args.dpi)
    plot_sample_contact_sheet(xy, ids, names, x_edges, y_edges, pooled, args.min_pooled,
                              args.vmax,
                              args.output_dir / "contact_sheet_enrichment.png",
                              args.contact_cols, args.dpi)
    log(f"\noutput: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
