#!/usr/bin/env python
"""Where each sample sits in the UMAP.

Three figures:

  contact_sheet.png         95 panels (10x10 grid), one per source parquet.
                            Grey = all 9.5M rows; colour = that sample's 100K rows.
  scatter_genotype.png      Single scatter, one colour per genotype, downsampled.
  genotype_centroids.png    95 centroids with 1-sigma ellipses, coloured by genotype,
                            sized by timepoint.

Sample names parsed as ``{genotype}{timepoint}-{replicate}-ppm{id}``,
e.g. ``ddbR12-b1-ppm0053.featuremap`` -> genotype ddb, timepoint R12.

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

GENOTYPE_COLOURS = {
    "wt":    "#3d6fb4",
    "csa":   "#d18b3c",
    "csb":   "#c4573f",
    "xpa":   "#4f9d69",
    "xpc":   "#2f7f8f",
    "xpd":   "#7fa93c",
    "ddb":   "#8a6bab",
    "other": "#8a8a8a",
}
FALLBACK_COLOURS = ["#b5793a", "#a8577e", "#6b8f3a", "#7a6ba8", "#3f8f7a",
                    "#9c6b4f", "#5f7fb5", "#8f8f3a", "#b0506b", "#4a6f8a"]
_assigned_colours: dict[str, str] = {}

# All repair timepoints in order, earliest first.
TIMEPOINT_ORDER = ["0", "R1", "R2", "R3", "R4", "R6", "R9", "R11", "R12"]

NAME_PATTERN = re.compile(r"^([A-Za-z]+?)(0|R\d+)-([^-]+)-ppm(\d+)")


def genotype_colour(genotype: str) -> str:
    if genotype in GENOTYPE_COLOURS:
        return GENOTYPE_COLOURS[genotype]
    if genotype not in _assigned_colours:
        _assigned_colours[genotype] = FALLBACK_COLOURS[
            len(_assigned_colours) % len(FALLBACK_COLOURS)]
    return _assigned_colours[genotype]


def genotype_order(present: set | dict) -> list[str]:
    known = [g for g in GENOTYPE_COLOURS if g in present]
    return known + sorted(g for g in present if g not in GENOTYPE_COLOURS)


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_sample_name(name: str) -> tuple[str, str]:
    match = NAME_PATTERN.match(name)
    if not match:
        return "other", "?"
    return match.group(1).lower(), match.group(2)


# ── figures ────────────────────────────────────────────────────────────────────

def plot_contact_sheet(xy: np.ndarray, ids: np.ndarray, names: list[str],
                       out_path: Path, ncols: int, dpi: int,
                       scatter_n: int = 30_000) -> None:
    """10x10 grid: grey background of all rows, coloured foreground = this sample."""
    nrows = math.ceil(len(names) / ncols)
    # Downsample the background once so each panel renders quickly.
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(len(xy), min(scatter_n, len(xy)), replace=False)
    bg = xy[bg_idx]

    figure, axes = plt.subplots(nrows, ncols,
                                figsize=(ncols * 1.8, nrows * 1.7), squeeze=False)
    for index in range(nrows * ncols):
        axis = axes[index // ncols][index % ncols]
        if index >= len(names):
            axis.axis("off")
            continue
        genotype, timepoint = parse_sample_name(names[index])
        colour = genotype_colour(genotype)
        pts = xy[ids == index]

        axis.scatter(bg[:, 0], bg[:, 1], s=0.3, c="#d0d0d0", linewidths=0,
                     rasterized=True)
        axis.scatter(pts[:, 0], pts[:, 1], s=0.4, c=[colour], linewidths=0,
                     alpha=0.4, rasterized=True)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#bbbbbb")
        axis.set_title(f"{genotype}{timepoint}", fontsize=5.5, pad=1.5,
                       color=colour)

    figure.suptitle(
        f"UMAP: source of each sample's rows  —  {len(names)} samples, "
        f"{len(xy):,} total rows",
        fontsize=9, y=1.001,
    )
    figure.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    log(f"  -> {out_path.name}  ({nrows}x{ncols} grid)")


def plot_scatter_genotype(xy: np.ndarray, ids: np.ndarray, names: list[str],
                          out_path: Path, dpi: int,
                          scatter_n_per_sample: int = 5_000) -> None:
    """Single scatter with one colour per genotype, downsampled for legibility."""
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(parse_sample_name(name)[0], []).append(index)

    figure, axis = plt.subplots(figsize=(9, 7))
    rng = np.random.default_rng(1)

    for genotype in genotype_order(groups):
        members = groups[genotype]
        mask = np.isin(ids, members)
        pts = xy[mask]
        if len(pts) == 0:
            continue
        sub = pts[rng.choice(len(pts), min(scatter_n_per_sample * len(members), len(pts)),
                             replace=False)]
        axis.scatter(sub[:, 0], sub[:, 1], s=1.0, c=[genotype_colour(genotype)],
                     linewidths=0, alpha=0.35, label=f"{genotype} ({len(members)})",
                     rasterized=True)

    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("UMAP coloured by genotype", fontsize=12)
    axis.legend(fontsize=9, frameon=False, loc="best", markerscale=6,
                title="genotype (n samples)", title_fontsize=9)
    axis.grid(True, linewidth=0.3, alpha=0.3)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    log(f"  -> {out_path.name}")


def plot_genotype_centroids(xy: np.ndarray, ids: np.ndarray, names: list[str],
                            out_path: Path, dpi: int) -> None:
    """Every sample as one centroid point, sized by timepoint, coloured by genotype."""
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
        colour = genotype_colour(genotype)
        tp_rank = TIMEPOINT_ORDER.index(timepoint) if timepoint in TIMEPOINT_ORDER else 0
        size = 30.0 + 15.0 * tp_rank

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
                        markerfacecolor=genotype_colour(g), markeredgecolor="white",
                        label=g)
                 for g in genotype_order(set(seen_genotypes))],
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

    parsed = [parse_sample_name(name) for name in names]
    tally: dict[str, int] = {}
    for genotype, _ in parsed:
        tally[genotype] = tally.get(genotype, 0) + 1
    log("  genotypes: " + ", ".join(f"{g}={tally[g]}" for g in genotype_order(tally)))
    timepoints = sorted({t for _, t in parsed},
                        key=lambda t: TIMEPOINT_ORDER.index(t) if t in TIMEPOINT_ORDER else 99)
    log(f"  timepoints: {', '.join(timepoints)}")
    unparsed = [n for n, (g, t) in zip(names, parsed) if g == "other"]
    if unparsed:
        log(f"  WARNING: {len(unparsed)} name(s) did not match pattern: "
            + ", ".join(unparsed[:5]) + (" ..." if len(unparsed) > 5 else ""))

    log("\nrendering ...")
    plot_contact_sheet(xy, ids, names,
                       args.output_dir / "contact_sheet.png",
                       args.contact_cols, args.dpi)
    plot_scatter_genotype(xy, ids, names,
                          args.output_dir / "scatter_genotype.png",
                          args.dpi)
    plot_genotype_centroids(xy, ids, names,
                            args.output_dir / "genotype_centroids.png",
                            args.dpi)
    log(f"\noutput: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
