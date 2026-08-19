#!/usr/bin/env python
"""Show which parquet file (sample) dominates each spatial region of the UMAP.

Bins the UMAP into a grid, finds the plurality source_file per bin, and
colours the UMAP accordingly. Also writes an Excel table:
rows = non-empty grid bins, columns = samples, values = row counts.

If --hdbscan-labels is given (a parquet with columns umap_1, umap_2, label)
the HDBSCAN cluster is used instead of a spatial grid bin, and the table
has one row per cluster label.

Usage::

    python umap_hdbscan_sweep/plot_cluster_dominance.py \\
        --stratified ~/pure-internship/umap_hdbscan_sweep/hdbscan/stratified_9M.parquet \\
        --output-dir ~/pure-internship/umap_hdbscan_sweep/hdbscan/sample_source_plots

    # with HDBSCAN labels already attached to the stratified parquet
    python umap_hdbscan_sweep/plot_cluster_dominance.py \\
        --stratified ~/pure-internship/umap_hdbscan_sweep/hdbscan/stratified_9M.parquet \\
        --label-col label \\
        --output-dir ~/pure-internship/umap_hdbscan_sweep/hdbscan/sample_source_plots
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
import matplotlib.patches as mpatches
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
    "other": "#aaaaaa",
}
FALLBACK_COLOURS = ["#b5793a", "#a8577e", "#6b8f3a", "#7a6ba8", "#3f8f7a",
                    "#9c6b4f", "#5f7fb5", "#8f8f3a", "#b0506b", "#4a6f8a"]
_assigned: dict[str, str] = {}

NAME_PATTERN = re.compile(r"^([A-Za-z]+?)(0|R\d+)-([^-]+)-ppm(\d+)")


def genotype_colour(g: str) -> str:
    if g in GENOTYPE_COLOURS:
        return GENOTYPE_COLOURS[g]
    if g not in _assigned:
        _assigned[g] = FALLBACK_COLOURS[len(_assigned) % len(FALLBACK_COLOURS)]
    return _assigned[g]


def hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def parse_genotype(name: str) -> str:
    m = NAME_PATTERN.match(name)
    return m.group(1).lower() if m else "other"


def log(msg: str) -> None:
    print(msg, flush=True)


# ── spatial grid binning ───────────────────────────────────────────────────────

def build_dominance_grid(xy: np.ndarray, genotypes: np.ndarray,
                         samples: np.ndarray, bins: int, min_count: int,
                         clip_pct: float) -> dict:
    """For each grid cell, find plurality genotype and sample."""
    lo = (100 - clip_pct) / 2
    hi = 100 - lo
    x0, x1 = np.percentile(xy[:, 0], [lo, hi])
    y0, y1 = np.percentile(xy[:, 1], [lo, hi])
    xe = np.linspace(x0, x1, bins + 1)
    ye = np.linspace(y0, y1, bins + 1)

    xi = np.clip(np.searchsorted(xe, xy[:, 0], side="right") - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(ye, xy[:, 1], side="right") - 1, 0, bins - 1)
    cell = xi * bins + yi

    unique_genotypes = sorted(set(genotypes))
    unique_samples   = sorted(set(samples))
    g_idx = {g: i for i, g in enumerate(unique_genotypes)}
    s_idx = {s: i for i, s in enumerate(unique_samples)}

    n_cells = bins * bins
    g_counts = np.zeros((n_cells, len(unique_genotypes)), dtype=np.int32)
    s_counts = np.zeros((n_cells, len(unique_samples)),   dtype=np.int32)

    np.add.at(g_counts, (cell, np.array([g_idx[g] for g in genotypes])), 1)
    np.add.at(s_counts, (cell, np.array([s_idx[s] for s in samples])),   1)

    total = g_counts.sum(axis=1)
    occupied = total >= min_count

    dominant_g = np.full(n_cells, -1, dtype=np.int32)
    dominant_s = np.full(n_cells, -1, dtype=np.int32)
    dominant_g[occupied] = g_counts[occupied].argmax(axis=1)
    dominant_s[occupied] = s_counts[occupied].argmax(axis=1)

    return {
        "xe": xe, "ye": ye, "bins": bins,
        "g_counts": g_counts, "s_counts": s_counts,
        "total": total, "occupied": occupied,
        "dominant_g": dominant_g, "dominant_s": dominant_s,
        "unique_genotypes": unique_genotypes,
        "unique_samples": unique_samples,
        "g_idx": g_idx, "s_idx": s_idx,
    }


# ── cluster-based (HDBSCAN label column) ──────────────────────────────────────

def build_cluster_dominance(labels: np.ndarray, genotypes: np.ndarray,
                            samples: np.ndarray) -> dict:
    """Per HDBSCAN cluster, count genotype and sample membership."""
    unique_labels    = sorted(set(labels))
    unique_genotypes = sorted(set(genotypes))
    unique_samples   = sorted(set(samples))
    l_idx = {l: i for i, l in enumerate(unique_labels)}
    g_idx = {g: i for i, g in enumerate(unique_genotypes)}
    s_idx = {s: i for i, s in enumerate(unique_samples)}

    n = len(unique_labels)
    g_counts = np.zeros((n, len(unique_genotypes)), dtype=np.int32)
    s_counts = np.zeros((n, len(unique_samples)),   dtype=np.int32)
    np.add.at(g_counts, (np.array([l_idx[l] for l in labels]),
                         np.array([g_idx[g] for g in genotypes])), 1)
    np.add.at(s_counts, (np.array([l_idx[l] for l in labels]),
                         np.array([s_idx[s] for s in samples])),   1)

    total = g_counts.sum(axis=1)
    dominant_g = g_counts.argmax(axis=1)
    dominant_s = s_counts.argmax(axis=1)

    return {
        "unique_labels": unique_labels, "l_idx": l_idx,
        "g_counts": g_counts, "s_counts": s_counts,
        "total": total,
        "dominant_g": dominant_g, "dominant_s": dominant_s,
        "unique_genotypes": unique_genotypes,
        "unique_samples": unique_samples,
    }


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_grid_dominance(xy: np.ndarray, grid: dict, out_path: Path, dpi: int,
                        mode: str = "genotype") -> None:
    """UMAP heatmap coloured by plurality genotype (or sample) per grid cell."""
    bins = grid["bins"]
    xe, ye = grid["xe"], grid["ye"]
    occupied = grid["occupied"]

    if mode == "genotype":
        dominant = grid["dominant_g"]
        labels   = grid["unique_genotypes"]
        colour_fn = genotype_colour
    else:
        dominant = grid["dominant_s"]
        labels   = grid["unique_samples"]
        colour_fn = lambda s: genotype_colour(parse_genotype(s))

    # Build an RGBA image (bins × bins)
    img = np.ones((bins, bins, 4))  # white, transparent

    for cell in np.where(occupied)[0]:
        gx, gy = cell // bins, cell % bins
        dom = dominant[cell]
        if dom < 0:
            continue
        r, g, b = hex_to_rgb(colour_fn(labels[dom]))
        # Blend strength by how dominant the plurality is
        total = grid["total"][cell]
        counts = grid["g_counts"][cell] if mode == "genotype" else grid["s_counts"][cell]
        frac = counts[dom] / total if total > 0 else 0
        alpha = 0.3 + 0.7 * min(frac * len(labels), 1.0)  # weak if near-uniform
        img[gx, gy] = [r, g, b, alpha]

    figure, axis = plt.subplots(figsize=(9, 7))
    # Grey background scatter for context
    rng = np.random.default_rng(0)
    bg = xy[rng.choice(len(xy), min(80_000, len(xy)), replace=False)]
    axis.scatter(bg[:, 0], bg[:, 1], s=0.3, c="#e0e0e0", linewidths=0,
                 rasterized=True, zorder=1)

    axis.imshow(img.transpose(1, 0, 2),  # (x,y) -> (row=y, col=x)
                origin="lower", aspect="auto", zorder=2,
                extent=(xe[0], xe[-1], ye[0], ye[-1]),
                interpolation="nearest")

    seen: set[str] = set()
    for cell in np.where(occupied)[0]:
        dom = dominant[cell]
        if dom < 0:
            continue
        lbl = labels[dom]
        g = lbl if mode == "genotype" else parse_genotype(lbl)
        seen.add(g)

    axis.legend(
        handles=[mpatches.Patch(facecolor=genotype_colour(g), label=g)
                 for g in sorted(seen, key=lambda x: list(GENOTYPE_COLOURS).index(x)
                                 if x in GENOTYPE_COLOURS else 99)],
        fontsize=9, frameon=False, title="plurality genotype", title_fontsize=9,
    )
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title(
        f"Plurality {'genotype' if mode == 'genotype' else 'sample'} per UMAP region\n"
        f"colour strength = how dominant the plurality is  ({bins}×{bins} grid)",
        fontsize=10,
    )
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    log(f"  -> {out_path.name}")


def plot_cluster_centroids(xy: np.ndarray, labels: np.ndarray, genotypes: np.ndarray,
                           cluster_info: dict, out_path: Path, dpi: int) -> None:
    """Scatter of HDBSCAN cluster centroids, coloured by plurality genotype."""
    ul = cluster_info["unique_labels"]
    ug = cluster_info["unique_genotypes"]
    dom_g = cluster_info["dominant_g"]

    figure, axis = plt.subplots(figsize=(9, 7))
    rng = np.random.default_rng(0)
    bg = xy[rng.choice(len(xy), min(80_000, len(xy)), replace=False)]
    axis.scatter(bg[:, 0], bg[:, 1], s=0.3, c="#e0e0e0", linewidths=0,
                 rasterized=True, zorder=1)

    seen: set[str] = set()
    for li, label in enumerate(ul):
        mask = labels == label
        pts = xy[mask]
        if len(pts) == 0:
            continue
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        genotype = ug[dom_g[li]]
        colour = genotype_colour(genotype)
        seen.add(genotype)
        size = max(20, min(800, cluster_info["total"][li] / 500))
        axis.scatter([cx], [cy], s=size, c=[colour], linewidths=0.5,
                     edgecolors="white", alpha=0.85, zorder=3)
        axis.annotate(str(label), (cx, cy), xytext=(2, 2),
                      textcoords="offset points", fontsize=5, color=colour, zorder=4)

    axis.legend(
        handles=[mpatches.Patch(facecolor=genotype_colour(g), label=g)
                 for g in sorted(seen, key=lambda x: list(GENOTYPE_COLOURS).index(x)
                                 if x in GENOTYPE_COLOURS else 99)],
        fontsize=9, frameon=False, title="plurality genotype", title_fontsize=9,
    )
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("HDBSCAN cluster centroids, coloured by plurality genotype\n"
                   "marker size ∝ cluster size", fontsize=10)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    log(f"  -> {out_path.name}")


# ── Excel export ───────────────────────────────────────────────────────────────

def write_excel_grid(grid: dict, out_path: Path) -> None:
    """One row per occupied grid bin, columns = all samples + summary cols."""
    occupied_cells = np.where(grid["occupied"])[0]
    bins = grid["bins"]
    s_counts = grid["s_counts"]
    g_counts = grid["g_counts"]
    ug = grid["unique_genotypes"]
    us = grid["unique_samples"]

    rows = []
    for cell in occupied_cells:
        gx, gy = cell // bins, cell % bins
        dom_g = grid["dominant_g"][cell]
        dom_s = grid["dominant_s"][cell]
        total = grid["total"][cell]
        row = {
            "bin_x": int(gx), "bin_y": int(gy),
            "total_rows": int(total),
            "dominant_genotype": ug[dom_g] if dom_g >= 0 else "",
            "dominant_sample":   us[dom_s] if dom_s >= 0 else "",
        }
        for gi, g in enumerate(ug):
            row[f"genotype_{g}"] = int(g_counts[cell, gi])
        for si, s in enumerate(us):
            row[s] = int(s_counts[cell, si])
        rows.append(row)

    df = pl.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_excel(out_path)
    log(f"  -> {out_path.name}  ({len(rows):,} occupied bins × {len(us)} samples)")


def write_excel_clusters(cluster_info: dict, out_path: Path) -> None:
    """One row per HDBSCAN cluster."""
    ul = cluster_info["unique_labels"]
    ug = cluster_info["unique_genotypes"]
    us = cluster_info["unique_samples"]
    g_counts = cluster_info["g_counts"]
    s_counts = cluster_info["s_counts"]
    dom_g = cluster_info["dominant_g"]
    dom_s = cluster_info["dominant_s"]
    total = cluster_info["total"]

    rows = []
    for li, label in enumerate(ul):
        row = {
            "cluster": label,
            "total_rows": int(total[li]),
            "dominant_genotype": ug[dom_g[li]],
            "dominant_sample":   us[dom_s[li]],
        }
        for gi, g in enumerate(ug):
            row[f"genotype_{g}"] = int(g_counts[li, gi])
        for si, s in enumerate(us):
            row[s] = int(s_counts[li, si])
        rows.append(row)

    df = pl.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_excel(out_path)
    log(f"  -> {out_path.name}  ({len(ul)} clusters × {len(us)} samples)")


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stratified", type=Path, required=True,
                   help="stratified_9M.parquet (umap_1, umap_2, source_file, optionally label)")
    p.add_argument("--label-col", default=None,
                   help="column name of HDBSCAN cluster labels in --stratified "
                        "(if absent, spatial grid binning is used)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bins", type=int, default=60,
                   help="grid resolution per axis when not using HDBSCAN labels (default 60)")
    p.add_argument("--min-count", type=int, default=50,
                   help="min rows in a grid bin before it is shown (default 50)")
    p.add_argument("--clip-pct", type=float, default=99.5)
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cols = ["umap_1", "umap_2", "source_file"]
    if args.label_col:
        cols.append(args.label_col)

    log(f"loading {args.stratified} ...")
    frame = pl.read_parquet(args.stratified, columns=cols)
    xy = frame.select(["umap_1", "umap_2"]).to_numpy().astype(np.float32)
    source = frame["source_file"].to_list()
    genotypes = np.array([parse_genotype(s) for s in source])
    samples   = np.array(source)

    log(f"  {len(xy):,} rows, {len(set(source)):,} unique samples")

    log("\nrendering ...")
    if args.label_col and args.label_col in frame.columns:
        labels = frame[args.label_col].to_numpy()
        unique_labels = sorted(set(labels))
        log(f"  using HDBSCAN labels from '{args.label_col}': "
            f"{len(unique_labels)} clusters  "
            f"(noise={sum(labels == -1):,} rows)" if -1 in set(labels) else
            f"  using HDBSCAN labels: {len(unique_labels)} clusters")

        ci = build_cluster_dominance(labels, genotypes, samples)
        plot_cluster_centroids(xy, labels, genotypes, ci,
                               args.output_dir / "cluster_dominance_centroids.png", args.dpi)
        write_excel_clusters(ci, args.output_dir / "cluster_sample_counts.xlsx")
    else:
        if args.label_col:
            log(f"  WARNING: --label-col '{args.label_col}' not found; "
                "falling back to spatial grid")
        log(f"  using {args.bins}×{args.bins} spatial grid  "
            f"(min {args.min_count} rows per bin)")
        grid = build_dominance_grid(xy, genotypes, samples,
                                    args.bins, args.min_count, args.clip_pct)
        occupied = int(grid["occupied"].sum())
        log(f"  {occupied:,} occupied bins of {args.bins**2:,}")

        plot_grid_dominance(xy, grid,
                            args.output_dir / "umap_plurality_genotype.png",
                            args.dpi, mode="genotype")
        write_excel_grid(grid, args.output_dir / "grid_sample_counts.xlsx")

    log(f"\noutput: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
