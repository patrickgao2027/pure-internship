#!/usr/bin/env python
"""Contact sheet showing which UMAP region each source parquet occupies.

For each of the 95 samples in the stratified embedding produced by
build_stratified_embed.py:

  <sample>.png
      All reads in light grey; this sample's reads in colour.
      Immediately shows whether the sample is evenly spread (mixed) or
      clumped in a specific UMAP region (sample-specific signal).

  contact_sheet_sample_source.png
      All 95 mini-panels tiled in a 10 × 10 grid, labelled by sample name.
      One-glance overview of inter-sample variability.

Usage::

    python umap_hdbscan_sweep/plot_sample_source.py \\
        --stratified ~/pure-internship/umap_hdbscan_sweep/hdbscan/stratified_9M.parquet \\
        --output-dir ~/pure-internship/umap_hdbscan_sweep/hdbscan/sample_source_plots \\
        --dpi 150

Pass --no-per-sample to skip individual PNGs and only produce the contact sheet.
"""
from __future__ import annotations

import argparse
import math
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


# ── palette ────────────────────────────────────────────────────────────────────

_TAB = (list(plt.cm.tab20.colors) +
        list(plt.cm.tab20b.colors) +
        list(plt.cm.tab20c.colors))   # 60 distinct colours before cycling

def sample_colour(rank: int) -> str:
    return mc.to_hex(_TAB[rank % len(_TAB)])


def log(msg: str) -> None:
    print(msg, flush=True)


# ── loading ────────────────────────────────────────────────────────────────────

def load_stratified(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xy [N,2], source_ids [N] as int32) and the sorted sample names."""
    frame = pl.read_parquet(path, columns=["umap_1", "umap_2", "source_file"])
    xy = frame.select(["umap_1", "umap_2"]).to_numpy().astype(np.float32)
    names = sorted(frame["source_file"].drop_nulls().unique().to_list())
    name_to_id = {n: i for i, n in enumerate(names)}
    ids = np.array([name_to_id.get(v, -1)
                    for v in frame["source_file"].to_list()], dtype=np.int32)
    return xy, ids, names


# ── per-sample panel ───────────────────────────────────────────────────────────

def plot_single_sample(xy: np.ndarray, ids: np.ndarray,
                       sample_idx: int, sample_name: str,
                       colour: str,
                       out_path: Path,
                       point_size: float = 1.5, dpi: int = 150,
                       bg_alpha: float = 0.12,
                       bg_subsample: int = 200_000) -> None:
    """Full-resolution panel for one sample."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # grey backdrop — subsample to keep rendering fast
    bg_mask = ids != sample_idx
    bg_idx  = np.where(bg_mask)[0]
    if bg_idx.size > bg_subsample:
        rng = np.random.default_rng(0)
        bg_idx = rng.choice(bg_idx, size=bg_subsample, replace=False)
    ax.scatter(xy[bg_idx, 0], xy[bg_idx, 1],
               s=point_size, c="#bbbbbb", linewidths=0,
               alpha=bg_alpha, rasterized=True, zorder=1)

    # this sample
    fg_mask = ids == sample_idx
    ax.scatter(xy[fg_mask, 0], xy[fg_mask, 1],
               s=point_size * 1.5, c=colour, linewidths=0,
               alpha=0.70, rasterized=True, zorder=2)

    n = int(fg_mask.sum())
    ax.set_title(f"{sample_name}  ({n:,} reads)", fontsize=10)
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── contact sheet ──────────────────────────────────────────────────────────────

def _mini_panel(ax, xy: np.ndarray, ids: np.ndarray,
                sample_idx: int, sample_name: str,
                colour: str,
                point_size: float,
                bg_alpha: float,
                bg_xy: np.ndarray) -> None:
    """Draw one mini-panel onto an existing axis."""
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#cccccc")

    # grey backdrop (pre-subsampled array shared across all panels)
    ax.scatter(bg_xy[:, 0], bg_xy[:, 1],
               s=point_size * 0.4, c="#cccccc", linewidths=0,
               alpha=bg_alpha, rasterized=True, zorder=1)

    # this sample's points
    mask = ids == sample_idx
    ax.scatter(xy[mask, 0], xy[mask, 1],
               s=point_size * 0.7, c=colour, linewidths=0,
               alpha=0.72, rasterized=True, zorder=2)

    ax.set_title(sample_name, fontsize=5, pad=1.5, color=colour)


def plot_contact_sheet(xy: np.ndarray, ids: np.ndarray, names: list[str],
                       out_path: Path,
                       ncols: int = 10,
                       point_size: float = 1.5,
                       dpi: int = 150,
                       bg_subsample: int = 150_000) -> None:
    nrows = math.ceil(len(names) / ncols)

    # shared grey backdrop subsample (computed once)
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(len(xy), size=min(bg_subsample, len(xy)), replace=False)
    bg_xy  = xy[bg_idx]

    panel_w, panel_h = 1.6, 1.35
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * panel_w, nrows * panel_h),
                             squeeze=False)

    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            ax  = axes[row][col]
            if idx >= len(names):
                ax.axis("off")
                continue
            _mini_panel(ax, xy, ids, idx, names[idx],
                        sample_colour(idx), point_size, 0.18, bg_xy)

    fig.suptitle(
        f"Sample source contact sheet  —  {len(names)} samples  "
        f"({len(xy):,} total reads, equal representation)",
        fontsize=9, y=1.002
    )
    fig.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"contact sheet -> {out_path}  ({nrows}×{ncols}, {len(names)} panels)")


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stratified", type=Path, required=True,
                   help="output of build_stratified_embed.py "
                        "(must contain umap_1, umap_2, source_file)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--no-per-sample", action="store_true",
                   help="skip individual per-sample PNGs; only produce the contact sheet")
    p.add_argument("--point-size", type=float, default=1.5)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--contact-cols", type=int, default=10)
    p.add_argument("--bg-subsample", type=int, default=150_000,
                   help="grey-backdrop point count in contact sheet panels (default 150 K)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading {args.stratified} ...")
    xy, ids, names = load_stratified(args.stratified)
    log(f"  {xy.shape[0]:,} rows, {len(names)} unique samples")

    if not args.no_per_sample:
        log(f"\nrendering {len(names)} per-sample panels ...")
        for rank, name in enumerate(names):
            out = args.output_dir / f"{name}.png"
            plot_single_sample(xy, ids, rank, name, sample_colour(rank),
                               out, point_size=args.point_size, dpi=args.dpi)
            if (rank + 1) % 10 == 0 or rank == len(names) - 1:
                log(f"  [{rank+1:>3}/{len(names)}] {name}")

    log(f"\nrendering contact sheet ...")
    plot_contact_sheet(
        xy, ids, names,
        out_path=args.output_dir / "contact_sheet_sample_source.png",
        ncols=args.contact_cols,
        point_size=args.point_size,
        dpi=args.dpi,
        bg_subsample=args.bg_subsample,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
