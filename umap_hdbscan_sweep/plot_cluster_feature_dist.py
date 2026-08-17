#!/usr/bin/env python
"""Per-cluster feature distributions for every sweep cell.

For each cell, produces two figures written into <cell>/feature_distributions/:

  feature_dist_numeric.png
      Grid of box plots, one panel per numeric feature.  X-axis = top-N clusters
      (sorted by member count, largest first).  Each box shows Q1/median/Q3 plus
      1.5 × IQR whiskers, computed directly from the 2 M sample so no data is
      discarded.  Title shows feature name; cluster labels are on the x-axis.
      A horizontal grey band shows the whole-cohort Q1-Q3 for reference.

  feature_dist_categorical.png
      Grid of stacked bar charts, one panel per categorical feature.  Each bar is
      one cluster; segments are the fraction of reads in each category level.
      Top-12 levels are coloured; the rest collapse to "other".

Both figures share the same set of top-N clusters and the same row sample, so
comparisons between features within a cell are apples-to-apples.

Only the cluster_label column is cell-specific; all feature values come from
enriched.parquet, which is loaded once and reused across all cells.

Usage::

    python umap_hdbscan_sweep/plot_cluster_feature_dist.py \\
        --results-dir ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/cohort_reports_original \\
        --enriched    ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/enriched.parquet \\
        --output-dir  ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/cohort_reports_original \\
        --top-clusters 20 --workers 4

Pass --output-dir the same path as --results-dir to write feature_distributions/
directly into each cell's existing directory.
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


# ── constants ──────────────────────────────────────────────────────────────────

_NUMERIC_DTYPES = (pl.Float32, pl.Float64,
                   pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
_CATEGORICAL_DTYPES = (pl.Utf8, pl.String, pl.Categorical, pl.Boolean,
                       pl.Int8, pl.UInt8)  # small-int cols may be categorical

# Columns that exist in analysis / enriched but are not features.
_EXCLUDE = {"umap_1", "umap_2", "cluster_label", "cluster_probability", "POS",
            "__row", "locus_reads"}

_MAX_CAT_LEVELS = 12        # levels beyond this collapse to "other"
_OTHER_COLOUR    = "#bbbbbb"

# tab20 + tab20b for categorical palettes
_CAT_PALETTE = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)


def log(msg: str) -> None:
    print(msg, flush=True)


# ── helpers ────────────────────────────────────────────────────────────────────

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
                  ) -> tuple[np.ndarray, pl.DataFrame]:
    total = pl.scan_parquet(path).select(pl.len()).collect().item()
    rng = np.random.default_rng(seed)
    if 0 < sample_rows < total:
        positions = np.sort(rng.choice(total, size=sample_rows, replace=False))
    else:
        positions = np.arange(total)
    frame = pl.read_parquet(path)
    if positions.size < frame.height:
        frame = frame[positions.tolist()]
    return frame, positions


def load_labels(cell_path: Path, positions: np.ndarray) -> np.ndarray:
    frame = pl.read_parquet(cell_path / "analysis.parquet",
                            columns=["cluster_label"])
    if positions.size < frame.height:
        frame = frame[positions.tolist()]
    return frame["cluster_label"].to_numpy().astype(np.int32)


def top_clusters(labels: np.ndarray, n: int) -> list[int]:
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = np.argsort(-counts)
    return [int(unique[i]) for i in order[:n]]


# ── box-plot statistics ────────────────────────────────────────────────────────

def boxstats(values: np.ndarray) -> dict:
    """Pre-compute box stats so matplotlib draws from pre-aggregated data."""
    v = values[~np.isnan(values)]
    if v.size == 0:
        return {"med": 0, "q1": 0, "q3": 0, "whislo": 0, "whishi": 0, "fliers": []}
    q1, med, q3 = float(np.percentile(v, 25)), float(np.percentile(v, 50)), float(np.percentile(v, 75))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    whislo = float(v[v >= lo].min()) if (v >= lo).any() else q1
    whishi = float(v[v <= hi].max()) if (v <= hi).any() else q3
    return {"med": med, "q1": q1, "q3": q3, "whislo": whislo, "whishi": whishi, "fliers": []}


# ── numeric figure ─────────────────────────────────────────────────────────────

def _is_likely_categorical(col: pl.Series, numeric_thresh: int = 20) -> bool:
    """Small-integer columns with few unique values are better shown as bars."""
    if col.dtype not in (pl.Int8, pl.UInt8, pl.Int16, pl.UInt16):
        return False
    n_unique = col.drop_nulls().n_unique()
    return n_unique <= numeric_thresh


def plot_numeric_distributions(
    frame: pl.DataFrame,
    labels: np.ndarray,
    ordered_clusters: list[int],
    cell_title: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    numeric_cols = [
        c for c in frame.columns
        if c not in _EXCLUDE
        and frame[c].dtype in _NUMERIC_DTYPES
        and not _is_likely_categorical(frame[c])
        and frame[c].null_count() < frame.height
    ]
    if not numeric_cols:
        return

    ncols_grid = 6
    nrows_grid = int(np.ceil(len(numeric_cols) / ncols_grid))
    fig_w = max(18, ncols_grid * 3.2)
    fig_h = max(10, nrows_grid * 2.8)

    fig, axes = plt.subplots(nrows_grid, ncols_grid,
                             figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(f"{cell_title}  |  numeric feature distributions  "
                 f"(top {len(ordered_clusters)} clusters by size)",
                 fontsize=11, y=1.002)

    # precompute per-cluster masks once
    cluster_masks = {lab: (labels == lab) for lab in ordered_clusters}
    x_positions = list(range(len(ordered_clusters)))
    x_labels = [f"C{lab}" for lab in ordered_clusters]

    for idx, col in enumerate(numeric_cols):
        r, c = divmod(idx, ncols_grid)
        ax = axes[r][c]
        values = frame[col].to_numpy(allow_copy=True).astype(float)

        # whole-cohort Q1/Q3 as background reference band
        v_all = values[~np.isnan(values)]
        if v_all.size > 0:
            cq1, cq3 = np.percentile(v_all, [25, 75])
            ax.axhspan(cq1, cq3, color="#e0e0e0", alpha=0.5, zorder=0,
                       label="cohort Q1–Q3")

        stats = [boxstats(values[cluster_masks[lab]]) for lab in ordered_clusters]
        ax.bxp(stats, positions=x_positions, widths=0.6,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#4c72b0", alpha=0.75),
               medianprops=dict(color="#c44e52", linewidth=1.5),
               whiskerprops=dict(linewidth=0.8),
               capprops=dict(linewidth=0.8))

        ax.set_title(col, fontsize=8, pad=2)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    # blank unused panels
    for idx in range(len(numeric_cols), nrows_grid * ncols_grid):
        r, c = divmod(idx, ncols_grid)
        axes[r][c].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── categorical figure ─────────────────────────────────────────────────────────

def plot_categorical_distributions(
    frame: pl.DataFrame,
    labels: np.ndarray,
    ordered_clusters: list[int],
    cell_title: str,
    out_path: Path,
    max_levels: int = _MAX_CAT_LEVELS,
    dpi: int = 150,
) -> None:
    cat_cols = [
        c for c in frame.columns
        if c not in _EXCLUDE
        and (frame[c].dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Boolean)
             or _is_likely_categorical(frame[c]))
        and frame[c].null_count() < frame.height
    ]
    if not cat_cols:
        return

    ncols_grid = 4
    nrows_grid = int(np.ceil(len(cat_cols) / ncols_grid))
    fig_w = max(16, ncols_grid * 4.5)
    fig_h = max(10, nrows_grid * 3.5)

    fig, axes = plt.subplots(nrows_grid, ncols_grid,
                             figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(f"{cell_title}  |  categorical feature distributions  "
                 f"(top {len(ordered_clusters)} clusters by size)",
                 fontsize=11, y=1.002)

    cluster_masks = {lab: (labels == lab) for lab in ordered_clusters}
    x_positions = np.arange(len(ordered_clusters))
    x_labels = [f"C{lab}" for lab in ordered_clusters]

    for idx, col in enumerate(cat_cols):
        r, c = divmod(idx, ncols_grid)
        ax = axes[r][c]

        # resolve levels: top max_levels by whole-cohort frequency
        raw = frame[col].cast(pl.Utf8).fill_null("__null__")
        level_counts = raw.value_counts(sort=True)
        top_levels = level_counts[""].to_list()[:max_levels]
        has_other = level_counts.height > max_levels

        raw_np = raw.to_numpy(allow_copy=True)
        bottoms = np.zeros(len(ordered_clusters))

        for li, level in enumerate(top_levels):
            fracs = np.array([
                float((raw_np[cluster_masks[lab]] == level).mean())
                for lab in ordered_clusters
            ])
            colour = _CAT_PALETTE[li % len(_CAT_PALETTE)]
            ax.bar(x_positions, fracs, bottom=bottoms,
                   color=colour, width=0.75, label=str(level))
            bottoms += fracs

        if has_other:
            other_fracs = 1.0 - bottoms
            other_fracs = np.clip(other_fracs, 0, 1)
            ax.bar(x_positions, other_fracs, bottom=bottoms,
                   color=_OTHER_COLOUR, width=0.75, label="other")

        ax.set_title(col, fontsize=9, pad=2)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)

        n_legend = min(max_levels, level_counts.height) + (1 if has_other else 0)
        if n_legend <= 8:
            ax.legend(fontsize=5, loc="upper right", framealpha=0.6,
                      ncol=1, handlelength=0.8, borderpad=0.3)

    for idx in range(len(cat_cols), nrows_grid * ncols_grid):
        r, c = divmod(idx, ncols_grid)
        axes[r][c].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True,
                   help="directory containing cell subdirs (each with analysis.parquet)")
    p.add_argument("--enriched", type=Path, required=True,
                   help="enriched.parquet with umap_1/umap_2 and all 70 source columns")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="root for output; feature_distributions/ is created under each cell. "
                        "Pass the same path as --results-dir to write into existing cell dirs.")
    p.add_argument("--sample-rows", type=int, default=2_000_000,
                   help="rows to use from enriched.parquet (0 = all; default 2M)")
    p.add_argument("--top-clusters", type=int, default=20,
                   help="number of clusters to show per figure, ranked by member count")
    p.add_argument("--max-cat-levels", type=int, default=_MAX_CAT_LEVELS,
                   help="categorical levels shown individually; the rest collapse to 'other'")
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
    frame, positions = load_enriched(args.enriched, args.sample_rows, args.seed)
    log(f"  {frame.height:,} rows, {frame.width} columns")

    # Drop columns that are pure pipeline outputs and not features
    feature_frame = frame.drop([c for c in _EXCLUDE if c in frame.columns])

    for i, cell in enumerate(cells, 1):
        meta = parse_cell_name(cell)
        log(f"[{i:>2}/{len(cells)}] {cell.name}")

        labels = load_labels(cell, positions)
        ordered = top_clusters(labels, args.top_clusters)
        if not ordered:
            log(f"  no clusters found, skipping")
            continue
        log(f"  top {len(ordered)} clusters of "
            f"{int((labels >= 0).sum().item() if hasattr((labels >= 0).sum(), 'item') else (labels >= 0).sum()):,} "
            f"clustered rows")

        out_cell = args.output_dir / cell.name / "feature_distributions"

        plot_numeric_distributions(
            feature_frame, labels, ordered, meta,
            out_cell / "feature_dist_numeric.png", dpi=args.dpi)
        log(f"  feature_dist_numeric.png")

        plot_categorical_distributions(
            feature_frame, labels, ordered, meta,
            out_cell / "feature_dist_categorical.png",
            max_levels=args.max_cat_levels, dpi=args.dpi)
        log(f"  feature_dist_categorical.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
