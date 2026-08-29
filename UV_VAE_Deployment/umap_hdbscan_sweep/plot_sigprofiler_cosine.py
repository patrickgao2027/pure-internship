#!/usr/bin/env python
"""UMAP colored by SigProfiler signature-reconstruction cosine similarity.

SigProfiler assigns mutational signatures to each HDBSCAN cluster and reports
how well the assigned mixture reconstructs the cluster's observed SBS spectrum
(Cosine Similarity column of Assignment_Solution_Samples_Stats.txt).  This
script paints that per-cluster value onto every point in the UMAP so you can
see which regions of latent space have clean signature structure (high cosine
sim, yellow) versus ambiguous or mixed regions (low cosine sim, purple).

Points whose cluster is absent from the stats file (e.g. noise label -1, or
clusters too small for SigProfiler) are drawn in grey.

Inputs
------
--cells-dir      parent of the per-cell sigprofiler result dirs
                 (e.g. results/param_sweep/cells)
--analysis-dir   parent of the per-cell analysis.parquet dirs
                 (e.g. results/cohort_reports_original)
--enriched       enriched.parquet with umap_1 / umap_2 columns

The two dirs must share the same cell subdirectory names
(fit500000_mcs250_ms5_eom, etc.).

Usage::

    python umap_hdbscan_sweep/plot_sigprofiler_cosine.py \\
        --cells-dir    ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/param_sweep/cells \\
        --analysis-dir ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/cohort_reports_original \\
        --enriched     ~/pure-internship/umap_hdbscan_sweep/hdbscan/enriched.parquet \\
        --output-dir   ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/cluster_sweep_plots \\
        --sample-rows 2000000 --dpi 150
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
import math
import numpy as np
import polars as pl


# ── constants ──────────────────────────────────────────────────────────────────

_NOISE_COLOUR = "#cccccc"   # grey for noise / missing cosine sim


def log(msg: str) -> None:
    print(msg, flush=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_cell_label(path: Path) -> str:
    m = re.search(r"fit(\d+)_mcs(\d+)_ms(\d+)", path.name)
    if not m:
        return path.name
    fit = int(m.group(1))
    fit_s = f"{fit // 1_000_000}M" if fit >= 1_000_000 else f"{fit // 1_000}K"
    return f"fit {fit_s}  mcs {m.group(2)}  ms {m.group(3)}"


def find_stats_file(cell_dir: Path) -> Path | None:
    """Glob for Assignment_Solution_Samples_Stats.txt anywhere under cell_dir."""
    hits = list(cell_dir.glob(
        "sigprofilerassignment*/output/Assignment_Solution/"
        "Solution_Stats/Assignment_Solution_Samples_Stats.txt"))
    return hits[0] if hits else None


def parse_cosine_sim(stats_path: Path) -> dict[int, float]:
    """Return {cluster_label: cosine_similarity} from a SigProfiler stats file.

    Sample names are expected to be 'cluster_N'; any row whose name cannot be
    parsed to an integer is skipped with a warning.
    """
    mapping: dict[int, float] = {}
    with open(stats_path) as fh:
        header = None
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            cols = line.split("\t")
            if header is None:
                header = [c.strip() for c in cols]
                # find the cosine similarity column (case-insensitive)
                try:
                    cos_idx = next(i for i, h in enumerate(header)
                                   if "cosine" in h.lower())
                except StopIteration:
                    raise SystemExit(
                        f"No 'Cosine' column found in {stats_path}\n"
                        f"Header: {header}")
                continue
            sample_name = cols[0].strip()
            # extract integer label from e.g. "cluster_42"
            m = re.search(r"(\d+)$", sample_name)
            if m is None:
                log(f"  skipping unparseable sample name: {sample_name!r}")
                continue
            label = int(m.group(1))
            try:
                mapping[label] = float(cols[cos_idx])
            except (IndexError, ValueError):
                pass
    return mapping


# ── data loading ───────────────────────────────────────────────────────────────

def load_xy(enriched: Path, sample_rows: int, seed: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """Load UMAP coords and the row positions used."""
    total = pl.scan_parquet(enriched).select(pl.len()).collect().item()
    rng = np.random.default_rng(seed)
    if 0 < sample_rows < total:
        positions = np.sort(rng.choice(total, size=sample_rows, replace=False))
    else:
        positions = np.arange(total)
    frame = pl.read_parquet(enriched, columns=["umap_1", "umap_2"])
    if positions.size < frame.height:
        frame = frame[positions.tolist()]
    return frame.to_numpy().astype(np.float32), positions


def load_labels(analysis_parquet: Path, positions: np.ndarray) -> np.ndarray:
    frame = pl.read_parquet(analysis_parquet, columns=["cluster_label"])
    if positions.size < frame.height:
        frame = frame[positions.tolist()]
    return frame["cluster_label"].to_numpy().astype(np.int32)


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_sigprofiler_cosine_umap(
    xy: np.ndarray,
    labels: np.ndarray,
    cosine_map: dict[int, float],
    cell_label: str,
    out_path: Path,
    point_size: float = 1.5,
    dpi: int = 150,
) -> None:
    # assign per-point cosine sim; NaN for noise / missing clusters
    sim = np.full(len(labels), np.nan, dtype=np.float32)
    for lab, val in cosine_map.items():
        sim[labels == lab] = val

    has_sim = ~np.isnan(sim)
    no_sim  = ~has_sim          # noise or clusters not in stats file

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # grey background: points without a cosine sim
    if no_sim.any():
        ax.scatter(xy[no_sim, 0], xy[no_sim, 1], s=point_size,
                   c=_NOISE_COLOUR, linewidths=0, alpha=0.25,
                   rasterized=True, zorder=1)

    # coloured points: each coloured by its cluster's reconstruction quality
    if has_sim.any():
        sc = ax.scatter(xy[has_sim, 0], xy[has_sim, 1],
                        s=point_size, c=sim[has_sim],
                        cmap="viridis", vmin=0.0, vmax=1.0,
                        linewidths=0, alpha=0.78, rasterized=True, zorder=2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("SigProfiler cosine similarity\n(signature reconstruction quality)",
                     fontsize=8)

    n_clusters  = len(cosine_map)
    mean_cos    = float(np.nanmean(sim[has_sim])) if has_sim.any() else float("nan")
    pct_covered = 100 * has_sim.sum() / len(labels)

    ax.set_title(
        f"{cell_label}  |  SigProfiler cosine similarity\n"
        f"{n_clusters} clusters  ·  mean {mean_cos:.3f}  ·  "
        f"{pct_covered:.1f}% points covered",
        fontsize=9,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(umap_paths: list[Path], labels: list[str],
                        out_path: Path, ncols: int = 7, dpi: int = 120) -> None:
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
    fig.suptitle(
        "SigProfiler signature-reconstruction cosine similarity — all HDBSCAN parameterisations",
        fontsize=10, y=1.002)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"contact sheet -> {out_path}  ({nrows}×{ncols}, {len(umap_paths)} cells)")


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells-dir", type=Path, required=True,
                   help="parent dir of per-cell sigprofiler outputs "
                        "(e.g. results/param_sweep/cells)")
    p.add_argument("--analysis-dir", type=Path, required=True,
                   help="parent dir of per-cell analysis.parquet files "
                        "(e.g. results/cohort_reports_original)")
    p.add_argument("--enriched", type=Path, required=True,
                   help="enriched.parquet with umap_1 / umap_2")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-rows", type=int, default=2_000_000)
    p.add_argument("--contact-cols", type=int, default=7)
    p.add_argument("--point-size", type=float, default=1.5)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log("loading UMAP coordinates from enriched.parquet ...")
    xy, positions = load_xy(args.enriched, args.sample_rows, args.seed)
    log(f"  {xy.shape[0]:,} points")

    # cells defined by what has a sigprofiler stats file
    cell_dirs = sorted(d for d in args.cells_dir.iterdir() if d.is_dir())
    log(f"found {len(cell_dirs)} cell dirs under {args.cells_dir}")

    sheet_paths:  list[Path] = []
    sheet_labels: list[str]  = []

    for i, cell_dir in enumerate(cell_dirs, 1):
        cell_name  = cell_dir.name
        cell_label = parse_cell_label(cell_dir)
        log(f"[{i:>2}/{len(cell_dirs)}] {cell_name}")

        stats_path = find_stats_file(cell_dir)
        if stats_path is None:
            log("  no sigprofiler stats file found, skipping")
            continue

        analysis_parquet = args.analysis_dir / cell_name / "analysis.parquet"
        if not analysis_parquet.exists():
            log(f"  analysis.parquet not found at {analysis_parquet}, skipping")
            continue

        cosine_map = parse_cosine_sim(stats_path)
        log(f"  {len(cosine_map)} clusters with cosine sim  "
            f"(mean {np.mean(list(cosine_map.values())):.3f})")

        labels  = load_labels(analysis_parquet, positions)
        out_path = args.output_dir / cell_name / "sigprofiler_cosine_umap.png"
        plot_sigprofiler_cosine_umap(xy, labels, cosine_map, cell_label, out_path,
                                     point_size=args.point_size, dpi=args.dpi)
        log(f"  -> {out_path.name}")
        sheet_paths.append(out_path)
        sheet_labels.append(cell_label)

    if sheet_paths:
        make_contact_sheet(sheet_paths, sheet_labels,
                           args.output_dir / "sigprofiler_cosine_contact_sheet.png",
                           ncols=args.contact_cols)
    else:
        log("no cells produced output — check --cells-dir and --analysis-dir paths")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
