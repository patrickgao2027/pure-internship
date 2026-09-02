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

``--color-by sigprofiler``
    Every cluster painted by its *dominant SigProfiler-assigned signature* -- the actual
    fitting result, not a summary of raw counts. Reads each cell's
    ``Assignment_Solution_Activities.txt`` (one row per cluster, one column per COSMIC
    signature, values = mutations assigned to that signature) and picks the column with
    the largest count per cluster. By default this is the ``sigprofilerassignment_uv_only``
    run that exists for all 28 cells -- and because that reference set only contains UV
    signatures (SBS7A/B/C/D, SBS38), every cluster is forced into one of those 5 regardless
    of its true content (see Phase 0: uv_only cannot fit ~70% of clusters well, median
    cosine 0.28). Pass ``--sig-run cosmic_full_rerun`` for the 4 cells that have a correct
    full-COSMIC rerun instead.

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
# COSMIC's T>A is #CAC9C9, so noise has to sit well clear of it or the two read as one
# colour. Pale blue is the closest thing to "grey" that no substitution occupies.
NOISE_COLOUR = "#eceff4"

# The 5 signatures every uv_only run's Activities.txt is restricted to. A cosmic_full_rerun
# can contain any COSMIC signature, so its palette is built on the fly (tab20 cycle) instead.
UV_ONLY_SIGNATURES = ["SBS7A", "SBS7B", "SBS7C", "SBS7D", "SBS38"]
UV_ONLY_COLOURS = {"SBS7A": "#E32926", "SBS7B": "#03BCEE", "SBS7C": "#A1CE63",
                   "SBS7D": "#EBC6C4", "SBS38": "#7B4FA3"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coords", type=Path, default=DEFAULT_COORDS)
    p.add_argument("--cells-root", type=Path, default=DEFAULT_CELLS)
    p.add_argument("--cells", nargs="*", default=None,
                   help="cell labels to plot; default is a spread across the grid")
    p.add_argument("--all", action="store_true", help="plot every finished cell")
    p.add_argument("--color-by", choices=["cluster", "substitution", "sigprofiler"],
                   default="cluster")
    p.add_argument("--sig-run", default="sigprofilerassignment_uv_only_grch38_v3.5",
                   help="which SigProfiler output dir to read Activities.txt from "
                        "(default: the uv_only run present in all 28 cells; pass "
                        "'cosmic_full_rerun' for the 4 cells with the correct reference)")
    p.add_argument("--sample-rows", type=int, default=750_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--point-size", type=float, default=0.6)
    p.add_argument("--ncols", type=int, default=0, help="0 = choose automatically")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--output", type=Path, default=Path("umap_plots"),
                   help="output directory (created if needed); one PNG per cell")
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


def dominant_signature(activities_path: Path) -> tuple[dict[int, str], list[str]]:
    """cluster id -> name of that cluster's largest-activity SigProfiler signature.

    ``Assignment_Solution_Activities.txt``: one row per cluster (``Samples`` column holds
    ``cluster_<id>``), one column per signature the run's reference set contains, values
    are mutation counts SigProfiler assigned to that signature. Also returns the ordered
    list of signature columns actually present, so the caller can build a matching legend
    even for a non-uv_only run whose signature set isn't known ahead of time.
    """
    import polars as pl

    frame = pl.read_csv(activities_path, separator="\t")
    sig_columns = frame.columns[1:]
    ids = frame[:, 0].to_list()
    counts = frame.select(sig_columns).to_numpy()

    out: dict[int, str] = {}
    for row_id, row_counts in zip(ids, counts):
        match = re.search(r"(\d+)$", str(row_id))
        if match is None or row_counts.sum() <= 0:
            continue
        out[int(match.group(1))] = sig_columns[int(np.argmax(row_counts))]
    return out, list(sig_columns)


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


def panel(ax, xy, labels, title, colour_by, dominant, point_size, categories=None,
          colour_of=None):
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
        # One scatter call per category keeps the legend honest and avoids building a
        # 750k-long RGBA array.
        for cat in categories:
            ids = {cid for cid, s in dominant.items() if s == cat}
            if not ids:
                continue
            mask = clustered & np.isin(labels, np.fromiter(ids, dtype=labels.dtype))
            if mask.any():
                ax.scatter(xy[mask, 0], xy[mask, 1], s=point_size, c=colour_of[cat],
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

    out_dir = args.output if args.output.suffix == "" else args.output.parent / args.output.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for cell in cells:
        cell_dir = args.cells_root / cell
        labels = np.asarray(np.load(cell_dir / "cohort_labels.npy",
                                    mmap_mode="r")[positions])
        dominant: dict[int, str] = {}
        categories: list[str] = []
        colour_of: dict[str, str] = {}

        if args.color_by == "substitution":
            hits = sorted(cell_dir.glob("**/cluster_sbs96_matrix.tsv"))
            if not hits:
                print(f"  {cell}: no cluster_sbs96_matrix.tsv, drawing it grey")
            else:
                dominant = dominant_substitution(hits[0])
            categories, colour_of = SUBSTITUTIONS, SUB_COLOURS

        elif args.color_by == "sigprofiler":
            hits = sorted(cell_dir.glob(f"**/{args.sig_run}/**/Assignment_Solution_Activities.txt"))
            if not hits:
                print(f"  {cell}: no {args.sig_run} Activities.txt, drawing it grey")
            else:
                dominant, categories = dominant_signature(hits[0])
                colour_of = {s: UV_ONLY_COLOURS.get(
                    s, plt.get_cmap("tab20")(i % 20)) for i, s in enumerate(categories)}

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        n = panel(ax, xy, labels, cell, args.color_by, dominant, args.point_size,
                 categories, colour_of)

        if args.color_by in ("substitution", "sigprofiler"):
            handles = [Line2D([], [], marker="o", linestyle="", markersize=7,
                              color=colour_of[c], label=c) for c in categories]
            handles.append(Line2D([], [], marker="o", linestyle="", markersize=7,
                                  color=NOISE_COLOUR, label="noise"))
            fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7),
                       frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
            if args.color_by == "sigprofiler" and args.sig_run.startswith("sigprofilerassignment_uv_only"):
                fig.text(0.5, 0.965, "uv_only reference: every cluster forced into a UV "
                         "signature regardless of true content (see Phase 0)",
                         ha="center", fontsize=8, color="#a04040")

        # Leave headroom when the uv_only banner is drawn at y=0.965, or tight_layout gives
        # the axes the full height and the panel titles land on top of it.
        _bottom = 0.04 if args.color_by in ("substitution", "sigprofiler") else 0
        _top = 0.94 if (args.color_by == "sigprofiler"
                        and args.sig_run.startswith("sigprofilerassignment_uv_only")) else 1
        fig.tight_layout(rect=(0, _bottom, 1, _top))
        out_path = out_dir / f"{cell}_{args.color_by}.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  {cell:42} {n:5,} clusters  {100 * (labels < 0).mean():5.2f}% noise  -> {out_path.name}")

    print(f"saved {len(cells)} images -> {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
