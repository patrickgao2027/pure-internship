#!/usr/bin/env python
"""``write_cluster_profiles``, rebuilt to survive 500+ clusters, with the text box fixed.

``run_variant_cluster_pipeline.write_cluster_profiles`` is left untouched -- every result in
this repo was produced with it and it stays the reference. This is a drop-in replacement for
the cohort scale it was never asked to handle, and it is a strict rewrite of only the parts
that do not survive the jump from 8 clusters to 505:

**One scan, not one per cluster.** The original calls ``read_cluster_points`` inside the
cluster loop, so an N-cluster run reads ``analysis.parquet`` N times. At 8 clusters that is
invisible; at 505 it is the whole runtime. ``read_cluster_points_by_label`` reads the file
once and splits it by label with a sort -- O(rows log rows), independent of cluster count.

**Panels rendered in parallel.** Each panel is 13 matplotlib axes rasterised at 180 dpi, which
is CPU-bound rasterisation with no array maths for a GPU to take over (cuDF/cuML accelerate
the clustering, not Agg). The work is embarrassingly parallel across clusters, so it goes to a
process pool. The shared UMAP backdrop is pushed to workers once via the pool initializer
rather than pickled per task.

**The summary box moved inside the axes.** The original anchors it at ``x=1.02`` in UMAP-axes
coordinates -- in the gap between the UMAP panel and the ``A>C`` bar chart, where it lands on
top of that chart's y-axis and first bars. It is unreadable in every profile of the
2026-07-09 trial run. Here it sits at ``x=0.02``, inside the UMAP panel's own empty top-left.

Everything else -- figure size, GridSpec, palette, context colours, y-scaling, file naming,
manifest schema -- is deliberately identical, so output is comparable with the trial run's.

Also runnable directly, to regenerate profiles for any existing ``analysis.parquet``
(including the 2026-07-09 trial run, which is how its overlapping text gets fixed)::

    python umap_hdbscan_sweep/cluster_profiles.py \\
        --analysis train_then_cluster_20260709T140251Z/clustering/analysis.parquet \\
        --output-dir train_then_cluster_20260709T140251Z/clustering/cluster_profiles_fixed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Walk up to the directory that holds uv_vae/ rather than counting levels: these scripts
# get reorganised into subfolders (umap/, hdbscan/, tests/), and a hard-coded parents[N]
# silently resolves to the wrong root the moment one moves, failing on `import uv_vae`.
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, REPO_ROOT / "uv_vae" / "scripts",
                  Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Before importing the pipeline: it installs a GPU budget and calls cuml.accel.install() at
# module scope, neither of which belongs in a process that only draws pictures.
os.environ.setdefault("UV_VAE_DISABLE_CUML", "1")

import numpy as np
import polars as pl
import pyarrow.parquet as pq


def _pipeline():
    """The reference module, imported lazily -- it is heavy and pulls in SigProfiler."""
    import run_variant_cluster_pipeline as rvcp

    return rvcp


# ── reading points ─────────────────────────────────────────────────────────────

def read_cluster_points_by_label(
    analysis_path: Path,
    threads: int,
    database_path: Path | None = None,
    temp_directory: Path | None = None,
    memory_limit: str | None = None,
) -> dict[int, np.ndarray]:
    """Every cluster's points in ONE scan: ``{label: (n, 3) [umap_1, umap_2, probability]}``.

    Peak memory is the whole clustered set (~24 B/row) against one cluster for the pipeline's
    per-query path, which is why ``write_cluster_profiles`` falls back above ``bulk_rows``.
    """
    rvcp = _pipeline()
    sql = f"""
        SELECT CAST("cluster_label" AS INTEGER) AS cluster_label,
               "umap_1", "umap_2", "cluster_probability"
        FROM read_parquet({rvcp.sql_quote(str(analysis_path))})
        WHERE "cluster_label" >= 0
    """
    frame = rvcp.query_frame(
        sql,
        threads=threads,
        database_path=database_path,
        temp_directory=temp_directory,
        memory_limit=memory_limit,
    )
    labels = frame["cluster_label"].to_numpy()
    points = frame.select(["umap_1", "umap_2", "cluster_probability"]).to_numpy()

    # Group by sorting once. A boolean mask per label would put the cluster count back into
    # the cost, which is the thing this function exists to remove.
    order = np.argsort(labels, kind="stable")
    labels_sorted = labels[order]
    points_sorted = points[order]
    unique, starts = np.unique(labels_sorted, return_index=True)
    bounds = [*starts.tolist(), len(labels_sorted)]
    return {
        int(label): points_sorted[bounds[index]:bounds[index + 1]]
        for index, label in enumerate(unique)
    }


# ── rendering ──────────────────────────────────────────────────────────────────

# Set once per worker by the pool initializer: the UMAP backdrop is identical for every panel
# and large enough that pickling it per task would dominate the render. The substitution and
# context lists ride along for a different reason -- they are the ONLY things the renderer
# needed from run_variant_cluster_pipeline, and importing that module inside a worker drags
# in torch, umap and SigProfiler for the sake of two constant lists. Measured at ~5 s of
# startup per worker under Windows' spawn; forwarding them keeps the parent as the single
# source of truth while leaving workers needing nothing heavier than matplotlib.
_BACKDROP: dict[str, object] = {}


def _init_worker(plot_x, plot_y, limits, context_colors, substitutions, contexts) -> None:
    import matplotlib

    matplotlib.use("Agg")
    _BACKDROP.update(plot_x=plot_x, plot_y=plot_y, limits=limits,
                     context_colors=context_colors, substitutions=substitutions,
                     contexts=contexts)


def _render_profile(task: tuple) -> str:
    """Draw one cluster's panel. Module level and argument-driven so it can be pickled."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    (label, cluster_size, cluster_color, cluster_points, summary_lines,
     values_by_substitution, y_max, plot_path) = task

    plot_x = _BACKDROP["plot_x"]
    plot_y = _BACKDROP["plot_y"]
    limits = _BACKDROP["limits"]
    context_colors = _BACKDROP["context_colors"]
    substitutions = _BACKDROP["substitutions"]
    contexts = _BACKDROP["contexts"]

    fig = plt.figure(figsize=(19, 12), dpi=180)
    grid = GridSpec(4, 4, figure=fig, width_ratios=[2.4, 1, 1, 1], hspace=0.45, wspace=0.28)

    umap_axis = fig.add_subplot(grid[:, 0])
    umap_axis.scatter(plot_x, plot_y, s=2, c="#d9d9d9", alpha=0.22, linewidths=0)
    umap_axis.scatter(cluster_points[:, 0], cluster_points[:, 1], s=4, c=cluster_color,
                      alpha=0.88, linewidths=0)
    umap_axis.set_xlim(limits[0], limits[1])
    umap_axis.set_ylim(limits[2], limits[3])
    umap_axis.set_xlabel("UMAP 1")
    umap_axis.set_ylabel("UMAP 2")
    umap_axis.set_title(f"Cluster {label} on UMAP")
    # x=0.02, not the reference's x=1.02. See the module docstring: 1.02 puts the box outside
    # the UMAP axes and directly over the A>C chart.
    umap_axis.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=umap_axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        zorder=5,
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    for index, substitution in enumerate(substitutions):
        grid_row = index // 3
        grid_col = 1 + (index % 3)
        axis = fig.add_subplot(grid[grid_row, grid_col])
        axis.bar(
            range(len(contexts)),
            values_by_substitution[substitution],
            color=[context_colors[context] for context in contexts],
            width=0.82,
        )
        axis.set_title(substitution, fontsize=10)
        axis.set_ylim(0, y_max)
        axis.set_xticks(range(len(contexts)))
        axis.set_xticklabels(contexts, rotation=90, fontsize=7)
        if grid_col == 1:
            axis.set_ylabel("Fraction")
        else:
            axis.set_yticklabels([])
        axis.grid(axis="y", alpha=0.15)

    fig.suptitle(
        f"HDBSCAN Cluster {label}: strand-specific trinucleotide distribution (192)",
        fontsize=16,
        y=0.995,
    )
    # GridSpec + suptitle is exactly what tight_layout warns it cannot solve, once per panel.
    # constrained_layout is not a drop-in (it fights width_ratios), so keep the rect and
    # silence the noise rather than change the layout the trial run used.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
        fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return str(plot_path)


# ── the drop-in ────────────────────────────────────────────────────────────────

def write_cluster_profiles(
    output_dir: Path,
    plot_df: pl.DataFrame,
    analysis_path: Path,
    cluster_stats: pl.DataFrame,
    trinuc192_counts: pl.DataFrame,
    threads: int,
    database_path: Path | None = None,
    temp_directory: Path | None = None,
    memory_limit: str | None = None,
    max_workers: int | None = None,
    bulk_rows: int = 60_000_000,
    progress_every: int = 25,
) -> tuple[Path, Path]:
    """Same signature and same outputs as the pipeline's, plus ``max_workers``/``bulk_rows``.

    ``max_workers=1`` renders sequentially (useful when a traceback is wanted); ``bulk_rows``
    is the row count above which the single-scan read is judged too memory-hungry and the
    pipeline's per-cluster queries are used instead.
    """
    rvcp = _pipeline()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir.parent / "cluster_profiles_manifest.json"
    trinuc192_path = output_dir.parent / "cluster_trinuc192.parquet"
    trinuc192_counts.write_parquet(trinuc192_path)

    plot_x = plot_df["umap_1"].to_numpy()
    plot_y = plot_df["umap_2"].to_numpy()
    limits = (
        float(plot_x.min() - 0.5),
        float(plot_x.max() + 0.5),
        float(plot_y.min() - 0.5),
        float(plot_y.max() + 0.5),
    )
    context_colors = {
        context: color
        for context, color in zip(
            rvcp.SBS192_CONTEXTS,
            ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
             "#a6761d", "#666666", "#1f78b4", "#b2df8a", "#fb9a99", "#fdbf6f",
             "#cab2d6", "#6a3d9a", "#ff7f00", "#b15928"],
            strict=True,
        )
    }
    ordered_labels = cluster_stats["cluster_label"].to_list()
    palette = rvcp.build_palette(len(ordered_labels))
    label_to_color = {int(label): palette[index] for index, label in enumerate(ordered_labels)}

    count_lookup = {
        (int(row["cluster_label"]), row["ref_alt"], row["context16"]): int(row["count"])
        for row in trinuc192_counts.to_dicts()
    }
    # One grouped pass instead of a filter+sort per cluster -- the same reason the points are
    # read in bulk. head(6) per group reproduces the reference's per-cluster top-6 exactly.
    top_lookup: dict[int, list[dict[str, object]]] = {
        int(label): [] for label in ordered_labels
    }
    if trinuc192_counts.height:
        for (label,), group in (
            trinuc192_counts.sort("count", descending=True).group_by("cluster_label")
        ):
            if int(label) in top_lookup:
                top_lookup[int(label)] = group.head(6).to_dicts()

    total_rows = pq.ParquetFile(analysis_path).metadata.num_rows
    points_by_label: dict[int, np.ndarray] | None = None
    if total_rows <= bulk_rows:
        points_by_label = read_cluster_points_by_label(
            analysis_path=analysis_path, threads=threads, database_path=database_path,
            temp_directory=temp_directory, memory_limit=memory_limit,
        )

    tasks: list[tuple] = []
    manifest: list[dict[str, object]] = []
    for row in cluster_stats.to_dicts():
        label = int(row["cluster_label"])
        cluster_size = int(row["cluster_size"])
        if points_by_label is not None:
            cluster_points = points_by_label.get(label, np.empty((0, 3), dtype=np.float64))
        else:
            cluster_points = rvcp.read_cluster_points(
                analysis_path=analysis_path, cluster_label=label, threads=threads,
                database_path=database_path, temp_directory=temp_directory,
                memory_limit=memory_limit,
            )

        summary_lines = [
            f"n = {cluster_size:,}",
            f"mean p = {float(row['mean_cluster_probability']):.3f}",
            f"median p = {float(row['median_cluster_probability']):.3f}",
            "Top trinucs:",
        ]
        summary_lines.extend(
            f"{item['trinuc192']}  {int(item['count']):,} ({float(item['fraction']):.2%})"
            for item in top_lookup[label]
        )

        max_fraction = 0.0
        values_by_substitution: dict[str, np.ndarray] = {}
        for substitution in rvcp.SBS192_SUBSTITUTIONS:
            values = np.array(
                [count_lookup.get((label, substitution, context), 0) / cluster_size
                 for context in rvcp.SBS192_CONTEXTS],
                dtype=np.float64,
            )
            values_by_substitution[substitution] = values
            if values.size:
                max_fraction = max(max_fraction, float(values.max()))
        y_max = max_fraction * 1.12 if max_fraction > 0 else 1.0

        plot_path = output_dir / f"cluster_{label:02d}_n{cluster_size}.png"
        tasks.append((label, cluster_size, label_to_color[label], cluster_points,
                      summary_lines, values_by_substitution, y_max, plot_path))
        manifest.append({
            "cluster_label": label,
            "cluster_size": cluster_size,
            "mean_probability": float(row["mean_cluster_probability"]),
            "median_probability": float(row["median_cluster_probability"]),
            "plot_path": str(plot_path),
            "top_trinucs": top_lookup[label],
        })

    workers = max_workers if max_workers is not None else min(8, os.cpu_count() or 1)
    workers = max(1, min(workers, len(tasks))) if tasks else 1
    initargs = (plot_x, plot_y, limits, context_colors,
                rvcp.SBS192_SUBSTITUTIONS, rvcp.SBS192_CONTEXTS)
    print(f"    rendering {len(tasks):,} profiles on {workers} worker(s)", flush=True)

    if workers == 1:
        _init_worker(*initargs)
        for done, task in enumerate(tasks, start=1):
            _render_profile(task)
            if progress_every and done % progress_every == 0:
                print(f"      {done:,}/{len(tasks):,}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=initargs) as pool:
            # Consume the iterator so a worker exception surfaces here rather than being
            # swallowed when the pool shuts down.
            for done, _ in enumerate(pool.map(_render_profile, tasks, chunksize=1), start=1):
                if progress_every and done % progress_every == 0:
                    print(f"      {done:,}/{len(tasks):,}", flush=True)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path, trinuc192_path


# ── standalone ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate cluster profile panels for an existing analysis.parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="where the cluster_<id>_n<size>.png files go")
    parser.add_argument("--plot-sample", type=Path, default=None,
                        help="parquet supplying the grey UMAP backdrop "
                             "(default: plot_sample.parquet beside --analysis, else a "
                             "subsample of --analysis itself)")
    parser.add_argument("--plot-rows", type=int, default=300_000)
    parser.add_argument("--max-clusters", type=int, default=0, help="0 = every cluster")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    rvcp = _pipeline()
    analysis_path = args.analysis.resolve()

    plot_path = args.plot_sample or (analysis_path.parent / "plot_sample.parquet")
    if plot_path.exists():
        plot_df = pl.read_parquet(plot_path)
        print(f"backdrop from {plot_path.name}: {plot_df.height:,} rows")
    else:
        plot_df = pl.read_parquet(analysis_path, columns=["umap_1", "umap_2"])
        if plot_df.height > args.plot_rows:
            plot_df = plot_df.sample(n=args.plot_rows, seed=args.seed)
        print(f"backdrop subsampled from {analysis_path.name}: {plot_df.height:,} rows")

    cluster_stats = rvcp.query_cluster_stats(analysis_path, threads=args.threads)
    if cluster_stats.height == 0:
        raise SystemExit("no non-noise clusters in that analysis.parquet")
    if args.max_clusters and cluster_stats.height > args.max_clusters:
        print(f"{cluster_stats.height:,} clusters -> profiling the {args.max_clusters} largest")
        cluster_stats = cluster_stats.head(args.max_clusters)

    trinuc192 = rvcp.query_trinuc192_counts(analysis_path, cluster_stats, threads=args.threads)
    manifest_path, trinuc_path = write_cluster_profiles(
        output_dir=args.output_dir.resolve(),
        plot_df=plot_df,
        analysis_path=analysis_path,
        cluster_stats=cluster_stats,
        trinuc192_counts=trinuc192,
        threads=args.threads,
        max_workers=args.workers,
    )
    print(f"{cluster_stats.height:,} profiles -> {args.output_dir}")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
