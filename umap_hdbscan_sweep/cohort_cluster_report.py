#!/usr/bin/env python
"""``run_variant_cluster_pipeline``'s plots and cluster profiles, for the cohort sweep cells.

The trial run in ``train_then_cluster_20260709T140251Z/clustering/`` produced two directories
that nothing in the cohort pipeline reproduces:

    plots/              umap_<column>.png per numeric column, umap_clusters.png,
                        umap_cluster_probability.png, umap_contact_sheet.png
    cluster_profiles/   one 192-trinucleotide profile panel per cluster

That run clustered ~1 M rows from a single parquet. The cohort sweep instead keeps a shared
2-D embedding and a per-cell ``cohort_labels.npy`` over 157.5 M rows, with the feature values
left behind in stage 0's dedup parquets -- so none of the pipeline's queries can be pointed at
it directly. This script rebuilds the input those functions expect and then **calls them**:
``numeric_umap_plot``, ``cluster_umap_plot``, ``probability_umap_plot``, ``make_contact_sheet``
and ``write_cluster_profiles`` are imported from ``run_variant_cluster_pipeline``, not
reimplemented. ``hdbscan_param_sweep``'s docstring makes the argument for that -- a second
implementation would be a second answer to compare against, not a shortcut -- and it is the
only way "match the plots" is actually true rather than approximately true.

How the pieces line up
----------------------
Row *i* is the same variant in all three places, and that is the only join used:

    latent.npy                row i  <- stage 1, written in dedup-manifest order
    dedup/chrom=*.parquet     row i  <- stage 0, same order (feature values live here)
    <cell>/cohort_labels.npy  row i  <- the sweep, labelled against the same coords

What it writes, per cell::

    <output-root>/<cell>/
        analysis.parquet               the sampled frame the pipeline functions consume
        plot_sample.parquet            the smaller subsample the scatter plots use
        plots/                         umap_<col>.png, umap_clusters.png, umap_contact_sheet.png
        cluster_profiles/              cluster_<id>_n<size>.png
        cluster_profiles_manifest.json
        cluster_trinuc192.parquet
        report.json                    what was sampled, which cell, which encoder

Two things the cohort cannot give you
-------------------------------------
**Membership probabilities.** ``hdbscan_param_sweep`` drops ``held_probabilities`` after
recording its mean (it keeps the labels only), so there is no per-row probability to plot.
Without ``--probabilities`` this writes no ``umap_cluster_probability.png`` and the profile
panels report ``mean p = nan`` rather than inventing a value.

**DP, MAPQ, RAW_VAF, SNVQ.** The trial run coloured by these, but they are ranking columns,
not VAE features, so ``build_selected_columns`` never carried them into the dedup output. Only
the intersection with what stage 0 actually wrote gets plotted; the rest are reported as
skipped. ``--all-numeric-features`` swaps the trial's list for every numeric VAE input instead.

Usage (on miletus)::

    python umap_hdbscan_sweep/cohort_cluster_report.py \\
        --cells-root umap_hdbscan_sweep/umap_tests/param_sweep/cells --all \\
        --embed-summary <stage1>/embed_summary.json \\
        --encoder umap_hdbscan_sweep/umap_tests/final_models/13_BEST_25M_nn15_md0.1_umap.pt \\
        --output-root plots/cohort_reports

Pass ``--coords <coords.npy>`` instead of ``--encoder`` when the sweep's own coordinate file is
still on disk -- it is the array the labels were assigned against, so it removes any question
of whether the encoder reproduced them bitwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, REPO_ROOT / "uv_vae" / "scripts",
                  Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import polars as pl

# Must be set BEFORE run_variant_cluster_pipeline is imported: it installs a GPU budget at
# module scope and calls cuml.accel.install(), both of which are wrong for a process that is
# only drawing pictures. Same guard, same reason, as stage2_sweep.run_sigprofiler.
os.environ.setdefault("UV_VAE_DISABLE_CUML", "1")

# Trinucleotide context columns write_cluster_profiles needs, plus the identity columns the
# trial run's analysis.parquet carried.
CONTEXT_COLUMNS = ["CHROM", "POS", "REF", "ALT", "X_PREV1", "X_NEXT1"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells-root", type=Path, required=True,
                   help="directory holding <cell>/cohort_labels.npy")
    p.add_argument("--cells", nargs="*", default=None, help="cell names (default: --all)")
    p.add_argument("--all", action="store_true", help="every cell with cohort_labels.npy")
    p.add_argument("--embed-summary", type=Path, required=True,
                   help="stage 1 embed_summary.json (latent.npy + dedup manifest)")

    p.add_argument("--coords", type=Path, default=None,
                   help="coords.npy (N,2) the sweep labelled against -- preferred")
    p.add_argument("--encoder", type=Path, default=None,
                   help="parametric encoder .pt, used when --coords is absent")
    p.add_argument("--hidden", default="256,256,128")
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument("--probabilities", type=Path, default=None,
                   help="optional (N,) float per-row membership probability")

    p.add_argument("--analysis-rows", type=int, default=2_000_000,
                   help="rows sampled into analysis.parquet (stats, trinucs, profiles)")
    p.add_argument("--plot-rows", type=int, default=300_000,
                   help="rows subsampled from that for the scatter plots")
    p.add_argument("--max-clusters", type=int, default=60,
                   help="profile panels for the N largest clusters (0 = every cluster). "
                        "write_cluster_profiles issues one query per cluster, so 500+ "
                        "clusters is hours, not minutes")
    p.add_argument("--color-columns", default=None,
                   help="comma-separated override of the trial run's colour columns")
    p.add_argument("--all-numeric-features", action="store_true",
                   help="colour by every numeric VAE input instead of the trial's list")
    p.add_argument("--feature-spec-path", type=Path,
                   default=REPO_ROOT / "uv_vae" / "ml_features.json")

    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--skip-existing", action="store_true",
                   help="leave cells that already have a report.json alone")
    return p.parse_args()


# ── inputs ─────────────────────────────────────────────────────────────────────

def discover_cells(cells_root: Path) -> list[str]:
    return sorted(d.name for d in cells_root.iterdir()
                  if (d / "cohort_labels.npy").exists())


def sample_positions(total: int, wanted: int, seed: int) -> np.ndarray:
    """Sorted positions -- sorted so the gathers below stay mostly sequential over a memmap."""
    if wanted >= total:
        return np.arange(total)
    return np.sort(np.random.default_rng(seed).choice(total, size=wanted, replace=False))


def encoder_coords(encoder_path: Path, latent_sample: np.ndarray,
                   hidden: tuple[int, ...], device_name: str) -> np.ndarray:
    import torch

    import parametric_umap as pu

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name
    )
    blob = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder = pu.ParametricEncoder(latent_sample.shape[1], output_dim=2, hidden=hidden).to(device)
    encoder.load_state_dict(blob["state_dict"])
    encoder.eval()
    mode = blob.get("mode", "umap")
    log(f"  encoder {encoder_path.name} (mode={mode}) on {device}")
    return pu.ParametricUmap(encoder=encoder, device=device, mode=mode).transform(
        latent_sample.astype(np.float32, copy=False)
    )


def read_dedup_columns(manifest_path: Path, positions: np.ndarray,
                       columns: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """Gather ``columns`` at the given cohort row positions from the dedup parquet parts.

    The parts are chromosome-split and concatenate, in manifest order, into exactly the row
    order stage 1 encoded -- so a position maps to (part, offset within part) by a running
    sum of each part's ``loci`` count. Only the parts a sample actually touches are opened.
    """
    manifest = json.loads(manifest_path.read_text())
    parts = manifest["per_chromosome"]

    first = Path(parts[0]["path"])
    if not first.exists():
        raise SystemExit(f"dedup part missing: {first}\nRun this where stage 0's output lives.")
    present = set(pl.scan_parquet(first).collect_schema().names())
    wanted = [name for name in columns if name in present]
    skipped = [name for name in columns if name not in present]

    bounds = np.cumsum([0] + [int(entry["loci"]) for entry in parts])
    if positions[-1] >= bounds[-1]:
        raise SystemExit(f"position {positions[-1]:,} beyond {bounds[-1]:,} dedup rows")

    blocks: list[pl.DataFrame] = []
    for entry, start, stop in zip(parts, bounds[:-1], bounds[1:]):
        lo = int(np.searchsorted(positions, start, side="left"))
        hi = int(np.searchsorted(positions, stop, side="left"))
        if lo == hi:
            continue
        local = (positions[lo:hi] - start).tolist()
        frame = pl.read_parquet(Path(entry["path"]), columns=wanted)
        blocks.append(frame[local])

    # Parts are visited in ascending position order and `positions` is sorted, so a plain
    # vertical concat is already in sampling order -- no argsort needed.
    return pl.concat(blocks, how="vertical"), skipped


def resolve_color_columns(args, available: set[str], trial_default: list[str]) -> list[str]:
    from uv_vae.features import load_feature_specs

    if args.color_columns:
        requested = [part.strip() for part in args.color_columns.split(",") if part.strip()]
    elif args.all_numeric_features:
        requested = [s.name for s in load_feature_specs(args.feature_spec_path) if s.is_numeric]
    else:
        requested = list(trial_default)
    return requested


# ── per cell ───────────────────────────────────────────────────────────────────

def build_cell_report(cell: str, cell_dir: Path, out_dir: Path, xy: np.ndarray,
                      positions: np.ndarray, context: pl.DataFrame,
                      color_columns: list[str], probabilities: np.ndarray | None,
                      args, rvcp) -> dict:
    plots_dir = out_dir / "plots"
    profiles_dir = out_dir / "cluster_profiles"
    plots_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(np.load(cell_dir / "cohort_labels.npy", mmap_mode="r")[positions],
                        dtype=np.int32)

    if probabilities is not None:
        probability_column = probabilities.astype(np.float32, copy=False)
        have_probabilities = True
    else:
        # NaN, not 1.0: the profile panels print this number, and a fabricated 1.000 would
        # read as a real measurement. NaN formats as "nan" and is obviously absent.
        probability_column = np.full(labels.shape[0], np.nan, dtype=np.float32)
        have_probabilities = False

    analysis = context.with_columns([
        pl.Series("umap_1", xy[:, 0].astype(np.float32, copy=False)),
        pl.Series("umap_2", xy[:, 1].astype(np.float32, copy=False)),
        pl.Series("cluster_label", labels),
        pl.Series("cluster_probability", probability_column),
    ])
    analysis_path = out_dir / "analysis.parquet"
    analysis.write_parquet(analysis_path)

    plot_columns = [c for c in color_columns if c in analysis.columns]
    plot_df = analysis.select(
        rvcp.unique_columns(["umap_1", "umap_2", "cluster_label", "cluster_probability",
                             *plot_columns])
    )
    if plot_df.height > args.plot_rows:
        plot_df = plot_df.sample(n=args.plot_rows, seed=args.seed + 1)
    plot_df.write_parquet(out_dir / "plot_sample.parquet")

    cluster_stats = rvcp.query_cluster_stats(analysis_path, threads=args.threads)
    if cluster_stats.height == 0:
        log(f"  {cell}: no non-noise clusters in the sample, skipping")
        return {"cell": cell, "skipped": "no non-noise clusters"}

    # The cap limits how many PROFILE PANELS get drawn, nothing else. cluster_umap_plot and
    # the trinucleotide table still see every cluster -- otherwise the excluded ones fall
    # through cluster_umap_plot's palette lookup and render as anonymous dark grey, which
    # reads as a different clustering rather than as a truncated report. Both lists are
    # ordered by size DESC and build_palette is indexed, so the profiled clusters keep
    # exactly the colour they have in umap_clusters.png.
    profiled_stats = cluster_stats
    if args.max_clusters and cluster_stats.height > args.max_clusters:
        log(f"  {cluster_stats.height:,} clusters in sample -> profiling the "
            f"{args.max_clusters} largest (--max-clusters 0 for all)")
        profiled_stats = cluster_stats.head(args.max_clusters)

    trinuc192 = rvcp.query_trinuc192_counts(analysis_path, cluster_stats, threads=args.threads)

    plot_items: list[tuple[str, Path]] = []
    for column in plot_columns:
        path = plots_dir / f"umap_{column}.png"
        rvcp.numeric_umap_plot(plot_df, column, path)
        plot_items.append((column, path))

    cluster_sizes = {int(r["cluster_label"]): int(r["cluster_size"])
                     for r in cluster_stats.select(["cluster_label", "cluster_size"]).to_dicts()}
    clusters_path = plots_dir / "umap_clusters.png"
    rvcp.cluster_umap_plot(plot_df, cluster_sizes, clusters_path)
    plot_items.append(("clusters", clusters_path))

    if have_probabilities:
        probability_path = plots_dir / "umap_cluster_probability.png"
        rvcp.probability_umap_plot(plot_df, probability_path)
        plot_items.append(("cluster_probability", probability_path))

    rvcp.make_contact_sheet(plot_items, plots_dir / "umap_contact_sheet.png", columns=2)

    manifest_path, trinuc_path = rvcp.write_cluster_profiles(
        output_dir=profiles_dir,
        plot_df=plot_df,
        analysis_path=analysis_path,
        cluster_stats=profiled_stats,
        trinuc192_counts=trinuc192,
        threads=args.threads,
    )

    cell_metrics = {}
    metrics_file = cell_dir / "metrics.json"
    if metrics_file.exists():
        payload = json.loads(metrics_file.read_text())
        cell_metrics = {k: payload.get(k) for k in
                        ("fit_rows", "min_cluster_size", "min_samples",
                         "cluster_selection_method", "cohort_n_clusters",
                         "cohort_noise_fraction", "dbcv")}

    return {
        "cell": cell,
        "cell_metrics": cell_metrics,
        "sampled_rows": int(labels.shape[0]),
        "plot_rows": int(plot_df.height),
        "clusters_in_sample": len(cluster_sizes),
        "noise_fraction_in_sample": float((labels < 0).mean()),
        "profiled_clusters": int(profiled_stats.height),
        "color_columns": plot_columns,
        "has_probabilities": have_probabilities,
        "analysis_path": str(analysis_path),
        "plots_dir": str(plots_dir),
        "cluster_profiles_dir": str(profiles_dir),
        "cluster_profiles_manifest": str(manifest_path),
        "cluster_trinuc192": str(trinuc_path),
    }


def main() -> int:
    args = parse_args()
    hidden = tuple(int(p) for p in args.hidden.split(",") if p.strip())

    import run_variant_cluster_pipeline as rvcp

    summary = json.loads(args.embed_summary.read_text())
    latent_path = Path(summary["latent_path"])
    manifest_path = Path(summary["dedup_manifest"])

    cells = (discover_cells(args.cells_root) if (args.all or not args.cells)
             else list(args.cells))
    if not cells:
        raise SystemExit(f"no cells with cohort_labels.npy under {args.cells_root}")
    missing = [c for c in cells if not (args.cells_root / c / "cohort_labels.npy").exists()]
    if missing:
        raise SystemExit(f"no cohort_labels.npy for: {', '.join(missing)}")
    log(f"{len(cells)} cell(s): {', '.join(cells)}")

    latent = np.load(latent_path, mmap_mode="r")
    total_rows = int(latent.shape[0])
    positions = sample_positions(total_rows, args.analysis_rows, args.seed)
    log(f"cohort {total_rows:,} rows -> sampling {positions.size:,}")

    if args.coords is not None:
        coords = np.load(args.coords, mmap_mode="r")
        if coords.shape[0] != total_rows:
            raise SystemExit(f"{args.coords.name} has {coords.shape[0]:,} rows, "
                             f"latent has {total_rows:,}")
        xy = np.asarray(coords[positions], dtype=np.float32)
        coords_source = str(args.coords)
    elif args.encoder is not None:
        xy = encoder_coords(args.encoder,
                            np.ascontiguousarray(latent[positions], dtype=np.float32),
                            hidden, args.device)
        coords_source = str(args.encoder)
    else:
        raise SystemExit("pass --coords or --encoder")

    probabilities = None
    if args.probabilities is not None:
        probabilities = np.asarray(
            np.load(args.probabilities, mmap_mode="r")[positions], dtype=np.float32)

    color_columns = resolve_color_columns(args, set(), rvcp.DEFAULT_COLOR_COLUMNS)
    wanted = rvcp.unique_columns([*CONTEXT_COLUMNS, *color_columns])
    log(f"reading {len(wanted)} columns from the dedup parts")
    context, skipped = read_dedup_columns(manifest_path, positions, wanted)
    if skipped:
        log(f"  not in the dedup output, skipped: {', '.join(skipped)}")
    if context.height != positions.size:
        raise SystemExit(f"gathered {context.height:,} rows, expected {positions.size:,}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    for position, cell in enumerate(cells, start=1):
        out_dir = args.output_root / cell
        if args.skip_existing and (out_dir / "report.json").exists():
            log(f"=== [{position}/{len(cells)}] {cell} -- already done, skipping ===")
            index.append(json.loads((out_dir / "report.json").read_text()))
            continue

        log(f"=== [{position}/{len(cells)}] {cell} ===")
        started = perf_counter()
        record = build_cell_report(
            cell, args.cells_root / cell, out_dir, xy, positions, context,
            color_columns, probabilities, args, rvcp,
        )
        record.update({
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "coords_source": coords_source,
            "embed_summary": str(args.embed_summary),
            "cohort_rows": total_rows,
            "seed": args.seed,
            "seconds": round(perf_counter() - started, 1),
        })
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(json.dumps(record, indent=2))
        index.append(record)
        if "skipped" not in record:
            log(f"  {record['clusters_in_sample']:,} clusters, "
                f"{record['noise_fraction_in_sample'] * 100:.1f}% noise, "
                f"{record['profiled_clusters']} profiles -> {out_dir}")

    (args.output_root / "index.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "coords_source": coords_source,
        "cells": index,
    }, indent=2))
    log(f"done -> {args.output_root}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
