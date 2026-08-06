"""Decompose the HDBSCAN cell timer, the way phase_a_timing.py decomposed the UMAP one.

`UMAP_RANKING_PLAN.md` §1 records a per-cell cost of **3827-4155 s** covering, in one
undecomposed timer: HDBSCAN fit on the fit set, ``approximate_predict`` of 152.5 M rows,
metrics, a 157.5 M-row parquet write, and SigProfiler. The handoff blamed
``approximate_predict``; the plan doubts it on two indirect grounds -- cell time moves only
8% while ``min_cluster_size`` moves 25x, and every cell also writes a multi-GB parquet that
nobody subtracted. Neither position has been measured.

Phase A did exactly this for UMAP and overturned the assumption there (transform, not fit,
was 98% of the stage). This script does it for HDBSCAN:

    1. fit             HDBSCAN on the fit set, per min_cluster_size
    2. predict         approximate_predict at several probe sizes, then extrapolate
    3. parquet write   a real analysis parquet at probe scale, then extrapolate

Two things make the extrapolation in step 2 honest rather than assumed. It is measured at
several probe sizes and the per-million-row cost is reported for each, so a superlinear
cost shows up as a rising column instead of hiding inside a single ratio. And it is
repeated across ``min_cluster_size``, because the whole dispute is whether predict cost
tracks cluster structure -- if it does, that column moves; if the plan is right, it does
not.

The reconstructed total is printed against the recorded 3827-4155 s. Agreement means the
decomposition is complete; a large shortfall means the real cost is in something none of
these three steps covers, which is itself the finding.

Coordinates. HDBSCAN clusters the 2-D UMAP embedding, not the latent, so the script needs
2-D coordinates for both sets. By default it refits UMAP on the fit rows (~115 s) and
transforms the probe, which keeps it self-contained and matched to the other phase scripts.
``--coords-parquet`` instead reads the ``umap_1`` / ``umap_2`` columns a completed stage-2
cell already wrote, skipping the refit.

    python umap_hdbscan_sweep/phase_a_hdbscan_timing.py \\
        --embed-dir <stage1-output> \\
        --output-dir umap_hdbscan_sweep/umap_tests
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np

import sweep_core as core

TOTAL_COHORT_ROWS = 157_501_580
FULL_COHORT_HELD_ROWS = 152_501_580

# UMAP_RANKING_PLAN.md §1, from uv_vae/runs/nn15_md0.0_nc2/. The bundle this script splits.
RECORDED_CELL_SECONDS = (3827, 4155)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def load_coordinates(args, latent, fit_idx: np.ndarray, probe_idx: np.ndarray):
    """2-D coordinates for the fit and probe sets, plus the seconds UMAP cost to get them.

    The refit path reproduces what stage 2 does. The parquet path reuses a completed
    cell's ``umap_1`` / ``umap_2`` columns, which are the same coordinates stage 2 handed
    to HDBSCAN -- cheaper, but it ties the measurement to that run's UMAP config.
    """
    if args.coords_parquet:
        import polars as pl

        path = Path(args.coords_parquet).resolve()
        log(f"reading coordinates from {path.name} (skipping the UMAP refit)")
        t0 = perf_counter()
        frame = pl.read_parquet(path, columns=["umap_1", "umap_2"])
        coordinates = frame.to_numpy().astype(np.float32, copy=False)
        seconds = perf_counter() - t0
        if coordinates.shape[0] != latent.shape[0]:
            raise SystemExit(
                f"{path.name} has {coordinates.shape[0]:,} rows but latent.npy has "
                f"{latent.shape[0]:,} -- they must describe the same cohort"
            )
        log(f"  read {coordinates.shape[0]:,} coordinate rows in {seconds:.1f}s")
        return coordinates[fit_idx], coordinates[probe_idx], seconds, "parquet"

    config = core.UmapConfig(
        n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist,
        n_components=2, seed=args.seed, n_epochs=200,
    )
    log(f"fitting UMAP on {len(fit_idx):,} rows to produce coordinates ...")
    t0 = perf_counter()
    fitted = core.fit_umap(np.ascontiguousarray(latent[fit_idx]), config)
    fit_seconds = perf_counter() - t0
    log(f"  fit in {fit_seconds:.1f}s (backend={fitted.backend})")

    log(f"transforming {len(probe_idx):,} probe rows ...")
    t0 = perf_counter()
    probe_coordinates = fitted.transform(np.ascontiguousarray(latent[probe_idx]))
    transform_seconds = perf_counter() - t0
    log(f"  transform in {transform_seconds:.1f}s")
    return (
        fitted.embedding, probe_coordinates, fit_seconds + transform_seconds, "refit",
    )


def time_parquet_write(embed_dir: Path, rows: int, coordinates: np.ndarray,
                       output_dir: Path) -> dict:
    """Write a real analysis parquet at probe scale and extrapolate to the full cohort.

    Uses the actual context columns when ``context.parquet`` is there, because the write
    cost is driven by column count, dtypes and compressibility -- a synthetic frame of
    float noise would compress unlike real genomic strings and understate the cost.
    """
    import polars as pl

    context_path = embed_dir / "context.parquet"
    if context_path.exists():
        context = pl.read_parquet(context_path, n_rows=rows)
        source = f"context.parquet ({context.width} columns)"
    else:
        # Shape-matched stand-in: stage 2 writes 11 columns, 9 of them context.
        context = pl.DataFrame({
            f"col_{i}": np.random.default_rng(i).integers(0, 1000, rows, dtype=np.int32)
            for i in range(9)
        })
        source = "synthetic 9-column stand-in (context.parquet absent)"
    log(f"parquet write test on {rows:,} rows from {source}")

    labels = np.random.default_rng(0).integers(-1, 500, rows).astype(np.int32)
    probabilities = np.random.default_rng(1).random(rows).astype(np.float32)
    columns = {name: context[name] for name in context.columns}
    columns["cluster_label"] = pl.Series("cluster_label", labels)
    columns["cluster_probability"] = pl.Series("cluster_probability", probabilities)
    columns["umap_1"] = pl.Series("umap_1", coordinates[:rows, 0])
    columns["umap_2"] = pl.Series("umap_2", coordinates[:rows, 1])

    path = output_dir / "_parquet_write_probe.parquet"
    t0 = perf_counter()
    pl.DataFrame(columns).write_parquet(path)
    seconds = perf_counter() - t0
    size_mb = path.stat().st_size / 1e6
    path.unlink()

    full_seconds = seconds * TOTAL_COHORT_ROWS / rows
    log(f"  wrote {rows:,} rows ({size_mb:.1f} MB) in {seconds:.2f}s "
        f"-> {full_seconds / 60:.1f} min for {TOTAL_COHORT_ROWS:,}")
    return {
        "probe_rows": int(rows),
        "probe_seconds": round(seconds, 3),
        "probe_size_mb": round(size_mb, 1),
        "extrapolated_full_seconds": round(full_seconds, 1),
        "extrapolated_full_minutes": round(full_seconds / 60, 1),
        "extrapolated_full_gb": round(size_mb * TOTAL_COHORT_ROWS / rows / 1000, 2),
        "source": source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the HDBSCAN cell timer into fit / predict / parquet write"
    )
    parser.add_argument("--embed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-rows", type=int, default=5_000_000,
                        help="rows HDBSCAN is fit on. The recorded run used 5M; stage 2's "
                             "own default is every deduplicated locus.")
    parser.add_argument("--probe-sizes", default="50000,100000,200000",
                        help="approximate_predict is timed at each, so a superlinear cost "
                             "shows up as a rising per-million-row column rather than "
                             "hiding inside one extrapolation")
    parser.add_argument("--min-cluster-sizes", default="100,500,2500",
                        help="the ends and middle of the swept grid (100,250,500,1000,2500). "
                             "The dispute is whether predict cost tracks cluster structure.")
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--coords-parquet", default=None,
                        help="analysis parquet from a finished cell; reads its umap_1 / "
                             "umap_2 columns instead of refitting UMAP")
    parser.add_argument("--skip-parquet-write", action="store_true")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-budget-gb", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    started = perf_counter()
    args = parse_args()

    embed_dir = Path(args.embed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_path = embed_dir / "latent.npy"
    if not latent_path.exists():
        raise SystemExit(f"latent.npy not found in {embed_dir}")

    if core.gpu_available() and args.gpu_budget_gb is not None:
        core.apply_gpu_budget("apply", budget_gb=args.gpu_budget_gb)

    probe_sizes = sorted({int(p) for p in args.probe_sizes.split(",") if p.strip()})
    min_cluster_sizes = sorted({int(m) for m in args.min_cluster_sizes.split(",") if m.strip()})

    latent = np.load(latent_path, mmap_mode="r")
    total_rows, latent_dim = latent.shape
    log(f"{latent_path.name}: {total_rows:,} rows x {latent_dim} dims")

    # Same draw as the other phase scripts, so fit and probe sets are comparable across them.
    rng = np.random.default_rng(args.seed)
    fit_idx = np.sort(rng.choice(total_rows, size=args.fit_rows, replace=False))
    outside = np.setdiff1d(np.arange(total_rows), fit_idx, assume_unique=True)
    probe_idx = np.sort(
        np.random.default_rng(args.seed + 1).choice(
            outside, size=min(max(probe_sizes), len(outside)), replace=False
        )
    )
    log(f"fit set {len(fit_idx):,} rows, probe set {len(probe_idx):,} held-out rows")

    fit_coordinates, probe_coordinates, coordinate_seconds, coordinate_source = (
        load_coordinates(args, latent, fit_idx, probe_idx)
    )

    cells = []
    for min_cluster_size in min_cluster_sizes:
        config = core.HdbscanConfig(
            min_cluster_size=min_cluster_size, min_samples=args.min_samples,
        )
        log(f"[mcs={min_cluster_size}] fitting HDBSCAN on {len(fit_idx):,} rows ...")
        t0 = perf_counter()
        fitted = core.fit_hdbscan(fit_coordinates, config)
        hdbscan_fit_seconds = perf_counter() - t0
        n_clusters = int(fitted.labels.max()) + 1 if fitted.labels.size else 0
        noise_fraction = float((fitted.labels == -1).mean())
        log(f"[mcs={min_cluster_size}]   fit in {hdbscan_fit_seconds:.1f}s "
            f"(backend={fitted.backend}, {n_clusters:,} clusters, "
            f"{noise_fraction:.1%} noise)")

        # Warm up first: the initial call pays one-off allocation and, on cuML, kernel
        # setup. Charging that to the smallest probe would fake a superlinear curve.
        fitted.predict(probe_coordinates[: min(2000, probe_sizes[0])])

        predictions = []
        for size in probe_sizes:
            t0 = perf_counter()
            labels, _ = fitted.predict(probe_coordinates[:size])
            seconds = perf_counter() - t0
            per_million = seconds / (size / 1e6)
            full_seconds = per_million * (FULL_COHORT_HELD_ROWS / 1e6)
            log(f"[mcs={min_cluster_size}]   predict {size:>7,} rows in {seconds:6.2f}s "
                f"= {per_million:6.2f} s/M -> {full_seconds / 60:5.1f} min full cohort")
            predictions.append({
                "probe_rows": int(size),
                "seconds": round(seconds, 3),
                "seconds_per_million": round(per_million, 3),
                "extrapolated_full_seconds": round(full_seconds, 1),
                "extrapolated_full_minutes": round(full_seconds / 60, 2),
                "noise_fraction": round(float((labels == -1).mean()), 5),
            })

        # Flat within noise means the extrapolation from the largest probe is sound.
        rates = [entry["seconds_per_million"] for entry in predictions]
        linearity = round(max(rates) / min(rates), 3) if min(rates) > 0 else float("nan")
        cells.append({
            "min_cluster_size": min_cluster_size,
            "min_samples": args.min_samples,
            "backend": fitted.backend,
            "n_clusters": n_clusters,
            "fit_noise_fraction": round(noise_fraction, 5),
            "hdbscan_fit_seconds": round(hdbscan_fit_seconds, 2),
            "predict": predictions,
            "predict_rate_spread": linearity,
        })
        del fitted

    parquet = None
    if not args.skip_parquet_write:
        parquet = time_parquet_write(
            embed_dir, min(probe_sizes[-1], len(probe_coordinates)),
            probe_coordinates, output_dir,
        )

    # Reconstruct the recorded bundle from the pieces, using the largest probe for each.
    largest = {cell["min_cluster_size"]: cell["predict"][-1] for cell in cells}
    reconstructions = []
    for cell in cells:
        predict_seconds = largest[cell["min_cluster_size"]]["extrapolated_full_seconds"]
        write_seconds = parquet["extrapolated_full_seconds"] if parquet else 0.0
        total = cell["hdbscan_fit_seconds"] + predict_seconds + write_seconds
        reconstructions.append({
            "min_cluster_size": cell["min_cluster_size"],
            "hdbscan_fit_seconds": cell["hdbscan_fit_seconds"],
            "predict_seconds": predict_seconds,
            "parquet_write_seconds": round(write_seconds, 1),
            "reconstructed_cell_seconds": round(total, 1),
            "recorded_cell_seconds": list(RECORDED_CELL_SECONDS),
            "fraction_of_recorded": round(total / np.mean(RECORDED_CELL_SECONDS), 3),
        })

    results = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_rows": int(total_rows),
        "latent_dim": int(latent_dim),
        "fit_rows": int(len(fit_idx)),
        "probe_sizes": probe_sizes,
        "coordinates": {
            "source": coordinate_source,
            "seconds": round(coordinate_seconds, 2),
            "umap_n_neighbors": args.umap_n_neighbors,
            "umap_min_dist": args.umap_min_dist,
        },
        "cells": cells,
        "parquet_write": parquet,
        "reconstruction": reconstructions,
        "total_seconds": round(perf_counter() - started, 1),
    }
    out_path = output_dir / "phase_a_hdbscan_timing.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"wrote {out_path}")

    print("\n-- HDBSCAN cell, decomposed --------------------------------------")
    print(f"fit set {len(fit_idx):,} rows   probe {probe_sizes[-1]:,} rows   "
          f"coordinates: {coordinate_source}")
    print()
    print("  mcs   clusters   noise      fit      predict->152.5M    s/M spread")
    for cell in cells:
        last = cell["predict"][-1]
        print(f"{cell['min_cluster_size']:>5}  {cell['n_clusters']:>9,}  "
              f"{cell['fit_noise_fraction']:>6.1%}  {cell['hdbscan_fit_seconds']:>7.1f}s  "
              f"{last['extrapolated_full_minutes']:>12.1f} min  "
              f"{cell['predict_rate_spread']:>10.2f}x")
    print()
    if parquet:
        print(f"parquet write   {parquet['probe_rows']:,} rows in "
              f"{parquet['probe_seconds']:.2f}s -> "
              f"{parquet['extrapolated_full_minutes']:.1f} min "
              f"({parquet['extrapolated_full_gb']:.1f} GB for {TOTAL_COHORT_ROWS:,} rows)")
        print()
    print(f"reconstructed cell vs recorded {RECORDED_CELL_SECONDS[0]}-"
          f"{RECORDED_CELL_SECONDS[1]}s:")
    for entry in reconstructions:
        print(f"  mcs={entry['min_cluster_size']:>5}   "
              f"fit {entry['hdbscan_fit_seconds']:>6.0f}s + "
              f"predict {entry['predict_seconds']:>6.0f}s + "
              f"write {entry['parquet_write_seconds']:>6.0f}s = "
              f"{entry['reconstructed_cell_seconds']:>6.0f}s "
              f"({entry['fraction_of_recorded']:.0%} of recorded)")
    print("------------------------------------------------------------------")

    spreads = [cell["predict_rate_spread"] for cell in cells]
    if max(spreads) > 1.5:
        print("\nPredict cost per row is NOT flat across probe size (spread "
              f"{max(spreads):.2f}x). The extrapolation to 152.5 M is unsafe -- rerun with")
        print("larger --probe-sizes before believing any full-cohort number above.")

    predict_minutes = [cell["predict"][-1]["extrapolated_full_minutes"] for cell in cells]
    fit_seconds = [cell["hdbscan_fit_seconds"] for cell in cells]
    covered = np.mean([entry["fraction_of_recorded"] for entry in reconstructions])
    if covered < 0.7:
        print(f"\nThese three steps account for only {covered:.0%} of the recorded cell. The")
        print("rest is in something unmeasured here -- metrics (DBCV is the suspect, it is")
        print("O(n^2)-ish), or SigProfiler. Decompose those next; optimising fit, predict")
        print("or the write cannot recover time they do not consume.")
    elif max(predict_minutes) > 30:
        print(f"\napproximate_predict is the cell's bottleneck ({max(predict_minutes):.0f} min),")
        print("so the handoff was right and nearest-centroid assignment is worth building.")
    else:
        print(f"\napproximate_predict costs {min(predict_minutes):.0f}-{max(predict_minutes):.0f} min "
              f"and HDBSCAN fit {min(fit_seconds):.0f}-{max(fit_seconds):.0f}s, confirming the")
        print("plan's reading: the handoff's proposed fix targets a minority of the cell.")
        if parquet and parquet["extrapolated_full_minutes"] > max(predict_minutes):
            print(f"The {parquet['extrapolated_full_minutes']:.0f}-min parquet write is the larger "
                  "item, and it is pure I/O -- drop columns")
            print("or stop writing it per cell rather than optimising the clusterer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
