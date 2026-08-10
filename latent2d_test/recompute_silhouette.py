"""Recompute internal-validity metrics with HDBSCAN noise excluded.

Why this exists
---------------
``umap_metrics.cluster_quality`` hands the label array to sklearn untouched, so
HDBSCAN's noise label (``-1``) is scored as though it were an ordinary cluster.
That is defensible for cross-run *comparison* -- every run is contaminated the same
way, so the ranking survives -- but it makes the absolute numbers unreadable. At
52-86% noise, a silhouette of -0.42 is mostly reporting "the noise 'cluster' is
spatially diffuse", which is true by construction and says nothing about whether
the real clusters are well separated.

This script recomputes the three sklearn metrics on non-noise rows only. It does
**not** re-fit anything: every input it needs is already in each cell's
``analysis.parquet`` (``cluster_label``, ``latent_1``, ``latent_2``), which the
clustering run wrote for all rows.

What it reports per cell
------------------------
``silhouette_incl_noise``
    Recomputed on the *same* 20k subsample the original run used, by replaying the
    tier RNG. This is a correctness check, not a result: it must match the saved
    ``metrics.silhouette``. A mismatch means the subsample was not reproduced --
    usually a ``--seed`` / ``--knn-rows`` / ``--pair-rows`` disagreement with the
    clustering run -- and every other number here should then be distrusted.

``silhouette_excl_noise``
    The headline. Drawn as a fresh subsample from non-noise rows only, so it keeps
    the full ``--pair-rows`` sample size instead of whatever fraction of the
    original 20k happened not to be noise. At 86% noise that difference is 20,000
    rows versus ~2,800, which is the difference between a stable number and one
    dominated by draw luck.

``silhouette_excl_noise_same_subsample``
    The original 20k subsample with its noise rows dropped. Smaller and noisier
    than the above, but it isolates the effect of *excluding noise* from the effect
    of *drawing different rows*. When the two disagree, the fresh draw is the one
    to trust and the gap is sampling error.

``davies_bouldin_excl_noise`` / ``calinski_harabasz_excl_noise``
    Both are linear (centroids only), so they run on every non-noise row rather
    than a subsample -- matching how the original run computed them on every row.
    Use ``--full-cap`` if the non-noise population is too large for the machine.

Results are written back into each cell's ``metrics.json`` under a new
``metrics_excl_noise`` key. The original ``metrics`` block is never modified, so
the contaminated numbers stay available and nothing downstream that reads the old
schema breaks.

Typical use on miletus::

    python recompute_silhouette.py \\
        --cluster-root ~/pure-internship/uv_vae/runs/latent2d_cohort/latent2d_cluster

Add ``--cell mcs1000_ms5`` (repeatable) to restrict to particular cells.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

NOISE_LABEL = -1


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute silhouette / Davies-Bouldin / Calinski-Harabasz with HDBSCAN "
            "noise excluded, from saved analysis.parquet files. No re-fitting."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cluster-root",
        required=True,
        help="Directory holding the per-cell subdirectories (each with metrics.json "
             "and analysis.parquet).",
    )
    parser.add_argument(
        "--cell",
        action="append",
        dest="cells",
        default=None,
        help="Cell directory name to process. Repeatable. Default: every cell found.",
    )
    # These three must match the clustering run, or the incl-noise reproduction check
    # cannot pass. They are exposed rather than hard-coded so a run that overrode them
    # can still be verified.
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the clustering run's --seed (default 42)")
    parser.add_argument("--knn-rows", type=int, default=250_000,
                        help="Must match the clustering run's --knn-rows (default 250000)")
    parser.add_argument("--pair-rows", type=int, default=20_000,
                        help="Must match the clustering run's --pair-rows (default 20000)")
    parser.add_argument(
        "--full-cap",
        type=int,
        default=None,
        help="Cap the row count used for Davies-Bouldin / Calinski-Harabasz. Default: "
             "use every non-noise row, matching how the original run used every row.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="How far the reproduced incl-noise silhouette may sit from the saved value "
             "before it is reported as a mismatch (default 1e-4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print, but do not write metrics.json.",
    )
    return parser.parse_args()


def pair_positions_for(total_rows: int, seed: int, knn_rows: int, pair_rows: int) -> np.ndarray:
    """Replay the clustering run's metric-tier draw.

    Must stay byte-identical to ``latent2d_cluster.main``: one generator seeded at
    ``seed + 1``, the knn tier drawn first from the whole population, then the pair
    tier drawn *from the knn tier*. Drawing pair independently would consume the
    generator differently and silently produce a different subsample.
    """
    tier_rng = np.random.default_rng(seed + 1)
    knn_positions = np.sort(tier_rng.choice(
        total_rows, size=min(knn_rows, total_rows), replace=False))
    return np.sort(tier_rng.choice(
        knn_positions, size=min(pair_rows, knn_positions.size), replace=False))


def silhouette_of(coordinates: np.ndarray, labels: np.ndarray) -> float | None:
    """Silhouette, or None when it is undefined (fewer than two distinct labels)."""
    from sklearn.metrics import silhouette_score

    if len(set(labels.tolist())) < 2:
        return None
    try:
        return round(float(silhouette_score(
            np.asarray(coordinates, dtype=np.float32), np.asarray(labels)
        )), 5)
    except Exception as exc:  # pragma: no cover - sklearn edge cases
        log(f"    silhouette failed: {exc}")
        return None


def centroid_metrics_of(coordinates: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

    result: dict[str, float | None] = {}
    for name, function in (
        ("davies_bouldin", davies_bouldin_score),
        ("calinski_harabasz", calinski_harabasz_score),
    ):
        if len(set(labels.tolist())) < 2:
            result[name] = None
            continue
        try:
            result[name] = round(float(function(
                np.asarray(coordinates, dtype=np.float32), np.asarray(labels)
            )), 5)
        except Exception as exc:  # pragma: no cover
            log(f"    {name} failed: {exc}")
            result[name] = None
    return result


def process_cell(cell_dir: Path, args: argparse.Namespace) -> dict | None:
    metrics_path = cell_dir / "metrics.json"
    analysis_path = cell_dir / "analysis.parquet"
    if not metrics_path.is_file():
        log(f"  [{cell_dir.name}] no metrics.json, skipping")
        return None
    payload = json.loads(metrics_path.read_text())
    if "error" in payload:
        log(f"  [{cell_dir.name}] cell recorded an error, skipping")
        return None

    # metrics.json records the absolute path from the machine that produced it. Prefer
    # the sibling file so a directory that was copied elsewhere still works.
    if not analysis_path.is_file():
        recorded = payload.get("analysis_path")
        if recorded and Path(recorded).is_file():
            analysis_path = Path(recorded)
        else:
            log(f"  [{cell_dir.name}] analysis.parquet not found (looked at {analysis_path} "
                f"and the recorded path), skipping")
            return None

    started = perf_counter()
    frame = pl.read_parquet(analysis_path, columns=["cluster_label", "latent_1", "latent_2"])
    labels = frame["cluster_label"].to_numpy()
    space = np.column_stack([
        frame["latent_1"].to_numpy(),
        frame["latent_2"].to_numpy(),
    ]).astype(np.float32, copy=False)
    del frame
    total_rows = labels.shape[0]

    recorded_points = payload.get("metrics", {}).get("n_points")
    if recorded_points is not None and int(recorded_points) != total_rows:
        log(f"  [{cell_dir.name}] !! analysis.parquet has {total_rows:,} rows but metrics.json "
            f"recorded n_points={int(recorded_points):,}. Refusing -- the subsample replay "
            f"below indexes into the population and would read the wrong rows.")
        return None

    clean_mask = labels != NOISE_LABEL
    clean_count = int(clean_mask.sum())
    noise_fraction = 1.0 - clean_count / total_rows
    log(f"  [{cell_dir.name}] {total_rows:,} rows, {clean_count:,} non-noise "
        f"({noise_fraction * 100:.2f}% noise)")

    if clean_count == 0:
        log(f"  [{cell_dir.name}] every row is noise; nothing to score")
        return None

    result: dict = {
        "n_rows": total_rows,
        "n_rows_excl_noise": clean_count,
        "noise_fraction": round(noise_fraction, 5),
        "n_clusters": int(len(set(labels[clean_mask].tolist()))),
    }

    # --- 1. reproduce the original subsample, to prove the replay is aligned ---------
    pair_positions = pair_positions_for(
        total_rows, args.seed, args.knn_rows, args.pair_rows
    )
    original_coordinates = space[pair_positions]
    original_labels = labels[pair_positions]
    result["silhouette_incl_noise"] = silhouette_of(original_coordinates, original_labels)
    result["silhouette_incl_noise_rows"] = int(original_labels.size)

    saved = payload.get("metrics", {}).get("silhouette")
    if saved is not None and result["silhouette_incl_noise"] is not None:
        delta = abs(float(saved) - result["silhouette_incl_noise"])
        result["reproduction_delta"] = round(delta, 8)
        result["reproduction_ok"] = bool(delta <= args.tolerance)
        if not result["reproduction_ok"]:
            log(f"    !! reproduced incl-noise silhouette {result['silhouette_incl_noise']} "
                f"but metrics.json saved {saved} (delta {delta:.2e}). The subsample was NOT "
                f"reproduced -- check --seed / --knn-rows / --pair-rows against the run. "
                f"Every excl-noise number below is still valid (it does not depend on the "
                f"replay), but the incl/excl comparison is not like-for-like.")
        else:
            log(f"    reproduction check OK (silhouette {saved} reproduced)")
    else:
        result["reproduction_delta"] = None
        result["reproduction_ok"] = None

    # --- 2. same subsample, noise dropped -------------------------------------------
    same_clean = original_labels != NOISE_LABEL
    if int(same_clean.sum()) >= 2:
        result["silhouette_excl_noise_same_subsample"] = silhouette_of(
            original_coordinates[same_clean], original_labels[same_clean]
        )
    else:
        result["silhouette_excl_noise_same_subsample"] = None
    result["silhouette_excl_noise_same_subsample_rows"] = int(same_clean.sum())

    # --- 3. fresh full-size draw from non-noise rows only ---------------------------
    # Seeded independently of the tier RNG so it cannot disturb the replay above, and
    # so re-running this script gives the same answer.
    clean_positions = np.flatnonzero(clean_mask)
    draw_size = min(args.pair_rows, clean_positions.size)
    fresh_rng = np.random.default_rng(args.seed + 1000)
    fresh_positions = np.sort(fresh_rng.choice(clean_positions, size=draw_size, replace=False))
    result["silhouette_excl_noise"] = silhouette_of(
        space[fresh_positions], labels[fresh_positions]
    )
    result["silhouette_excl_noise_rows"] = int(draw_size)

    # --- 4. centroid metrics on every non-noise row ---------------------------------
    if args.full_cap is not None and clean_count > args.full_cap:
        cap_rng = np.random.default_rng(args.seed + 2000)
        full_positions = np.sort(cap_rng.choice(
            clean_positions, size=args.full_cap, replace=False))
        result["centroid_metric_rows"] = int(args.full_cap)
        result["centroid_metric_capped"] = True
    else:
        full_positions = clean_positions
        result["centroid_metric_rows"] = clean_count
        result["centroid_metric_capped"] = False
    centroid = centroid_metrics_of(space[full_positions], labels[full_positions])
    result["davies_bouldin_excl_noise"] = centroid["davies_bouldin"]
    result["calinski_harabasz_excl_noise"] = centroid["calinski_harabasz"]

    result["seconds"] = round(perf_counter() - started, 1)

    log(f"    silhouette: {result['silhouette_incl_noise']} (incl noise) -> "
        f"{result['silhouette_excl_noise']} (excl noise, {draw_size:,} rows)")
    log(f"    davies_bouldin: {payload.get('metrics', {}).get('davies_bouldin')} -> "
        f"{result['davies_bouldin_excl_noise']}")
    log(f"    calinski_harabasz: {payload.get('metrics', {}).get('calinski_harabasz')} -> "
        f"{result['calinski_harabasz_excl_noise']}")

    if not args.dry_run:
        payload["metrics_excl_noise"] = result
        metrics_path.write_text(json.dumps(payload, indent=2))
        log(f"    wrote metrics_excl_noise into {metrics_path}")

    return result


def main() -> None:
    args = parse_args()
    cluster_root = Path(args.cluster_root).expanduser().resolve()
    if not cluster_root.is_dir():
        raise SystemExit(f"--cluster-root is not a directory: {cluster_root}")

    if args.cells:
        cell_dirs = [cluster_root / name for name in args.cells]
        missing = [d for d in cell_dirs if not d.is_dir()]
        if missing:
            raise SystemExit(
                "these --cell directories do not exist: "
                + ", ".join(str(d) for d in missing)
            )
    else:
        cell_dirs = sorted(d for d in cluster_root.iterdir()
                           if d.is_dir() and (d / "metrics.json").is_file())
    if not cell_dirs:
        raise SystemExit(f"no cells with a metrics.json found under {cluster_root}")

    log(f"Recomputing noise-excluded metrics for {len(cell_dirs)} cell(s) under {cluster_root}")
    if args.dry_run:
        log("DRY RUN: metrics.json will not be modified")

    results: dict[str, dict] = {}
    for cell_dir in cell_dirs:
        outcome = process_cell(cell_dir, args)
        if outcome is not None:
            results[cell_dir.name] = outcome

    if not results:
        raise SystemExit("no cells produced results")

    # Comparison table. Printed to stdout (not the log) so it can be redirected cleanly.
    print()
    header = (
        f"{'cell':<24} {'noise%':>7} {'clusters':>9} "
        f"{'sil(incl)':>10} {'sil(excl)':>10} {'DB(excl)':>9} {'CH(excl)':>13}"
    )
    print(header)
    print("-" * len(header))
    for name, outcome in sorted(
        results.items(),
        key=lambda item: (item[1]["silhouette_excl_noise"] is None,
                          -(item[1]["silhouette_excl_noise"] or 0.0)),
    ):
        def show(value, spec: str) -> str:
            return "n/a" if value is None else format(value, spec)
        print(
            f"{name:<24} "
            f"{outcome['noise_fraction'] * 100:>6.2f}% "
            f"{outcome['n_clusters']:>9,} "
            f"{show(outcome['silhouette_incl_noise'], '>10.5f')} "
            f"{show(outcome['silhouette_excl_noise'], '>10.5f')} "
            f"{show(outcome['davies_bouldin_excl_noise'], '>9.4f')} "
            f"{show(outcome['calinski_harabasz_excl_noise'], '>13,.0f')}"
        )
    print()

    unverified = [name for name, outcome in results.items()
                  if outcome.get("reproduction_ok") is False]
    if unverified:
        print(
            "WARNING: the incl-noise column could not be reproduced for: "
            + ", ".join(unverified)
            + ". Those cells' incl/excl comparison is not like-for-like."
        )

    summary_path = cluster_root / "silhouette_excl_noise.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps({
            "cluster_root": str(cluster_root),
            "seed": args.seed,
            "knn_rows": args.knn_rows,
            "pair_rows": args.pair_rows,
            "full_cap": args.full_cap,
            "cells": results,
        }, indent=2))
        log(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
