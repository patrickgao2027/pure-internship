#!/usr/bin/env python
"""Is a cell's clustering reproducible? Refit it on independent subsamples and compare.

Why this exists
---------------
Every internal validity index -- silhouette, Davies-Bouldin, Calinski-Harabasz, connectivity,
even DBCV -- asks whether the clusters are compact and separated *in this space*. The space
is a UMAP embedding, and manufacturing compact, separated neighbourhoods is UMAP's objective
function. Grading HDBSCAN with those indices substantially grades UMAP, which is why the
2026-08-12 sweep saw connectivity sit at 0.998-1.000 across a 15x change in cluster count.

Stability asks a different question -- is the structure *reproducible* -- and it assumes
nothing about cluster shape, so it is not circular with respect to the embedding. For this
project it is also the question that matters: whether a model fitted on a subsample recovers
the structure of the whole is the thesis, not a diagnostic for it.

The experiment
--------------
The UMAP coordinates are fixed on disk and HDBSCAN is deterministic given its input, so the
only stochastic ingredient in a cell is *which rows land in the fit set*. That makes the
design clean:

    reserve a probe -> for each replicate, draw a fit set from the remaining rows,
    fit, label the probe -> compare the replicates' probe labellings pairwise

The probe is held out of every fit set, so the labels being compared are all out-of-sample
predictions produced the same way -- no replicate gets to grade its own training rows.

Scored per pair, by ``cross_size_ari.compare``:

* **ARI / AMI-family** -- pair-counting agreement, with and without noise rows.
* **Variation of information** (Meila 2007) -- a true metric on partitions; its conditional
  half ``H(b|a) == 0`` is an exact test that b is a clean merge of a rather than a
  disagreement.
* **Per-cluster Jaccard** (Hennig 2007) -- the one to lead with. Above 0.75 a cluster is
  reproducible, below 0.5 it has dissolved. Reported by cluster count and by point share,
  which answer different questions.
* **Noise agreement** -- all four cells of clustered/noise against clustered/noise.

The caveat that must travel with the result
-------------------------------------------
Stability rises as clusterings get coarser -- a 2-cluster solution is nearly always stable
(von Luxburg 2010). A stability score is therefore only interpretable next to the cluster
count, exactly like connectivity. Read the stability-vs-granularity curve; do not pick the
single highest number.

References
----------
* Hennig (2007), "Cluster-wise assessment of cluster stability", Comput. Stat. Data Anal.
  52(1):258-271.
* Ben-Hur, Elisseeff & Guyon (2002), "A stability based method for discovering structure in
  clustered data", Pacific Symposium on Biocomputing 7:6-17.
* Tibshirani & Walther (2005), "Cluster Validation by Prediction Strength", J. Comput. Graph.
  Stat. 14(3):511-528.
* Lange, Roth, Braun & Buhmann (2004), "Stability-based validation of clustering solutions",
  Neural Computation 16(6):1299-1323.
* von Luxburg (2010), "Clustering Stability: An Overview", Found. Trends ML 2(3):235-274.
* Meila (2007), "Comparing clusterings -- an information based distance", J. Multivariate
  Analysis 98(5):873-895.

Usage
-----
    python umap_hdbscan_sweep/stability_sweep.py \
        --coords  <coords.npy> \
        --output-dir umap_tests/stability \
        --fit-sizes 500000 1000000 5000000 \
        --min-cluster-sizes 250 500 1000 2500 \
        --min-samples 5 --replicates 3

    # what it will cost, without running anything
    python umap_hdbscan_sweep/stability_sweep.py --coords <coords.npy> --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cross_size_ari import compare  # noqa: E402
from hdbscan_param_sweep import (  # noqa: E402
    Cell, _fit_hdbscan, _to_host, build_grid, log, projected_fit_seconds,
)

DEFAULT_PROBE_ROWS = 5_000_000
DEFAULT_REPLICATES = 3
# The probe draw must not move when the grid or the replicate seeds change, or replicates
# would be compared on different rows.
PROBE_SEED = 20260813


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coords", type=Path, required=True,
                   help="coords.npy, the fixed UMAP embedding of the whole cohort")
    p.add_argument("--output-dir", type=Path, required=True)
    # Comma-separated to match hdbscan_param_sweep.py, so one shell runner can drive both.
    p.add_argument("--fit-sizes", default="500000,1000000,5000000")
    p.add_argument("--min-cluster-sizes", default="250,500,1000,2500")
    p.add_argument("--min-samples", default="5")
    p.add_argument("--methods", default="eom")
    p.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES,
                   help="independent fits per cell; pairs scored = R*(R-1)/2")
    p.add_argument("--probe-rows", type=int, default=DEFAULT_PROBE_ROWS,
                   help="held-out rows every replicate labels. Excluded from all fit sets.")
    p.add_argument("--seed", type=int, default=42,
                   help="base seed; replicate r draws its fit rows with seed+r")
    p.add_argument("--backend", default="rbc", choices=["rbc", "brute", "sklearn"])
    p.add_argument("--batch-rows", type=int, default=5_000_000)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--dry-run", action="store_true", help="print the grid and cost, then stop")
    p.add_argument("--overwrite", action="store_true",
                   help="rerun cells that already have stability.json")
    args = p.parse_args()
    args.fit_sizes = [int(v) for v in args.fit_sizes.split(",") if v]
    args.min_cluster_sizes = [int(v) for v in args.min_cluster_sizes.split(",") if v]
    args.min_samples = [int(v) for v in args.min_samples.split(",") if v]
    args.methods = [v for v in args.methods.split(",") if v]
    return args


def probe_and_pool(total_rows: int, probe_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the cohort into a fixed probe and the pool every fit set is drawn from.

    Holding the probe out of every fit matters: a replicate that trained on a probe row knows
    its label exactly, while the others only predict it, and that asymmetry would show up as
    agreement that has nothing to do with stability.
    """
    rows = min(probe_rows, total_rows // 2)
    probe = np.sort(np.random.default_rng(PROBE_SEED).choice(total_rows, size=rows,
                                                             replace=False))
    mask = np.ones(total_rows, dtype=bool)
    mask[probe] = False
    return probe, np.nonzero(mask)[0]


def label_probe(cell: Cell, coords: np.ndarray, pool: np.ndarray, probe: np.ndarray,
                seed: int, args) -> tuple[np.ndarray, dict]:
    """One replicate: draw a fit set from the pool, fit, and label the probe."""
    import fast_predict

    rng = np.random.default_rng(seed)
    fit_indices = pool[np.sort(rng.choice(pool.size, size=cell.fit_rows, replace=False))]
    fit_coords = np.asarray(coords[fit_indices], dtype=np.float32)

    started = perf_counter()
    clusterer = _fit_hdbscan(fit_coords, cell, args)
    fit_seconds = round(perf_counter() - started, 1)
    fit_labels = np.asarray(_to_host(clusterer.labels_)).astype(np.int32)
    n_clusters = int(fit_labels.max()) + 1 if (fit_labels >= 0).any() else 0

    started = perf_counter()
    tables = fast_predict.build_tables(clusterer, n_fit=cell.fit_rows,
                                       min_samples=cell.min_samples)
    index = fast_predict.build_index(fit_coords, 2 * tables.min_samples, args.backend)
    probe_labels, probe_probabilities = fast_predict.predict(
        tables, fit_coords, np.asarray(coords[probe], dtype=np.float32),
        backend=args.backend, batch_rows=args.batch_rows, index=index)
    label_seconds = round(perf_counter() - started, 1)

    probe_labels = np.asarray(probe_labels).astype(np.int32)
    return probe_labels, {
        "seed": seed, "fit_seconds": fit_seconds, "label_seconds": label_seconds,
        "n_clusters_fit": n_clusters,
        "fit_noise_fraction": round(float((fit_labels < 0).mean()), 4),
        "probe_n_clusters": int(probe_labels.max()) + 1 if (probe_labels >= 0).any() else 0,
        "probe_noise_fraction": round(float((probe_labels < 0).mean()), 4),
        "probe_mean_probability": round(float(np.asarray(probe_probabilities).mean()), 4),
    }


def summarise_pairs(pairs: list[dict]) -> dict:
    """Mean and worst case over the replicate pairs.

    The minimum is reported beside the mean because a cell that reproduces well twice and
    badly once is not a stable cell, and averaging hides exactly that.
    """
    keys = ["ari", "ari_with_noise", "variation_of_information", "both_clustered_fraction",
            "jaccard_median_a_to_b", "frac_clusters_stable_a_to_b",
            "point_share_stable_a_to_b", "point_share_dissolved_a_to_b"]
    out: dict = {"n_pairs": len(pairs)}
    for k in keys:
        values = [p[k] for p in pairs if k in p and p[k] is not None]
        if not values:
            continue
        out[f"{k}_mean"] = round(float(np.mean(values)), 4)
        out[f"{k}_min"] = round(float(np.min(values)), 4)
    return out


def run_cell(cell: Cell, coords: np.ndarray, pool: np.ndarray, probe: np.ndarray,
             cell_dir: Path, args) -> dict:
    cell_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"cell": asdict(cell), "label": cell.label(),
                    "started": datetime.now(timezone.utc).isoformat(),
                    "probe_rows": int(probe.size), "replicates": args.replicates}

    labellings, replicates = [], []
    for r in range(args.replicates):
        seed = args.seed + r
        log(f"    replicate {r + 1}/{args.replicates} (seed {seed})")
        labels, info = label_probe(cell, coords, pool, probe, seed, args)
        log(f"      {info['fit_seconds']}s fit, {info['n_clusters_fit']:,} clusters, "
            f"{info['probe_noise_fraction'] * 100:.2f}% probe noise")
        labellings.append(labels)
        replicates.append(info)
    record["replicate_runs"] = replicates

    counts = [i["n_clusters_fit"] for i in replicates]
    record["cluster_count_mean"] = float(np.mean(counts))
    record["cluster_count_spread"] = int(max(counts) - min(counts))

    pairs = []
    for (i, a), (j, b) in combinations(list(enumerate(labellings)), 2):
        pair = compare(a, b, mask=None)
        pair["replicates"] = [i, j]
        pairs.append(pair)
        log(f"      pair {i}-{j}: ARI {pair.get('ari')}, "
            f"stable clusters {pair.get('frac_clusters_stable_a_to_b')}, "
            f"point share stable {pair.get('point_share_stable_a_to_b')}")
    record["pairs"] = pairs
    record["summary"] = summarise_pairs(pairs)
    record["finished"] = datetime.now(timezone.utc).isoformat()

    (cell_dir / "stability.json").write_text(json.dumps(record, indent=2))
    return record


def main() -> int:
    args = parse_args()
    grid = build_grid(args.fit_sizes, args.min_cluster_sizes, args.min_samples,
                      args.methods, [0.0], max_min_samples_at={})

    per_cell = [args.replicates * projected_fit_seconds(c.fit_rows, min(args.min_samples))
                for c in grid]
    log(f"grid: {len(grid)} cells x {args.replicates} replicates = "
        f"{len(grid) * args.replicates} fits")
    log(f"projected: {sum(per_cell) / 3600:.1f} h of fitting "
        f"(+ probe labelling, ~{args.probe_rows / 157_500_000:.1%} of a cohort pass each)")
    for cell, seconds in zip(grid, per_cell):
        log(f"    {cell.label():44} ~{seconds / 60:7.1f} min for {args.replicates} fits")
    if args.dry_run:
        return 0

    coords = np.load(args.coords, mmap_mode="r")
    log(f"coords: {coords.shape[0]:,} x {coords.shape[1]}")
    probe, pool = probe_and_pool(coords.shape[0], args.probe_rows)
    log(f"probe {probe.size:,} rows held out; fit pool {pool.size:,} rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for n, cell in enumerate(grid, start=1):
        cell_dir = args.output_dir / "cells" / cell.label()
        if (cell_dir / "stability.json").exists() and not args.overwrite:
            log(f"[{n}/{len(grid)}] {cell.label()} -- already done, skipping")
            continue
        log(f"[{n}/{len(grid)}] {cell.label()}")
        record = run_cell(cell, coords, pool, probe, cell_dir, args)
        s = record["summary"]
        log(f"    ARI {s.get('ari_mean')} (min {s.get('ari_min')})   "
            f"stable point share {s.get('point_share_stable_a_to_b_mean')}   "
            f"cluster count {record['cluster_count_mean']:.0f} "
            f"+/-{record['cluster_count_spread']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
