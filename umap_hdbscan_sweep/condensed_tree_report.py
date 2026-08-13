#!/usr/bin/env python
"""Look at the condensed tree, the membership probabilities, and the persistences directly.

Three questions this answers that a metrics.json cannot.

**1. What does the hierarchy actually look like?** The condensed tree is HDBSCAN's own picture
of the clustering: bars are clusters, width is population, height is the density scale over
which they survive, and the selected clusters are the ones EOM kept. A grid of numbers cannot
show whether the clusters are deep stable structures or a thin crust shaved off one blob.

**2. Are the cuML numbers real?** The 2026-08-12 sweep reported ``persistence_mean == 1.0``
in 26 of 28 cells and ``prob_median == 1.0`` in all 28. Both are suspicious in the same
direction, and both come from cuML. This script fits the SAME rows with CPU hdbscan and
prints the two side by side. If the CPU run gives a spread where cuML gave a constant, the
cuML values are degenerate and every persistence and probability number in the sweep must be
re-read as unavailable rather than as evidence.

**3. Do the probabilities survive out of sample?** The sweep's ``prob_*`` are fit-set
numbers, but the product is 157.5M cohort labels assigned by ``fast_predict``. Those carry
their own probabilities, and in the sweep they fell from 0.91 at 500k fit rows to 0.006 at
25M -- so the fit-set confidence does not describe the labels that were actually produced.
``--probe-rows`` re-measures that gap on held-out rows.

Usage
-----
    python umap_hdbscan_sweep/condensed_tree_report.py \
        --coords umap_tests/hdbscan_scaling/coords.npy \
        --output-dir umap_tests/tree_report \
        --fit-rows 500000 --min-cluster-size 1000 --min-samples 15

Fitting on CPU is the point, so keep ``--fit-rows`` modest: 500k takes a few minutes, 5M does
not fit in a sensible interactive budget. The tree's shape is a property of the parameters,
not of the row count, so a 500k view is representative of the cell it names.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def log(message: str) -> None:
    from datetime import datetime
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coords", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fit-rows", type=int, default=500_000)
    p.add_argument("--min-cluster-size", type=int, default=1000)
    p.add_argument("--min-samples", type=int, default=15)
    p.add_argument("--cluster-selection-method", default="eom", choices=["eom", "leaf"])
    p.add_argument("--probe-rows", type=int, default=1_000_000,
                   help="held-out rows to re-label, to compare out-of-sample confidence "
                        "against the fit set. 0 to skip.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--compare-cuml", action="store_true",
                   help="also fit the same rows with cuML and print both, to test whether "
                        "the constant persistence/probability values are a cuML artefact")
    return p.parse_args()


def describe(values: np.ndarray, name: str) -> dict:
    """A distribution, not a mean. The mean is what hid the problem in the first place."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{name}_n": 0}
    return {
        f"{name}_n": int(values.size),
        f"{name}_mean": float(values.mean()),
        f"{name}_p01": float(np.percentile(values, 1)),
        f"{name}_p10": float(np.percentile(values, 10)),
        f"{name}_median": float(np.median(values)),
        f"{name}_p90": float(np.percentile(values, 90)),
        f"{name}_max": float(values.max()),
        # The tell for a degenerate array: everything pinned to one value.
        f"{name}_frac_exactly_1": float((values >= 1.0 - 1e-9).mean()),
        f"{name}_frac_exactly_0": float((values <= 1e-9).mean()),
        f"{name}_distinct_values": int(np.unique(np.round(values, 6)).size),
    }


def plot_tree(clusterer, out_path: Path, args) -> None:
    """The condensed tree, with the selected clusters outlined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 9))
    # log_size keeps a 1,000-cluster tree legible: without it the root bar is so wide that
    # every real cluster collapses to a hairline.
    clusterer.condensed_tree_.plot(select_clusters=True, log_size=True,
                                   selection_palette=None, axis=ax)
    ax.set_title(f"Condensed tree — {args.fit_rows:,} rows, "
                 f"mcs={args.min_cluster_size}, ms={args.min_samples}, "
                 f"{args.cluster_selection_method}\n"
                 f"bar width = population (log), height = density scale it survives; "
                 f"outlined bars are the clusters EOM selected")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log(f"  wrote {out_path}")


def plot_distributions(probabilities, persistences, held_probabilities,
                       out_path: Path, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 2 + (held_probabilities is not None)
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4.4))

    assigned = probabilities[probabilities > 0]
    axes[0].hist(assigned, bins=60, color="#3b6ea5")
    axes[0].set_title(f"Fit-set membership probability\n"
                      f"median {np.median(assigned):.3f}, "
                      f"{100 * (assigned >= 0.999).mean():.1f}% at 1.0")
    axes[0].set_xlabel("probability")

    axes[1].hist(persistences, bins=40, color="#8a5a9c")
    axes[1].set_title(f"Cluster persistence\n{len(persistences)} clusters, "
                      f"{len(np.unique(np.round(persistences, 6)))} distinct values")
    axes[1].set_xlabel("persistence")

    if held_probabilities is not None:
        held = np.asarray(held_probabilities)
        axes[2].hist(held[held > 0], bins=60, color="#a5643b")
        axes[2].set_title(f"HELD-OUT probability (fast_predict)\n"
                          f"mean {held.mean():.4f} — this is what the cohort labels carry")
        axes[2].set_xlabel("probability")

    for ax in axes:
        ax.set_ylabel("points" if ax is not axes[1] else "clusters")
    fig.suptitle(f"{args.fit_rows:,} rows, mcs={args.min_cluster_size}, "
                 f"ms={args.min_samples} — CPU hdbscan")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log(f"  wrote {out_path}")


def main() -> int:
    import hdbscan

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coords = np.load(args.coords, mmap_mode="r")
    total = coords.shape[0]
    rng = np.random.default_rng(args.seed)
    fit_index = np.sort(rng.choice(total, size=min(args.fit_rows, total), replace=False))
    fit_coords = np.asarray(coords[fit_index], dtype=np.float32)
    log(f"coords {total:,} x {coords.shape[1]}; fitting {fit_coords.shape[0]:,} rows on CPU")

    started = perf_counter()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
        cluster_selection_method=args.cluster_selection_method,
        metric="euclidean", prediction_data=True, core_dist_n_jobs=args.threads,
    ).fit(fit_coords)
    fit_seconds = perf_counter() - started

    labels = clusterer.labels_.astype(np.int32)
    probabilities = np.asarray(clusterer.probabilities_, dtype=np.float64)
    persistences = np.asarray(clusterer.cluster_persistence_, dtype=np.float64)
    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
    log(f"  {fit_seconds:.1f}s, {n_clusters:,} clusters, "
        f"{100 * (labels < 0).mean():.2f}% noise")

    record = {
        "fit_rows": int(fit_coords.shape[0]), "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "cluster_selection_method": args.cluster_selection_method,
        "backend": "cpu_hdbscan", "fit_seconds": round(fit_seconds, 1),
        "n_clusters": n_clusters, "noise_fraction": float((labels < 0).mean()),
        **describe(probabilities[labels >= 0], "probability"),
        **describe(persistences, "persistence"),
    }
    log(f"  CPU persistence: median {record.get('persistence_median'):.4f}, "
        f"{record.get('persistence_distinct_values')} distinct values, "
        f"{100 * record.get('persistence_frac_exactly_1', 0):.1f}% exactly 1.0")
    log(f"  CPU probability: median {record.get('probability_median'):.4f}, "
        f"{100 * record.get('probability_frac_exactly_1', 0):.1f}% exactly 1.0")

    held_probabilities = None
    if args.probe_rows:
        import fast_predict
        mask = np.ones(total, dtype=bool)
        mask[fit_index] = False
        pool = np.nonzero(mask)[0]
        probe = pool[np.sort(rng.choice(pool.size, size=min(args.probe_rows, pool.size),
                                        replace=False))]
        log(f"  labelling {probe.size:,} held-out rows to measure out-of-sample confidence")
        tables = fast_predict.build_tables(clusterer, n_fit=fit_coords.shape[0],
                                           min_samples=args.min_samples)
        index = fast_predict.build_index(fit_coords, 2 * tables.min_samples, "sklearn")
        held_labels, held_probabilities = fast_predict.predict(
            tables, fit_coords, np.asarray(coords[probe], dtype=np.float32),
            backend="sklearn", batch_rows=1_000_000, index=index)
        held_labels = np.asarray(held_labels)
        held_probabilities = np.asarray(held_probabilities, dtype=np.float64)
        record["held_noise_fraction"] = float((held_labels < 0).mean())
        record.update(describe(held_probabilities[held_labels >= 0], "held_probability"))
        # The comparison that matters: fit-set confidence vs the confidence the cohort
        # labels actually carry.
        record["confidence_gap"] = round(
            record["probability_mean"] - record.get("held_probability_mean", float("nan")), 4)
        log(f"  held-out probability mean {record.get('held_probability_mean'):.4f} "
            f"vs fit-set {record['probability_mean']:.4f}  "
            f"(gap {record['confidence_gap']:+.4f})")

    if args.compare_cuml:
        try:
            from cuml.cluster import HDBSCAN as CumlHDBSCAN
            log("  refitting the same rows with cuML for comparison")
            gpu = CumlHDBSCAN(
                min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
                cluster_selection_method=args.cluster_selection_method,
                metric="euclidean", prediction_data=True).fit(fit_coords)
            to_host = lambda a: np.asarray(a.get() if hasattr(a, "get") else a)
            gpu_labels = to_host(gpu.labels_)
            record["cuml"] = {
                "n_clusters": int(gpu_labels.max()) + 1 if (gpu_labels >= 0).any() else 0,
                "noise_fraction": float((gpu_labels < 0).mean()),
                **describe(to_host(gpu.probabilities_)[gpu_labels >= 0], "probability"),
                **describe(to_host(gpu.cluster_persistence_), "persistence"),
            }
            log(f"  cuML persistence: median "
                f"{record['cuml'].get('persistence_median'):.4f}, "
                f"{record['cuml'].get('persistence_distinct_values')} distinct values")
        except Exception as exc:  # noqa: BLE001
            record["cuml_error"] = f"{type(exc).__name__}: {exc}"
            log(f"  cuML comparison unavailable: {exc}")

    stem = (f"fit{args.fit_rows}_mcs{args.min_cluster_size}"
            f"_ms{args.min_samples}_{args.cluster_selection_method}")
    (args.output_dir / f"{stem}.json").write_text(json.dumps(record, indent=2))
    try:
        plot_tree(clusterer, args.output_dir / f"{stem}_condensed_tree.png", args)
        plot_distributions(probabilities[labels >= 0], persistences, held_probabilities,
                           args.output_dir / f"{stem}_distributions.png", args)
    except Exception as exc:  # noqa: BLE001 - a plotting failure must not lose the fit
        log(f"  plotting failed: {type(exc).__name__}: {exc}")
    log(f"  wrote {args.output_dir / f'{stem}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
