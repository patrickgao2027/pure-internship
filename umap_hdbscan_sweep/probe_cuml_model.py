#!/usr/bin/env python
"""Can a cuML HDBSCAN model be saved, reloaded, and used to label new rows?

Everything downstream of the sweep -- per_parquet_inference above all -- consumes a
*pickled* clusterer: it joblib.loads the model, hands it to fast_predict.build_tables, and
labels 157.5M rows against the tables that come back. The CPU `hdbscan` package supports
that path by design. cuML does not advertise it: its estimators pickle through CumlArray
serialisation, it exposes no `prediction_data_`, and its `labels_` may come back as a cupy
array rather than a numpy one. Whether the round trip survives is an empirical question, and
the honest place to answer it is here -- in two minutes on a small fit -- rather than 20
hours into a 95-sample run that turns out to have been labelling against a broken model.

The probe deliberately mirrors per_parquet_inference's own call sequence rather than testing
pickling in the abstract:

    joblib.dump -> joblib.load -> read labels_/min_samples -> build_tables -> build_index
    -> predict

and then checks the reloaded model predicts *identically* to the in-memory one. A model that
loads without raising but silently returns different labels is the dangerous outcome, not the
one that crashes, so agreement is asserted rather than assumed.

Usage:
    python umap_hdbscan_sweep/probe_cuml_model.py --coords <coords.npy>
    python umap_hdbscan_sweep/probe_cuml_model.py --coords <coords.npy> --backend cpu
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np


def log(message: str) -> None:
    print(message, flush=True)


def _to_host(array) -> np.ndarray:
    """cupy -> numpy, numpy -> itself. cuML returns device arrays from some attributes."""
    for attribute in ("get", "to_numpy"):
        method = getattr(array, attribute, None)
        if callable(method):
            return np.asarray(method())
    return np.asarray(array)


def build_clusterer(fit_coords: np.ndarray, mcs: int, ms: int, epsilon: float, backend: str):
    """Fit with the requested backend, returning (clusterer, backend_that_ran)."""
    common = dict(min_cluster_size=mcs, min_samples=ms,
                  cluster_selection_method="eom",
                  cluster_selection_epsilon=epsilon)
    if backend == "cpu":
        import hdbscan
        return hdbscan.HDBSCAN(**common, prediction_data=True).fit(fit_coords), "cpu"
    from cuml.cluster import HDBSCAN as CumlHDBSCAN
    return CumlHDBSCAN(**common).fit(fit_coords), "cuml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coords", required=True, help="cohort coords.npy")
    parser.add_argument("--backend", default="cuml", choices=["cuml", "cpu"])
    parser.add_argument("--fit-rows", type=int, default=200_000,
                        help="small on purpose -- this probes the round trip, not the fit")
    parser.add_argument("--query-rows", type=int, default=50_000)
    parser.add_argument("--mcs", type=int, default=2500)
    parser.add_argument("--ms", type=int, default=15)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-backend", default="auto",
                        help="fast_predict knn backend (auto/cuml/sklearn)")
    parser.add_argument("--keep", default=None,
                        help="write the pickle here instead of a temp dir")
    args = parser.parse_args()

    import fast_predict

    coords = np.load(args.coords, mmap_mode="r")
    total = coords.shape[0]
    log(f"coords          : {args.coords}  {coords.shape} {coords.dtype}")

    rng = np.random.default_rng(args.seed)
    fit_idx = np.sort(rng.choice(total, size=min(args.fit_rows, total), replace=False))
    fit_coords = np.asarray(coords[fit_idx], dtype=np.float32)

    held_mask = np.ones(total, dtype=bool)
    held_mask[fit_idx] = False
    held_idx = np.nonzero(held_mask)[0]
    query_idx = held_idx[rng.choice(len(held_idx), size=min(args.query_rows, len(held_idx)),
                                    replace=False)]
    query_coords = np.asarray(coords[np.sort(query_idx)], dtype=np.float32)

    log(f"fit rows        : {len(fit_idx):,}     query rows: {len(query_coords):,}")
    log(f"params          : mcs={args.mcs} ms={args.ms} eps={args.epsilon}")

    # ── 1. fit ──────────────────────────────────────────────────────────────────
    started = perf_counter()
    clusterer, backend = build_clusterer(fit_coords, args.mcs, args.ms, args.epsilon,
                                         args.backend)
    fit_seconds = perf_counter() - started
    labels_before = _to_host(clusterer.labels_).ravel().astype(np.int32)
    n_clusters = int(labels_before.max()) + 1 if (labels_before >= 0).any() else 0
    log(f"\n[1] fit          : {fit_seconds:.1f}s  backend={backend}  "
        f"{n_clusters} clusters  {(labels_before < 0).mean() * 100:.2f}% noise")
    if n_clusters == 0:
        log("    no clusters -- raise --fit-rows or lower --mcs; probe is inconclusive")
        return 1

    # ── 2. predict with the IN-MEMORY model (the reference answer) ──────────────
    started = perf_counter()
    tables_mem = fast_predict.build_tables(clusterer, n_fit=len(fit_idx),
                                           min_samples=args.ms)
    index = fast_predict.build_index(fit_coords, 2 * tables_mem.min_samples,
                                     args.predict_backend)
    labels_mem, probabilities_mem = fast_predict.predict(
        tables_mem, fit_coords, query_coords, backend=args.predict_backend, index=index)
    log(f"[2] in-memory    : build_tables + predict OK ({perf_counter() - started:.1f}s), "
        f"{(labels_mem < 0).mean() * 100:.2f}% noise on the query set")

    # ── 3. round trip ───────────────────────────────────────────────────────────
    temp_dir = None
    if args.keep:
        pkl_path = Path(args.keep)
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory()
        pkl_path = Path(temp_dir.name) / "probe_model.pkl"

    import joblib
    try:
        started = perf_counter()
        joblib.dump(clusterer, pkl_path)
        size_mb = pkl_path.stat().st_size / 1e6
        log(f"[3] joblib.dump  : OK  {size_mb:,.1f} MB  ({perf_counter() - started:.1f}s)"
            f"  -> {pkl_path}")
    except Exception as exc:  # noqa: BLE001 - the whole point is to report, not raise
        log(f"[3] joblib.dump  : FAILED  {type(exc).__name__}: {exc}")
        log("\nVERDICT: this backend cannot be saved. Downstream scripts that take "
            "--hdbscan-model cannot use it.")
        return 2

    try:
        started = perf_counter()
        reloaded = joblib.load(pkl_path)
        log(f"[4] joblib.load  : OK  ({perf_counter() - started:.1f}s)  "
            f"{type(reloaded).__module__}.{type(reloaded).__name__}")
    except Exception as exc:  # noqa: BLE001
        log(f"[4] joblib.load  : FAILED  {type(exc).__name__}: {exc}")
        log("\nVERDICT: the pickle is written but cannot be read back.")
        return 2

    # ── 5. the attributes the consumers actually read ───────────────────────────
    problems = []
    try:
        labels_after = _to_host(reloaded.labels_).ravel().astype(np.int32)
        if labels_after.shape != labels_before.shape:
            problems.append(f"labels_ shape changed {labels_before.shape} -> {labels_after.shape}")
        elif not np.array_equal(labels_after, labels_before):
            problems.append(f"labels_ changed on reload "
                            f"({(labels_after != labels_before).sum():,} rows differ)")
        else:
            log(f"[5] labels_      : preserved  ({len(labels_after):,} rows identical)")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"labels_ unreadable ({type(exc).__name__}: {exc})")

    model_ms = getattr(reloaded, "min_samples", None)
    log(f"    min_samples  : {model_ms!r}"
        + ("" if model_ms else "   <- absent; consumers fall back to the CLI value"))
    for attribute in ("probabilities_", "cluster_persistence_", "outlier_scores_",
                      "exemplars_", "prediction_data_", "condensed_tree_"):
        present = getattr(reloaded, attribute, None) is not None
        log(f"    {attribute:<22}{'present' if present else '-'}")

    # ── 6. predict with the RELOADED model, and require agreement ───────────────
    try:
        started = perf_counter()
        tables_disk = fast_predict.build_tables(reloaded, n_fit=len(fit_idx),
                                                min_samples=args.ms)
        labels_disk, probabilities_disk = fast_predict.predict(
            tables_disk, fit_coords, query_coords, backend=args.predict_backend, index=index)
        log(f"[6] reloaded     : build_tables + predict OK "
            f"({perf_counter() - started:.1f}s)")
    except Exception as exc:  # noqa: BLE001
        log(f"[6] reloaded     : FAILED  {type(exc).__name__}: {exc}")
        log("\nVERDICT: the model reloads but cannot drive fast_predict. "
            "per_parquet_inference would fail on it.")
        return 2

    disagree = int((labels_disk != labels_mem).sum())
    if disagree:
        problems.append(f"{disagree:,}/{len(labels_mem):,} query rows "
                        f"({disagree / len(labels_mem) * 100:.3f}%) label differently "
                        f"after the round trip")
    else:
        log(f"[7] agreement    : all {len(labels_mem):,} query rows identical, "
            f"max |Δp| = {np.abs(probabilities_disk - probabilities_mem).max():.2e}")

    if temp_dir is not None:
        temp_dir.cleanup()

    log("")
    if problems:
        log("VERDICT: NOT SAFE for --hdbscan-model")
        for problem in problems:
            log(f"  - {problem}")
        return 3
    log(f"VERDICT: SAFE. A {backend} model can be passed to per_parquet_inference "
        f"as --hdbscan-model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
