#!/usr/bin/env python
"""Exact `approximate_predict`, with the brute-force kNN swapped for an indexed one.

Why this exists
---------------
The scaling sweep measured cuML's ``approximate_predict`` at 4.35 s per M-probe-rows per
M-fit-rows -- dead linear in the fit-set size across a 50x range (500K -> 25M). That constant
is the signature of a brute-force neighbour search: every query point compared against every
fit point. At 25M fit rows, labelling the 157.5M-row cohort costs ~4 hours, 87% of a cell.

But only ONE step of approximate_predict scales with the fit set, and it is not intrinsic to
HDBSCAN. From the reference implementation (hdbscan/prediction.py) the algorithm is:

    1. query the 2*min_samples nearest fit points          <-- the only O(N_fit) step
    2. mr = max(core_dist(neighbour), core_dist_est(query), dist) for each neighbour, where
       core_dist_est(query) is the (min_samples)-th raw neighbour distance
    3. pick the neighbour minimising mr; lambda = 1 / mr
    4. walk up the condensed tree from that neighbour's leaf while lambda_val >= lambda
    5. label = cluster_map[cluster] (else -1); prob = min(max_lambda, lambda) / max_lambda

Steps 2-5 are O(tree depth) and independent of N_fit. Replacing step 1 with an indexed
neighbour search -- leaving 2-5 byte-identical -- reproduces the output EXACTLY while
removing the term that scales.

cuML already ships the index: ``NearestNeighbors(algorithm='rbc')``, Cayton's Random Ball
Cover. It partitions the space and prunes with the triangle inequality, answering queries in
O(sqrt n) after an O(n sqrt n) build, and it is EXACT -- not an approximation. It supports
euclidean/l2 (only haversine carries the <=3-dimension restriction) and the embedding here is
2-D, the regime it was designed for. At 25M fit rows sqrt(n) ~ 5,000 versus n = 25M.

Exactness is a claim to check, not assert. ``--validate`` re-labels the saved eval probe and
compares against the ``approximate_predict`` labels the sweep already wrote, so agreement is
measured against real ground truth before anything depends on it.

    # 1. does the fitted model expose what this needs? (seconds -- run this first)
    python umap_hdbscan_sweep/fast_predict.py --inspect --model models/fit25000000.joblib

    # 2. exactness against the saved eval-probe labels
    python umap_hdbscan_sweep/fast_predict.py --validate \
        --model models/fit25000000.joblib --coords coords.npy \
        --fit-indices labels/fit25000000_indices.npy \
        --probe-indices eval_probe_indices.npy \
        --reference labels/fit25000000_evalprobe_labels.npy

    # 3. label the full cohort
    python umap_hdbscan_sweep/fast_predict.py --apply \
        --model models/fit25000000.joblib --coords coords.npy \
        --fit-indices labels/fit25000000_indices.npy --out cohort_labels.npy

The tree-descent core is pure numpy and is tested against hdbscan's own ``approximate_predict``
in tests/test_fast_predict.py -- that part needs no GPU.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

DBL_MAX = np.finfo(np.float64).max


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}", flush=True)


# ── the tables the descent needs, densified into arrays ────────────────────────

@dataclass
class PredictionTables:
    """Everything from the fitted model that steps 2-5 touch, as flat arrays.

    The reference implementation carries dicts and re-scans the raw tree per point. Indexing
    by cluster id into arrays instead is what lets the whole descent run vectorised over
    millions of query points at once.
    """

    core_distances: np.ndarray      # (n_fit,)   core distance of every fit point
    point_parent: np.ndarray        # (n_fit,)   condensed-tree cluster each point falls in
    point_lambda: np.ndarray        # (n_fit,)   lambda_val of that point's tree row
    parent_of: np.ndarray           # (n_clusters+1,) cluster -> parent cluster
    lambda_of: np.ndarray           # (n_clusters+1,) cluster -> lambda_val (-inf if absent)
    cluster_label: np.ndarray       # (n_clusters+1,) cluster -> final label, -1 if unselected
    max_lambda: np.ndarray          # (n_clusters+1,) cluster -> max lambda in that cluster
    tree_root: int
    min_samples: int
    n_fit: int


def build_tables(clusterer, n_fit: int, min_samples: int | None = None) -> PredictionTables:
    """Densify a fitted HDBSCAN (cuML or CPU) into PredictionTables.

    Raises with a specific message when an attribute is missing rather than failing deep in
    the descent, because "which internals does this cuML build expose" is the one thing that
    cannot be settled off the GPU box.
    """
    raw_tree = _raw_condensed_tree(clusterer)
    min_samples = int(min_samples or getattr(clusterer, "min_samples", None)
                      or clusterer.min_cluster_size)

    core_distances = _core_distances(clusterer, n_fit)
    if core_distances.shape[0] != n_fit:
        raise ValueError(
            f"core_distances has {core_distances.shape[0]} entries but the fit set has "
            f"{n_fit} rows -- the --fit-indices file does not match this model"
        )

    parents = raw_tree["parent"].astype(np.int64)
    children = raw_tree["child"].astype(np.int64)
    lambdas = raw_tree["lambda_val"].astype(np.float64)
    child_sizes = raw_tree["child_size"].astype(np.int64)

    # Rows whose child is an individual point (child_size == 1) give each point its leaf
    # cluster; rows whose child is itself a cluster form the cluster tree walked in step 4.
    is_point = children < n_fit
    point_parent = np.zeros(n_fit, dtype=np.int64)
    point_lambda = np.zeros(n_fit, dtype=np.float64)
    point_parent[children[is_point]] = parents[is_point]
    point_lambda[children[is_point]] = lambdas[is_point]

    is_cluster_row = child_sizes > 1
    max_cluster = int(max(parents.max(), children.max())) + 1

    parent_of = np.arange(max_cluster, dtype=np.int64)
    lambda_of = np.full(max_cluster, -np.inf, dtype=np.float64)
    parent_of[children[is_cluster_row]] = parents[is_cluster_row]
    lambda_of[children[is_cluster_row]] = lambdas[is_cluster_row]

    # Each cluster's own max lambda = the largest lambda_val among rows having it as parent.
    own_max_lambda = np.zeros(max_cluster, dtype=np.float64)
    np.maximum.at(own_max_lambda, parents, lambdas)

    cluster_label = _cluster_labels(clusterer, raw_tree, max_cluster, parent_of,
                                    is_cluster_row, parents, children)

    # A cluster below a selected one inherits the SELECTED ancestor's max lambda, not its
    # own (PredictionData.__init__: `self.max_lambdas[sub_cluster] = self.max_lambdas[cluster]`).
    # Probabilities are normalised by the selected cluster's scale, so using each cluster's
    # own maximum here silently inflates the probability of points that land on a descendant.
    # The selected cluster is the topmost ancestor still carrying the same label.
    max_lambda = own_max_lambda[_selected_ancestor(cluster_label, parent_of)]

    return PredictionTables(
        core_distances=core_distances.astype(np.float64),
        point_parent=point_parent,
        point_lambda=point_lambda,
        parent_of=parent_of,
        lambda_of=lambda_of,
        cluster_label=cluster_label,
        max_lambda=max_lambda,
        tree_root=int(parents[is_cluster_row].min()) if is_cluster_row.any()
        else int(parents.min()),
        min_samples=min_samples,
        n_fit=n_fit,
    )


def _selected_ancestor(cluster_label: np.ndarray, parent_of: np.ndarray) -> np.ndarray:
    """For each cluster, the topmost ancestor carrying the same label -- i.e. the selected
    cluster it descends from. Unlabelled clusters map to themselves (their max lambda is
    never read, since probability is forced to 0 wherever the label is -1).

    Pointer-jumping instead of recursion: one pass per tree level, over all clusters at once.
    """
    top = np.arange(cluster_label.shape[0], dtype=np.int64)
    for _ in range(64):
        parent = parent_of[top]
        can_rise = (cluster_label[top] >= 0) & (parent != top) & (
            cluster_label[parent] == cluster_label[top])
        if not can_rise.any():
            break
        top[can_rise] = parent[can_rise]
    return top


def _raw_condensed_tree(clusterer) -> np.ndarray:
    tree = getattr(clusterer, "condensed_tree_", None)
    if tree is None:
        raise AttributeError(
            "fitted model has no `condensed_tree_`. Without the condensed tree the lambda "
            "descent (steps 4-5) cannot be reproduced and only approximate label transfer "
            "is possible. Run with --inspect to see what this model does expose."
        )
    for attribute in ("_raw_tree", "to_numpy"):
        value = getattr(tree, attribute, None)
        if value is None:
            continue
        return value() if callable(value) else value
    if isinstance(tree, np.ndarray):
        return tree
    try:  # cuML returns a cudf DataFrame from to_pandas()/to_numpy() in some versions
        return tree.to_pandas().to_records(index=False)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        raise AttributeError(
            f"could not get a raw condensed tree out of {type(tree)!r}: {exc}"
        ) from exc


def _core_distances(clusterer, n_fit: int) -> np.ndarray:
    prediction_data = getattr(clusterer, "prediction_data_", None)
    for holder, attribute in ((prediction_data, "core_distances"),
                              (clusterer, "core_distances_"),
                              (clusterer, "core_distances")):
        if holder is None:
            continue
        value = getattr(holder, attribute, None)
        if value is not None:
            return _to_host(value).ravel()
    raise AttributeError(
        "fitted model exposes no core distances (looked at prediction_data_.core_distances, "
        "core_distances_, core_distances). They can be recomputed as the min_samples-th "
        "nearest-neighbour distance within the fit set -- pass --recompute-core-distances."
    )


def _cluster_labels(clusterer, raw_tree, max_cluster, parent_of,
                    is_cluster_row, parents, children) -> np.ndarray:
    """cluster id -> final label, with selected clusters' descendants inheriting the label.

    The reference builds this by walking `_clusters_below` per selected cluster; here the
    selected set is taken from the model and descendants inherit by iterated pointer-jumping,
    which is the same mapping without the recursion.
    """
    labels = np.full(max_cluster, -1, dtype=np.int64)

    prediction_data = getattr(clusterer, "prediction_data_", None)
    cluster_map = getattr(prediction_data, "cluster_map", None) if prediction_data else None
    if cluster_map:
        for cluster, label in cluster_map.items():
            labels[int(cluster)] = int(label)
        return labels

    # No prediction_data_ (the cuML case): recover the selected clusters from the fit labels.
    # Each fit point's leaf cluster maps to that point's final label; propagating up gives
    # every selected cluster and its descendants.
    fit_labels = _to_host(clusterer.labels_).ravel()
    is_point = children < len(fit_labels)
    point_leaf = np.zeros(len(fit_labels), dtype=np.int64)
    point_leaf[children[is_point]] = parents[is_point]

    clustered = fit_labels >= 0
    labels[point_leaf[clustered]] = fit_labels[clustered]

    # Push each labelled cluster's label up to ancestors that carry the same label, so a
    # query landing on an ancestor resolves the same way the reference's cluster_map does.
    for _ in range(64):
        unlabelled = np.nonzero(labels < 0)[0]
        if unlabelled.size == 0:
            break
        inherited = labels[parent_of[unlabelled]]
        if not (inherited >= 0).any():
            break
        labels[unlabelled] = inherited
    return labels


def _to_host(array) -> np.ndarray:
    for attribute in ("to_numpy", "get"):
        method = getattr(array, attribute, None)
        if callable(method):
            return np.asarray(method())
    return np.asarray(array)


# ── neighbour search backends ──────────────────────────────────────────────────

def knn_query(fit_coords: np.ndarray, query_coords: np.ndarray, k: int,
              backend: str = "rbc", batch_rows: int = 5_000_000,
              index=None) -> tuple[np.ndarray, np.ndarray]:
    """k nearest fit points for each query point. All backends here are EXACT.

    'rbc'     cuML Random Ball Cover -- O(sqrt n) queries, exact, 2-D euclidean is its
              intended regime. This is the one that removes the O(N_fit) term.
    'brute'   cuML brute force -- what approximate_predict does internally. Reference.
    'sklearn' CPU KDTree -- what the CPU hdbscan reference uses. For local testing.
    """
    if backend == "sklearn":
        from sklearn.neighbors import KDTree
        tree = index if index is not None else KDTree(fit_coords)
        return tree.query(query_coords, k=k)

    from cuml.neighbors import NearestNeighbors
    if index is None:
        index = NearestNeighbors(n_neighbors=k, algorithm=backend, metric="euclidean")
        index.fit(fit_coords)

    distance_chunks, index_chunks = [], []
    for start in range(0, query_coords.shape[0], batch_rows):
        chunk = query_coords[start:start + batch_rows]
        distances, indices = index.kneighbors(chunk, n_neighbors=k)
        distance_chunks.append(_to_host(distances).astype(np.float64))
        index_chunks.append(_to_host(indices).astype(np.int64))
    return np.concatenate(distance_chunks), np.concatenate(index_chunks)


def build_index(fit_coords: np.ndarray, k: int, backend: str):
    if backend == "sklearn":
        from sklearn.neighbors import KDTree
        return KDTree(fit_coords)
    from cuml.neighbors import NearestNeighbors
    index = NearestNeighbors(n_neighbors=k, algorithm=backend, metric="euclidean")
    index.fit(fit_coords)
    return index


# ── steps 2-5, vectorised ──────────────────────────────────────────────────────

def predict_from_neighbours(tables: PredictionTables, neighbour_distances: np.ndarray,
                            neighbour_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Steps 2-5 of approximate_predict, over all query points at once.

    Mirrors hdbscan's `_find_neighbor_and_lambda` + `_find_cluster_and_probability` exactly,
    including the detail that the query point's own core distance is estimated as the
    (min_samples)-th raw neighbour distance -- index min_samples into the 2*min_samples
    neighbours, not the last one.
    """
    min_samples = tables.min_samples
    n_query = neighbour_indices.shape[0]

    # step 2: mutual reachability to each candidate neighbour
    point_core = neighbour_distances[:, min_samples][:, None]
    mr = np.maximum(tables.core_distances[neighbour_indices], point_core)
    np.maximum(mr, neighbour_distances, out=mr)

    # step 3: the neighbour minimising mutual reachability (NOT raw distance)
    best = mr.argmin(axis=1)
    rows = np.arange(n_query)
    mr_min = mr[rows, best]
    nearest = neighbour_indices[rows, best]
    lambda_ = np.where(mr_min > 0.0, 1.0 / np.where(mr_min > 0.0, mr_min, 1.0), DBL_MAX)

    # step 4: walk up the cluster tree while the edge's lambda_val >= the point's lambda.
    # The reference loops per point; here every point that can still step takes one step per
    # iteration, so the loop runs once per tree level rather than once per point.
    potential = tables.point_parent[nearest].copy()
    walking = np.nonzero(tables.point_lambda[nearest] > lambda_)[0]
    while walking.size:
        current = potential[walking]
        can_step = (current > tables.tree_root) & (tables.lambda_of[current] >= lambda_[walking])
        walking = walking[can_step]
        if walking.size == 0:
            break
        potential[walking] = tables.parent_of[potential[walking]]

    # step 5
    labels = tables.cluster_label[potential]
    max_lambda = tables.max_lambda[potential]
    clamped = np.minimum(max_lambda, lambda_)
    with np.errstate(divide="ignore", invalid="ignore"):
        probabilities = np.where(max_lambda > 0.0, clamped / max_lambda, 1.0)
    probabilities = np.where(labels >= 0, probabilities, 0.0)
    return labels.astype(np.int32), probabilities.astype(np.float32)


def predict(tables: PredictionTables, fit_coords: np.ndarray, query_coords: np.ndarray,
            backend: str = "rbc", batch_rows: int = 5_000_000,
            index=None) -> tuple[np.ndarray, np.ndarray]:
    k = 2 * tables.min_samples
    distances, indices = knn_query(fit_coords, query_coords, k, backend=backend,
                                   batch_rows=batch_rows, index=index)
    return predict_from_neighbours(tables, distances, indices)


# ── CLI ────────────────────────────────────────────────────────────────────────

def load_model(path: Path):
    import joblib
    return joblib.load(path)


def do_inspect(model) -> None:
    """Report which internals this build exposes. Run before anything long."""
    print(f"model type: {type(model)}")
    interesting = ["labels_", "probabilities_", "condensed_tree_", "prediction_data_",
                   "core_distances_", "core_distances", "min_samples", "min_cluster_size",
                   "single_linkage_tree_", "minimum_spanning_tree_"]
    for name in interesting:
        value = getattr(model, name, None)
        if value is None:
            print(f"  {name:26} MISSING")
            continue
        shape = getattr(value, "shape", None)
        print(f"  {name:26} {type(value).__name__}"
              + (f" shape={shape}" if shape is not None else ""))

    tree = getattr(model, "condensed_tree_", None)
    if tree is not None:
        print(f"\n  condensed_tree_ attributes: "
              f"{[a for a in dir(tree) if not a.startswith('__')][:12]}")
    prediction_data = getattr(model, "prediction_data_", None)
    if prediction_data is not None:
        print(f"  prediction_data_ attributes: "
              f"{[a for a in dir(prediction_data) if not a.startswith('__')][:12]}")

    print("\nWhat this script needs: condensed_tree_ (raw), core distances, labels_.")
    print("If core distances are MISSING, pass --recompute-core-distances "
          "(they are just the min_samples-th NN distance inside the fit set).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true", help="report model internals, exit")
    mode.add_argument("--validate", action="store_true",
                      help="relabel the eval probe and compare against saved reference labels")
    mode.add_argument("--apply", action="store_true", help="label rows and write them out")

    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--coords", type=Path, help="coords.npy for the whole cohort")
    parser.add_argument("--fit-indices", type=Path, help="rows the model was fit on")
    parser.add_argument("--probe-indices", type=Path, help="rows to label (default: all held)")
    parser.add_argument("--reference", type=Path, help="approximate_predict labels to check")
    parser.add_argument("--reference-probabilities", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--backend", default="rbc", choices=["rbc", "brute", "sklearn"])
    parser.add_argument("--batch-rows", type=int, default=5_000_000)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--recompute-core-distances", action="store_true",
                        help="derive core distances from the fit set instead of the model")
    args = parser.parse_args()

    log(f"loading {args.model}")
    model = load_model(args.model)

    if args.inspect:
        do_inspect(model)
        return

    if not (args.coords and args.fit_indices):
        raise SystemExit("--coords and --fit-indices are required for --validate/--apply")

    coords = np.load(args.coords, mmap_mode="r")
    fit_idx = np.load(args.fit_indices)
    fit_coords = np.array(coords[fit_idx], dtype=np.float32)
    log(f"fit set {fit_coords.shape[0]:,} x {fit_coords.shape[1]}")

    tables = build_tables(model, n_fit=fit_coords.shape[0], min_samples=args.min_samples)
    log(f"tables built: min_samples={tables.min_samples}, "
        f"{int((tables.cluster_label >= 0).sum()):,} mapped clusters, "
        f"tree_root={tables.tree_root}")

    if args.recompute_core_distances:
        log("recomputing core distances inside the fit set ...")
        distances, _ = knn_query(fit_coords, fit_coords, tables.min_samples,
                                 backend=args.backend, batch_rows=args.batch_rows)
        tables.core_distances = distances[:, -1].astype(np.float64)

    if args.probe_indices:
        query_idx = np.load(args.probe_indices)
    else:
        mask = np.ones(coords.shape[0], dtype=bool)
        mask[fit_idx] = False
        query_idx = np.nonzero(mask)[0]
    query_coords = np.array(coords[query_idx], dtype=np.float32)
    log(f"labelling {query_coords.shape[0]:,} rows via backend={args.backend}")

    started = perf_counter()
    index = build_index(fit_coords, 2 * tables.min_samples, args.backend)
    build_seconds = perf_counter() - started
    log(f"  index built in {build_seconds:.1f}s")

    started = perf_counter()
    labels, probabilities = predict(tables, fit_coords, query_coords,
                                    backend=args.backend, batch_rows=args.batch_rows,
                                    index=index)
    elapsed = perf_counter() - started
    rate = elapsed / (query_coords.shape[0] / 1e6)
    log(f"  labelled in {elapsed:.1f}s = {rate:.2f} s/M "
        f"-> {rate * 157.5 / 60:.1f} min for the 157.5M cohort")

    if args.validate:
        if not args.reference:
            raise SystemExit("--validate needs --reference")
        reference = np.load(args.reference)
        if reference.shape != labels.shape:
            raise SystemExit(f"reference has {reference.shape} labels, got {labels.shape}")
        agree = labels == reference
        print(f"\nexact label agreement : {agree.mean() * 100:.4f}% "
              f"({int((~agree).sum()):,} of {labels.size:,} differ)")

        both = (labels >= 0) & (reference >= 0)
        print(f"  clustered in both     : {both.mean() * 100:.2f}%")
        print(f"  noise here only       : {float(((labels < 0) & (reference >= 0)).mean()) * 100:.4f}%")
        print(f"  noise reference only  : {float(((labels >= 0) & (reference < 0)).mean()) * 100:.4f}%")
        if (~agree).any():
            from cross_size_ari import adjusted_rand, contingency
            counts, rows, cols = contingency(labels, reference)
            print(f"  ARI vs reference      : {adjusted_rand(counts, rows, cols):.6f}")
            disagreeing = np.nonzero(~agree)[0][:10]
            print(f"  first disagreements   : "
                  f"{[(int(i), int(labels[i]), int(reference[i])) for i in disagreeing]}")
        if args.reference_probabilities:
            reference_probabilities = np.load(args.reference_probabilities)
            delta = np.abs(probabilities - reference_probabilities)
            print(f"  max |prob delta|      : {delta.max():.3e}  "
                  f"(mean {delta.mean():.3e})")
        summary = args.model.parent / f"{args.model.stem}_fastpredict_validation.json"
        summary.write_text(json.dumps({
            "backend": args.backend,
            "rows": int(labels.size),
            "exact_agreement": float(agree.mean()),
            "seconds": round(elapsed, 2),
            "seconds_per_million": round(rate, 3),
            "index_build_seconds": round(build_seconds, 2),
        }, indent=2))
        print(f"\nwrote {summary}")

    if args.apply:
        out = args.out or Path("fast_predict_labels.npy")
        np.save(out, labels)
        np.save(out.with_name(out.stem + "_probabilities.npy"), probabilities)
        np.save(out.with_name(out.stem + "_indices.npy"), query_idx)
        log(f"wrote {out} ({labels.size:,} labels, "
            f"{float((labels < 0).mean()) * 100:.2f}% noise)")


if __name__ == "__main__":
    main()
