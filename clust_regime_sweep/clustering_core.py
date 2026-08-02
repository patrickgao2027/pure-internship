"""Clustering primitives for the HPC clustering testing regime (CPU-only).

Everything here fits on ALL rows (no fit-sample / approximate_predict). HDBSCAN is
run as a small ``min_cluster_size`` grid at fixed ``min_samples`` and the best cut is
selected by DBCV (``relative_validity_``). A shared joblib ``memory`` cache means the
expensive mutual-reachability / MST computation is done ONCE per call and reused
across the grid (only the cheap tree-condensing step re-runs per ``min_cluster_size``).

Heavy optional deps (umap, sklearn.decomposition) are imported lazily; ``hdbscan`` is
imported at module top because it is the core dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import hdbscan
from joblib import Memory as JoblibMemory


def _noise_fraction(labels: np.ndarray) -> float:
    return float(np.mean(labels == -1)) if labels.size else float("nan")


def _n_clusters(labels: np.ndarray) -> int:
    return int(np.unique(labels[labels != -1]).size)


@dataclass
class ClusteringResult:
    """Selected clustering plus the full grid it was chosen from."""

    labels: np.ndarray
    probabilities: np.ndarray
    clusterer: object
    min_cluster_size: int
    min_samples: int
    dbcv: float | None
    grid: list = field(default_factory=list)


def hdbscan_grid(
    X: np.ndarray,
    min_cluster_sizes: list[int],
    min_samples: int = 25,
    metric: str = "euclidean",
    core_dist_n_jobs: int = 1,
    gen_min_span_tree: bool = True,
    cache_dir: str | None = None,
) -> ClusteringResult:
    """Fit HDBSCAN over a ``min_cluster_size`` grid on all N rows; select by DBCV.

    ``min_samples`` is fixed so a shared ``memory`` cache lets the grid reuse one MST.
    Selection rule: maximize DBCV among cuts with >=2 clusters; tie-break on lower
    noise fraction. Every grid point is retained in ``result.grid`` so the
    metric-vs-min_cluster_size curve is available downstream.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    grid: list[dict] = []
    best: ClusteringResult | None = None
    best_key: tuple | None = None

    _memory = JoblibMemory(cache_dir, verbose=0) if cache_dir else JoblibMemory(None, verbose=0)
    for min_cluster_size in min_cluster_sizes:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            metric=metric,
            cluster_selection_method="eom",
            core_dist_n_jobs=int(core_dist_n_jobs),
            gen_min_span_tree=bool(gen_min_span_tree),
            memory=_memory,  # shared cache => MST/core-dist computed once across the grid
            prediction_data=False,
        )
        clusterer.fit(X)
        labels = clusterer.labels_.astype(np.int32, copy=False)
        try:
            dbcv = float(clusterer.relative_validity_)
        except Exception:
            dbcv = None

        cluster_count = _n_clusters(labels)
        noise = _noise_fraction(labels)
        grid.append(
            {
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
                "n_clusters": cluster_count,
                "noise_fraction": noise,
                "dbcv": dbcv,
            }
        )

        # Selection: prefer higher DBCV among cuts that formed >=2 clusters; if DBCV is
        # unavailable or the cut collapsed, its score is -inf so a valid cut always wins.
        score = dbcv if (dbcv is not None and cluster_count >= 2) else float("-inf")
        key = (score, -noise)
        if best_key is None or key > best_key:
            best_key = key
            best = ClusteringResult(
                labels=labels,
                probabilities=clusterer.probabilities_.astype(np.float32, copy=False),
                clusterer=clusterer,
                min_cluster_size=int(min_cluster_size),
                min_samples=int(min_samples),
                dbcv=dbcv,
            )

    if best is None:  # no grid values supplied
        raise ValueError("min_cluster_sizes was empty")
    best.grid = grid
    return best


def fit_umap(
    X: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    metric: str = "euclidean",
    seed: int = 42,
) -> np.ndarray:
    """Fit UMAP on all N rows and return 2D coordinates.

    ``random_state`` is set for reproducibility, which forces single-threaded UMAP --
    the main CPU cost at large N. Callers that need speed over exact reproducibility
    can pass ``seed=None`` (handled by UMAP as multi-threaded, non-deterministic).
    """
    from umap import UMAP

    X = np.asarray(X, dtype=np.float32)
    # low_memory=True switches pynndescent to a more memory-efficient kNN algorithm;
    # required when N > ~1M to avoid C-extension segfaults from OOM in the graph build.
    model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        low_memory=True,
    )
    return model.fit_transform(X).astype(np.float32, copy=False)


def one_hot(indices: np.ndarray, cardinality: int) -> np.ndarray:
    """One-hot encode integer category indices into a (N, cardinality) float32 matrix."""
    indices = np.asarray(indices, dtype=np.int64)
    encoded = np.zeros((indices.shape[0], int(cardinality)), dtype=np.float32)
    valid = (indices >= 0) & (indices < cardinality)
    encoded[np.arange(indices.shape[0])[valid], indices[valid]] = 1.0
    return encoded


def _to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "numpy"):
        return tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor.numpy()
    return np.asarray(tensor)


def build_preprocessed_matrix(cat_tensor, num_tensor, categorical_specs) -> np.ndarray:
    """Variant C feature space: standardized numeric columns + one-hot categoricals.

    ``cat_tensor`` (N, n_cat) holds integer category indices; ``num_tensor`` (N, n_num)
    holds the already-standardized numeric columns from ``preprocess.transform_frame``.
    """
    num = _to_numpy(num_tensor).astype(np.float32, copy=False)
    cat = _to_numpy(cat_tensor).astype(np.int64, copy=False)

    blocks: list[np.ndarray] = []
    if num.ndim == 2 and num.shape[1]:
        blocks.append(num)
    for column_index, spec in enumerate(categorical_specs):
        cardinality = int(getattr(spec, "cardinality", 0))
        column = cat[:, column_index]
        if cardinality <= 0:
            cardinality = int(column.max()) + 1 if column.size else 1
        blocks.append(one_hot(column, cardinality))

    if not blocks:
        raise ValueError("no numeric or categorical features to build the preprocessed matrix")
    return np.concatenate(blocks, axis=1)


def pca_reduce(X: np.ndarray, n_components: int = 16, seed: int = 42) -> np.ndarray:
    """Reduce a matrix to ``n_components`` via PCA (fit on all N)."""
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=np.float64)
    n_components = int(min(n_components, X.shape[1], max(1, X.shape[0] - 1)))
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(X)
    return reduced.astype(np.float32, copy=False)
