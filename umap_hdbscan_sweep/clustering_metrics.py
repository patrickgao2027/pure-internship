"""Unsupervised clustering-quality metrics for the HPC clustering testing regime.

All metrics here work WITHOUT ground-truth labels (the data is unlabeled):

* Internal validity indices score a clustering using only the feature coordinates
  and the cluster ids HDBSCAN produced: silhouette, Davies-Bouldin,
  Calinski-Harabasz, mean within-cluster sum of squares, entropy, noise fraction.
* Agreement indices (ARI / AMI / NMI) compare TWO predicted clusterings of the same
  points (subsample-vs-full, or variant-vs-variant). This is a stability/agreement
  use and needs no ground truth -- only two aligned label vectors.

Relative validity (DBCV) is read off the fitted HDBSCAN clusterer in
``clustering_core`` (it needs the min-spanning-tree), not here.

scikit-learn is imported lazily inside the functions that need it so the pure-numpy
helpers (entropy, noise fraction, WCSS) remain importable without sklearn present.
"""
from __future__ import annotations

import math

import numpy as np

NOISE_LABEL = -1


def noise_fraction(labels: np.ndarray) -> float:
    """Fraction of points assigned to the HDBSCAN noise cluster (label -1)."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return float("nan")
    return float(np.mean(labels == NOISE_LABEL))


def n_clusters(labels: np.ndarray) -> int:
    """Number of non-noise clusters."""
    labels = np.asarray(labels)
    return int(np.unique(labels[labels != NOISE_LABEL]).size)


def cluster_size_entropy(labels: np.ndarray, exclude_noise: bool = True) -> tuple[float, float]:
    """Shannon entropy (nats) of the cluster-size distribution, and its 0-1 normalized form.

    Normalized = H / log(k); 1.0 means perfectly balanced clusters, ~0 means one
    cluster dominates. Returns (nan, nan) when there are no clusters.
    """
    labels = np.asarray(labels)
    if exclude_noise:
        labels = labels[labels != NOISE_LABEL]
    if labels.size == 0:
        return float("nan"), float("nan")
    _, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    entropy = float(-np.sum(proportions * np.log(proportions)))
    k = int(proportions.size)
    normalized = float(entropy / math.log(k)) if k > 1 else 0.0
    return entropy, normalized


def mean_wcss(X: np.ndarray, labels: np.ndarray, exclude_noise: bool = True) -> float:
    """Mean within-cluster sum of squares PER POINT (inertia / n_points).

    Dividing by the number of clustered points makes this comparable across
    subsample sizes. HDBSCAN is not centroid-based, so this is descriptive, not the
    quantity it optimizes.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    mask = labels != NOISE_LABEL if exclude_noise else np.ones(labels.shape, dtype=bool)
    if not mask.any():
        return float("nan")
    Xm = X[mask]
    lm = labels[mask]
    total = 0.0
    for cluster in np.unique(lm):
        points = Xm[lm == cluster]
        centroid = points.mean(axis=0)
        total += float(((points - centroid) ** 2).sum())
    return float(total / Xm.shape[0])


def _subsample_indices(n: int, k: int, seed: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=k, replace=False)


def internal_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    eval_rows: int = 50_000,
    seed: int = 42,
) -> dict:
    """Compute the internal-validity panel on the clustering space + predicted labels.

    silhouette / Davies-Bouldin / Calinski-Harabasz are O(N^2)-ish, so they are
    evaluated on a seeded subsample (``eval_rows``) of the non-noise points. This is a
    read-only scoring of the labels the all-N model already produced -- it does not
    reduce what the clustering model saw. Undefined indices (fewer than 2 clusters)
    come back as NaN with an ``internal_reason`` string rather than raising.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)

    entropy, entropy_norm = cluster_size_entropy(labels)
    result: dict = {
        "n_points": int(labels.size),
        "n_clusters": n_clusters(labels),
        "noise_fraction": noise_fraction(labels),
        "entropy": entropy,
        "entropy_normalized": entropy_norm,
        "mean_wcss": mean_wcss(X, labels),
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
        "internal_eval_points": 0,
        "internal_reason": None,
    }

    mask = labels != NOISE_LABEL
    n_nonnoise = int(mask.sum())
    if result["n_clusters"] < 2 or n_nonnoise < 3:
        result["internal_reason"] = (
            f"need >=2 clusters and >=3 non-noise points "
            f"(got k={result['n_clusters']}, n={n_nonnoise})"
        )
        return result

    X_nonnoise = X[mask]
    labels_nonnoise = labels[mask]
    idx = _subsample_indices(X_nonnoise.shape[0], eval_rows, seed)
    X_eval = X_nonnoise[idx]
    labels_eval = labels_nonnoise[idx]

    if np.unique(labels_eval).size < 2:
        result["internal_eval_points"] = int(X_eval.shape[0])
        result["internal_reason"] = "evaluation subsample collapsed to <2 clusters"
        return result

    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    result["silhouette"] = float(silhouette_score(X_eval, labels_eval))
    result["davies_bouldin"] = float(davies_bouldin_score(X_eval, labels_eval))
    result["calinski_harabasz"] = float(calinski_harabasz_score(X_eval, labels_eval))
    result["internal_eval_points"] = int(X_eval.shape[0])
    return result


def clustering_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """ARI / AMI / NMI between two aligned predicted labelings of the SAME points.

    Both label vectors must have the same length and index the same rows (align them
    on variant identity before calling). Noise (-1) is treated as its own category, so
    agreement also credits both runs calling the same points noise.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError(f"label arrays must be aligned and equal length: {a.shape} vs {b.shape}")
    if a.size == 0:
        return {"ari": float("nan"), "ami": float("nan"), "nmi": float("nan"), "n": 0}

    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
    )

    return {
        "ari": float(adjusted_rand_score(a, b)),
        "ami": float(adjusted_mutual_info_score(a, b)),
        "nmi": float(normalized_mutual_info_score(a, b)),
        "n": int(a.size),
    }
