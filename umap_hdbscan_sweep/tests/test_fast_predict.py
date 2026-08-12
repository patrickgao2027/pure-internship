"""fast_predict must reproduce hdbscan's approximate_predict exactly, not approximately.

The whole argument for the RBC swap is that only the neighbour search changes and steps 2-5
stay byte-identical. That is a testable claim, and this is where it gets tested -- on CPU,
against the reference implementation, with no GPU involved. If these pass, the only untested
surface left on the GPU box is whether cuML's RBC returns the same neighbours as a KDTree
(it is documented as exact) and whether the cuML model exposes the internals build_tables
needs (that is what --inspect is for).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

hdbscan = pytest.importorskip("hdbscan")

from fast_predict import build_tables, predict  # noqa: E402


def make_blobs(n=6000, seed=0):
    """2-D blobs of uneven density plus uniform noise -- the shape a UMAP embedding has."""
    rng = np.random.default_rng(seed)
    parts = []
    for centre, spread, count in [((0, 0), 0.35, n // 3), ((5, 5), 0.9, n // 3),
                                  ((-4, 6), 0.2, n // 6), ((8, -3), 1.4, n // 6)]:
        parts.append(rng.normal(centre, spread, size=(count, 2)))
    parts.append(rng.uniform(-8, 12, size=(n // 10, 2)))
    data = np.vstack(parts).astype(np.float64)
    rng.shuffle(data)
    return data


@pytest.mark.parametrize("min_samples,min_cluster_size", [(5, 50), (3, 25), (15, 100)])
def test_matches_reference_exactly(min_samples, min_cluster_size):
    rng = np.random.default_rng(7)
    fit = make_blobs(6000, seed=1)
    query = np.vstack([make_blobs(1500, seed=2), rng.uniform(-12, 16, size=(500, 2))])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                cluster_selection_method="eom", prediction_data=True)
    clusterer.fit(fit)

    reference_labels, reference_probabilities = hdbscan.approximate_predict(clusterer, query)

    tables = build_tables(clusterer, n_fit=fit.shape[0])
    labels, probabilities = predict(tables, fit, query, backend="sklearn")

    assert labels.shape == reference_labels.shape
    mismatched = int((labels != reference_labels).sum())
    assert mismatched == 0, (
        f"{mismatched}/{labels.size} labels differ at min_samples={min_samples}; "
        f"first: {[(int(i), int(labels[i]), int(reference_labels[i])) for i in np.nonzero(labels != reference_labels)[0][:5]]}"
    )
    np.testing.assert_allclose(probabilities, reference_probabilities, atol=1e-6)


def test_noise_points_are_reproduced():
    """Points far outside every cluster must come back -1 with probability 0, as the
    reference does -- an assigner that forces everything into a cluster is not equivalent."""
    fit = make_blobs(4000, seed=3)
    far = np.array([[500.0, 500.0], [-400.0, 250.0], [1e3, -1e3]])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5,
                                cluster_selection_method="eom", prediction_data=True)
    clusterer.fit(fit)
    reference_labels, _ = hdbscan.approximate_predict(clusterer, far)

    tables = build_tables(clusterer, n_fit=fit.shape[0])
    labels, probabilities = predict(tables, fit, far, backend="sklearn")

    assert (reference_labels == -1).all(), "reference should call these noise"
    np.testing.assert_array_equal(labels, reference_labels)
    assert (probabilities[labels < 0] == 0).all()


def test_batching_does_not_change_the_answer():
    """predict() batches the whole pipeline, not just the kNN, because the neighbour arrays
    (2*min_samples per row, distances + indices + the mutual-reachability copy) are what
    would need ~32 GB on the 132.5M-row cohort. Chunking must be transparent: a batch
    boundary falling mid-run cannot alter any label or probability."""
    fit = make_blobs(5000, seed=8)
    query = make_blobs(2000, seed=9)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5,
                                cluster_selection_method="eom", prediction_data=True)
    clusterer.fit(fit)
    tables = build_tables(clusterer, n_fit=fit.shape[0])

    whole = predict(tables, fit, query, backend="sklearn", batch_rows=10**9)
    for batch_rows in (1, 7, 333, 1999, 2000, 2001):
        chunked = predict(tables, fit, query, backend="sklearn", batch_rows=batch_rows)
        np.testing.assert_array_equal(chunked[0], whole[0])
        np.testing.assert_array_equal(chunked[1], whole[1])


def test_cluster_labels_recovered_without_prediction_data():
    """cuML models have no `prediction_data_`, so build_tables falls back to recovering the
    cluster->label map from `labels_`. That fallback must give the same answer as the map."""
    fit = make_blobs(5000, seed=4)
    query = make_blobs(1200, seed=5)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5,
                                cluster_selection_method="eom", prediction_data=True)
    clusterer.fit(fit)
    with_map = predict(build_tables(clusterer, fit.shape[0]), fit, query, backend="sklearn")[0]

    stripped = _StrippedModel(clusterer)
    without_map = predict(build_tables(stripped, fit.shape[0]), fit, query,
                          backend="sklearn")[0]

    disagreement = float((with_map != without_map).mean())
    assert disagreement == 0.0, f"fallback label map differs on {disagreement:.2%} of points"


def test_falls_back_when_model_max_lambdas_is_unusable():
    """A max_lambdas dict that exists but does not cover the mapped clusters leaves
    max_lambda at 0, and `prob = where(max_lambda > 0, ..., 1.0)` then reports every labelled
    point as fully confident. Labels stay correct, so it is invisible without checking
    probabilities -- exactly the failure seen against the 25M cuML model. build_tables must
    notice the coverage gap and reconstruct instead."""
    fit = make_blobs(5000, seed=10)
    query = make_blobs(1500, seed=11)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5,
                                cluster_selection_method="eom", prediction_data=True)
    clusterer.fit(fit)
    reference_labels, reference_probabilities = hdbscan.approximate_predict(clusterer, query)

    crippled = _CrippledMaxLambdas(clusterer)
    tables = build_tables(crippled, n_fit=fit.shape[0])
    assert "reconstructed" in tables.max_lambda_source, tables.max_lambda_source

    labels, probabilities = predict(tables, fit, query, backend="sklearn")
    np.testing.assert_array_equal(labels, reference_labels)
    np.testing.assert_allclose(probabilities, reference_probabilities, atol=1e-6)


class _CrippledMaxLambdas:
    """Real prediction data except max_lambdas, which covers almost nothing."""

    def __init__(self, clusterer):
        self.condensed_tree_ = clusterer.condensed_tree_
        self.labels_ = clusterer.labels_
        self.min_samples = clusterer.min_samples
        self.min_cluster_size = clusterer.min_cluster_size

        real = clusterer.prediction_data_
        one_key = next(iter(real.max_lambdas))
        self.prediction_data_ = type("_PD", (), {
            "core_distances": real.core_distances,
            "cluster_map": real.cluster_map,
            "max_lambdas": {one_key: real.max_lambdas[one_key]},
        })()


def test_falls_back_when_model_max_lambdas_are_sentinels():
    """The real miletus failure: a joblib-loaded cuML model regenerates prediction data with
    max_lambdas full of FLT_MAX (3.403e38). Every value is > 0, so a coverage check passes,
    and prob = lambda/3.4e38 ~ 0 everywhere while labels stay correct. The bound that catches
    it is that a max lambda is a lambda_val from the condensed tree and cannot exceed the
    largest one in it."""
    fit = make_blobs(5000, seed=12)
    query = make_blobs(1500, seed=13)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5,
                                cluster_selection_method="eom", prediction_data=True)
    clusterer.fit(fit)
    reference_labels, reference_probabilities = hdbscan.approximate_predict(clusterer, query)

    poisoned = _SentinelMaxLambdas(clusterer, sentinel=np.finfo(np.float32).max)
    tables = build_tables(poisoned, n_fit=fit.shape[0])
    assert "reconstructed" in tables.max_lambda_source, tables.max_lambda_source

    labels, probabilities = predict(tables, fit, query, backend="sklearn")
    np.testing.assert_array_equal(labels, reference_labels)
    np.testing.assert_allclose(probabilities, reference_probabilities, atol=1e-6)
    assert probabilities.mean() > 0.01, "sentinel max_lambda would crush probabilities to ~0"


def test_sentinel_rejected_even_when_the_tree_holds_an_infinite_lambda():
    """The regression from the 25M cuML model.

    The sentinel guard bounds max_lambda by the largest lambda_val in the condensed tree, on
    the reasoning that a max lambda IS one of those values. But lambda = 1/distance, and cuML
    computes it in float32, so coincident points -- a certainty in a 25M-row 2-D embedding --
    put `inf` in the tree. The bound becomes `values <= inf`, nothing fails it, and a model
    whose every mapped cluster was FLT_MAX sailed through: labels correct, probability mean
    0.0076 against cuML's 0.9160. CPU hdbscan clips those lambdas to finite, which is why no
    naturally-fitted model reproduces it -- the inf has to be injected to get the cuML shape.

    A non-finite lambda also poisons the reconstruction path, since own_max_lambda takes a
    per-cluster maximum: one inf sets a whole cluster's scale to inf and every probability in
    it to lambda/inf = 0. Both paths have to survive it, so this asserts on probabilities and
    not merely on which source was chosen.
    """
    fit = make_blobs(5000, seed=14)
    query = make_blobs(1500, seed=15)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5,
                                cluster_selection_method="eom", prediction_data=True)
    clusterer.fit(fit)
    reference_labels, reference_probabilities = hdbscan.approximate_predict(clusterer, query)

    poisoned = _SentinelMaxLambdas(clusterer, sentinel=np.finfo(np.float32).max,
                                   inject_infinite_lambda=True)
    assert not np.isfinite(poisoned.condensed_tree_._raw_tree["lambda_val"]).all()

    tables = build_tables(poisoned, n_fit=fit.shape[0])
    assert "reconstructed" in tables.max_lambda_source, (
        f"FLT_MAX slipped through when the tree held an inf: {tables.max_lambda_source}")
    assert np.isfinite(tables.max_lambda).all(), "an inf lambda leaked into the max_lambda table"

    labels, probabilities = predict(tables, fit, query, backend="sklearn")
    assert probabilities.mean() > 0.5, (
        f"probabilities crushed to {probabilities.mean():.4f} -- the miletus symptom")

    # The injected inf sits on one leaf row, so the labelling is unchanged for everything that
    # does not descend through it.
    agreement = float((labels == reference_labels).mean())
    assert agreement > 0.99, f"only {agreement:.4%} of labels survived the injected inf"


class _SentinelMaxLambdas:
    """Real prediction data except max_lambdas, which is every-key-is-FLT_MAX.

    With `inject_infinite_lambda` the condensed tree also gets one non-finite lambda_val, the
    way cuML's float32 `1/0` does on coincident points. CPU hdbscan never emits one, so it has
    to be planted to reproduce the model that actually defeated the guard.
    """

    def __init__(self, clusterer, sentinel, inject_infinite_lambda=False):
        self.condensed_tree_ = clusterer.condensed_tree_
        self.labels_ = clusterer.labels_
        self.min_samples = clusterer.min_samples
        self.min_cluster_size = clusterer.min_cluster_size

        if inject_infinite_lambda:
            raw = clusterer.condensed_tree_._raw_tree.copy()
            leaf = np.nonzero(raw["child_size"] == 1)[0][0]
            raw["lambda_val"][leaf] = np.inf
            self.condensed_tree_ = type("_CT", (), {"_raw_tree": raw})()

        real = clusterer.prediction_data_
        self.prediction_data_ = type("_PD", (), {
            "core_distances": real.core_distances,
            "cluster_map": real.cluster_map,
            "max_lambdas": {k: float(sentinel) for k in real.max_lambdas},
        })()


class _StrippedModel:
    """A fitted model with `prediction_data_` hidden, standing in for the cuML shape."""

    prediction_data_ = None

    def __init__(self, clusterer):
        self.condensed_tree_ = clusterer.condensed_tree_
        self.labels_ = clusterer.labels_
        self.min_samples = clusterer.min_samples
        self.min_cluster_size = clusterer.min_cluster_size
        self.core_distances = clusterer.prediction_data_.core_distances
