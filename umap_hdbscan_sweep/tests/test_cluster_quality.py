"""Geometry metrics: the properties that make each one readable, and the traps.

Each metric here has a way of looking good for the wrong reason -- connectivity rises when
clusters merge, probability is renormalised per cluster, DBCV refuses large cluster counts.
Those behaviours are pinned deliberately so nobody reads a number without its caveat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster_quality import (  # noqa: E402
    connectivity,
    dbcv,
    persistence_stats,
    probability_stats,
    summarise,
)


def blobs(per_cluster=300, spread=0.25, seed=0):
    rng = np.random.default_rng(seed)
    centres = [(0, 0), (6, 6), (-6, 5), (7, -4)]
    space = np.vstack([rng.normal(c, spread, size=(per_cluster, 2)) for c in centres])
    labels = np.repeat(np.arange(len(centres)), per_cluster).astype(np.int32)
    return space.astype(np.float64), labels


# ── probabilities ──────────────────────────────────────────────────────────────

def test_probability_stats_ignore_noise_rows():
    """Noise carries probability 0 by definition, so folding it in would make this a
    restatement of the noise fraction rather than a statement about membership strength."""
    probabilities = np.array([0.9, 0.8, 0.0, 0.0])
    labels = np.array([0, 1, -1, -1], dtype=np.int32)

    stats = probability_stats(probabilities, labels)
    assert stats["prob_n_assigned"] == 2
    assert stats["prob_mean"] == pytest.approx(0.85)
    assert stats["prob_frac_above_05"] == pytest.approx(1.0)


def test_probability_thresholds_are_reported_alongside_the_mean():
    """The mean hides shape: half at 1.0 and half at 0.2 averages to a respectable 0.6 while
    40% of assigned points sit below 0.5."""
    probabilities = np.concatenate([np.full(50, 1.0), np.full(50, 0.2)])
    labels = np.zeros(100, dtype=np.int32)

    stats = probability_stats(probabilities, labels)
    assert stats["prob_mean"] == pytest.approx(0.6)
    assert stats["prob_frac_above_05"] == pytest.approx(0.5)
    assert stats["prob_median"] == pytest.approx(0.6)


def test_probability_stats_on_an_all_noise_labelling():
    stats = probability_stats(np.zeros(10), np.full(10, -1, dtype=np.int32))
    assert stats == {"prob_n_assigned": 0}


# ── persistence ────────────────────────────────────────────────────────────────

def test_persistence_stats_read_the_models_own_array():
    class Model:
        cluster_persistence_ = np.array([0.1, 0.5, 0.9])

    stats = persistence_stats(Model())
    assert stats["persistence_mean"] == pytest.approx(0.5)
    assert stats["persistence_sum"] == pytest.approx(1.5)
    assert stats["persistence_min"] == pytest.approx(0.1)


def test_persistence_absent_when_the_backend_does_not_expose_it():
    assert persistence_stats(object()) == {}


def test_persistence_drops_non_finite_entries():
    class Model:
        cluster_persistence_ = np.array([0.4, np.inf, np.nan, 0.6])

    assert persistence_stats(Model())["persistence_mean"] == pytest.approx(0.5)


# ── connectivity ───────────────────────────────────────────────────────────────

def test_connectivity_is_one_for_well_separated_clusters():
    space, labels = blobs(spread=0.15)
    stats = connectivity(space, labels, n_neighbours=5)
    assert stats["connectivity_mean"] == pytest.approx(1.0)
    assert stats["connectivity_frac_pure"] == pytest.approx(1.0)


def test_connectivity_excludes_the_query_point_itself():
    """Including the self-match would add a constant 1/(k+1) and make a random labelling
    look better than chance."""
    rng = np.random.default_rng(1)
    space = rng.normal(size=(600, 2))
    labels = rng.integers(0, 3, size=600).astype(np.int32)   # no structure at all

    stats = connectivity(space, labels, n_neighbours=9)
    # Three random labels -> ~1/3 agreement. With the self-match it would be ~0.40.
    assert 0.25 < stats["connectivity_mean"] < 0.42


def test_connectivity_is_monotone_in_cluster_count():
    """THE caveat: merging every cluster into one makes connectivity perfect. This is why it
    must never be read without n_clusters beside it."""
    space, labels = blobs()
    merged = np.zeros_like(labels)

    split = connectivity(space, labels, n_neighbours=5)["connectivity_mean"]
    whole = connectivity(space, merged, n_neighbours=5)["connectivity_mean"]
    assert whole == pytest.approx(1.0)
    assert whole >= split


def test_connectivity_subsample_is_deterministic_and_flagged():
    space, labels = blobs(per_cluster=500)
    first = connectivity(space, labels, n_neighbours=5, max_rows=300, seed=7)
    second = connectivity(space, labels, n_neighbours=5, max_rows=300, seed=7)
    assert first["connectivity_n"] == 300
    assert first["connectivity_mean"] == second["connectivity_mean"]


def test_connectivity_handles_a_labelling_with_too_few_points():
    stats = connectivity(np.zeros((3, 2)), np.full(3, -1, dtype=np.int32), n_neighbours=10)
    assert stats == {"connectivity_n": 0}


# ── DBCV ───────────────────────────────────────────────────────────────────────

def test_dbcv_scores_separated_blobs_positively():
    pytest.importorskip("hdbscan")
    space, labels = blobs(per_cluster=200, spread=0.2)
    result = dbcv(space, labels, max_rows=800)
    assert result["dbcv"] is not None
    assert result["dbcv"] > 0.5, result


def test_dbcv_refuses_above_the_cluster_ceiling():
    """validity_index divides by per-cluster internal distances and raises outright when
    clusters have too few sampled points -- measured at 2,000 clusters over 12,500 rows. The
    ceiling must return a reason, not a number and not an exception."""
    space = np.random.default_rng(0).normal(size=(1000, 2))
    labels = np.arange(1000, dtype=np.int32)
    result = dbcv(space, labels, max_clusters=500)
    assert result["dbcv"] is None
    assert "exceeds max_clusters" in result["dbcv_note"]


def test_dbcv_sees_absorbed_scatter_that_connectivity_misses():
    """The reason DBCV earns its cost despite being a subsample score.

    Measured: adding 4% uniform scatter that the clustering then absorbs drops DBCV from
    +0.905 to +0.153 -- six-fold -- while connectivity moves 1.000 -> 0.999 and probability
    and persistence do not react at all. Connectivity only ever asks whether a point's
    neighbours agree, and a scattered point absorbed into a big cluster has neighbours that
    agree. So these two metrics are not redundant, and reporting connectivity alone would
    call a contaminated clustering clean.
    """
    rng = np.random.default_rng(0)
    clean = np.vstack([rng.normal(c, s, size=(n, 2)) for c, s, n in
                       [((0, 0), 0.4, 800), ((8, 8), 0.6, 800), ((-7, 6), 0.3, 600)]])
    contaminated = np.vstack([clean, rng.uniform(-12, 14, size=(100, 2))])

    clean_labels = np.repeat([0, 1, 2], [800, 800, 600]).astype(np.int32)
    # Every scattered point absorbed into its nearest cluster -- what HDBSCAN does here.
    centres = np.array([(0, 0), (8, 8), (-7, 6)], dtype=np.float64)
    scatter_labels = np.argmin(
        ((contaminated[clean.shape[0]:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
    dirty_labels = np.concatenate([clean_labels, scatter_labels]).astype(np.int32)

    clean_dbcv = dbcv(clean, clean_labels, max_rows=2200)["dbcv"]
    dirty_dbcv = dbcv(contaminated, dirty_labels, max_rows=2300)["dbcv"]
    clean_connect = connectivity(clean, clean_labels, n_neighbours=10)["connectivity_mean"]
    dirty_connect = connectivity(contaminated, dirty_labels,
                                 n_neighbours=10)["connectivity_mean"]

    assert dirty_dbcv < clean_dbcv / 2, (clean_dbcv, dirty_dbcv)
    assert dirty_connect > clean_connect - 0.05, (clean_connect, dirty_connect)


def test_dbcv_failure_is_reported_not_raised():
    """A DBCV failure must not cost a cell that took hours to fit."""
    space = np.zeros((50, 2))            # every point identical -> degenerate distances
    labels = np.zeros(50, dtype=np.int32)
    result = dbcv(space, labels, max_rows=50)
    assert result["dbcv"] is None and "dbcv_note" in result


# ── together ───────────────────────────────────────────────────────────────────

def test_summarise_returns_numbers_without_ranking_them():
    """No score, no verdict, no combined index -- just the measurements."""
    pytest.importorskip("hdbscan")
    space, labels = blobs(per_cluster=150)
    probabilities = np.full(labels.size, 0.9)

    class Model:
        cluster_persistence_ = np.array([0.3, 0.4, 0.5, 0.6])

    record = summarise(space, labels, probabilities=probabilities, clusterer=Model(),
                       connectivity_rows=400, dbcv_rows=400)

    assert record["n_clusters"] == 4
    assert record["noise_fraction"] == 0.0
    for key in ["prob_mean", "persistence_mean", "connectivity_mean", "dbcv"]:
        assert key in record
    # Nothing that implies an ordering.
    assert not any(k in record for k in ("score", "rank", "best", "eligible", "verdict"))
