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
    MIN_PER_CLUSTER,
    _allocate,
    connectivity,
    dbcv,
    persistence_stats,
    probability_stats,
    relative_validity,
    stratified_sample,
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

    # per_cluster_rows=None for the fixed-budget path: detecting 4% absorbed scatter needs
    # enough points per cluster to sample the scatter at all, and the default 50 draws about
    # two of them. That is a real sensitivity limit of the per-cluster mode, not a test
    # artefact -- see test_per_cluster_resolution_bounds_contamination_sensitivity.
    clean_dbcv = dbcv(clean, clean_labels, max_rows=2200, per_cluster_rows=None)["dbcv"]
    dirty_dbcv = dbcv(contaminated, dirty_labels, max_rows=2300,
                      per_cluster_rows=None)["dbcv"]
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


# ── stratified sampling ────────────────────────────────────────────────────────

def test_allocation_spends_the_whole_budget():
    """Runtime is set by the TOTAL sampled points, so an allocation that under-spends pays
    the same order of cost for a worse estimate."""
    quota = _allocate(np.array([1000, 1000, 1000, 1000]), 400)
    assert quota.sum() == 400
    assert set(quota) == {100}


def test_allocation_redistributes_the_surplus_from_small_clusters():
    """Two clusters can only supply 3 points each; the other 394 must go somewhere, or a
    skewed clustering would silently sample far less than the budget allows."""
    quota = _allocate(np.array([3, 3, 1000, 1000]), 400)
    assert quota.tolist() == [3, 3, 197, 197]
    assert quota.sum() == 400


def test_allocation_never_asks_for_more_points_than_a_cluster_has():
    sizes = np.array([5, 12, 400, 9000])
    quota = _allocate(sizes, 2000)
    assert (quota <= sizes).all()


def test_allocation_degrades_gracefully_below_one_point_per_cluster():
    quota = _allocate(np.array([10] * 100), 50)
    assert quota.sum() == 50
    assert set(np.unique(quota)) <= {0, 1}


def test_stratified_sample_covers_every_cluster_where_uniform_does_not():
    """The whole reason this exists. A heavy size skew starves small clusters under uniform
    sampling until validity_index drops them; equal shares of the same budget do not."""
    rng = np.random.default_rng(0)
    sizes = np.concatenate([np.full(300, 200), np.full(10, 40_000)])
    labels = np.repeat(np.arange(sizes.size), sizes).astype(np.int32)

    strat = stratified_sample(labels, 12_000, seed=0)
    per_strat = np.bincount(labels[strat], minlength=sizes.size)
    assert (per_strat >= MIN_PER_CLUSTER).all()
    assert per_strat.size == sizes.size

    uniform = rng.choice(labels.size, size=12_000, replace=False)
    per_uniform = np.bincount(labels[uniform], minlength=sizes.size)
    assert (per_uniform < MIN_PER_CLUSTER).sum() > 0, "uniform should starve small clusters"


def test_stratified_sample_stays_within_budget_and_is_deterministic():
    labels = np.repeat(np.arange(50), 1000).astype(np.int32)
    first = stratified_sample(labels, 5_000, seed=7)
    second = stratified_sample(labels, 5_000, seed=7)
    assert first.size <= 5_000
    assert np.array_equal(first, second)


def test_stratified_sample_ignores_noise_rows():
    labels = np.array([-1] * 500 + [0] * 100 + [1] * 100, dtype=np.int32)
    sample = stratified_sample(labels, 100, seed=0)
    assert (labels[sample] >= 0).all()


def test_stratified_sample_on_an_all_noise_labelling():
    assert stratified_sample(np.full(50, -1, dtype=np.int32), 100).size == 0


def test_stratified_dbcv_agrees_with_uniform_where_both_work():
    """Stratifying changes which points are drawn, so the reweighted aggregate must still
    estimate the same quantity -- otherwise scores are not comparable to the uniform ones
    already recorded in the sweep."""
    pytest.importorskip("hdbscan")
    space, labels = blobs(per_cluster=400, spread=0.3)
    a = dbcv(space, labels, max_rows=1200, seed=0, stratified=False)["dbcv"]
    b = dbcv(space, labels, max_rows=1200, seed=0, stratified=True)["dbcv"]
    assert a is not None and b is not None
    assert abs(a - b) < 0.1, (a, b)


def test_stratified_dbcv_reports_the_per_cluster_distribution():
    """The resolution a single aggregate cannot give: which clusters are poor, and how much
    of the data sits in them."""
    pytest.importorskip("hdbscan")
    space, labels = blobs(per_cluster=300)
    record = dbcv(space, labels, max_rows=1000, seed=0, stratified=True)
    for key in ["dbcv_sample_weighted", "dbcv_cluster_median", "dbcv_cluster_min",
                "dbcv_frac_clusters_negative", "dbcv_point_share_negative",
                "dbcv_sampling", "dbcv_sampled_rows"]:
        assert key in record, key
    assert record["dbcv_sampling"].startswith("stratified")
    assert 0.0 <= record["dbcv_point_share_negative"] <= 1.0


def test_per_cluster_mode_gives_every_cluster_the_same_resolution():
    """THE property that makes DBCV comparable between models. A fixed TOTAL budget scores a
    20-cluster model at 50 points per cluster and a 200-cluster model at 5 -- different
    sampling resolution, different bias, scores that cannot be read against each other.
    Fixing points-per-cluster fixes the bias."""
    small = np.repeat(np.arange(20), 500).astype(np.int32)
    large = np.repeat(np.arange(200), 500).astype(np.int32)

    for labels in (small, large):
        drawn = np.bincount(labels[stratified_sample(labels, per_cluster=40, seed=0)])
        assert set(np.unique(drawn)) == {40}

    # the fixed-budget mode does NOT have this property -- pinned so the difference is not
    # quietly lost in a refactor
    a = np.bincount(small[stratified_sample(small, budget=1000, seed=0)]).max()
    b = np.bincount(large[stratified_sample(large, budget=1000, seed=0)]).max()
    assert a != b


def test_per_cluster_mode_never_over_draws_a_small_cluster():
    labels = np.repeat([0, 1, 2], [500, 12, 7]).astype(np.int32)
    drawn = np.bincount(labels[stratified_sample(labels, per_cluster=40, seed=0)])
    assert drawn.tolist() == [40, 12, 7]


def test_sampler_refuses_both_modes_at_once():
    """Silently preferring one would make the resolution -- and so the comparability -- depend
    on an argument the caller thought was ignored."""
    labels = np.repeat(np.arange(4), 100).astype(np.int32)
    with pytest.raises(ValueError):
        stratified_sample(labels, budget=100, per_cluster=10)
    with pytest.raises(ValueError):
        stratified_sample(labels)


def test_dbcv_records_the_resolution_it_used():
    """Comparability has to be auditable, not assumed: two cells' scores are only comparable
    if they were scored at the same points-per-cluster."""
    pytest.importorskip("hdbscan")
    space, labels = blobs(per_cluster=300)
    record = dbcv(space, labels, per_cluster_rows=40, seed=0)
    assert record["dbcv_points_per_cluster_median"] == 40
    assert record["dbcv_sampling"] == "stratified_40_per_cluster"


def test_dbcv_per_cluster_respects_the_minimum():
    """Asking for fewer points than score_cell's drop threshold would silently discard every
    cluster."""
    pytest.importorskip("hdbscan")
    space, labels = blobs(per_cluster=100)
    record = dbcv(space, labels, per_cluster_rows=1, seed=0)
    assert record["dbcv_points_per_cluster_median"] >= MIN_PER_CLUSTER


def test_relative_validity_is_extracted_when_the_fit_provides_it():
    pytest.importorskip("hdbscan")
    import hdbscan as _h

    space, _ = blobs(per_cluster=200, spread=0.3)
    fitted = _h.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True).fit(space)
    stats = relative_validity(fitted)
    assert "relative_validity" in stats
    assert -1.0 <= stats["relative_validity"] <= 1.0


def test_relative_validity_absent_without_the_min_span_tree():
    """gen_min_span_tree defaults to False, and then the attribute raises rather than
    existing -- so the helper must return nothing rather than propagate."""
    assert relative_validity(object()) == {}


def test_dbcv_has_no_cluster_ceiling_by_default():
    """The old default refused above 500 clusters, which excluded most of the real grid.
    Stratified sampling removes the reason for the ceiling."""
    pytest.importorskip("hdbscan")
    space, labels = blobs(per_cluster=200)
    record = dbcv(space, labels, max_rows=800, seed=0)
    assert "exceeds max_clusters" not in record.get("dbcv_note", "")
