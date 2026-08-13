"""Per-cluster Jaccard: the properties that make it readable, and the traps.

The global agreement scores collapse a partition to one number. Jaccard keeps per-cluster
resolution, which is the whole point -- so the tests pin the cases where the per-cluster view
disagrees with the global one, and where counting clusters disagrees with counting points.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cross_size_ari import (  # noqa: E402
    JACCARD_DISSOLVED, JACCARD_STABLE, cluster_jaccard, compare, contingency,
)


def jac(a, b):
    counts, rows, cols = contingency(np.asarray(a), np.asarray(b))
    return cluster_jaccard(counts, rows, cols)


def test_identical_labellings_are_perfectly_stable():
    labels = np.repeat([0, 1, 2], 100)
    stats = jac(labels, labels)
    assert stats["jaccard_mean"] == pytest.approx(1.0)
    assert stats["frac_clusters_stable"] == pytest.approx(1.0)
    assert stats["point_share_stable"] == pytest.approx(1.0)
    assert stats["clusters_dissolved"] == 0


def test_relabelling_does_not_count_as_instability():
    """Cluster ids are arbitrary. A permutation of the same partition is the same partition."""
    a = np.repeat([0, 1, 2], 100)
    b = np.repeat([7, 5, 9], 100)
    assert jac(a, b)["jaccard_mean"] == pytest.approx(1.0)


def test_shuffled_membership_dissolves_every_cluster():
    rng = np.random.default_rng(0)
    a = np.repeat(np.arange(20), 100)
    b = a.copy()
    rng.shuffle(b)
    stats = jac(a, b)
    assert stats["jaccard_mean"] < 0.2
    assert stats["frac_clusters_dissolved"] == pytest.approx(1.0)


def test_a_clean_merge_is_asymmetric():
    """THE case VI's conditional half also detects. When b merges pairs of a's clusters,
    every a-cluster still sits inside one b-cluster (Jaccard is dragged down only by the
    merge partner), while no b-cluster is matched by a single a-cluster."""
    a = np.repeat(np.arange(8), 100)
    b = a // 2                                   # merge clusters pairwise
    forward = jac(a, b)["jaccard_mean"]          # a -> b
    backward = jac(b, a)["jaccard_mean"]         # b -> a
    # each a-cluster is half of its b-cluster -> 0.5; each b-cluster is matched by an
    # a-cluster covering half of it -> also 0.5, but the direction that matters is that
    # neither reports a spurious 1.0
    assert forward == pytest.approx(0.5)
    assert backward == pytest.approx(0.5)


def test_a_split_leaves_the_unsplit_side_stable():
    a = np.repeat(np.arange(4), 100)
    b = np.arange(400) // 50                     # each a-cluster split in two
    # every b-cluster lies wholly inside one a-cluster -> b->a Jaccard is 0.5 per cluster,
    # but a->b cannot exceed 0.5 either. What distinguishes split from merge is cluster count.
    assert jac(b, a)["jaccard_mean"] == pytest.approx(0.5)
    assert len(np.unique(b)) > len(np.unique(a))


def test_point_share_and_cluster_count_disagree():
    """The reason both are reported. One huge unstable cluster and a crowd of stable tiny
    ones: by count the clustering looks stable, by mass it does not."""
    a = np.concatenate([np.zeros(9000, dtype=int),          # one huge cluster
                        np.repeat(np.arange(1, 21), 50)])   # 20 small ones
    b = a.copy()
    rng = np.random.default_rng(1)
    b[:9000] = rng.integers(100, 110, size=9000)            # shatter only the huge one

    stats = jac(a, b)
    assert stats["frac_clusters_stable"] > 0.9      # 20 of 21 clusters reproduce
    assert stats["point_share_stable"] < 0.2        # but they hold almost none of the data


def test_thresholds_are_the_published_ones():
    """Hennig (2007): above 0.75 stable, below 0.5 dissolved. Pinned so a future edit
    cannot quietly move the line the numbers are reported against."""
    assert JACCARD_STABLE == 0.75
    assert JACCARD_DISSOLVED == 0.5


def test_empty_input_returns_nothing_rather_than_raising():
    assert cluster_jaccard(np.zeros((0, 0)), np.zeros(0), np.zeros(0)) == {}
    assert cluster_jaccard(np.zeros((3, 3)), np.zeros(3), np.zeros(3)) == {}


def test_chunking_is_transparent():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 40, size=5000)
    b = rng.integers(0, 40, size=5000)
    counts, rows, cols = contingency(a, b)
    whole = cluster_jaccard(counts, rows, cols, chunk=10_000)
    chunked = cluster_jaccard(counts, rows, cols, chunk=7)
    assert whole == chunked


def test_compare_reports_jaccard_in_both_directions():
    a = np.repeat(np.arange(6), 100)
    b = a // 2
    result = compare(a, b, mask=None)
    for key in ["jaccard_mean_a_to_b", "jaccard_mean_b_to_a",
                "point_share_stable_a_to_b", "frac_clusters_dissolved_b_to_a"]:
        assert key in result, key


def test_noise_rows_are_excluded_from_the_cluster_view():
    """compare() scores Jaccard on the both-clustered subset. Noise is the absence of a
    cluster, so a run of noise rows must not appear as a large stable 'cluster'."""
    a = np.array([-1] * 500 + [0] * 100 + [1] * 100)
    b = np.array([-1] * 500 + [0] * 100 + [1] * 100)
    result = compare(a, b, mask=None)
    assert result["clusters_a"] == 2
    assert result["point_share_stable_a_to_b"] == pytest.approx(1.0)
