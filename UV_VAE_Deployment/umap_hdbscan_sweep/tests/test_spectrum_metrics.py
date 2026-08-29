"""The arithmetic the parameter choice will rest on.

Two things here are load-bearing and easy to get silently wrong. Normalisation: without it
L1 and L2 rank clusters by SIZE, and since the sweep varies min_cluster_size that would make
the metric a restatement of the parameter. And the pairwise means: they are computed via a
gram matrix and a blocked loop rather than the obvious double loop, so they are checked
against the obvious double loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectrum_metrics import (  # noqa: E402
    load_spectra,
    normalise,
    pairwise_stats,
    per_cluster_table,
    summarise,
    top_channel_stats,
)

CHANNELS = [f"ch{i:02d}" for i in range(96)]


def write_matrix(path: Path, columns: dict[str, list[float]]) -> Path:
    pl.DataFrame({"Type": CHANNELS, **columns}).write_csv(path, separator="\t")
    return path


def one_hot(channel: int, scale: float = 1.0) -> list[float]:
    column = [0.0] * 96
    column[channel] = float(scale)
    return column


def spread(channels: list[int], scale: float = 1.0) -> list[float]:
    column = [0.0] * 96
    for channel in channels:
        column[channel] = float(scale) / len(channels)
    return column


# ── normalisation ──────────────────────────────────────────────────────────────

def test_normalise_makes_columns_sum_to_one_and_drops_empties():
    matrix = np.array([[5.0, 0.0, 1.0], [15.0, 0.0, 3.0]])
    spectra, names = normalise(matrix, ["a", "b", "c"])
    assert names == ["a", "c"]
    np.testing.assert_allclose(spectra.sum(axis=0), [1.0, 1.0])


def test_pairwise_ignores_cluster_size():
    """The same two shapes at 1,000 and at 400,000 mutations must score identically.

    min_cluster_size is the swept parameter and it moves cluster size directly, so a distance
    that responded to size would report the parameter back rather than the geometry.
    """
    small = np.array([spread([0, 1]), spread([2, 3])], dtype=np.float64).T
    large = small * np.array([1_000.0, 400_000.0])

    normalised_small, _ = normalise(small, ["a", "b"])
    normalised_large, _ = normalise(large, ["a", "b"])
    for key, value in pairwise_stats(normalised_small).items():
        other = pairwise_stats(normalised_large)[key]
        if isinstance(value, float) and not np.isnan(value):
            np.testing.assert_allclose(value, other, rtol=1e-12)


# ── concentration ──────────────────────────────────────────────────────────────

def test_top_channel_share_of_a_one_hot_cluster_is_one():
    spectra = np.array([one_hot(7)], dtype=np.float64).T
    statistics = top_channel_stats(spectra, CHANNELS)
    assert statistics["top_channel_share_median"] == pytest.approx(1.0)
    assert statistics["most_common_dominant"][0] == ("ch07", 1)


def test_top_channel_share_of_a_flat_cluster_is_one_over_96():
    spectra = np.array([[1 / 96] * 96], dtype=np.float64).T
    assert top_channel_stats(spectra, CHANNELS)["top_channel_share_median"] == pytest.approx(1 / 96)


def test_fraction_above_thresholds_counts_clusters_not_mutations():
    """Three concentrated clusters and one spread one: 0.75 above both thresholds,
    regardless of the fact that the spread cluster could hold most of the mutations."""
    columns = [one_hot(0), one_hot(1), one_hot(2), spread(list(range(96)))]
    spectra = np.array(columns, dtype=np.float64).T
    statistics = top_channel_stats(spectra, CHANNELS)
    assert statistics["frac_above_05"] == pytest.approx(0.75)
    assert statistics["frac_above_08"] == pytest.approx(0.75)


def test_clusters_per_dominant_channel_detects_within_context_subdivision():
    """Six clusters over two dominant channels is 3.0 -- the signal that the partition is
    splitting within a context rather than separating processes."""
    columns = [spread([0, 1], 1), spread([0, 2], 1), spread([0, 3], 1),
               spread([5, 6], 1), spread([5, 7], 1), spread([5, 8], 1)]
    spectra = np.array(columns, dtype=np.float64).T
    statistics = top_channel_stats(spectra, CHANNELS)
    assert statistics["distinct_dominant_channels"] == 2
    assert statistics["clusters_per_dominant_channel"] == pytest.approx(3.0)


# ── pairwise ───────────────────────────────────────────────────────────────────

def test_identical_spectra_are_cosine_one_and_distance_zero():
    spectra = np.array([spread([0, 1]), spread([0, 1]), spread([0, 1])], dtype=np.float64).T
    statistics = pairwise_stats(spectra)
    assert statistics["cosine_mean"] == pytest.approx(1.0)
    assert statistics["l1_mean"] == pytest.approx(0.0, abs=1e-12)
    assert statistics["l2_mean"] == pytest.approx(0.0, abs=1e-12)


def test_disjoint_spectra_hit_the_l1_ceiling():
    """L1 between two distributions is twice their total variation distance, so disjoint
    support is exactly 2.0 -- the value that says these clusters share no mutation type."""
    spectra = np.array([one_hot(0), one_hot(1)], dtype=np.float64).T
    statistics = pairwise_stats(spectra)
    assert statistics["cosine_mean"] == pytest.approx(0.0)
    assert statistics["l1_mean"] == pytest.approx(2.0)
    assert statistics["l2_mean"] == pytest.approx(np.sqrt(2.0))


@pytest.mark.parametrize("n_clusters", [2, 5, 17, 140])
def test_pairwise_means_match_the_obvious_double_loop(n_clusters):
    """cosine and L2 go through a gram matrix and L1 through a blocked loop; all three must
    equal the naive pairwise computation, including across a block boundary (block=128)."""
    rng = np.random.default_rng(3)
    raw = rng.random((96, n_clusters)) ** 3
    spectra, _ = normalise(raw, [f"c{i}" for i in range(n_clusters)])

    points = spectra.T
    cosines, l1s, l2s = [], [], []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            a, b = points[i], points[j]
            cosines.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            l1s.append(np.abs(a - b).sum())
            l2s.append(np.linalg.norm(a - b))

    statistics = pairwise_stats(spectra)
    assert statistics["cosine_mean"] == pytest.approx(float(np.mean(cosines)))
    assert statistics["l1_mean"] == pytest.approx(float(np.mean(l1s)))
    assert statistics["l2_mean"] == pytest.approx(float(np.mean(l2s)))


def test_subsampling_is_deterministic_and_flagged():
    rng = np.random.default_rng(5)
    spectra, _ = normalise(rng.random((96, 300)), [f"c{i}" for i in range(300)])

    first = pairwise_stats(spectra, max_clusters=50, seed=11)
    second = pairwise_stats(spectra, max_clusters=50, seed=11)
    assert first["pairwise_subsampled"] is True and first["pairwise_n"] == 50
    assert first["cosine_mean"] == second["cosine_mean"]
    assert pairwise_stats(spectra, max_clusters=1000)["pairwise_subsampled"] is False


def test_single_cluster_gives_nan_rather_than_a_fake_zero():
    """One cluster has no pairs. Reporting 0.0 would read as 'perfectly distinct'."""
    statistics = pairwise_stats(np.array([one_hot(4)], dtype=np.float64).T)
    assert np.isnan(statistics["cosine_mean"]) and np.isnan(statistics["l1_mean"])


# ── end to end ─────────────────────────────────────────────────────────────────

def test_summarise_separates_a_context_clustering_from_a_signature_clustering(tmp_path):
    """The discriminating case, built to mimic what was measured: one cell whose clusters are
    single-channel (like the UMAP runs at 0.87) and one whose clusters are broad mixtures."""
    context_like = write_matrix(tmp_path / "context.tsv", {
        f"cluster_{i}": one_hot(i % 8, scale=10_000) for i in range(24)})
    signature_like = write_matrix(tmp_path / "signature.tsv", {
        f"cluster_{i}": spread(list(range(i, i + 40)), scale=10_000) for i in range(24)})

    context = summarise(context_like)
    signature = summarise(signature_like)

    assert context["top_channel_share_median"] == pytest.approx(1.0)
    assert context["frac_above_08"] == pytest.approx(1.0)
    assert signature["top_channel_share_median"] < 0.1
    assert signature["frac_above_05"] == 0.0
    # Duplicated one-hots: many clusters, few distinct spectra.
    assert context["clusters_per_dominant_channel"] == pytest.approx(3.0)
    assert signature["cosine_mean"] > context["cosine_mean"]


def test_per_cluster_table_reports_channels_needed_for_90_percent(tmp_path):
    path = write_matrix(tmp_path / "m.tsv", {
        "cluster_0": one_hot(3, 1000.0),
        "cluster_1": spread(list(range(10)), 1000.0),
    })
    table = per_cluster_table(path).sort("cluster")
    assert table["top_channel"][0] == "ch03"
    assert table["channels_for_90pct"][0] == 1
    assert table["channels_for_90pct"][1] == 9   # 9 of 10 equal channels reach 90%


def test_load_spectra_keeps_channel_and_cluster_order(tmp_path):
    path = write_matrix(tmp_path / "m.tsv",
                        {"cluster_5": one_hot(1), "cluster_2": one_hot(2)})
    channels, names, matrix = load_spectra(path)
    assert channels == CHANNELS
    assert names == ["cluster_5", "cluster_2"]
    assert matrix.shape == (96, 2)
