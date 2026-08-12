"""The parts of the sweep that must be right before 17 hours of GPU time goes into it.

Everything here runs on CPU with no cuML and no SigProfiler. What it pins down is the
arithmetic the ranking depends on: the SBS96 canonicalisation (a wrong strand convention
silently mixes C>T with G>A and every downstream cosine is garbage), the count matrix, and
the cluster-size statistics the decision rule thresholds on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdbscan_param_sweep import (  # noqa: E402
    MUTATION_CEILING,
    MUTATION_FLOOR,
    SBS7_SUBTYPES,
    UV_SIGNATURES,
    Cell,
    build_grid,
    cluster_size_stats,
    projected_fit_seconds,
    sbs96_counts,
    sbs96_expr,
    sigprofiler_metrics,
)


def sbs96_of(ref, alt, previous, following):
    frame = pl.DataFrame({"REF": [ref], "ALT": [alt],
                          "X_PREV1": [previous], "X_NEXT1": [following]})
    return frame.select(sbs96_expr().alias("sbs96"))["sbs96"][0]


def test_pyrimidine_reference_is_left_alone():
    assert sbs96_of("C", "T", "A", "G") == "A[C>T]G"
    assert sbs96_of("T", "A", "G", "C") == "G[T>A]C"


def test_purine_reference_is_reverse_complemented_with_flanks_swapped():
    """T[A>G]C read off the other strand is G[T>C]A: the substitution complements (A>G to
    T>C) AND the flanks swap sides as well as complementing.

    The swap is the part that is easy to get wrong. Without it this returns A[T>C]G -- a
    real channel, just the wrong one -- so the counts stay plausible while landing in the
    mirror-image context and every cosine downstream is quietly corrupted. Asymmetric flanks
    (T on the left, C on the right) are what make the two cases distinguishable.
    """
    assert sbs96_of("A", "G", "T", "C") == "G[T>C]A"
    assert sbs96_of("G", "T", "A", "A") == "T[C>A]T"


def test_non_snv_contexts_come_back_null():
    assert sbs96_of("C", "C", "A", "G") is None   # not a substitution
    assert sbs96_of("N", "T", "A", "G") is None   # unknown base


def test_every_canonical_context_is_reachable_and_distinct():
    """All 96 channels must be produced, exactly once each, by the pyrimidine enumeration --
    otherwise the matrix has dead rows and SigProfiler is fitting a padded spectrum."""
    bases = ["A", "C", "G", "T"]
    produced = set()
    for reference, alternates in (("C", ["A", "G", "T"]), ("T", ["A", "C", "G"])):
        for alternate in alternates:
            for previous in bases:
                for following in bases:
                    produced.add(sbs96_of(reference, alternate, previous, following))
    assert len(produced) == 96, f"expected 96 channels, got {len(produced)}"


def test_sbs96_counts_scatter_matches_a_loop():
    rng = np.random.default_rng(0)
    n, n_clusters = 5000, 7
    index = rng.integers(-1, 96, size=n).astype(np.int8)
    labels = rng.integers(-1, n_clusters, size=n).astype(np.int32)

    counts = sbs96_counts(index, labels, n_clusters)

    expected = np.zeros((96, n_clusters), dtype=np.int64)
    for row, label in zip(index, labels):
        if row >= 0 and 0 <= label < n_clusters:
            expected[row, label] += 1
    np.testing.assert_array_equal(counts, expected)
    # Noise rows and null contexts are dropped, never folded into cluster 0.
    assert counts.sum() == int(((index >= 0) & (labels >= 0)).sum())


def test_sbs96_counts_respects_read_weights():
    index = np.array([0, 0, 5], dtype=np.int8)
    labels = np.array([0, 1, 0], dtype=np.int32)
    weights = np.array([10, 3, 2], dtype=np.int32)

    counts = sbs96_counts(index, labels, 2, weights=weights)
    assert counts[0, 0] == 10 and counts[0, 1] == 3 and counts[5, 0] == 2


def test_cluster_size_stats_measures_the_decision_variables():
    """p10/p90 and the in-band mutation share are what the ranking thresholds on, so they
    are computed over the FULL cohort labelling and must ignore noise entirely."""
    labels = np.concatenate([
        np.full(500, -1),                       # noise
        np.full(1_000, 0),                      # below the 3k floor
        np.full(50_000, 1),                     # in band
        np.full(80_000, 2),                     # in band
        np.full(400_000, 3),                    # above the 300k ceiling
    ]).astype(np.int32)

    stats = cluster_size_stats(labels)

    assert stats["n_clusters"] == 4
    assert stats["clusters_below_floor"] == 1
    assert stats["clusters_above_ceiling"] == 1
    assert stats["min"] == 1_000 and stats["max"] == 400_000
    assert stats["total_assigned"] == 531_000
    np.testing.assert_allclose(stats["mutation_share_in_band"], 130_000 / 531_000)
    np.testing.assert_allclose(stats["noise_fraction"], 500 / 531_500)


def test_cluster_size_stats_entropy_penalises_one_giant_cluster():
    balanced = np.repeat(np.arange(4), 1000).astype(np.int32)
    lopsided = np.concatenate([np.zeros(3970), np.arange(1, 4).repeat(10)]).astype(np.int32)

    assert cluster_size_stats(balanced)["size_entropy_normalised"] == pytest.approx(1.0)
    assert cluster_size_stats(lopsided)["size_entropy_normalised"] < 0.3


def test_cluster_size_stats_handles_an_all_noise_labelling():
    stats = cluster_size_stats(np.full(100, -1, dtype=np.int32))
    assert stats["n_clusters"] == 0 and stats["noise_fraction"] == 1.0


def test_grid_holds_min_cluster_size_fixed_across_fit_sizes():
    """The whole point of the sweep: scaling_results.json used mcs = N * 1e-4, so cluster
    count fell 1338 -> 442 while mcs rose 50x and neither effect can be attributed. Every
    fit size must see the same mcs ladder."""
    grid = build_grid([1_000_000, 5_000_000], [250, 1000], [5], ["eom"], [0.0], {})
    by_size = {}
    for cell in grid:
        by_size.setdefault(cell.fit_rows, set()).add(cell.min_cluster_size)
    assert by_size[1_000_000] == by_size[5_000_000] == {250, 1000}


def test_grid_applies_the_min_samples_ceiling():
    """min_samples=15 at 25M OOMed on the 47 GB card, so the grid must not schedule it."""
    grid = build_grid([5_000_000, 25_000_000], [500], [5, 15], ["eom"], [0.0],
                      {25_000_000: 5})
    at_25m = {cell.min_samples for cell in grid if cell.fit_rows == 25_000_000}
    at_5m = {cell.min_samples for cell in grid if cell.fit_rows == 5_000_000}
    assert at_25m == {5}
    assert at_5m == {5, 15}


def test_grid_runs_cheapest_first():
    grid = build_grid([25_000_000, 500_000], [500], [5], ["eom"], [0.0], {})
    assert [cell.fit_rows for cell in grid] == [500_000, 25_000_000]


def test_cell_labels_are_unique_and_encode_epsilon():
    cells = build_grid([1_000_000], [500], [5], ["eom", "leaf"], [0.0, 0.1], {})
    labels = [cell.label() for cell in cells]
    assert len(set(labels)) == len(labels)
    assert any("eps0.1" in label for label in labels)
    assert not any(label.endswith("eps0.0") for label in labels)


def test_projected_fit_seconds_reproduces_the_measured_points():
    """The projection drives the budget the user commits GPU-days to, so it has to land on
    the measurements it interpolates rather than near them."""
    for rows, seconds in [(1_000_000, 23.6), (5_000_000, 469.9), (25_000_000, 11684.0)]:
        assert projected_fit_seconds(rows, 5) == pytest.approx(seconds, rel=1e-6)
    # Between measured points it stays monotone and inside the bracket.
    middle = projected_fit_seconds(2_000_000, 5)
    assert 23.6 < middle < 469.9


def test_thresholds_match_the_plan():
    assert (MUTATION_FLOOR, MUTATION_CEILING) == (3_000, 300_000)
    assert Cell(1, 2, 3).cluster_selection_method == "eom"


def test_uv_signature_set_matches_the_uv_only_database():
    """The uv_only reference is SBS7A-D **plus SBS38**
    (run_variant_cluster_pipeline.SBS96_SIGNATURES). Dropping SBS38 loses the "does SBS38
    still appear" question the plan asks by name, and it is the one UV signature that is not
    an SBS7 subtype, so it cannot be recovered from the subtype count."""
    assert set(UV_SIGNATURES) == {"SBS7A", "SBS7B", "SBS7C", "SBS7D", "SBS38"}
    assert set(SBS7_SUBTYPES) == {"SBS7A", "SBS7B", "SBS7C", "SBS7D"}
    assert "SBS38" not in SBS7_SUBTYPES


@pytest.mark.parametrize("spelling", ["upper", "lower"])
def test_signature_metrics_survive_both_column_spellings(tmp_path, spelling):
    """COSMIC ships `SBS7a`; write_uv_only_signature_database renames to `SBS7A`. So the
    Activities columns are lowercase under --signature-set full and uppercase under uv_only.

    Matching one spelling literally reports zero UV signatures under the other -- and it
    reports it as a number, not an error, so it reads as a clustering result. That is the
    failure this pins.
    """
    names = (["SBS7A", "SBS7B", "SBS7C", "SBS7D", "SBS38"] if spelling == "upper"
             else ["SBS7a", "SBS7b", "SBS7c", "SBS7d", "SBS38"])
    stats_dir = tmp_path / "Assignment_Solution" / "Solution_Stats"
    activities_dir = tmp_path / "Assignment_Solution" / "Activities"
    stats_dir.mkdir(parents=True)
    activities_dir.mkdir(parents=True)

    clusters = [f"cluster_{i}" for i in range(5)]
    pl.DataFrame({"Samples": clusters,
                  "Cosine Similarity": [0.95, 0.85, 0.75, 0.65, 0.20]}
                 ).write_csv(stats_dir / "Assignment_Solution_Samples_Stats.txt",
                             separator="\t")
    # Each cluster is dominated by a different signature: all four SBS7 subtypes separate
    # and SBS38 dominates the fifth.
    activity = {"Samples": clusters}
    for position, name in enumerate(names):
        activity[name] = [1000 if row == position else 1 for row in range(5)]
    pl.DataFrame(activity).write_csv(
        activities_dir / "Assignment_Solution_Activities.txt", separator="\t")

    mutations = {name: value for name, value in
                 zip(clusters, [100_000.0, 50_000.0, 20_000.0, 10_000.0, 5_000.0])}
    metrics = sigprofiler_metrics(tmp_path, clusters, mutations)

    assert metrics["n_sbs7_subtypes_separated"] == 4, metrics
    assert metrics["sbs38_present"] is True
    assert metrics["sbs38_dominates_a_cluster"] is True
    assert metrics["signatures_used"] == 5

    # Mutation share above 0.7 must be size-weighted, not a cluster fraction: three of five
    # clusters clear it, but they hold 170k of 185k mutations.
    assert metrics["clusters_above_07"] == 3
    np.testing.assert_allclose(metrics["mutation_share_above_07"], 170_000 / 185_000)


def test_sbs38_absence_is_reported_without_hiding_sbs7_separation(tmp_path):
    """SBS38 missing and one SBS7 subtype missing are different findings; a single combined
    UV count would score both as 4 and conflate them."""
    stats_dir = tmp_path / "Assignment_Solution" / "Solution_Stats"
    activities_dir = tmp_path / "Assignment_Solution" / "Activities"
    stats_dir.mkdir(parents=True)
    activities_dir.mkdir(parents=True)

    clusters = [f"cluster_{i}" for i in range(4)]
    pl.DataFrame({"Samples": clusters, "Cosine Similarity": [0.9, 0.9, 0.9, 0.9]}
                 ).write_csv(stats_dir / "Assignment_Solution_Samples_Stats.txt",
                             separator="\t")
    activity = {"Samples": clusters}
    for position, name in enumerate(["SBS7A", "SBS7B", "SBS7C", "SBS7D"]):
        activity[name] = [1000 if row == position else 0 for row in range(4)]
    activity["SBS38"] = [0, 0, 0, 0]
    pl.DataFrame(activity).write_csv(
        activities_dir / "Assignment_Solution_Activities.txt", separator="\t")

    metrics = sigprofiler_metrics(tmp_path, clusters, dict.fromkeys(clusters, 1000.0))
    assert metrics["n_sbs7_subtypes_separated"] == 4
    assert metrics["sbs38_present"] is False
    assert metrics["sbs38_dominates_a_cluster"] is False
