"""Parsing and weighting of SigProfiler's per-cluster stats.

Nothing here recomputes a fit statistic -- SigProfiler already did that. What is tested is the
handling around it, where the two ways to get a wrong answer are picking the raw norm instead
of the percentage (which makes the metric track cluster size, i.e. the swept parameter) and
averaging over clusters when the question is about mutations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignment_metrics import load_stats, summarise  # noqa: E402

HEADER = ["Sample Names", "Total Mutations", "Cosine Similarity", "L1 Norm", "L1_Norm_%",
          "L2 Norm", "L2_Norm_%", "KL Divergence", "Correlation"]


def write_stats(path: Path, rows: list[dict]) -> Path:
    frame = pl.DataFrame({
        "Sample Names": [r["name"] for r in rows],
        "Total Mutations": [r["mutations"] for r in rows],
        "Cosine Similarity": [r["cosine"] for r in rows],
        "L1 Norm": [r["l1_abs"] for r in rows],
        "L1_Norm_%": [f"{r['l1_pct']}%" for r in rows],
        "L2 Norm": [r["l2_abs"] for r in rows],
        "L2_Norm_%": [f"{r['l2_pct']}%" for r in rows],
        "KL Divergence": [r["kl"] for r in rows],
        "Correlation": [r["correlation"] for r in rows],
    })
    frame.write_csv(path, separator="\t")
    return path


def make_row(name, mutations, cosine, l1_pct=50.0, l2_pct=25.0, kl=1.0, correlation=0.9):
    # SigProfiler reports the raw norm in mutation counts; the percentage is that over the
    # cluster total. The helper keeps the two consistent so a test that confuses them fails.
    return {"name": name, "mutations": mutations, "cosine": cosine,
            "l1_abs": mutations * l1_pct / 100, "l1_pct": l1_pct,
            "l2_abs": mutations * l2_pct / 100, "l2_pct": l2_pct,
            "kl": kl, "correlation": correlation}


def test_percentage_columns_parse_despite_the_trailing_percent(tmp_path):
    path = write_stats(tmp_path / "s.txt", [make_row("cluster_0", 1000, 0.9, l1_pct=123.813)])
    stats = load_stats(path)
    assert stats["l1_pct"][0] == pytest.approx(123.813)
    assert stats["l2_pct"][0] == pytest.approx(25.0)


def test_raw_and_percentage_norms_are_kept_apart(tmp_path):
    """'L1 Norm' and 'L1_Norm_%' differ by one character and one is a size-scaled trap.
    Matching the wrong one would rank cells by cluster size."""
    path = write_stats(tmp_path / "s.txt", [make_row("cluster_0", 100_000, 0.9, l1_pct=60.0)])
    stats = load_stats(path)
    assert stats["l1_pct"][0] == pytest.approx(60.0)
    assert stats["l1_abs"][0] == pytest.approx(60_000.0)


def test_percentage_norms_ignore_cluster_size_but_raw_norms_do_not(tmp_path):
    """Two cells with identical per-cluster fit quality and 100x different cluster sizes.
    The percentage means must match exactly; the raw means must not."""
    small = write_stats(tmp_path / "small.txt",
                        [make_row(f"cluster_{i}", 1_000, 0.5, l1_pct=80.0) for i in range(4)])
    large = write_stats(tmp_path / "large.txt",
                        [make_row(f"cluster_{i}", 100_000, 0.5, l1_pct=80.0) for i in range(4)])

    small_summary, large_summary = summarise(small), summarise(large)
    assert small_summary["l1_pct_mean"] == pytest.approx(large_summary["l1_pct_mean"])
    assert small_summary["l1_abs_mean"] * 100 == pytest.approx(large_summary["l1_abs_mean"])


def test_mutation_weighting_differs_from_cluster_averaging(tmp_path):
    """One big well-fit cluster and three tiny badly-fit ones. The cluster mean says the cell
    is poor; the mutation-weighted mean says most of the cohort is well explained. Both are
    true and they answer different questions, so both must be reported."""
    rows = [make_row("cluster_0", 970_000, 0.9)] + \
           [make_row(f"cluster_{i}", 10_000, 0.1) for i in (1, 2, 3)]
    summary = summarise(write_stats(tmp_path / "s.txt", rows))

    assert summary["cosine_mean"] == pytest.approx((0.9 + 0.1 * 3) / 4)
    assert summary["cosine_weighted_mean"] == pytest.approx(
        (0.9 * 970_000 + 0.1 * 30_000) / 1_000_000)
    assert summary["cosine_weighted_mean"] > 0.87 > summary["cosine_mean"]


def test_mutation_share_above_threshold_is_not_a_cluster_fraction(tmp_path):
    """The 2026-08-04 finding: cluster count above 0.7 moved 544 -> 68 while the mutation
    share they carried stayed at ~10.5%. Conflating the two is what made cluster count look
    like a result."""
    rows = [make_row("cluster_0", 900_000, 0.95)] + \
           [make_row(f"cluster_{i}", 10_000, 0.2) for i in range(1, 11)]
    summary = summarise(write_stats(tmp_path / "s.txt", rows))

    assert summary["clusters_above_07"] == 1
    assert summary["frac_clusters_above_07"] == pytest.approx(1 / 11)
    assert summary["mutation_share_above_07"] == pytest.approx(900_000 / 1_000_000)


def test_thresholds_are_strict_inequalities(tmp_path):
    """A cluster sitting exactly on 0.8 must not be counted above it, or a cell can be
    flattered by a pile-up at the boundary -- and pile-ups at exact values are real here
    (40 clusters shared cosine 0.296)."""
    rows = [make_row("cluster_0", 1000, 0.8), make_row("cluster_1", 1000, 0.80001)]
    summary = summarise(write_stats(tmp_path / "s.txt", rows))
    assert summary["clusters_above_08"] == 1


def test_l2_rebased_on_total_mutations_is_shape_independent(tmp_path):
    """SigProfiler's L2_Norm_% divides by ||obs||_2, which is 1.0x total mutations for a
    one-hot spectrum and total/sqrt(k) for one spread over k channels. Two clusters with the
    SAME absolute L2 error therefore get very different L2_Norm_% purely from shape -- which
    is why L2_Norm_% ranks cells opposite to L1_Norm_%. The rebased version divides by total
    mutations, like L1 does, so equal absolute error gives equal score.
    """
    concentrated = make_row("cluster_0", 1000, 0.5, l2_pct=40.0)   # ||obs||_2 ~ total
    spread = make_row("cluster_1", 1000, 0.5, l2_pct=80.0)         # ||obs||_2 ~ total/4
    spread["l2_abs"] = concentrated["l2_abs"]                       # identical actual error

    summary = summarise(write_stats(tmp_path / "s.txt", [concentrated, spread]))

    # SigProfiler's own percentage says these differ two-fold...
    assert summary["l2_pct_mean"] == pytest.approx(60.0)
    # ...the rebased one correctly says they are identical.
    assert summary["l2_over_total_pct_mean"] == pytest.approx(
        100 * concentrated["l2_abs"] / 1000)


def test_l2_rebased_matches_l1_convention_on_a_clean_case(tmp_path):
    """Both norms divided by total mutations must agree when the residual sits in a single
    channel, since there ||d||_1 == ||d||_2."""
    row = make_row("cluster_0", 10_000, 0.5, l1_pct=30.0)
    row["l2_abs"] = row["l1_abs"]
    summary = summarise(write_stats(tmp_path / "s.txt", [row]))
    assert summary["l2_over_total_pct_mean"] == pytest.approx(summary["l1_pct_mean"])


def test_missing_optional_columns_do_not_break_the_summary(tmp_path):
    """Older or reduced stats files may lack KL/Correlation. Cosine is required; the rest
    should simply be absent rather than raising or appearing as zero."""
    pl.DataFrame({"Sample Names": ["cluster_0"], "Total Mutations": [1000],
                  "Cosine Similarity": [0.75]}).write_csv(tmp_path / "s.txt", separator="\t")
    summary = summarise(tmp_path / "s.txt")
    assert summary["cosine_mean"] == pytest.approx(0.75)
    assert "kl_mean" not in summary and "l1_pct_mean" not in summary


def test_a_missing_cosine_column_raises_rather_than_returning_nothing(tmp_path):
    pl.DataFrame({"Sample Names": ["cluster_0"], "Total Mutations": [1000]}).write_csv(
        tmp_path / "s.txt", separator="\t")
    with pytest.raises(ValueError, match="no cosine column"):
        summarise(tmp_path / "s.txt")


def test_matches_the_real_file_shape():
    """Guards against a SigProfiler column rename silently disabling a metric."""
    real = Path("uv_vae/runs/nn15_md0.0_nc2/mcs1000_ms5/"
                "sigprofilerassignment_uv_only_grch38_v3.5/output/Assignment_Solution/"
                "Solution_Stats/Assignment_Solution_Samples_Stats.txt")
    if not real.exists():
        pytest.skip("reference run not present")
    summary = summarise(real)
    for key in ["cosine_mean", "l1_pct_mean", "l2_pct_mean", "l2_over_total_pct_mean",
                "kl_mean", "correlation_mean", "mutation_share_above_07"]:
        assert key in summary, f"{key} missing -- a column name may have changed"
    assert summary["n_clusters"] == 1150
    assert 0.0 <= summary["cosine_mean"] <= 1.0
    assert np.isfinite(summary["l1_pct_mean"])
