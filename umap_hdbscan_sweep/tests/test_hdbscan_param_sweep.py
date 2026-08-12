"""The grid arithmetic, before GPU-days go into it.

The sweep itself is fit -> label -> save. Everything downstream of the labels (SBS96 counts,
SigProfiler, cluster-size tables) lives in run_variant_cluster_pipeline and is tested there,
so what is left to pin here is the grid: that min_cluster_size really is held fixed across
fit sizes (the confound the whole sweep exists to break), that the min_samples ceiling keeps
the OOM cell off the schedule, and that the cost projection the budget is committed against
lands on its measured points.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO_ROOT / "uv_vae" / "scripts"))

from hdbscan_param_sweep import (  # noqa: E402
    Cell,
    build_grid,
    projected_fit_seconds,
    trinuc_counts_from_labels,
)


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


def test_default_cell_selection_method():
    assert Cell(1, 2, 3).cluster_selection_method == "eom"


# ── the hand-off to run_variant_cluster_pipeline ───────────────────────────────

def write_context(path, rows):
    pl.DataFrame({
        "REF": [r[0] for r in rows], "ALT": [r[1] for r in rows],
        "X_PREV1": [r[2] for r in rows], "X_NEXT1": [r[3] for r in rows],
    }).write_parquet(path)
    return path


def test_trinuc_counts_drop_noise_and_aggregate_duplicates(tmp_path):
    """Noise rows must not become a cluster, and identical contexts within a cluster must
    collapse to one counted row -- that is the shape rvcp.annotate_trinuc_counts expects."""
    context = write_context(tmp_path / "c.parquet", [
        ("C", "T", "A", "G"), ("C", "T", "A", "G"), ("C", "T", "A", "G"),
        ("T", "A", "G", "C"),
        ("C", "T", "A", "G"),
    ])
    labels = np.array([0, 0, 0, 1, -1], dtype=np.int32)

    counts = trinuc_counts_from_labels(context, labels)

    assert set(counts["cluster_label"].to_list()) == {0, 1}
    row = counts.filter(pl.col("cluster_label") == 0)
    assert row.height == 1 and row["count"][0] == 3
    assert counts["count"].sum() == 4   # the noise row is gone


def test_trinuc_counts_refuse_a_length_mismatch(tmp_path):
    context = write_context(tmp_path / "c.parquet", [("C", "T", "A", "G")] * 3)
    with pytest.raises(SystemExit, match="must come from the same stage-1 run"):
        trinuc_counts_from_labels(context, np.zeros(5, dtype=np.int32))


def test_counts_feed_rvcp_annotate_unchanged(tmp_path):
    """End to end into the REAL run_variant_cluster_pipeline.annotate_trinuc_counts.

    The point of routing through rvcp is that the SBS96 canonicalisation stays the single
    implementation every earlier result in this repo used. If the column names this produces
    ever drift from what rvcp reads, this fails instead of silently emitting nulls.
    """
    rvcp = pytest.importorskip("run_variant_cluster_pipeline")

    context = write_context(tmp_path / "c.parquet", [
        ("C", "T", "A", "G"),          # already pyrimidine -> A[C>T]G
        ("A", "G", "T", "C"),          # purine -> reverse complement -> G[T>C]A
    ])
    labels = np.array([0, 0], dtype=np.int32)

    counts = trinuc_counts_from_labels(context, labels)
    sizes = pl.DataFrame({"cluster_label": [0], "cluster_size": [2]})
    annotated = rvcp.annotate_trinuc_counts(counts, sizes)

    assert "sbs96" in annotated.columns
    assert set(annotated["sbs96"].to_list()) == {"A[C>T]G", "G[T>C]A"}
