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
    build_sbs96_index,
    load_or_build_index,
    projected_fit_seconds,
    sbs96_counts,
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


# ── the SBS96 counting path ────────────────────────────────────────────────────

def write_context(path, rows):
    pl.DataFrame({
        "REF": [r[0] for r in rows], "ALT": [r[1] for r in rows],
        "X_PREV1": [r[2] for r in rows], "X_NEXT1": [r[3] for r in rows],
    }).write_parquet(path)
    return path


CHANNELS = [f"{p}[{r}>{a}]{n}"
            for r, alts in (("C", "AGT"), ("T", "ACG"))
            for a in alts for p in "ACGT" for n in "ACGT"]


def test_sbs96_counts_scatter_matches_a_loop():
    """bincount on a flattened (channel, cluster) index replaces np.add.at, which is
    unbuffered and runs at ~1M updates/sec. Same arithmetic, so it must match exactly."""
    rng = np.random.default_rng(0)
    index = rng.integers(-1, 96, size=5000).astype(np.int8)
    labels = rng.integers(-1, 7, size=5000).astype(np.int32)

    counts = sbs96_counts(index, labels, 7)

    expected = np.zeros((96, 7), dtype=np.int64)
    for channel, cluster in zip(index, labels):
        if channel >= 0 and 0 <= cluster < 7:
            expected[channel, cluster] += 1
    np.testing.assert_array_equal(counts, expected)
    # Noise rows and contextless rows are dropped, never folded into cluster 0.
    assert counts.sum() == int(((index >= 0) & (labels >= 0)).sum())


def test_sbs96_counts_are_chunk_invariant():
    """The chunking exists only to bound scratch memory; a chunk boundary must not move a
    single count."""
    rng = np.random.default_rng(4)
    index = rng.integers(-1, 96, size=3000).astype(np.int8)
    labels = rng.integers(-1, 5, size=3000).astype(np.int32)

    whole = sbs96_counts(index, labels, 5, chunk_rows=10**9)
    for chunk in (1, 7, 999, 3000, 3001):
        np.testing.assert_array_equal(sbs96_counts(index, labels, 5, chunk_rows=chunk), whole)


def test_index_build_uses_the_repos_own_canonicalisation(tmp_path):
    """The int8 index must agree with stage3_apply_full.sbs96_expr, which is itself
    documented as identical to rvcp.annotate_trinuc_counts. Building a third
    canonicalisation is exactly what this cache must not become.

    A[C>T]G is already pyrimidine and passes through; T[A>G]C has a purine reference and
    must come back G[T>C]A -- complemented AND with the flanks swapped. Getting the swap
    wrong yields A[T>C]G, a real channel, so the counts would stay plausible while landing
    in the mirror-image context.
    """
    context = write_context(tmp_path / "c.parquet", [
        ("C", "T", "A", "G"),      # -> A[C>T]G
        ("A", "G", "T", "C"),      # -> G[T>C]A
        ("C", "C", "A", "G"),      # not a substitution -> -1
    ])
    index = build_sbs96_index(context, CHANNELS, tmp_path / "idx.npy")

    assert CHANNELS[index[0]] == "A[C>T]G"
    assert CHANNELS[index[1]] == "G[T>C]A"
    assert index[2] == -1
    assert (tmp_path / "idx.npy").exists()


def test_index_cache_is_reused_rather_than_rebuilt(tmp_path):
    context = write_context(tmp_path / "c.parquet", [("C", "T", "A", "G")] * 4)
    first = load_or_build_index(context, CHANNELS, tmp_path)
    # Deleting the source proves the second call read the cache and did not re-scan.
    context.unlink()
    second = load_or_build_index(context, CHANNELS, tmp_path)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_index_matches_rvcp_annotate_on_the_same_rows(tmp_path):
    """End to end against the REAL run_variant_cluster_pipeline, so a drift between the
    cached index and the pipeline's own SBS96 labels fails here rather than silently
    producing a differently-canonicalised matrix."""
    rvcp = pytest.importorskip("run_variant_cluster_pipeline")

    rows = [("C", "T", "A", "G"), ("A", "G", "T", "C"), ("G", "T", "A", "A"),
            ("T", "A", "G", "C")]
    context = write_context(tmp_path / "c.parquet", rows)
    index = build_sbs96_index(context, CHANNELS, tmp_path / "idx.npy")

    counts = pl.DataFrame({
        "cluster_label": [0] * len(rows),
        "REF": [r[0] for r in rows], "ALT": [r[1] for r in rows],
        "X_PREV1": [r[2] for r in rows], "X_NEXT1": [r[3] for r in rows],
        "count": [1] * len(rows),
    })
    annotated = rvcp.annotate_trinuc_counts(
        counts, pl.DataFrame({"cluster_label": [0], "cluster_size": [len(rows)]}))

    for position, expected in enumerate(annotated["sbs96"].to_list()):
        assert CHANNELS[index[position]] == expected
