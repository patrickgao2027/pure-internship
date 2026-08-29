#!/usr/bin/env python
"""Check that enriched.parquet really is analysis.parquet plus every source column.

Four questions, because "the columns are present" is the weakest of them and a file can pass
it while being silently wrong:

1. **Coverage** -- is every source column now present, and did nothing get dropped from the
   analysis file on the way through?
2. **Fidelity** -- do the columns that were already in analysis.parquet still hold exactly
   the values they held there? The join must have *added* columns, not perturbed existing
   ones, and the row order must be unchanged: the plotting code plays a positional join
   (latent row i == plot row i), so a reordered file would attach every colour to the wrong
   point without changing a single value.
3. **Population** -- how much of each recovered column is actually non-null? A column that
   joined but came back all-null is a failed recovery wearing the costume of a successful
   one.
4. **Plottability** -- which columns are constant, and therefore expected to render as a
   flat panel. ``st``/``et``/``FILT`` are pinned by the row filter and *should* be constant;
   anything else appearing here is worth a look.

Exit status is 0 only if coverage and fidelity both pass. Population and plottability are
reported, not enforced -- an all-null column can be a true fact about the data.

Usage::

    python umap_hdbscan_sweep/verify_enriched.py \\
        --enriched enriched.parquet \\
        --analysis <cell>/analysis.parquet \\
        --source-glob '/data/lab/ppmseq_parquets/*.parquet'
"""
from __future__ import annotations

import argparse
import glob as globmod
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

# Pinned by the default row filter (st = 'MIXED' AND et = 'MIXED' AND FILT = 1), so a single
# value in these is the filter working, not a recovery failure.
EXPECTED_CONSTANT = {"st", "et", "FILT"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enriched", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True,
                   help="the analysis.parquet that was passed to recover_source_columns.py")
    p.add_argument("--source-glob", required=True)
    p.add_argument("--sample-rows", type=int, default=200_000,
                   help="rows compared for the fidelity check (0 = all)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    matches = sorted(globmod.glob(args.source_glob))
    if not matches:
        raise SystemExit(f"no files match {args.source_glob}")
    source_columns = list(pq.read_schema(matches[0]).names)

    enriched_schema = pq.read_schema(args.enriched)
    analysis_schema = pq.read_schema(args.analysis)
    enriched_columns = set(enriched_schema.names)

    n_enriched = pq.ParquetFile(args.enriched).metadata.num_rows
    n_analysis = pq.ParquetFile(args.analysis).metadata.num_rows

    print(f"enriched : {args.enriched.name}  {n_enriched:,} rows x "
          f"{len(enriched_schema.names)} columns")
    print(f"analysis : {args.analysis.name}  {n_analysis:,} rows x "
          f"{len(analysis_schema.names)} columns")
    print(f"source   : {Path(matches[0]).name}  {len(source_columns)} columns "
          f"({len(matches)} files)")

    # ── 1. coverage ────────────────────────────────────────────────────────────
    print("\n=== 1. coverage ===")
    if n_enriched != n_analysis:
        failures.append(f"row count changed: {n_analysis:,} -> {n_enriched:,}")
        print(f"  FAIL  row count changed: {n_analysis:,} -> {n_enriched:,}")
    else:
        print(f"  ok    row count preserved ({n_enriched:,})")

    missing_source = [c for c in source_columns if c not in enriched_columns]
    missing_analysis = [c for c in analysis_schema.names if c not in enriched_columns]
    if missing_source:
        failures.append(f"{len(missing_source)} source columns absent")
        print(f"  FAIL  {len(missing_source)} source columns absent: "
              f"{', '.join(missing_source)}")
    else:
        print(f"  ok    all {len(source_columns)} source columns present")
    if missing_analysis:
        failures.append(f"{len(missing_analysis)} analysis columns lost")
        print(f"  FAIL  {len(missing_analysis)} analysis columns lost: "
              f"{', '.join(missing_analysis)}")
    else:
        print(f"  ok    all {len(analysis_schema.names)} analysis columns retained")

    extra = sorted(enriched_columns - set(source_columns) - set(analysis_schema.names))
    if extra:
        print(f"  note  {len(extra)} columns from neither input: {', '.join(extra)}")

    # ── 2. fidelity ────────────────────────────────────────────────────────────
    # Compares the carried-over columns value-by-value in file order. Reading both files
    # whole and slicing the same window keeps the row correspondence positional, which is
    # exactly the property being tested.
    print("\n=== 2. fidelity (carried columns unchanged, in the same order) ===")
    shared = [c for c in analysis_schema.names if c in enriched_columns]
    limit = n_analysis if args.sample_rows in (0, None) else min(args.sample_rows, n_analysis)

    left = pl.read_parquet(args.analysis, columns=shared).head(limit)
    right = pl.read_parquet(args.enriched, columns=shared).head(limit)
    print(f"  comparing {len(shared)} shared columns over the first {limit:,} rows")

    mismatched = []
    for column in shared:
        a, b = left[column], right[column]
        # Null positions must line up too, so compare the null masks alongside the values;
        # equality alone reports null == null as null and would hide a difference.
        same = (a.is_null() == b.is_null()).all() and (
            a.fill_null(0).eq(b.fill_null(0)).all()
            if a.dtype.is_numeric() else
            a.fill_null("\x00").eq(b.fill_null("\x00")).all()
        )
        if not same:
            mismatched.append(column)

    if mismatched:
        failures.append(f"{len(mismatched)} carried columns changed value")
        print(f"  FAIL  {len(mismatched)} changed: {', '.join(mismatched)}")
        print("        the join altered existing data or reordered rows -- the positional")
        print("        join used downstream would attach colours to the wrong points")
    else:
        print(f"  ok    all {len(shared)} carried columns identical, row order preserved")

    # ── 3. population ──────────────────────────────────────────────────────────
    print("\n=== 3. population of the recovered columns ===")
    # Restricted to columns the file actually has: coverage above already reported any that
    # are missing, and selecting one here would raise instead of letting the rest be checked.
    recovered = [c for c in source_columns
                 if c not in analysis_schema.names and c in enriched_columns]
    frame = pl.read_parquet(args.enriched, columns=recovered)
    stats = []
    for column in recovered:
        series = frame[column]
        stats.append((column, float(series.is_not_null().mean() * 100), series.n_unique()))

    all_null = [c for c, pct, _ in stats if pct == 0.0]
    print(f"  {len(recovered)} columns recovered; "
          f"{len(recovered) - len(all_null)} carry data, {len(all_null)} are all-null")
    print(f"\n  {'column':<16} {'non-null':>9} {'unique':>9}")
    print(f"  {'-' * 16} {'-' * 9} {'-' * 9}")
    for column, pct, unique in sorted(stats, key=lambda r: -r[1]):
        print(f"  {column:<16} {pct:>8.2f}% {unique:>9,}")
    if all_null:
        print(f"\n  all-null: {', '.join(all_null)}")
        print("  (either genuinely empty upstream, or the join missed -- check the match")
        print("   rate recover_source_columns.py printed)")

    # ── 4. plottability ────────────────────────────────────────────────────────
    print("\n=== 4. columns that will render flat ===")
    constant = [c for c, pct, unique in stats if unique <= 1 and pct > 0]
    unexpected = [c for c in constant if c not in EXPECTED_CONSTANT]
    if constant:
        print(f"  constant: {', '.join(constant)}")
    else:
        print("  none")
    if unexpected:
        print(f"  note    {', '.join(unexpected)} are constant but not pinned by the row")
        print("          filter -- worth checking they are meant to be")
    pinned = [c for c in EXPECTED_CONSTANT if c in constant]
    if pinned:
        print(f"  ok      {', '.join(pinned)} constant as expected (pinned by the row filter)")

    print()
    if failures:
        print(f"FAILED: {'; '.join(failures)}")
        return 1
    print(f"PASSED: {len(source_columns)} source columns present, "
          f"{len(shared)} carried columns unchanged, row order intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
