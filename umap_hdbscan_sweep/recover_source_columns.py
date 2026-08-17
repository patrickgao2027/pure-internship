#!/usr/bin/env python
"""Rebuild a cell's analysis.parquet with **every** source column, not the curated 35.

The problem
-----------
``analysis.parquet`` carries 35 data columns. The source parquets have 70. The missing 35
(QUAL, MAPQ, DP, DP_FILT, RAW_VAF, VAF, AD_A..AD_INS, RPA, RU, STR, X_IC, X_IL, X_PREV2/3,
X_NEXT2/3, RN, DUP, MQUAL, SNVQ, FILT, DP_MAPQ1, ADJ_REF_DIFF, tm, a3, st, et, MI, DS) were
never selected by stage 0, so they exist upstream and nowhere downstream.

Getting them back means answering one question per row: **which source read is this?**
Stage 0 collapsed ~32 reads per locus down to one, so (CHROM, POS, REF, ALT) alone does not
answer it -- it names the locus, not the read.

Why not ``max()`` per locus
---------------------------
``cohort_cluster_report.read_source_columns`` recovers five columns that way. It is
defensible only for QUAL -- the first ranking key, where the survivor *is* the maximum. For
everything else it is wrong: MAPQ/DP/RAW_VAF are ranked within the QUAL-tied subset, so the
locus maximum can come from a read that lost; VAF/AD_*/DUP take the maximum over all reads,
i.e. a different read from the one the row's other 35 columns came from; and RU/X_IC/RN/MI/
tm/st/et are strings, where ``max()`` is lexicographic and means nothing. The result is a
chimera -- a row that never existed in the data.

Why not replay stage 0's ranking
--------------------------------
The obvious fix is to re-run the window: same filter, same PARTITION BY, same ORDER BY QUAL,
MAPQ, DP, RAW_VAF DESC, same tie-break, take rn = 1. That works, but its correctness rests
entirely on reproducing a sort. QUAL, MAPQ and DP are discrete and there are ~32 reads per
locus, so ties on all four ranking keys are common and the tie-break -- a column list derived
from the training checkpoint -- decides a large share of loci. Point it at the wrong
checkpoint and it silently selects different reads. No error, just wrong values.

What this does instead: identify the row, do not re-derive it
-------------------------------------------------------------
The 35 columns already in ``analysis.parquet`` are read-level measurements -- rq, BCSQ,
SMQ_BEFORE/AFTER, EDIST, RL, INDEX, l1..l7, q2..q6, HAMDIST. Together they fingerprint the
read. Joining source to analysis on all of them plus the identity columns lands on the exact
source row with no ORDER BY to reproduce and no checkpoint to supply. The values were copied
through stage 0 without arithmetic, so they compare bit-for-bit.

Ties are handled by construction rather than by a rule: two reads matching on all 35
measurements are indistinguishable in every column the pipeline ever used, so either serves.
The script reports how many loci that happened at instead of hiding it.

One run covers every cell
-------------------------
``cohort_cluster_report.main`` draws ``positions`` once and passes the same array to every
cell, and the UMAP coordinates come from one ``coords.npy``. So all 28 cells' analysis.parquet
files are identical row-for-row except ``cluster_label``/``cluster_probability`` -- verified
by direct comparison. Run this once, plot once.

Usage::

    python umap_hdbscan_sweep/recover_source_columns.py \\
        --analysis <any-cell>/analysis.parquet \\
        --source-glob '/data/lab/ppmseq_parquets/*.parquet' \\
        --output enriched.parquet
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import polars as pl
import pyarrow.parquet as pq

from uv_vae.data import connect_duckdb, quote_ident

from stage0_dedup import DEFAULT_ROW_FILTER, IDENTITY_COLUMNS

# Pipeline outputs, not source columns: carried across from the analysis file untouched.
PIPELINE_COLUMNS = ["umap_1", "umap_2", "cluster_label", "cluster_probability"]
# Written by stage 0, absent from the source, and not a read measurement -- it must not join.
DERIVED_COLUMNS = ["locus_reads"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis", type=Path, required=True,
                   help="any cell's analysis.parquet -- supplies the rows and the row order")
    p.add_argument("--source-glob", required=True,
                   help="the 95 source parquets, e.g. '/data/lab/ppmseq_parquets/*.parquet'")
    p.add_argument("--row-filter", default=DEFAULT_ROW_FILTER,
                   help="must match the filter stage 0 ran with")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--memory-limit", default=None, help="e.g. 64GB")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = perf_counter()

    analysis = pl.read_parquet(args.analysis)
    analysis = analysis.with_row_index("__row")
    log(f"{args.analysis.name}: {analysis.height:,} rows x {analysis.width - 1} columns")

    source_columns = list(pq.read_schema(_first_source(args.source_glob)).names)
    log(f"source schema: {len(source_columns)} columns")

    # The fingerprint: every column the two files share, minus pipeline outputs and stage 0's
    # own derived columns. Identity names the locus; the measurements name the read.
    excluded = set(PIPELINE_COLUMNS) | set(DERIVED_COLUMNS) | {"__row"}
    fingerprint = [c for c in analysis.columns if c in source_columns and c not in excluded]
    measurements = [c for c in fingerprint if c not in IDENTITY_COLUMNS]
    recovered = [c for c in source_columns if c not in fingerprint]

    missing_identity = [c for c in IDENTITY_COLUMNS if c not in fingerprint]
    if missing_identity:
        raise SystemExit(f"analysis.parquet lacks identity columns {missing_identity}; "
                         "the join cannot be anchored")
    if not measurements:
        raise SystemExit("no read-level measurements shared with the source -- the identity "
                         "columns alone name the locus, not the read, so the join would be "
                         "ambiguous across every read at it")

    log(f"fingerprint: {len(IDENTITY_COLUMNS)} identity + {len(measurements)} measurements")
    log(f"to recover:  {len(recovered)} columns")

    on_clause = " AND ".join(
        # NOT DISTINCT FROM, not =: a null on either side must match a null, and SQL equality
        # returns unknown there, which would drop every row with a null measurement.
        f's.{quote_ident(c)} IS NOT DISTINCT FROM k.{quote_ident(c)}' for c in fingerprint
    )
    select_recovered = ", ".join(f"s.{quote_ident(c)}" for c in recovered)

    with connect_duckdb(threads=args.threads) as conn:
        if args.memory_limit:
            conn.execute(f"SET memory_limit='{args.memory_limit}'")
        conn.register("keys", analysis.select(["__row", *fingerprint]))

        log("scanning source parquets (one pass, joined on the fingerprint)")
        # DISTINCT ON collapses the case where two reads carry identical measurements: they
        # are indistinguishable in every column the pipeline used, so either serves, and the
        # count is reported below rather than passed over.
        frame = conn.execute(f"""
            SELECT k."__row", {select_recovered}
            FROM keys AS k
            JOIN read_parquet({sql_quote(args.source_glob)}) AS s
              ON {on_clause}
            WHERE ({args.row_filter})
        """).pl()

    matched = frame.select(pl.col("__row").n_unique()).item()
    duplicates = frame.height - matched
    log(f"matched {matched:,} of {analysis.height:,} rows "
        f"({100 * matched / analysis.height:.4f}%)")
    if duplicates:
        log(f"  {duplicates:,} extra matches: loci where two reads share all "
            f"{len(measurements)} measurements; keeping one, they are interchangeable")
    if matched < analysis.height:
        log(f"  WARNING: {analysis.height - matched:,} rows found no source row. Check that "
            f"--row-filter matches the filter stage 0 ran with (currently: {args.row_filter})")

    frame = frame.unique(subset=["__row"], keep="first")
    merged = (analysis.join(frame, on="__row", how="left")
                      .sort("__row")
                      .drop("__row"))
    if merged.height != analysis.height:
        raise SystemExit(f"join produced {merged.height:,} rows from {analysis.height:,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(args.output, compression="zstd")
    log(f"wrote {args.output}  ({merged.height:,} rows x {merged.width} columns, "
        f"{args.output.stat().st_size / 1e6:.0f} MB, {perf_counter() - started:.0f}s)")
    log(f"recovered: {', '.join(recovered)}")
    return 0


def _first_source(source_glob: str) -> str:
    import glob as globmod

    matches = sorted(globmod.glob(source_glob))
    if not matches:
        raise SystemExit(f"no files match {source_glob}")
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
