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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter

# Walk up to the directory that holds uv_vae/ rather than counting levels: these scripts
# get reorganised into subfolders (umap/, hdbscan/, tests/), and a hard-coded parents[N]
# silently resolves to the wrong root the moment one moves, failing on `import uv_vae`.
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import polars as pl
import pyarrow.parquet as pq

from uv_vae.data import connect_duckdb, quote_ident

# Inlined rather than imported from stage0_dedup: that module lives wherever this repo's
# umap_hdbscan_sweep/ subtree has been reorganised to on a given host, and neither its
# location nor its own REPO_ROOT bootstrapping (which assumes a fixed nesting depth) is this
# script's business just to reach two constants. If DEFAULT_ROW_FILTER ever changes in
# stage0_dedup.py, change it here too -- CLAUDE.md documents it as the project-wide default.
DEFAULT_ROW_FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"
IDENTITY_COLUMNS = ["CHROM", "POS", "REF", "ALT"]

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
    p.add_argument("--threads", type=int, default=16,
                   help="total DuckDB worker threads, shared across --parallel-chromosomes")
    p.add_argument("--parallel-chromosomes", type=int, default=6,
                   help="chromosome queries in flight at once. Each is I/O-bound (waiting on "
                        "the source filesystem, not CPU), so running several concurrently "
                        "keeps more read requests outstanding instead of one at a time; the "
                        "shared 569 GB source lives on what looks like a network mount, which "
                        "usually serves concurrent reads better than one sequential stream. "
                        "Start conservative and raise it if the filesystem tolerates more.")
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

    # The join is anchored on the four identity columns ALONE, with the measurements applied
    # afterwards as a filter. That split is not cosmetic: putting all 35 columns in the ON
    # clause -- and they must be null-safe comparisons, since SQL equality returns unknown
    # against a null and would silently drop every row with a null measurement -- gives the
    # planner a join condition it cannot hash on, and it degrades to a nested loop. Two
    # million keys against five billion source rows never finishes that way. Identity is a
    # plain equi-join the planner hashes, cutting the candidates to the ~32 reads at each
    # locus, and the measurements then pick the one read out of those. Same result set,
    # because an inner join on 35 columns is exactly an inner join on 4 filtered by the
    # other 31.
    match_clause = " AND ".join(
        f's.{quote_ident(c)} IS NOT DISTINCT FROM k.{quote_ident(c)}' for c in measurements
    )
    select_recovered = ", ".join(f"s.{quote_ident(c)}" for c in recovered)

    # One chromosome per query, mirroring stage 0: the source files are sorted by
    # (CHROM, POS), so pinning CHROM lets row-group statistics prune every group that cannot
    # match. Several of those queries run concurrently rather than one at a time, because each
    # is I/O-bound on the source filesystem rather than CPU-bound -- one query in flight
    # leaves the disk/network path underused while it waits, so overlapping --parallel-
    # chromosomes of them keeps more reads outstanding at once. Total bytes read are the same
    # either way; this only changes how much of that waiting happens concurrently.
    chromosomes = sorted(analysis.select("CHROM").unique().to_series().to_list())
    parallel = max(1, min(args.parallel_chromosomes, len(chromosomes)))
    log(f"scanning {len(chromosomes)} chromosomes, {parallel} at a time")

    keys_by_chrom = {
        chromosome: analysis.filter(pl.col("CHROM") == chromosome).select(
            ["__row", *fingerprint])
        for chromosome in chromosomes
    }

    # One cursor per worker THREAD, not per task: duckdb connections/cursors are not meant to
    # be touched from more than one thread concurrently, and creating a fresh cursor per task
    # from multiple threads at once would itself be a race. threading.local() gives each of
    # the pool's worker threads exactly one cursor, created lazily the first time that thread
    # picks up a task, and every later task on that same thread reuses it.
    local = threading.local()

    def query_chromosome(conn, chromosome: str) -> tuple[str, "pl.DataFrame"]:
        if not hasattr(local, "cursor"):
            local.cursor = conn.cursor()
        cursor = local.cursor
        view = f"keys_{threading.get_ident()}"
        cursor.register(view, keys_by_chrom[chromosome])
        # CHROM (and now row_filter alongside it) is filtered inside the scanned subquery, not
        # in an outer WHERE after the join. A post-join predicate on the probe side is only
        # pushed into the parquet scan if the optimiser chooses to hoist it there, which is not
        # guaranteed once the other join operand is a registered non-parquet relation --
        # filtering here makes the row-group pruning unconditional instead of a planner guess.
        # This was the difference between reading ~1/24th of the source and reading all 5.08B
        # rows while a hash table for one chromosome was built underneath it.
        block = cursor.execute(f"""
            SELECT k."__row", {select_recovered}
            FROM {quote_ident(view)} AS k
            JOIN (
                SELECT *
                FROM read_parquet({sql_quote(args.source_glob)})
                WHERE "CHROM" = {sql_quote(chromosome)}
                  AND ({args.row_filter})
            ) AS s
              ON s."CHROM" = k."CHROM" AND s."POS" = k."POS"
             AND s."REF" = k."REF" AND s."ALT" = k."ALT"
            WHERE {match_clause}
        """).pl()
        cursor.unregister(view)
        return chromosome, block

    blocks_by_chrom: dict[str, "pl.DataFrame"] = {}
    with connect_duckdb(threads=args.threads) as conn:
        if args.memory_limit:
            conn.execute(f"SET memory_limit='{args.memory_limit}'")

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(query_chromosome, conn, chromosome): chromosome
                      for chromosome in chromosomes}
            for completed, future in enumerate(as_completed(futures), start=1):
                chromosome, block = future.result()
                blocks_by_chrom[chromosome] = block
                log(f"  [{completed:>2}/{len(chromosomes)}] {chromosome:<6} "
                    f"{keys_by_chrom[chromosome].height:>9,} keys -> "
                    f"{block.height:>9,} matched ({perf_counter() - started:.0f}s)")

    # Reassembled in the original chromosome order (not completion order) so the row-order
    # guarantee downstream -- keys.join(..., how='left').sort('__row') -- does not depend on
    # which query happened to finish first.
    frame = (pl.concat([blocks_by_chrom[c] for c in chromosomes], how="vertical")
             if blocks_by_chrom else pl.DataFrame())

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
