#!/usr/bin/env python
"""Attach ``source_file`` to every row of the stage 0 dedup set, once.

Stage 0 collapses ~32 reads per locus down to one and writes
``(CHROM, POS, REF, ALT)``-unique rows per chromosome. It does **not** record which of
the 95 source parquets the surviving read came from -- that column simply is not in the
output schema. Any question of the form "where do sample X's mutations land" therefore
cannot be answered from the dedup set alone, and neither can stratified-by-sample
sampling, which needs the group label before it can draw equal counts per group.

This script produces that label and nothing else: ``CHROM, POS, REF, ALT, source_file``,
one row per dedup row, written per chromosome. It is a few GB rather than the ~30 GB a
full re-materialisation of the dedup rows would cost, and because ``(CHROM, POS, REF,
ALT)`` is unique in a global dedup, it joins back to the dedup parquets on the identity
columns alone -- no fingerprint needed downstream.

How the label is recovered
--------------------------
The same fingerprint join ``recover_source_columns.py`` uses, and for the same reason.
The alternatives both fail:

* ``max()`` per locus mixes columns across reads and is meaningless on the string ones.
* Replaying stage 0's ``ORDER BY QUAL, MAPQ, DP, RAW_VAF`` window rests entirely on
  reproducing a sort whose tie-break is a column list derived from the training
  checkpoint. Ties on all four ranking keys are common at ~32 reads per locus, so a
  wrong tie-break list silently attributes loci to the wrong sample. No error, just
  wrong labels -- which is exactly the thing this output is for.

Instead the dedup row is *identified* rather than re-derived. The ~37 read-level
measurement columns the dedup set shares with the source (rq, BCSQ, SMQ_BEFORE/AFTER,
EDIST, RL, INDEX, l1..l7, q2..q6, HAMDIST, the X_* context columns, ...) were copied
through stage 0 without arithmetic, so they compare bit-for-bit and together pin down
the exact source read. Where two reads match on all of them they are indistinguishable
in every column the pipeline ever used, so either serves; the script counts those and
keeps one deterministically.

Cost: one pass over the cohort (~5.08 B rows), chromosome by chromosome, with row-group
pruning on CHROM. Run it once -- the output is reusable by every downstream sampler.

Usage::

    python umap_hdbscan_sweep/label_dedup_source.py \\
        --dedup-glob '~/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/stage0_dedup/dedup/*.parquet' \\
        --source-glob '/data/lab/ppmseq_parquets/*.parquet' \\
        --output-dir ~/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/stage0_dedup/source_labels
"""
from __future__ import annotations

import argparse
import glob as globmod
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for _c in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import pyarrow.parquet as pq

from uv_vae.data import connect_duckdb, quote_ident

DEFAULT_ROW_FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"
IDENTITY_COLUMNS = ["CHROM", "POS", "REF", "ALT"]
# Written by stage 0, absent from the source, and not a read measurement -- must not join.
DERIVED_COLUMNS = ["locus_reads"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dedup-glob", required=True,
                   help="stage 0 output, e.g. '<run>/stage0_dedup/dedup/*.parquet'")
    p.add_argument("--source-glob", required=True,
                   help="the 95 source parquets")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--row-filter", default=DEFAULT_ROW_FILTER,
                   help="must match the filter stage 0 ran with")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--parallel-chromosomes", type=int, default=4,
                   help="chromosome queries in flight at once; each is I/O bound on the "
                        "source filesystem, so overlapping a few keeps more reads outstanding")
    p.add_argument("--memory-limit", default="200GB")
    p.add_argument("--temp-directory", default=None,
                   help="DuckDB spill directory; needs room for the join's hash tables")
    p.add_argument("--overwrite", action="store_true",
                   help="re-run chromosomes whose output already exists (default: skip)")
    return p.parse_args()


def chromosome_of(path: Path) -> str:
    """'chrom=chr5.parquet' -> 'chr5'; falls back to the stem."""
    m = re.search(r"chrom=([^/\\.]+)", path.name)
    return m.group(1) if m else path.stem


def main() -> int:
    args = parse_args()
    started = perf_counter()

    dedup_files = sorted(Path(p) for p in globmod.glob(str(Path(args.dedup_glob).expanduser())))
    if not dedup_files:
        raise SystemExit(f"no files match {args.dedup_glob}")
    source_files = sorted(globmod.glob(args.source_glob))
    if not source_files:
        raise SystemExit(f"no files match {args.source_glob}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"{len(dedup_files)} dedup chromosomes, {len(source_files)} source parquets")

    dedup_columns  = list(pq.read_schema(dedup_files[0]).names)
    source_columns = set(pq.read_schema(source_files[0]).names)

    excluded = set(DERIVED_COLUMNS) | set(IDENTITY_COLUMNS)
    measurements = [c for c in dedup_columns if c in source_columns and c not in excluded]

    missing_identity = [c for c in IDENTITY_COLUMNS if c not in dedup_columns]
    if missing_identity:
        raise SystemExit(f"dedup output lacks identity columns {missing_identity}")
    if not measurements:
        raise SystemExit("no read-level measurements shared with the source -- the identity "
                         "columns alone name the locus, not the read")

    log(f"fingerprint: {len(IDENTITY_COLUMNS)} identity + {len(measurements)} measurements")

    # Identity is a plain equi-join the planner hashes; the measurements are applied
    # afterwards as a filter. Putting all ~41 columns in the ON clause -- and they must be
    # null-safe, since SQL equality is unknown against a null and would drop every row with
    # a null measurement -- gives a condition the planner cannot hash on, degrading to a
    # nested loop that never finishes against 5.08 B source rows.
    match_clause = " AND ".join(
        f"s.{quote_ident(c)} IS NOT DISTINCT FROM d.{quote_ident(c)}" for c in measurements
    )
    source_list = "[" + ", ".join(sql_quote(p) for p in source_files) + "]"

    pending = []
    for path in dedup_files:
        chromosome = chromosome_of(path)
        out_path = args.output_dir / f"chrom={chromosome}.parquet"
        if out_path.exists() and not args.overwrite:
            log(f"  {chromosome:<6} already labelled, skipping (use --overwrite to redo)")
            continue
        pending.append((chromosome, path, out_path))

    if not pending:
        log("every chromosome already labelled; nothing to do")
        return 0

    local = threading.local()

    def label_chromosome(conn, chromosome: str, dedup_path: Path, out_path: Path):
        if not hasattr(local, "cursor"):
            local.cursor = conn.cursor()
        cursor = local.cursor

        # CHROM and the row filter go inside the scanned subquery rather than an outer
        # WHERE: a post-join predicate is only pushed into the parquet scan if the
        # optimiser chooses to hoist it, which is not guaranteed. Filtering here makes the
        # row-group pruning unconditional instead of a planner guess -- the difference
        # between reading ~1/24th of the cohort and reading all 5.08 B rows.
        #
        # QUALIFY keeps one row per dedup locus. Two source reads matching on all
        # measurements are interchangeable, so ordering by the extracted name makes the
        # choice deterministic rather than dependent on scan order.
        cursor.execute(f"""
            COPY (
                SELECT d."CHROM", d."POS", d."REF", d."ALT", s.source_file
                FROM read_parquet({sql_quote(str(dedup_path))}) AS d
                JOIN (
                    SELECT *,
                           regexp_extract(filename, '([^/\\\\]+)\\.parquet$', 1) AS source_file
                    FROM read_parquet({source_list}, filename=true)
                    WHERE "CHROM" = {sql_quote(chromosome)}
                      AND ({args.row_filter})
                ) AS s
                  ON s."CHROM" = d."CHROM" AND s."POS" = d."POS"
                 AND s."REF"   = d."REF"   AND s."ALT" = d."ALT"
                WHERE {match_clause}
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY d."CHROM", d."POS", d."REF", d."ALT"
                    ORDER BY s.source_file
                ) = 1
            ) TO {sql_quote(str(out_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        labelled = cursor.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_quote(str(out_path))})").fetchone()[0]
        total = cursor.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_quote(str(dedup_path))})").fetchone()[0]
        return chromosome, int(labelled), int(total)

    temp_directory = (Path(args.temp_directory) if args.temp_directory
                      else args.output_dir / "_duckdb_tmp")
    temp_directory.mkdir(parents=True, exist_ok=True)

    total_labelled = total_rows = 0
    with connect_duckdb(threads=args.threads, temp_directory=temp_directory,
                        memory_limit=args.memory_limit) as conn:
        parallel = max(1, min(args.parallel_chromosomes, len(pending)))
        log(f"labelling {len(pending)} chromosomes, {parallel} at a time")

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(label_chromosome, conn, c, dp, op): c
                       for c, dp, op in pending}
            for done, fut in enumerate(as_completed(futures), start=1):
                chromosome, labelled, total = fut.result()
                total_labelled += labelled
                total_rows     += total
                pct = 100 * labelled / total if total else 0.0
                flag = "" if labelled == total else "   <-- UNMATCHED ROWS"
                log(f"  [{done:>2}/{len(pending)}] {chromosome:<6} "
                    f"{labelled:>11,} / {total:>11,} ({pct:.4f}%) "
                    f"({perf_counter() - started:.0f}s){flag}")

    log(f"\nlabelled {total_labelled:,} of {total_rows:,} dedup rows "
        f"({100 * total_labelled / total_rows if total_rows else 0:.4f}%) "
        f"in {perf_counter() - started:.0f}s")
    if total_labelled < total_rows:
        log(f"  WARNING: {total_rows - total_labelled:,} dedup rows found no source row. "
            f"Check --row-filter matches what stage 0 ran with (currently: {args.row_filter})")
    log(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
