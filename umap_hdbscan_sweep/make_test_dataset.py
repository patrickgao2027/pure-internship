"""Build the test dataset for the clustering regime: a row sample of the filtered parquet.

Purpose: a smaller stand-in for the full featuremap that exercises the WHOLE pipeline --
dedup, VAE inference, UMAP, HDBSCAN, the metric panel, and SigProfiler -- so the regime
can be validated cheaply before running at full scale. Point SOURCE_PARQUET at the
output and nothing else changes.

Method: DuckDB reservoir sample streamed straight to parquet (no full materialization in
Python), single-threaded because REPEATABLE is only deterministic that way. No dedup --
this is a row sample just like the source, and the regime dedups per cell later.

Every column is carried through. That is a hard requirement, not a preference: dedup
partitions on CHROM/POS/REF/ALT, and SigProfiler builds trinucleotide contexts from REF
with X_PREV1/X_NEXT1 as the flanking bases. Dropping them silently degrades dedup and
leaves SigProfiler with nothing to read, so the script refuses to write a file missing
them (--required-columns to override).

    python scripts/make_test_dataset.py \
        --parquet-path /cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet \
        --output-path /cta/users/patrickgao765/parquet_files/test7M.featuremap.parquet \
        --sample-rows 7000000
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow.parquet as pq

from uv_vae.data import connect_duckdb, get_row_count

DEFAULT_FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"

# Without these the pipeline still runs but silently produces garbage: dedup loses its
# partition key and SigProfiler cannot build a trinucleotide context.
REQUIRED_COLUMNS = ["CHROM", "POS", "REF", "ALT", "X_PREV1", "X_NEXT1"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a test dataset from the filtered parquet")
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--sample-rows", type=int, default=7_000_000)
    parser.add_argument("--row-filter", default=DEFAULT_FILTER)
    parser.add_argument("--seed", type=int, default=99,
                        help="test-set draw seed; kept separate from the pipeline seed (42)")
    parser.add_argument("--required-columns", default=",".join(REQUIRED_COLUMNS),
                        help="comma-separated; empty string disables the check")
    parser.add_argument("--memory-limit", default="32GB")
    parser.add_argument("--temp-directory", default=None,
                        help="DuckDB spill directory; defaults to <output>.duckdb_tmp")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
        log(f"Test dataset already exists: {output_path} (pass --overwrite to rebuild)")
        return 0
    # A zero-byte leftover from a failed COPY would otherwise look like a valid file.
    if output_path.exists():
        output_path.unlink()

    available = set(pq.read_schema(args.parquet_path).names)
    required = [name for name in args.required_columns.split(",") if name.strip()]
    missing = [name for name in required if name not in available]
    if missing:
        raise SystemExit(
            f"Source parquet is missing columns the pipeline needs: {missing}. "
            f"Dedup needs CHROM/POS/REF/ALT; SigProfiler needs REF/X_PREV1/X_NEXT1."
        )
    log(f"Carrying through all {len(available)} columns; required columns present")

    temp_directory = Path(args.temp_directory) if args.temp_directory else \
        output_path.parent / f".{output_path.stem}.duckdb_tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    # threads=1 -- DuckDB's REPEATABLE sampling is only deterministic single-threaded.
    with connect_duckdb(
        threads=1,
        temp_directory=temp_directory,
        memory_limit=args.memory_limit,
    ) as conn:
        eligible = get_row_count(conn, args.parquet_path, where=args.row_filter)
        log(f"Filter: {args.row_filter}")
        if eligible == 0:
            raise SystemExit(f"No rows pass the filter: {args.row_filter}")
        take = min(args.sample_rows, eligible)
        log(f"{eligible:,} rows pass the filter; taking {take:,}"
            + (" (all of them -- requested more than eligible)" if take < args.sample_rows else ""))

        # COPY streams to parquet instead of materializing the sample in Python.
        conn.execute(f"""
            COPY (
                SELECT * FROM (
                    SELECT * FROM read_parquet({sql_quote(str(args.parquet_path))})
                    WHERE {args.row_filter}
                ) AS filtered_rows
                USING SAMPLE reservoir({int(take)} ROWS) REPEATABLE ({int(args.seed)})
            ) TO {sql_quote(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"COPY produced no output at {output_path}")

    written = pq.read_metadata(output_path)
    if written.num_rows == 0:
        output_path.unlink()
        raise RuntimeError("Wrote a parquet with 0 rows; removed it")

    log(f"Wrote {written.num_rows:,} rows x {written.num_columns} columns to {output_path} "
        f"({output_path.stat().st_size / 1e9:.2f} GB) in {perf_counter() - started:.1f}s")
    if written.num_rows != take:
        log(f"WARNING: expected {take:,} rows, got {written.num_rows:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
