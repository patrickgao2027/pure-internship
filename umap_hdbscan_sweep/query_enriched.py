#!/usr/bin/env python
"""Run SQL against the enriched-views database without the duckdb CLI.

miletus has the duckdb *Python package* but not the ``duckdb`` command, so the usual
``duckdb file.duckdb -c "SELECT ..."`` is unavailable. This is that, in the form that is
actually installed: pass SQL, get a table.

    python umap_hdbscan_sweep/query_enriched.py "SELECT * FROM sample_index LIMIT 5"

With no SQL it prints what the database contains, which is the thing you want first and
cannot remember the view names for.

Results stream to the terminal by default and are truncated at ``--limit`` rows so a query
that forgot its own LIMIT cannot fill the scrollback with five billion rows. ``--csv`` and
``--parquet`` write the *whole* result to a file instead, with no truncation.

The connection is read-only. These views describe the join between the source parquets and
the assignment files; nothing here should ever write to either.

Usage::

    # what is in here?
    python umap_hdbscan_sweep/query_enriched.py

    # a query
    python umap_hdbscan_sweep/query_enriched.py \\
        "SELECT cluster_label, count(*) FROM csb0_1_ppm0058 GROUP BY 1 ORDER BY 2 DESC"

    # from a file, saving the full result
    python umap_hdbscan_sweep/query_enriched.py -f cluster_report.sql --parquet out.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

DEFAULT_DATABASE = (Path.home() / "pure-internship" / "umap_hdbscan_sweep" / "enriched.duckdb")

OVERVIEW = """
SELECT view_name, labelled_rows, round(noise_pct, 3) AS noise_pct, source
FROM sample_index ORDER BY view_name
"""


def render(columns: list[str], rows: list[tuple], truncated: bool, limit: int) -> None:
    """Print a result set as an aligned table.

    Column widths come from the data actually being shown, so a wide column that happens to
    be narrow in these rows does not pad every line to its declared type width.
    """
    if not rows:
        print("(no rows)")
        return
    cells = [[("NULL" if value is None else
               f"{value:.6g}" if isinstance(value, float) else str(value))
              for value in row] for row in rows]
    widths = [max(len(str(columns[i])), max((len(row[i]) for row in cells), default=0))
              for i in range(len(columns))]
    widths = [min(w, 40) for w in widths]

    def line(values: list[str]) -> str:
        return "  ".join(v[:w].ljust(w) for v, w in zip(values, widths))

    print(line([str(c) for c in columns]))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print(line(row))
    print()
    note = f"{len(rows):,} rows"
    if truncated:
        note += (f" (truncated at --limit {limit:,}; use --csv/--parquet to write the full "
                 f"result, or add your own LIMIT)")
    print(note)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sql", nargs="?", default=None,
                        help="SQL to run; omit to describe the database")
    parser.add_argument("-f", "--file", type=Path, default=None,
                        help="read SQL from a file instead ('-' for stdin)")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--limit", type=int, default=50,
                        help="max rows to print (default 50); does not apply to --csv/--parquet")
    parser.add_argument("--csv", type=Path, default=None, help="write the full result here")
    parser.add_argument("--parquet", type=Path, default=None, help="write the full result here")
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--temp-dir", default=None,
                        help="duckdb spill directory. The default is on the root partition, "
                             "which is nearly full on miletus -- point this at /data/lab for "
                             "anything that scans all_samples.")
    args = parser.parse_args()

    import duckdb

    if not args.database.exists():
        print(f"ERROR: no database at {args.database}", file=sys.stderr)
        print("  Build it with umap_hdbscan_sweep/build_enriched_views.py", file=sys.stderr)
        return 1

    if args.file:
        sql = sys.stdin.read() if str(args.file) == "-" else args.file.read_text()
    elif args.sql:
        sql = args.sql
    else:
        sql = OVERVIEW

    # read_only is not merely tidy: these views point at the lab's source parquets, and a
    # stray DDL typo against a shared dataset is not a recoverable mistake.
    connection = duckdb.connect(str(args.database), read_only=True)
    connection.execute(f"SET memory_limit='{args.memory_limit}'")
    connection.execute(f"SET threads={args.threads}")
    if args.temp_dir:
        connection.execute(f"SET temp_directory='{args.temp_dir}'")

    started = perf_counter()
    try:
        if args.csv or args.parquet:
            target = args.csv or args.parquet
            fmt = "CSV, HEADER" if args.csv else "PARQUET"
            connection.execute(f"COPY ({sql}) TO '{target}' (FORMAT {fmt})")
            elapsed = perf_counter() - started
            size_mb = target.stat().st_size / 1e6
            print(f"wrote {target}  ({size_mb:,.1f} MB, {elapsed:.1f}s)")
            return 0

        relation = connection.sql(sql)
        if relation is None:                       # a statement that returns nothing
            print(f"ok ({perf_counter() - started:.1f}s)")
            return 0
        columns = relation.columns
        rows = relation.limit(args.limit + 1).fetchall()
    except Exception as exc:                       # noqa: BLE001
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        elapsed = perf_counter() - started

    truncated = len(rows) > args.limit
    render(columns, rows[:args.limit], truncated, args.limit)
    print(f"{elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
