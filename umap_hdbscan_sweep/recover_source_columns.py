#!/usr/bin/env python
"""Rebuild a cell's analysis.parquet with **every** source column, not the curated 35.

The problem
-----------
``analysis.parquet`` carries 35 data columns. The source parquets have 70. The missing 35
(QUAL, MAPQ, DP, DP_FILT, RAW_VAF, VAF, AD_A..AD_INS, RPA, RU, STR, X_IC, X_IL, X_PREV2/3,
X_NEXT2/3, RN, DUP, MQUAL, SNVQ, FILT, DP_MAPQ1, ADJ_REF_DIFF, tm, a3, st, et, MI, DS) were
never selected by stage 0, so they exist upstream and nowhere downstream.

Why a ``max()`` join is the wrong way to get them back
-----------------------------------------------------
``cohort_cluster_report.read_source_columns`` recovers five of them with ``max()`` per locus.
That is defensible only for QUAL -- the first ranking key, where the surviving row *is* the
maximum. It is wrong for everything else:

* ``MAPQ``/``DP``/``RAW_VAF`` are ranked *within* the rows tied on QUAL, so the locus maximum
  can come from a read that lost.
* ``VAF``, ``AD_*``, ``DUP``, ``locus``-varying numerics get the maximum over all reads at the
  locus, which is a different read from the one every other column in the row came from.
* ``RU``, ``X_IC``, ``RN``, ``MI``, ``tm``, ``st``, ``et`` are strings -- ``max()`` is
  lexicographic and means nothing at all.

Mixing those into a row whose other 35 columns come from one specific read produces a
chimera: a row that never existed in the data. Colouring a UMAP by it would be plotting an
artefact.

What this does instead
----------------------
Replays stage 0's selection. Same row filter, same ``PARTITION BY (CHROM, POS, REF, ALT)``,
same ``ORDER BY QUAL, MAPQ, DP, RAW_VAF DESC`` followed by the same stable tie-break -- which
is why ``--checkpoint-path`` is required: the tie-break column list is derived from the
checkpoint's feature report exactly as ``stage0_dedup`` derives it, so the row that wins here
is the row that won there. Every column then comes off that one surviving read.

Restricted to the sampled loci by a semi-join, so this is one pass that materialises ~2M rows
rather than re-deduplicating all 157.5M.

One run covers every cell
-------------------------
``cohort_cluster_report.main`` draws ``positions`` once and passes the same array to every
cell, and the UMAP coordinates come from one ``coords.npy``. So all 28 cells' analysis.parquet
files are identical row-for-row except ``cluster_label``/``cluster_probability`` -- verified
by direct comparison. Run this once, plot once; the feature panels are the same for all of
them.

Usage::

    python umap_hdbscan_sweep/recover_source_columns.py \\
        --analysis <any-cell>/analysis.parquet \\
        --source-glob '/data/lab/ppmseq_parquets/*.parquet' \\
        --checkpoint-path <the model.pt stage 0 was run with> \\
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
from uv_vae.pipeline_vae import build_selected_columns, load_feature_names, stable_order_by

# Imported rather than restated: if stage 0's notion of the filter or the identity/context
# columns ever changes, this must change with it or the replay silently diverges.
from stage0_dedup import CONTEXT_COLUMNS, DEFAULT_ROW_FILTER, IDENTITY_COLUMNS

# Carried from the analysis file, not recovered from source -- these are pipeline outputs.
CARRY_FROM_ANALYSIS = ["umap_1", "umap_2", "cluster_label", "cluster_probability"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis", type=Path, required=True,
                   help="any cell's analysis.parquet -- supplies the loci and the row order")
    p.add_argument("--source-glob", required=True,
                   help="the 95 source parquets, e.g. '/data/lab/ppmseq_parquets/*.parquet'")
    p.add_argument("--checkpoint-path", required=True,
                   help="the model.pt stage 0 used; pins the tie-break column list")
    p.add_argument("--row-filter", default=DEFAULT_ROW_FILTER)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--memory-limit", default=None, help="e.g. 64GB")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = perf_counter()

    keys = pl.read_parquet(args.analysis, columns=IDENTITY_COLUMNS + CARRY_FROM_ANALYSIS)
    # An explicit index because the LEFT JOIN below does not preserve input order and the
    # positional join downstream (latent row i == plot row i) depends on it absolutely.
    keys = keys.with_row_index("__row")
    log(f"{args.analysis.name}: {keys.height:,} loci")

    source_columns = list(pq.read_schema(_first_source(args.source_glob)).names)
    available = set(source_columns)
    log(f"source schema: {len(source_columns)} columns")

    # Identical derivation to stage0_dedup.main, so the same read survives each locus.
    wanted = build_selected_columns(load_feature_names(args.checkpoint_path))
    tie_break_columns = list(dict.fromkeys(
        name for name in [*wanted, *IDENTITY_COLUMNS, *CONTEXT_COLUMNS] if name in available
    ))
    tie_break = stable_order_by(tie_break_columns)
    log(f"tie-break over {len(tie_break_columns)} columns (from the checkpoint)")

    recovered = [c for c in source_columns if c not in IDENTITY_COLUMNS]
    select_list = ", ".join(quote_ident(name) for name in source_columns)

    with connect_duckdb(threads=args.threads) as conn:
        if args.memory_limit:
            conn.execute(f"SET memory_limit='{args.memory_limit}'")
        conn.register("sampled_keys", keys.select(IDENTITY_COLUMNS))

        log("scanning source parquets (one pass, semi-joined to the sampled loci)")
        frame = conn.execute(f"""
            WITH ranked AS (
                SELECT
                    {select_list},
                    ROW_NUMBER() OVER (
                        PARTITION BY "CHROM", "POS", "REF", "ALT"
                        ORDER BY
                            "QUAL" DESC NULLS LAST,
                            "MAPQ" DESC NULLS LAST,
                            "DP" DESC NULLS LAST,
                            "RAW_VAF" DESC NULLS LAST,
                            {tie_break}
                    ) AS rn
                FROM read_parquet({sql_quote(args.source_glob)}) AS s
                SEMI JOIN sampled_keys AS k
                  ON s."CHROM" = k."CHROM" AND s."POS" = k."POS"
                 AND s."REF" = k."REF" AND s."ALT" = k."ALT"
                WHERE ({args.row_filter})
            )
            SELECT {select_list} FROM ranked WHERE rn = 1
        """).pl()

    log(f"recovered {frame.height:,} loci x {frame.width} columns")
    if frame.height != keys.height:
        # Not fatal on its own -- a locus can drop out if the filter excludes every read at
        # it -- but it means some rows will be all-null, so say so rather than hide it.
        log(f"WARNING: {keys.height - frame.height:,} loci had no surviving row; "
            "those rows will be null in the output")

    merged = (keys.join(frame, on=IDENTITY_COLUMNS, how="left")
                  .sort("__row")
                  .drop("__row"))
    if merged.height != keys.height:
        raise SystemExit(
            f"join produced {merged.height:,} rows from {keys.height:,} -- "
            "(CHROM, POS, REF, ALT) is not unique in the recovered frame, which breaks the "
            "one-row-per-locus premise"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(args.output, compression="zstd")
    log(f"wrote {args.output}  ({merged.height:,} rows x {merged.width} columns, "
        f"{args.output.stat().st_size / 1e6:.0f} MB, {perf_counter() - started:.0f}s)")
    log(f"recovered columns: {', '.join(recovered)}")
    return 0


def _first_source(source_glob: str) -> str:
    import glob as globmod

    matches = sorted(globmod.glob(source_glob))
    if not matches:
        raise SystemExit(f"no files match {source_glob}")
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
