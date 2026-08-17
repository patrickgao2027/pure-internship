#!/usr/bin/env python
"""One PNG that shows -- not just states -- the enrichment merge worked.

``verify_enriched.py`` proves the same facts numerically and is the one that should gate
anything automated: it's exact, and it exits non-zero on failure. This is for eyeballing the
result quickly, and it catches the row-order failure mode in a way that is hard to misread --
a mismatch here isn't a subtle statistic, it's points refusing to sit on the diagonal.

Two panels:

left   row/column counts as bars -- analysis vs enriched should match on rows exactly and
       differ on columns by precisely the recovered count.
right  one shared column's value in analysis.parquet plotted against its value in
       enriched.parquet, row-aligned, for a sample of points. If the merge preserved values
       and row order, every point sits exactly on y = x. Any scatter off that line means rows
       were reordered or altered somewhere between the two files -- which is exactly the
       failure mode that would silently mis-colour every UMAP panel downstream, since that
       plotting code assumes row i in one file is row i in the other.

Usage::

    python umap_hdbscan_sweep/visualize_merge_check.py \\
        --enriched enriched.parquet --analysis <cell>/analysis.parquet \\
        --output merge_check.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enriched", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--check-column", default=None,
                   help="shared numeric column to scatter (default: first shared numeric)")
    p.add_argument("--sample-points", type=int, default=20_000,
                   help="points drawn on the diagonal-check scatter")
    p.add_argument("--output", type=Path, default=Path("merge_check.png"))
    return p.parse_args()


def main() -> int:
    args = parse_args()

    enriched_schema = pq.read_schema(args.enriched)
    analysis_schema = pq.read_schema(args.analysis)
    n_enriched = pq.ParquetFile(args.enriched).metadata.num_rows
    n_analysis = pq.ParquetFile(args.analysis).metadata.num_rows

    shared = [c for c in analysis_schema.names if c in enriched_schema.names]
    recovered = [c for c in enriched_schema.names if c not in analysis_schema.names]

    if args.check_column:
        column = args.check_column
        if column not in shared:
            raise SystemExit(f"{column!r} is not a shared column; choose from {shared}")
    else:
        # POS is excluded from the auto-pick: it is sorted within each chromosome, so a
        # correct merge draws a smooth monotone curve rather than a scattered cloud, which
        # reads as less obviously "on the diagonal" than a feature with no inherent order.
        numeric_types = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64)
        probe = pl.scan_parquet(args.analysis).select(shared).head(1).collect()
        candidates = [c for c in shared if c != "POS" and probe[c].dtype in numeric_types]
        column = candidates[0] if candidates else next(
            (c for c in shared if probe[c].dtype in numeric_types), shared[0])

    limit = min(args.sample_points, n_analysis)
    left = pl.scan_parquet(args.analysis).select(column).head(limit).collect()[column]
    right = pl.scan_parquet(args.enriched).select(column).head(limit).collect()[column]

    fig, (ax_counts, ax_diag) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── left: counts ──────────────────────────────────────────────────────────
    labels = ["rows\n(millions)", "columns"]
    a_vals = [n_analysis / 1e6, len(analysis_schema.names)]
    e_vals = [n_enriched / 1e6, len(enriched_schema.names)]
    x = range(len(labels))
    width = 0.35
    ax_counts.bar([i - width / 2 for i in x], a_vals, width, label="analysis.parquet",
                 color="#4c72b0")
    ax_counts.bar([i + width / 2 for i in x], e_vals, width, label="enriched.parquet",
                 color="#dd8452")
    for i, (a, e) in enumerate(zip(a_vals, e_vals)):
        ax_counts.text(i - width / 2, a, f"{a:.3g}", ha="center", va="bottom", fontsize=9)
        ax_counts.text(i + width / 2, e, f"{e:.3g}", ha="center", va="bottom", fontsize=9)
    ax_counts.set_xticks(list(x))
    ax_counts.set_xticklabels(labels)
    ax_counts.legend()
    match = "MATCH" if n_analysis == n_enriched else "MISMATCH"
    ax_counts.set_title(f"row count: {match}  ·  +{len(recovered)} columns recovered",
                        fontsize=10)

    # ── right: diagonal check ─────────────────────────────────────────────────
    ax_diag.scatter(left.to_numpy(), right.to_numpy(), s=4, alpha=0.3, linewidths=0,
                    color="#4c72b0", rasterized=True)
    lo = float(min(left.min(), right.min()))
    hi = float(max(left.max(), right.max()))
    ax_diag.plot([lo, hi], [lo, hi], color="#c44e52", linewidth=1, linestyle="--",
                label="y = x (expected)")
    on_diagonal = bool((left == right).all())
    ax_diag.set_xlabel(f"{column}  (analysis.parquet)")
    ax_diag.set_ylabel(f"{column}  (enriched.parquet)")
    ax_diag.legend(loc="upper left")
    verdict = "ON DIAGONAL -- values and row order preserved" if on_diagonal \
        else "OFF DIAGONAL -- rows changed or reordered"
    ax_diag.set_title(f"{verdict}\n({limit:,} rows checked)", fontsize=10,
                      color="#2ca02c" if on_diagonal else "#c44e52")

    fig.suptitle(f"{args.enriched.name}  vs  {args.analysis.name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"row count: {n_analysis:,} -> {n_enriched:,}  ({match})")
    print(f"columns:   {len(analysis_schema.names)} -> {len(enriched_schema.names)} "
          f"(+{len(recovered)} recovered)")
    print(f"diagonal check on {column!r}: {'PASS' if on_diagonal else 'FAIL'}")
    print(f"wrote {args.output}")
    return 0 if (n_analysis == n_enriched and on_diagonal) else 1


if __name__ == "__main__":
    raise SystemExit(main())
