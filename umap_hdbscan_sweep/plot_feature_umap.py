#!/usr/bin/env python
"""Colour a UMAP scatter by each raw numeric VAE input feature, one panel per feature.

Answers "does the UMAP layout just reflect one input feature" -- pick every numeric column
that actually went into the VAE (via ``ml_features.json``, restricted to the columns present
in the raw parquet -- 14 of the spec's numerics are 100% null and get dropped upstream) and
paint the UMAP embedding by that column's value. A feature whose colour forms a clean gradient
or a hard block boundary across the map is one the embedding is organised around; a feature
that comes out as salt-and-pepper noise is not visible to UMAP at all.

Usage (local, on a finished clust_regime_sweep variant)::

    python umap_hdbscan_sweep/plot_feature_umap.py \
        --analysis clust_regime_sweep/run1/vae_5pct/variant_A/analysis.parquet \
        --output feature_umap_plots

``--raw-parquet`` defaults to ``<analysis>/../../sample.parquet`` -- clust_regime_sweep's
layout, where ``<cell>/sample.parquet`` holds the raw (pre-VAE) features and
``<cell>/<variant>/analysis.parquet`` holds ``umap_1``/``umap_2``
(see ``clustering_regime_sweep.main``: both are sliced from the same in-memory frame with no
reordering in between). The two files are therefore joined **positionally by row index**, not
by a CHROM/POS/REF/ALT key -- cheap, and sidesteps null/dtype edge cases a real join would hit.
A row-count mismatch is treated as a hard error rather than silently truncating.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from uv_vae.features import load_feature_specs

DEFAULT_FEATURE_SPEC = REPO_ROOT / "uv_vae" / "ml_features.json"
NOISE_COLOUR = "#cccccc"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis", type=Path, required=True,
                   help="analysis.parquet with umap_1/umap_2 columns")
    p.add_argument("--raw-parquet", type=Path, default=None,
                   help="parquet with raw features, same row order as --analysis "
                        "(default: <analysis>/../../sample.parquet)")
    p.add_argument("--feature-spec-path", type=Path, default=DEFAULT_FEATURE_SPEC)
    p.add_argument("--features", nargs="*", default=None,
                   help="subset of numeric feature names to plot (default: all numeric "
                        "features from the spec that are present in --raw-parquet)")
    p.add_argument("--sample-rows", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--point-size", type=float, default=1.5)
    p.add_argument("--clip-percentile", type=float, default=1.0,
                   help="colour scale is clipped to [p, 100-p] percentiles of each "
                        "feature's sampled values, so a few outliers don't wash out the "
                        "gradient over the rest of the points")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--output", type=Path, default=Path("feature_umap_plots"),
                   help="output directory (created if needed); one PNG per feature")
    return p.parse_args()


def default_raw_parquet(analysis_path: Path) -> Path:
    return analysis_path.resolve().parent.parent / "sample.parquet"


def numeric_feature_names(spec_path: Path, available_columns: set[str],
                          requested: list[str] | None) -> list[str]:
    specs = load_feature_specs(spec_path)
    numeric = [s.name for s in specs if s.is_numeric and s.name in available_columns]
    if requested is None:
        return numeric
    missing = [name for name in requested if name not in numeric]
    if missing:
        raise SystemExit(f"not numeric / not present: {missing} (available: {numeric})")
    return requested


def sample_positions(total: int, wanted: int, seed: int) -> np.ndarray:
    if wanted >= total:
        return np.arange(total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=wanted, replace=False))


def main() -> int:
    args = parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import polars as pl

    analysis_path = args.analysis.resolve()
    raw_path = (args.raw_parquet or default_raw_parquet(analysis_path)).resolve()
    if not raw_path.exists():
        raise SystemExit(f"raw parquet not found: {raw_path} (pass --raw-parquet explicitly)")

    n_analysis = pl.scan_parquet(analysis_path).select(pl.len()).collect().item()
    n_raw = pl.scan_parquet(raw_path).select(pl.len()).collect().item()
    if n_analysis != n_raw:
        raise SystemExit(
            f"row count mismatch: {analysis_path.name} has {n_analysis:,} rows, "
            f"{raw_path.name} has {n_raw:,} -- positional join is unsafe, pick a matching pair"
        )

    raw_columns = set(pl.scan_parquet(raw_path).collect_schema().names())
    features = numeric_feature_names(args.feature_spec_path, raw_columns, args.features)
    if not features:
        raise SystemExit("no numeric features found in the raw parquet")
    print(f"Plotting {len(features)} numeric features: {', '.join(features)}")

    positions = sample_positions(n_analysis, args.sample_rows, args.seed)
    print(f"{n_analysis:,} rows total; sampling {positions.size:,}")

    umap_xy = (
        pl.scan_parquet(analysis_path)
        .select("umap_1", "umap_2")
        .collect()
        .to_numpy()[positions]
    )
    raw_values = (
        pl.scan_parquet(raw_path)
        .select([pl.col(name).cast(pl.Float32) for name in features])
        .collect()
        .to_numpy()[positions]
    )

    args.output.mkdir(parents=True, exist_ok=True)

    for col_index, feature in enumerate(features):
        values = raw_values[:, col_index]
        valid = np.isfinite(values)
        if not valid.any():
            print(f"  {feature}: all null in the sample, skipping")
            continue

        lo, hi = np.percentile(values[valid], [args.clip_percentile, 100 - args.clip_percentile])
        if lo == hi:
            lo, hi = float(values[valid].min()), float(values[valid].max())
            if lo == hi:
                hi = lo + 1.0

        fig, ax = plt.subplots(figsize=(9, 8))

        if (~valid).any():
            ax.scatter(umap_xy[~valid, 0], umap_xy[~valid, 1], s=args.point_size * 0.6,
                       c=NOISE_COLOUR, alpha=0.3, linewidths=0, rasterized=True,
                       label=f"null ({(~valid).sum():,})")

        scatter = ax.scatter(umap_xy[valid, 0], umap_xy[valid, 1], s=args.point_size,
                             c=values[valid], cmap=args.cmap, vmin=lo, vmax=hi,
                             alpha=0.6, linewidths=0, rasterized=True)
        fig.colorbar(scatter, ax=ax, label=feature, fraction=0.046, pad=0.04)

        ax.set_title(f"UMAP coloured by {feature}\n"
                     f"{positions.size:,} points shown "
                     f"(colour clipped to [{args.clip_percentile:g}, "
                     f"{100 - args.clip_percentile:g}] pct: [{lo:.3g}, {hi:.3g}])",
                     fontsize=10)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        out_path = args.output / f"{feature}_umap.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  {feature:16} range [{values[valid].min():.3g}, {values[valid].max():.3g}] "
              f"-> {out_path.name}")

    print(f"saved {len(features)} images -> {args.output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
