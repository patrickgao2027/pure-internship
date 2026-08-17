#!/usr/bin/env python
"""Colour a UMAP embedding by each raw numeric VAE input feature, one PNG per feature.

The question: *is the embedding organised around any single input feature?* A feature whose
colour forms a clean gradient or a hard boundary is one UMAP's geometry tracks; a feature that
comes out as salt-and-pepper inside every cluster is invisible to it. Answering that needs
2-D coordinates and raw feature values for the same rows, and this repo stores those two
things in two different layouts -- which is the only reason this script has two input modes.

Mode A -- cohort (``--embed-summary``)
    The 95-file pipeline. Coordinates come from a parametric encoder applied to sampled
    latent vectors (``--encoder``), or from a saved ``coords.npy`` (``--coords-npy``, faster
    when it exists). Feature values come from stage 0's chromosome-split dedup parquets.

Mode B -- single-run (``--analysis``)
    A ``clust_regime_sweep`` variant. Coordinates are the ``umap_1``/``umap_2`` columns
    already in ``analysis.parquet``; feature values come from the cell's ``sample.parquet``
    (``--raw-parquet``, defaulting to ``<analysis>/../../sample.parquet``).

**Both modes join positionally, by row index, and neither needs a key.** That is a property
of how the artefacts are written, not an assumption:

* Mode A: ``stage1_embed`` consumes the dedup parts in the order recorded in
  ``embed_summary.json``'s manifest and appends to ``latent.npy`` as it goes, so row *i* of
  the array is row *i* of the concatenated parts.
* Mode B: ``clustering_regime_sweep.main`` slices both the context columns and the clustered
  frame out of one in-memory frame with no reordering in between.

A row-count mismatch is therefore a hard error, not something to truncate past -- if the
counts disagree the premise is broken and every colour would be attached to the wrong point.

Usage::

    # cohort, via the settled encoder
    python umap_hdbscan_sweep/plot_feature_umap_cohort.py \\
        --embed-summary <stage1>/embed_summary.json \\
        --encoder umap_hdbscan_sweep/umap_tests/final_models/13_BEST_25M_nn15_md0.1_umap.pt \\
        --output plots/feature_umap_cohort

    # cohort, reusing coordinates the sweep already wrote
    python umap_hdbscan_sweep/plot_feature_umap_cohort.py \\
        --embed-summary <stage1>/embed_summary.json \\
        --coords-npy umap_hdbscan_sweep/umap_tests/hdbscan_scaling/coords.npy \\
        --output plots/feature_umap_cohort

    # a single clust_regime_sweep variant
    python umap_hdbscan_sweep/plot_feature_umap_cohort.py \\
        --analysis clust_regime_sweep/run1/vae_5pct/variant_A/analysis.parquet \\
        --output plots/feature_umap_vae5pct

Numerics only, by design -- categoricals need a discrete palette and a legend rather than a
colourbar. ``plot_feature_atlas.py`` renders both, scores every feature against
``cluster_label``, and supersedes this script wherever cluster labels are available; this one
remains the smaller tool for when they are not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from uv_vae.features import load_feature_specs

DEFAULT_FEATURE_SPEC = REPO_ROOT / "uv_vae" / "ml_features.json"
NULL_COLOUR = "#cccccc"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    mode_a = p.add_argument_group("mode A -- cohort")
    mode_a.add_argument("--embed-summary", type=Path, default=None,
                        help="embed_summary.json from stage1_embed.py (latent.npy + manifest)")
    mode_a.add_argument("--encoder", type=Path, default=None,
                        help="saved ParametricEncoder .pt from final_models/")
    mode_a.add_argument("--coords-npy", type=Path, default=None,
                        help="pre-computed (N, 2) coords; skips running the encoder")
    mode_a.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    mode_a.add_argument("--hidden", default="256,256,128")

    mode_b = p.add_argument_group("mode B -- single clust_regime_sweep run")
    mode_b.add_argument("--analysis", type=Path, default=None,
                        help="analysis.parquet carrying umap_1/umap_2")
    mode_b.add_argument("--raw-parquet", type=Path, default=None,
                        help="raw features in the same row order "
                             "(default: <analysis>/../../sample.parquet)")

    p.add_argument("--feature-spec-path", type=Path, default=DEFAULT_FEATURE_SPEC)
    p.add_argument("--features", nargs="*", default=None,
                   help="subset of numeric features (default: every numeric feature in the "
                        "spec that is present in the data)")
    p.add_argument("--sample-rows", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--point-size", type=float, default=1.5)
    p.add_argument("--clip-percentile", type=float, default=1.0,
                   help="colour scale clipped to [p, 100-p] percentiles per feature, so a "
                        "long tail cannot flatten the gradient over the bulk of the points")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--output", type=Path, default=Path("feature_umap"),
                   help="output directory; one PNG per feature")
    return p.parse_args()


def sample_positions(total: int, wanted: int, seed: int) -> np.ndarray:
    """Sorted, so the gathers below stay mostly sequential over a memmap or a parquet."""
    if wanted >= total:
        return np.arange(total)
    return np.sort(np.random.default_rng(seed).choice(total, size=wanted, replace=False))


def numeric_feature_names(spec_path: Path, available: set[str],
                          requested: list[str] | None) -> list[str]:
    """Numeric features from the spec that are actually present.

    Derived from the spec rather than hard-coded because 14 of its numerics are 100% null in
    this cohort and get dropped upstream; which ones survive is a property of the data.
    """
    specs = load_feature_specs(spec_path)
    numeric = [s.name for s in specs if s.is_numeric and s.name in available]
    if requested is None:
        return numeric
    missing = [name for name in requested if name not in numeric]
    if missing:
        raise SystemExit(f"not numeric / not present: {missing}\navailable: {numeric}")
    return requested


# ── mode A: cohort ─────────────────────────────────────────────────────────────

def encoder_transform(encoder_path: Path, latent_sample: np.ndarray,
                      hidden: tuple[int, ...], device_name: str) -> np.ndarray:
    """Run sampled latents through a saved ParametricEncoder -> (N, 2) float32."""
    import torch

    import parametric_umap as pu

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto"
        else device_name
    )
    blob = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder = pu.ParametricEncoder(latent_sample.shape[1], output_dim=2,
                                   hidden=hidden).to(device)
    encoder.load_state_dict(blob["state_dict"])
    encoder.eval()
    mode = blob.get("mode", "umap")
    print(f"  encoder {encoder_path.name} (mode={mode}) on {device}")
    return pu.ParametricUmap(encoder=encoder, device=device, mode=mode).transform(
        latent_sample.astype(np.float32, copy=False))


def read_dedup_features(manifest_path: Path, positions: np.ndarray,
                        names: list[str]) -> np.ndarray:
    """Gather feature values at the given cohort row positions from the dedup parts.

    Positions map to (part, offset) by a running sum of each part's ``loci`` count, and only
    the parts a sample actually touches are opened.
    """
    import polars as pl

    parts = json.loads(manifest_path.read_text())["per_chromosome"]
    bounds = np.cumsum([0] + [int(entry["loci"]) for entry in parts])
    if positions[-1] >= bounds[-1]:
        raise SystemExit(f"position {positions[-1]:,} beyond {bounds[-1]:,} dedup rows")

    blocks = []
    for entry, start, stop in zip(parts, bounds[:-1], bounds[1:]):
        lo = int(np.searchsorted(positions, start, side="left"))
        hi = int(np.searchsorted(positions, stop, side="left"))
        if lo == hi:
            continue
        local = (positions[lo:hi] - start).tolist()
        frame = pl.read_parquet(Path(entry["path"]), columns=names)
        blocks.append(frame.to_numpy(allow_copy=True)[local])

    # Parts are visited in ascending position order and `positions` is sorted, so the
    # concatenation is already in sampling order -- no argsort needed.
    return np.concatenate(blocks, axis=0).astype(np.float32, copy=False)


def load_cohort(args, hidden: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    import polars as pl

    summary = json.loads(args.embed_summary.read_text())
    latent_path = Path(summary["latent_path"])
    manifest_path = Path(summary["dedup_manifest"])
    if not latent_path.exists():
        raise SystemExit(f"latent.npy not found at {latent_path}")

    latent = np.load(latent_path, mmap_mode="r")
    total = int(latent.shape[0])
    positions = sample_positions(total, args.sample_rows, args.seed)
    print(f"latent.npy: {total:,} rows x {latent.shape[1]} dims -> sampling {positions.size:,}")

    if args.coords_npy is not None:
        coords = np.load(args.coords_npy, mmap_mode="r")
        if coords.shape[0] != total:
            raise SystemExit(f"{args.coords_npy.name} has {coords.shape[0]:,} rows, "
                             f"latent has {total:,}")
        xy = np.asarray(coords[positions], dtype=np.float32)
        label = args.coords_npy.stem
        print(f"  coords from {args.coords_npy.name}")
    elif args.encoder is not None:
        xy = encoder_transform(args.encoder,
                               np.ascontiguousarray(latent[positions], dtype=np.float32),
                               hidden, args.device)
        label = args.encoder.stem
    else:
        raise SystemExit("--embed-summary needs either --encoder or --coords-npy")

    parts = json.loads(manifest_path.read_text())["per_chromosome"]
    first = Path(parts[0]["path"])
    if not first.exists():
        raise SystemExit(f"dedup part not found: {first}\n"
                         "Run this where stage 0's output lives (miletus).")
    available = set(pl.scan_parquet(first).collect_schema().names())
    names = numeric_feature_names(args.feature_spec_path, available, args.features)
    if not names:
        raise SystemExit("no numeric features present in the dedup parts")
    print(f"reading {len(names)} numeric columns from {len(parts)} dedup parts")
    return xy, read_dedup_features(manifest_path, positions, names), names, label


# ── mode B: one clust_regime_sweep run ─────────────────────────────────────────

def load_analysis(args) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    import polars as pl

    analysis_path = args.analysis.resolve()
    raw_path = (args.raw_parquet or analysis_path.parent.parent / "sample.parquet").resolve()
    if not raw_path.exists():
        raise SystemExit(f"raw parquet not found: {raw_path} (pass --raw-parquet)")

    n_analysis = pl.scan_parquet(analysis_path).select(pl.len()).collect().item()
    n_raw = pl.scan_parquet(raw_path).select(pl.len()).collect().item()
    if n_analysis != n_raw:
        raise SystemExit(
            f"row count mismatch: {analysis_path.name} has {n_analysis:,}, "
            f"{raw_path.name} has {n_raw:,} -- the positional join is unsafe, so this is a "
            "hard stop rather than a truncation"
        )

    available = set(pl.scan_parquet(raw_path).collect_schema().names())
    names = numeric_feature_names(args.feature_spec_path, available, args.features)
    if not names:
        raise SystemExit("no numeric features present in the raw parquet")

    positions = sample_positions(n_analysis, args.sample_rows, args.seed)
    print(f"{analysis_path.name}: {n_analysis:,} rows -> sampling {positions.size:,}")
    print(f"reading {len(names)} numeric columns from {raw_path.name}")

    xy = (pl.scan_parquet(analysis_path).select("umap_1", "umap_2")
          .collect().to_numpy()[positions].astype(np.float32))
    values = (pl.scan_parquet(raw_path)
              .select([pl.col(n).cast(pl.Float32) for n in names])
              .collect().to_numpy()[positions])
    return xy, values, names, analysis_path.parent.name


# ── plotting ───────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    hidden = tuple(int(p) for p in args.hidden.split(",") if p.strip())

    if (args.embed_summary is None) == (args.analysis is None):
        raise SystemExit("pass exactly one of --embed-summary (cohort) or --analysis (run)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.embed_summary is not None:
        xy, values, names, source_label = load_cohort(args, hidden)
    else:
        xy, values, names, source_label = load_analysis(args)

    args.output.mkdir(parents=True, exist_ok=True)
    plotted = 0
    for index, feature in enumerate(names):
        column = values[:, index]
        valid = np.isfinite(column)
        if not valid.any():
            print(f"  {feature:16} all null in the sample, skipping")
            continue

        lo, hi = np.percentile(column[valid], [args.clip_percentile,
                                               100 - args.clip_percentile])
        if lo == hi:
            lo, hi = float(column[valid].min()), float(column[valid].max())
            if lo == hi:
                hi = lo + 1.0

        fig, axis = plt.subplots(figsize=(9, 8))
        if (~valid).any():
            axis.scatter(xy[~valid, 0], xy[~valid, 1], s=args.point_size * 0.6,
                         c=NULL_COLOUR, alpha=0.3, linewidths=0, rasterized=True)
        scatter = axis.scatter(xy[valid, 0], xy[valid, 1], s=args.point_size,
                               c=column[valid], cmap=args.cmap, vmin=lo, vmax=hi,
                               alpha=0.6, linewidths=0, rasterized=True)
        fig.colorbar(scatter, ax=axis, label=feature, fraction=0.046, pad=0.04)

        # The source label is derived, never hard-coded: an earlier version of this script
        # stamped "model 13: nn=15 md=0.1" on every panel regardless of which encoder was
        # actually passed, which is a caption that silently lies when you sweep encoders.
        axis.set_title(f"UMAP coloured by {feature}  [{source_label}]\n"
                       f"{xy.shape[0]:,} points · colour clipped to "
                       f"[{args.clip_percentile:g}, {100 - args.clip_percentile:g}] pct: "
                       f"[{lo:.3g}, {hi:.3g}]", fontsize=10)
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")

        out_path = args.output / f"{feature}_umap.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  {feature:16} [{column[valid].min():.3g}, {column[valid].max():.3g}]"
              f" -> {out_path.name}")
        plotted += 1

    print(f"\nsaved {plotted} images -> {args.output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
