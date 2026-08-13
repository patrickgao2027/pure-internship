#!/usr/bin/env python
"""Colour the full-cohort parametric UMAP by each raw numeric VAE input feature.

Same question as ``plot_feature_umap.py`` (which variant of the input is the UMAP organised
around?) but for the 95-file cohort pipeline, where the 2D coords come from a parametric
encoder and the raw feature values live in the dedup chromosome-split parquets written by
stage 0.

**What this joins on**: row positions in ``latent.npy`` correspond 1-to-1 to rows in the
concatenated dedup parquets (stage 1 reads them in the chromosome order recorded in
``embed_summary.json``'s ``dedup_manifest``). So the join is positional: row i of
``latent.npy`` = row i of the concatenated dedup parts, no key needed.

**2D coords**: the encoder is applied to the sampled latent vectors on-the-fly. A saved
``coords.npy`` from ``apply_parametric_full.py --save-coords`` is accepted via
``--coords-npy`` to skip the encoder (faster if you have it; ~1.3 GB on disk).

Usage (on miletus, after stage 0 + stage 1 have run)::

    python umap_hdbscan_sweep/plot_feature_umap_cohort.py \\
        --embed-summary umap_hdbscan_sweep/umap_tests/hdbscan_scaling/embed_summary.json \\
        --encoder umap_hdbscan_sweep/umap_tests/final_models/13_BEST_25M_nn15_md0.1_umap.pt \\
        --output plots/feature_umap_cohort \\
        --sample-rows 300000

    # if coords.npy was saved with --save-coords
    python umap_hdbscan_sweep/plot_feature_umap_cohort.py \\
        --embed-summary ... \\
        --coords-npy path/to/coords.npy \\
        --output plots/feature_umap_cohort
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
NOISE_COLOUR = "#cccccc"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embed-summary", type=Path, required=True,
                   help="embed_summary.json written by stage1_embed.py "
                        "(supplies latent.npy path + dedup_manifest path)")
    p.add_argument("--encoder", type=Path, default=None,
                   help="saved ParametricEncoder .pt from final_models/ "
                        "(not needed when --coords-npy is provided)")
    p.add_argument("--coords-npy", type=Path, default=None,
                   help="pre-computed (N_total, 2) float32 coords from "
                        "apply_parametric_full.py --save-coords; skips running the encoder")
    p.add_argument("--feature-spec-path", type=Path, default=DEFAULT_FEATURE_SPEC)
    p.add_argument("--features", nargs="*", default=None,
                   help="subset of numeric features to plot (default: all numeric from spec "
                        "that are present in the dedup parquets)")
    p.add_argument("--sample-rows", type=int, default=300_000,
                   help="points to sample and plot (default 300k)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--point-size", type=float, default=1.0)
    p.add_argument("--clip-percentile", type=float, default=1.0,
                   help="colour scale clipped to [p, 100-p] percentiles per feature")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument("--infer-batch-size", type=int, default=500_000)
    p.add_argument("--hidden", default="256,256,128")
    p.add_argument("--output", type=Path, default=Path("feature_umap_cohort"),
                   help="output directory; one PNG per feature")
    return p.parse_args()


def sample_positions(total: int, wanted: int, seed: int) -> np.ndarray:
    if wanted >= total:
        return np.arange(total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=wanted, replace=False))


def encoder_transform(encoder_path: Path, latent_sample: np.ndarray,
                      hidden: tuple[int, ...], device_str: str) -> np.ndarray:
    """Run latent_sample through the saved ParametricEncoder -> (N, 2) float32."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import parametric_umap as pu

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    latent_dim = latent_sample.shape[1]
    blob = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder = pu.ParametricEncoder(latent_dim, output_dim=2, hidden=hidden).to(device)
    encoder.load_state_dict(blob["state_dict"])
    encoder.eval()

    mode = blob.get("mode", "umap")
    model = pu.ParametricUmap(encoder=encoder, device=device, mode=mode)
    print(f"  encoder mode={mode} on {device}")
    return model.transform(latent_sample.astype(np.float32))


def load_dedup_features(manifest_path: Path, positions: np.ndarray,
                        feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Load feature values at the given row positions from the chromosome-split dedup parquets.

    The dedup parts are concatenated in manifest order and ``latent.npy`` was written in
    the same order, so row ``i`` in ``latent.npy`` is row ``i`` in the concatenation.

    Returns (values_array (N, n_features), available_feature_names).
    """
    import polars as pl

    manifest = json.loads(manifest_path.read_text())
    parts = manifest["per_chromosome"]

    # Check which features are actually in the parquets (first part defines the schema)
    first_path = Path(parts[0]["path"])
    if not first_path.exists():
        raise SystemExit(
            f"dedup part not found: {first_path}\n"
            "Run this script on miletus where the dedup output lives."
        )
    available_cols = set(pl.scan_parquet(first_path).collect_schema().names())
    wanted = [name for name in feature_names if name in available_cols]
    if not wanted:
        raise SystemExit(f"none of {feature_names} found in dedup parquets")
    missing = [name for name in feature_names if name not in available_cols]
    if missing:
        print(f"  WARNING: {missing} not in dedup parquets, skipping")

    # Build per-chromosome row-index offsets from the loci counts in the manifest.
    offsets = [0]
    for entry in parts:
        offsets.append(offsets[-1] + int(entry["loci"]))
    total = offsets[-1]
    if positions.max() >= total:
        raise SystemExit(f"sampled row {positions.max()} >= total loci {total}")

    rows_per_feature: list[np.ndarray] = []
    row_order: list[int] = []

    for chrom_index, (entry, chrom_start, chrom_end) in enumerate(
            zip(parts, offsets[:-1], offsets[1:])):
        # Which sampled positions fall in this chromosome?
        lo = np.searchsorted(positions, chrom_start, side="left")
        hi = np.searchsorted(positions, chrom_end, side="left")
        if lo == hi:
            continue

        global_rows = positions[lo:hi]
        local_rows = (global_rows - chrom_start).tolist()

        part_path = Path(entry["path"])
        frame = pl.read_parquet(part_path, columns=wanted)
        block = frame.to_numpy(allow_copy=True)[local_rows]

        rows_per_feature.append(block)
        row_order.extend(range(lo, hi))

    # Reassemble into the original sampling order.
    all_blocks = np.concatenate(rows_per_feature, axis=0)
    order_index = np.argsort(row_order)
    return all_blocks[order_index].astype(np.float32, copy=False), wanted


def main() -> int:
    args = parse_args()
    hidden = tuple(int(p) for p in args.hidden.split(",") if p.strip())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads(args.embed_summary.read_text())
    latent_path = Path(summary["latent_path"])
    manifest_path = Path(summary["dedup_manifest"])
    feature_names_from_model = summary.get("feature_names", [])

    if not latent_path.exists():
        raise SystemExit(f"latent.npy not found at {latent_path}")

    latent = np.load(latent_path, mmap_mode="r")
    total_rows = latent.shape[0]
    print(f"latent.npy: {total_rows:,} rows x {latent.shape[1]} latent dims")

    positions = sample_positions(total_rows, args.sample_rows, args.seed)
    print(f"Sampling {positions.size:,} rows")

    # --- 2D coords ---
    if args.coords_npy is not None:
        coords_full = np.load(args.coords_npy, mmap_mode="r")
        if coords_full.shape[0] != total_rows:
            raise SystemExit(
                f"coords.npy has {coords_full.shape[0]:,} rows but latent has {total_rows:,}"
            )
        xy = np.asarray(coords_full[positions], dtype=np.float32)
        print(f"Loaded 2D coords from {args.coords_npy.name}")
    else:
        if args.encoder is None:
            raise SystemExit("provide --encoder or --coords-npy")
        print(f"Running encoder {args.encoder.name} on {positions.size:,} rows ...")
        latent_sample = np.ascontiguousarray(latent[positions], dtype=np.float32)
        xy = encoder_transform(args.encoder, latent_sample, hidden, args.device)

    # --- features ---
    # Determine which numeric features to plot
    specs = load_feature_specs(args.feature_spec_path)
    numeric_names = [s.name for s in specs if s.is_numeric]
    if args.features:
        numeric_names = [n for n in args.features if n in numeric_names]

    print(f"Loading feature values from dedup parquets ({len(numeric_names)} numeric columns) ...")
    feature_values, available_features = load_dedup_features(
        manifest_path, positions, numeric_names)
    print(f"  available: {', '.join(available_features)}")

    # --- plot ---
    args.output.mkdir(parents=True, exist_ok=True)
    plotted = 0
    for col_index, feature in enumerate(available_features):
        values = feature_values[:, col_index]
        valid = np.isfinite(values)
        if not valid.any():
            print(f"  {feature}: all null, skipping")
            continue

        lo, hi = np.percentile(values[valid], [args.clip_percentile, 100 - args.clip_percentile])
        if lo == hi:
            lo, hi = float(values[valid].min()), float(values[valid].max())
            if lo == hi:
                hi = lo + 1.0

        fig, ax = plt.subplots(figsize=(9, 8))
        if (~valid).any():
            ax.scatter(xy[~valid, 0], xy[~valid, 1], s=args.point_size * 0.6,
                       c=NOISE_COLOUR, alpha=0.3, linewidths=0, rasterized=True,
                       label=f"null ({(~valid).sum():,})")

        scatter = ax.scatter(xy[valid, 0], xy[valid, 1], s=args.point_size,
                             c=values[valid], cmap=args.cmap, vmin=lo, vmax=hi,
                             alpha=0.6, linewidths=0, rasterized=True)
        fig.colorbar(scatter, ax=ax, label=feature, fraction=0.046, pad=0.04)
        ax.set_title(f"UMAP coloured by {feature}  [model 13: nn=15 md=0.1 umap]\n"
                     f"{positions.size:,} points (colour clipped to "
                     f"[{args.clip_percentile:g}, {100 - args.clip_percentile:g}] pct: "
                     f"[{lo:.3g}, {hi:.3g}])",
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
        print(f"  {feature:16} [{values[valid].min():.3g}, {values[valid].max():.3g}]"
              f" -> {out_path.name}")
        plotted += 1

    print(f"\nsaved {plotted} images -> {args.output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
