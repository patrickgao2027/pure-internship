"""Aggregate the clustering testing regime after the array job finishes.

Computes the cross-cell comparisons that individual cells cannot (they run in
separate array tasks). Cells are named by sweep:

  vae_<pct>    sweep 1+3 -- VAE trained on that fraction, ALL rows clustered
  rq_<thresh>  sweep 2   -- rq filter varied, VAE fixed
  rows_<n>     sweep 4   -- no VAE/UMAP, n rows clustered directly

  * vs-reference: every cell's clustering vs the same variant's ``vae_100pct``
    clustering (join on CHROM/POS/REF/ALT, so partial overlaps are handled).
  * D-vs-A: each rq threshold's variant-D clustering vs ``vae_100pct`` variant A.

Then writes ``regime_summary.json`` (a long table of per-cell/variant internal metrics
plus the agreement blocks) and, if matplotlib is available, a few summary plots.

Within-cell cross-variant agreement is already computed by clustering_regime_sweep.py
and is carried through from each ``cell_summary.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl

from uv_vae import clustering_metrics as metrics

KEY_COLUMNS = ["CHROM", "POS", "REF", "ALT"]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_labels(analysis_path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(analysis_path)
    keys = [name for name in KEY_COLUMNS if name in frame.columns]
    return frame.select([*keys, "cluster_label"])


def aligned_agreement(reference: pl.DataFrame, other: pl.DataFrame) -> dict:
    keys = [name for name in KEY_COLUMNS if name in reference.columns and name in other.columns]
    if not keys:
        return {"ari": float("nan"), "ami": float("nan"), "nmi": float("nan"), "n": 0,
                "note": "no shared key columns"}
    joined = other.join(reference, on=keys, how="inner", suffix="_ref")
    if joined.height == 0:
        return {"ari": float("nan"), "ami": float("nan"), "nmi": float("nan"), "n": 0,
                "note": "no overlapping variants"}
    return metrics.clustering_agreement(
        joined["cluster_label"].to_numpy(),
        joined["cluster_label_ref"].to_numpy(),
    )


def discover_cells(output_root: Path) -> dict[str, dict[str, Path]]:
    """Map cell_label -> {variant: analysis.parquet}."""
    cells: dict[str, dict[str, Path]] = {}
    for analysis in output_root.glob("*/variant_*/analysis.parquet"):
        variant = analysis.parent.name.replace("variant_", "")
        cell = analysis.parent.parent.name
        cells.setdefault(cell, {})[variant] = analysis
    return cells


def cell_size(output_root: Path, cell: str) -> int | None:
    """Value of this cell's swept variable -- VAE training rows, or clustered rows.

    In checkpoint mode the x-axis is how much data the VAE was TRAINED on (all cells
    cluster the same number of rows), so sample_rows_used would be a flat line.
    """
    summary_path = output_root / cell / "cell_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("swept_variable") == "vae_training_rows":
        return summary.get("vae_training_rows")
    return summary.get("sample_rows_used")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate clustering regime results")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    cells = discover_cells(output_root)
    if not cells:
        raise SystemExit(f"No cells found under {output_root}")
    log(f"Found {len(cells)} cells: {sorted(cells)}")

    # Sweep 1+3 cells are named vae_<pct>; sweep 2 rq_<threshold>; sweep 4 rows_<n>.
    vae_cells = {name: variants for name, variants in cells.items() if name.startswith("vae_")}
    rq_cells = {name: variants for name, variants in cells.items() if name.startswith("rq_")}
    row_cells = {name: variants for name, variants in cells.items() if name.startswith("rows_")}

    # Reference = the VAE trained on the most data (100pct); everything is compared to it.
    reference_cell = "vae_100pct"
    reference_variants = cells.get(reference_cell, {})
    ladder_cells = {**vae_cells, **row_cells}

    # Cache reference label frames once.
    reference_labels = {variant: load_labels(path) for variant, path in reference_variants.items()}

    subsample_vs_all: list[dict] = []
    if reference_variants:
        for cell, variants in sorted(ladder_cells.items()):
            if cell == reference_cell:
                continue
            size = cell_size(output_root, cell)
            for variant, path in sorted(variants.items()):
                if variant not in reference_labels:
                    continue
                agreement = aligned_agreement(reference_labels[variant], load_labels(path))
                subsample_vs_all.append({"variant": variant, "cell": cell, "sample_rows": size, **agreement})
                log(f"  subsample-vs-all {variant} {cell}: ARI={agreement.get('ari')}")
    else:
        log(f"WARNING: reference cell {reference_cell} not found; skipping vs-reference")

    d_vs_a_all: list[dict] = []
    reference_a = reference_labels.get("A")
    if reference_a is not None:
        for cell, variants in sorted(rq_cells.items()):
            path = variants.get("D")
            if path is None:
                continue
            size = cell_size(output_root, cell)
            agreement = aligned_agreement(reference_a, load_labels(path))
            d_vs_a_all.append({"cell": cell, "n_rows": size, **agreement})
            log(f"  D-vs-A-all {cell}: ARI={agreement.get('ari')}")
    else:
        log(f"WARNING: no {reference_cell}/variant_A reference; skipping D-vs-A")

    # Per-cell internal metrics + within-cell cross-variant agreement (from cell summaries).
    cell_metrics: dict[str, dict] = {}
    cross_variant_by_cell: dict[str, list] = {}
    for cell in sorted(cells):
        summary_path = output_root / cell / "cell_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        cross_variant_by_cell[cell] = summary.get("cross_variant_agreement", [])
        per_variant = {}
        for variant, payload in summary.get("variants", {}).items():
            if "metrics" in payload:
                per_variant[variant] = {
                    "sample_rows_used": summary.get("sample_rows_used"),
                    "swept_variable": summary.get("swept_variable"),
                    "swept_value": cell_size(output_root, cell),
                    "vae_training_rows": summary.get("vae_training_rows"),
                    "rq_threshold": summary.get("rq_threshold"),
                    "selected_min_cluster_size": payload.get("selected_min_cluster_size"),
                    "grid": payload.get("grid"),
                    **payload["metrics"],
                }
        cell_metrics[cell] = per_variant

    regime_summary = {
        "output_root": str(output_root),
        "cells": cell_metrics,
        "subsample_vs_all": subsample_vs_all,
        "d_vs_a_all": d_vs_a_all,
        "cross_variant_by_cell": cross_variant_by_cell,
    }
    summary_path = output_root / "regime_summary.json"
    summary_path.write_text(json.dumps(regime_summary, indent=2))
    log(f"Wrote {summary_path}")

    try:
        make_plots(output_root, cell_metrics, subsample_vs_all)
    except Exception as exc:  # plotting is a nicety, never fail aggregation on it
        log(f"Plotting skipped: {type(exc).__name__}: {exc}")

    return 0


def make_plots(output_root: Path, cell_metrics: dict, subsample_vs_all: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ARI vs sample_rows, one line per variant.
    if subsample_vs_all:
        by_variant: dict[str, list[tuple[int, float]]] = {}
        for row in subsample_vs_all:
            if row.get("sample_rows") is None:
                continue
            by_variant.setdefault(row["variant"], []).append((row["sample_rows"], row.get("ari", float("nan"))))
        fig, axis = plt.subplots(figsize=(8, 5))
        for variant, points in sorted(by_variant.items()):
            points.sort()
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            axis.plot(xs, ys, marker="o", label=variant)
        axis.set_xscale("log")
        axis.set_xlabel("VAE training rows  /  clustered rows")
        axis.set_ylabel("ARI vs vae_100pct")
        axis.set_title("Clustering stability vs the 100pct reference")
        axis.legend()
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "ari_vs_sample_rows.png", dpi=150)
        plt.close(fig)

    # Internal metrics vs sample rows, per variant.
    for metric_name in ("silhouette", "davies_bouldin", "calinski_harabasz", "noise_fraction", "relative_validity"):
        fig, axis = plt.subplots(figsize=(8, 5))
        drew = False
        by_variant: dict[str, list[tuple[int, float]]] = {}
        for cell, variants in cell_metrics.items():
            if not (cell.startswith("vae_") or cell.startswith("rows_")):
                continue
            for variant, payload in variants.items():
                size = payload.get("swept_value")
                value = payload.get(metric_name)
                if size is None or value is None:
                    continue
                by_variant.setdefault(variant, []).append((size, float(value)))
        for variant, points in sorted(by_variant.items()):
            points.sort()
            axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=variant)
            drew = True
        if not drew:
            plt.close(fig)
            continue
        axis.set_xscale("log")
        axis.set_xlabel("VAE training rows  /  clustered rows")
        axis.set_ylabel(metric_name)
        axis.set_title(f"{metric_name} vs sweep point")
        axis.legend()
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric_name}_vs_sample_rows.png", dpi=150)
        plt.close(fig)

    log(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
