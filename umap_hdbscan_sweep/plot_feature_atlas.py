#!/usr/bin/env python
"""Every column in the source parquets, painted onto the UMAP embedding.

``cohort_cluster_report.py`` colours by a curated list and only ever renders **numerics**, so
the context columns (X_PREV1, X_NEXT1, ALT, REF) have no panel at all while all-null columns
each get one. This renders the whole schema, categoricals included, so "no panel" stops
meaning "no information".

Two panel types, because one colour scale cannot serve both:

numeric      viridis, clipped to the 1st/99th percentile. The clip is not cosmetic: with a
             min->max scale a long tail eats the palette, so 99% of points land in the bottom
             few percent of the colormap and a feature that does vary reads as flat.
categorical  one colour per level from the pipeline's own ``build_palette``, drawn
             largest-level-first with a legend. Levels past ``--max-levels`` collapse into a
             single grey "other", so a 500-level column still yields a readable panel.

Panels are descriptive only -- no discrimination statistics. The title carries the value
count, the non-null share and the modal share, and ``feature_atlas.csv`` carries the same for
every feature, so a feature never vanishes silently even when its panel is uninformative.

Efficiency
----------
The cost here is I/O and rasterisation, not arithmetic, so:

* One sorted positional sample drives everything; each parquet is read **once**, for every
  column at once, rather than once per feature.
* Categorical levels are resolved to integer codes a single time in the parent. Workers get
  codes plus a colour list and never touch a string column.
* Panels render across a process pool. The shared xy backdrop goes to workers once via the
  pool initializer; nothing heavier than numpy and matplotlib is imported inside a worker
  (importing the pipeline there costs ~5 s per process for two constant lists -- measured
  while building ``cluster_profiles.py``).
* Scatter is rasterised, so PNG size tracks the figure, not the point count.

Usage::

    # the 35 columns already in a finished cell's analysis.parquet -- no extra I/O
    python umap_hdbscan_sweep/plot_feature_atlas.py \\
        --analysis <cell>/analysis.parquet \\
        --output-dir feature_atlas

    # all 70 source columns, after recover_source_columns.py has put them in one file
    python umap_hdbscan_sweep/plot_feature_atlas.py \\
        --analysis enriched.parquet \\
        --output-dir feature_atlas_full

Note that every cell's analysis.parquet is identical except ``cluster_label`` -- the sampled
positions and the UMAP coordinates are drawn once and shared -- so with scoring gone there is
nothing cell-specific left in these panels. Run this once, not once per cell.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

# Walk up to the directory that holds uv_vae/ rather than counting levels: these scripts
# get reorganised into subfolders (umap/, hdbscan/, tests/), and a hard-coded parents[N]
# silently resolves to the wrong root the moment one moves, failing on `import uv_vae`.
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, REPO_ROOT / "uv_vae" / "scripts",
                  Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

os.environ.setdefault("UV_VAE_DISABLE_CUML", "1")

import numpy as np
import polars as pl

# Coordinates, identifiers and the label itself are not features. POS is
# excluded there as a near-unique integer and the same reasoning applies to a colour panel.
EXCLUDE = {"umap_1", "umap_2", "cluster_label", "cluster_probability", "POS"}

NULL_COLOUR = "#d0d0d0"
OTHER_COLOUR = "#9e9e9e"


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_argument_group("input (either --analysis, or --embed-summary + coords)")
    source.add_argument("--analysis", type=Path, default=None,
                        help="analysis.parquet holding umap_1/umap_2 and feature columns")
    source.add_argument("--embed-summary", type=Path, default=None,
                        help="stage 1 embed_summary.json (latent.npy + dedup manifest)")
    source.add_argument("--coords", type=Path, default=None, help="coords.npy (N,2)")
    source.add_argument("--encoder", type=Path, default=None,
                        help="parametric encoder .pt, used when --coords is absent")
    source.add_argument("--labels", type=Path, default=None,
                        help="cohort_labels.npy (unused; kept for CLI compatibility)")
    source.add_argument("--source-glob", default=None,
                        help="also pull columns present only in the source parquets "
                             "(QUAL/MAPQ/DP/RAW_VAF/SNVQ), joined on CHROM/POS/REF/ALT")

    p.add_argument("--features", nargs="*", default=None, help="subset (default: all)")
    p.add_argument("--sample-rows", type=int, default=300_000)
    p.add_argument("--max-levels", type=int, default=12,
                   help="categorical levels drawn individually; the rest become 'other'")
    p.add_argument("--point-size", type=float, default=3.0)
    p.add_argument("--clip-percentile", type=float, default=1.0)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--hidden", default="256,256,128")
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-contact-sheet", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


# ── loading ────────────────────────────────────────────────────────────────────

def sample_positions(total: int, wanted: int, seed: int) -> np.ndarray:
    if wanted >= total:
        return np.arange(total)
    return np.sort(np.random.default_rng(seed).choice(total, size=wanted, replace=False))


def load_from_analysis(path: Path, wanted: int, seed: int) -> tuple[np.ndarray, pl.DataFrame,
                                                                   np.ndarray | None]:
    total = pl.scan_parquet(path).select(pl.len()).collect().item()
    positions = sample_positions(total, wanted, seed)
    frame = pl.read_parquet(path)
    if positions.size < total:
        # row-index selection; list form works across all Polars versions
        frame = frame[positions.tolist()]
    xy = frame.select(["umap_1", "umap_2"]).to_numpy().astype(np.float32)
    labels = (frame["cluster_label"].to_numpy().astype(np.int32)
              if "cluster_label" in frame.columns else None)
    features = frame.drop([c for c in frame.columns if c in EXCLUDE])
    return xy, features, labels


def load_from_cohort(args) -> tuple[np.ndarray, pl.DataFrame, np.ndarray | None]:
    import cohort_cluster_report as ccr

    summary = json.loads(args.embed_summary.read_text())
    latent = np.load(Path(summary["latent_path"]), mmap_mode="r")
    total = int(latent.shape[0])
    positions = sample_positions(total, args.sample_rows, args.seed)
    log(f"cohort {total:,} rows -> sampling {positions.size:,}")

    if args.coords is not None:
        coords = np.load(args.coords, mmap_mode="r")
        if coords.shape[0] != total:
            raise SystemExit(f"{args.coords.name} has {coords.shape[0]:,} rows, "
                             f"latent has {total:,}")
        xy = np.asarray(coords[positions], dtype=np.float32)
    elif args.encoder is not None:
        hidden = tuple(int(p) for p in args.hidden.split(",") if p.strip())
        xy = ccr.encoder_coords(args.encoder,
                                np.ascontiguousarray(latent[positions], dtype=np.float32),
                                hidden, args.device)
    else:
        raise SystemExit("pass --coords or --encoder alongside --embed-summary")

    # Every column stage 0 wrote, not a curated subset -- that is the point of this script.
    manifest = json.loads(Path(summary["dedup_manifest"]).read_text())
    first = Path(manifest["per_chromosome"][0]["path"])
    available = [c for c in pl.scan_parquet(first).collect_schema().names()
                 if c not in EXCLUDE]
    log(f"reading {len(available)} columns from the dedup parts")
    features, _ = ccr.read_dedup_columns(Path(summary["dedup_manifest"]), positions, available)

    if args.source_glob:
        missing = [c for c in ccr.RANKING_COLUMNS if c not in features.columns]
        if missing:
            log(f"recovering {', '.join(missing)} from {args.source_glob} (one cohort pass)")
            extra = ccr.read_source_columns(
                args.source_glob, features.select(["CHROM", "POS", "REF", "ALT"]),
                missing, args.threads)
            features = pl.concat([features, extra], how="horizontal")

    labels = None
    if args.labels is not None:
        labels = np.asarray(np.load(args.labels, mmap_mode="r")[positions], dtype=np.int32)

    return xy, features.drop([c for c in features.columns if c in EXCLUDE]), labels


def describe_feature(values_pd, numeric: bool) -> dict:
    """Basic descriptive stats for the panel title and CSV -- no scoring."""
    non_null = float(values_pd.notna().mean() * 100) if len(values_pd) else 0.0
    n_unique = int(values_pd.nunique(dropna=True))
    counts = values_pd.value_counts(normalize=True, dropna=True)
    mode_share = float(counts.iloc[0]) if len(counts) else 1.0
    return {"kind": "numeric" if numeric else "categorical",
            "non_null_pct": non_null, "n_unique": n_unique, "mode_share": mode_share}


# ── rendering ──────────────────────────────────────────────────────────────────

_BACKDROP: dict[str, object] = {}


def _init_worker(xy, point_size, dpi, cmap) -> None:
    import matplotlib

    matplotlib.use("Agg")
    _BACKDROP.update(xy=xy, point_size=point_size, dpi=dpi, cmap=cmap)


def _render(task: dict) -> str:
    """One panel. Everything it needs is precomputed; workers import no project code."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xy = _BACKDROP["xy"]
    size = _BACKDROP["point_size"]
    fig, axis = plt.subplots(figsize=(8.5, 6.5), dpi=_BACKDROP["dpi"])

    missing = task["missing"]
    if missing.any():
        axis.scatter(xy[missing, 0], xy[missing, 1], s=size, c=NULL_COLOUR,
                     alpha=0.45, linewidths=0, rasterized=True)

    if task["kind"] == "numeric":
        good = ~missing
        if good.any():
            scatter = axis.scatter(xy[good, 0], xy[good, 1], s=size, c=task["values"][good],
                                   alpha=0.78, linewidths=0, cmap=_BACKDROP["cmap"],
                                   vmin=task["vmin"], vmax=task["vmax"], rasterized=True)
            fig.colorbar(scatter, ax=axis).set_label(task["feature"])
    else:
        # Largest level first so small levels are drawn on top and stay visible.
        for code, (name, colour, count) in enumerate(task["levels"]):
            mask = task["codes"] == code
            if not mask.any():
                continue
            axis.scatter(xy[mask, 0], xy[mask, 1], s=size, c=colour, alpha=0.78,
                         linewidths=0, rasterized=True)
        handles = [Line2D([], [], marker="o", linestyle="", markersize=6, color=colour,
                          label=f"{name} ({count:,})")
                   for name, colour, count in task["levels"]]
        if missing.any():
            handles.append(Line2D([], [], marker="o", linestyle="", markersize=6,
                                  color=NULL_COLOUR, label=f"null ({int(missing.sum()):,})"))
        axis.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=False, fontsize=8)

    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title(task["title"], fontsize=10)
    fig.tight_layout()
    fig.savefig(task["path"], bbox_inches="tight")
    plt.close(fig)
    return str(task["path"])


def build_task(feature: str, series: pl.Series, values_pd, record: dict, args,
               output_dir: Path) -> dict:
    """Precompute everything the worker needs: codes, colours, limits, title."""
    import run_variant_cluster_pipeline as rvcp

    numeric = record["kind"] == "numeric"
    missing = values_pd.isna().to_numpy()

    title = (f"UMAP coloured by {feature}  [{record['kind']}]\n"
             f"{record['n_unique']:,} unique values · "
             f"{record['non_null_pct']:.1f}% non-null · "
             f"mode {record['mode_share'] * 100:.1f}%")

    task = {"feature": feature, "kind": record["kind"], "missing": missing,
            "path": output_dir / f"umap_{feature}.png", "title": title}

    if numeric:
        values = values_pd.to_numpy(dtype=np.float64, na_value=np.nan)
        finite = values[~missing]
        if finite.size:
            lo, hi = np.percentile(finite, [args.clip_percentile,
                                            100 - args.clip_percentile])
            if lo == hi:
                lo, hi = float(finite.min()), float(finite.max())
                if lo == hi:
                    hi = lo + 1.0
        else:
            lo, hi = 0.0, 1.0
        task.update(values=values, vmin=float(lo), vmax=float(hi))
        return task

    # Categorical: resolve levels once, in the parent. Codes are int16 so the array shipped
    # to a worker is a fraction of the string column it replaces.
    counts = series.drop_nulls().value_counts(sort=True)
    names = counts[series.name].to_list()
    sizes = counts["count"].to_list()
    keep = names[:args.max_levels]
    palette = rvcp.build_palette(len(keep))

    codes = np.full(missing.shape[0], -1, dtype=np.int16)
    raw = values_pd.to_numpy()
    for index, name in enumerate(keep):
        codes[raw == name] = index
    levels = [(str(name), palette[index], int(size))
              for index, (name, size) in enumerate(zip(keep, sizes[:len(keep)]))]

    overflow = names[args.max_levels:]
    if overflow:
        codes[(codes < 0) & ~missing] = len(keep)
        levels.append((f"other ({len(overflow)} levels)", OTHER_COLOUR,
                       int(sum(sizes[args.max_levels:]))))

    task.update(codes=codes, levels=levels)
    return task


def contact_sheet(items: list[tuple[str, Path]], path: Path) -> None:
    import run_variant_cluster_pipeline as rvcp

    if items:
        rvcp.make_contact_sheet(items, path, columns=3)


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    started = perf_counter()

    if args.analysis is not None:
        xy, features, labels = load_from_analysis(args.analysis, args.sample_rows, args.seed)
        log(f"{args.analysis.name}: {xy.shape[0]:,} rows sampled, "
            f"{len(features.columns)} feature columns")
    elif args.embed_summary is not None:
        xy, features, labels = load_from_cohort(args)
    else:
        raise SystemExit("pass --analysis, or --embed-summary with --coords/--encoder")

    if args.features:
        unknown = [f for f in args.features if f not in features.columns]
        if unknown:
            raise SystemExit(f"not in the input: {unknown}")
        features = features.select(args.features)
    if not features.columns:
        raise SystemExit("no feature columns left to plot")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log(f"preparing {len(features.columns)} features")
    tasks, records = [], []
    for feature in features.columns:
        series = features[feature]
        numeric = series.dtype.is_numeric()
        values_pd = series.to_pandas()
        record = describe_feature(values_pd, numeric)
        record["feature"] = feature
        records.append(record)
        tasks.append(build_task(feature, series, values_pd, record, args, args.output_dir))

    workers = args.workers if args.workers is not None else min(8, os.cpu_count() or 1)
    workers = max(1, min(workers, len(tasks)))
    log(f"rendering {len(tasks)} panels on {workers} worker(s)")
    initargs = (xy, args.point_size, args.dpi, args.cmap)
    if workers == 1:
        _init_worker(*initargs)
        for task in tasks:
            _render(task)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=initargs) as pool:
            for _ in pool.map(_render, tasks, chunksize=1):
                pass

    import pandas as pd

    table = pd.DataFrame(records)[["feature", "kind", "non_null_pct", "n_unique", "mode_share"]]
    csv_path = args.output_dir / "feature_atlas.csv"
    table.to_csv(csv_path, index=False)

    if not args.no_contact_sheet:
        for kind in ("numeric", "categorical"):
            items = [(t["feature"], t["path"]) for t in tasks if t["kind"] == kind]
            if items:
                contact_sheet(items, args.output_dir / f"contact_sheet_{kind}.png")
                log(f"  contact_sheet_{kind}.png ({len(items)} panels)")

    numeric_count = sum(1 for r in records if r["kind"] == "numeric")
    log(f"\n{len(records)} panels: {numeric_count} numeric, "
        f"{len(records) - numeric_count} categorical -> {args.output_dir}")
    log(f"\nmanifest -> {csv_path}   ({perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
