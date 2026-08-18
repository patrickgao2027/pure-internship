"""HPC clustering testing regime -- one dataset cell (see docs/clustering_regime_plan.md).

FOUR SWEEPS. The thing that varies is different in each, so read --mode carefully:

  mode=checkpoint  Sweep 1 + 3. The SUBSAMPLE LIVES IN THE VAE CHECKPOINT: each
                   pretrained checkpoint in sweep_results_full was trained on a
                   different row fraction (520K .. 52M). ALL deduplicated rows are
                   embedded through it and ALL of them flow into UMAP/HDBSCAN.
                     variant A = latent -> UMAP(2D) -> HDBSCAN   (full pipeline)
                     variant B = latent -> HDBSCAN               (UMAP dropped)
                   A and B share one embedding pass, so run them in the same task.

  mode=rq          Sweep 2. Row filter varies (rq < threshold) and the VAE is RETRAINED
                   on the rows passing that filter -- pass its checkpoint in via
                   --checkpoint-path (see scripts/train_rq_vae.py, which produces them).
                   Every row passing the filter flows through VAE -> UMAP -> HDBSCAN.
                     variant D

  mode=subsample   Sweep 4. No VAE, no UMAP -- so there is no model to train and the
                   subsample IS the clustered data. Rows are reservoir-sampled, then
                   clustered directly on the preprocessed matrix.
                     variant C1 = numeric + one-hot        -> HDBSCAN
                     variant C2 = that matrix -> PCA(16)   -> HDBSCAN
                   NOTE: --checkpoint-path is still required here, but ONLY to supply
                   the standardization stats and category vocabulary. Nothing is embedded.

This script never trains anything -- checkpoints are supplied ready-made (the ladder is
pretrained; the rq ones come from scripts/train_rq_vae.py). Post-VAE models fit on ALL
rows in the cell (no fit-sample, no approximate_predict). Rows are always FILTERED first,
then deduplicated, then sampled. SigProfiler runs in every sweep. Cross-cell comparisons
(ARI/AMI/NMI between cells) are done by aggregate_clustering_regime.py after the array;
this script computes only within-cell cross-variant agreement.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("NUMBA_CACHE_DIR", str((REPO_ROOT / ".numba_cache").resolve()))

import numpy as np
import polars as pl

from uv_vae import clustering_core as core
from uv_vae import clustering_metrics as metrics
from uv_vae.inference import LatentInference
from uv_vae.pipeline_vae import build_selected_columns, load_feature_names, write_deduplicated_sample
from uv_vae.preprocess import transform_frame

CONTEXT_COLUMNS = ["CHROM", "POS", "REF", "ALT", "X_PREV1", "X_NEXT1"]
LATENT_VARIANTS = {"A", "B", "D"}
UMAP_VARIANTS = {"A", "D"}
PREPROC_VARIANTS = {"C1", "C2"}

# Rows each sweep_results_full checkpoint was TRAINED on -- this is the subsample ladder
# for sweeps 1 and 3. Taken from each run's summary.json, NOT from the requested fraction:
# 100pct asked for 52M but the default filter only yields 50,591,627 eligible rows, and
# training clamps sample_rows to what exists. These are pre-deduplication counts.
VAE_TRAINING_ROWS = {
    "100pct": 50_591_627,
    "50pct": 26_000_000,
    "25pct": 13_000_000,
    "10pct": 5_200_000,
    "5pct": 2_600_000,
    "2pct": 1_300_000,   # fraction 0.025 -- the folder name is rounded
    "1pct": 520_000,
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part) for part in parse_csv(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clustering testing regime -- one cell")
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="VAE checkpoint (inference only, never trained). In checkpoint mode this IS "
             "the swept variable; in subsample mode it only supplies preprocessing stats.",
    )
    parser.add_argument("--feature-spec-path", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--mode",
        choices=["checkpoint", "rq", "subsample"],
        required=True,
        help="checkpoint: sweep the VAE (all data downstream) | rq: sweep the row filter "
             "| subsample: no VAE/UMAP, the subsample is the clustered data",
    )
    parser.add_argument(
        "--sample-rows",
        default=None,
        help="subsample mode only: N rows, or 'all'. Ignored in checkpoint/rq mode, "
             "where ALL rows passing the filter are used.",
    )
    parser.add_argument("--rq-threshold", type=float, default=None, help="rq mode: rq < threshold")
    parser.add_argument("--cell-label", default=None, help="override the derived cell directory name")
    parser.add_argument("--variants", default=None,
                        help="defaults per mode: checkpoint->A,B  rq->D  subsample->C1,C2")
    parser.add_argument("--row-filter", default="st = 'MIXED' AND et = 'MIXED' AND FILT = 1")
    parser.add_argument("--min-cluster-sizes", default="100,250,500,1000")
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.05)
    parser.add_argument("--umap-metric", default="euclidean")
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--sil-eval-rows", type=int, default=50_000)
    parser.add_argument("--embed-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--duckdb-memory-limit", default="32GB")
    parser.add_argument("--hdbscan-core-jobs", type=int, default=1)
    parser.add_argument("--no-sigprofiler", action="store_true", help="skip SigProfiler (debug/speed)")
    parser.add_argument("--sigprofiler-cpu", type=int, default=16)
    parser.add_argument("--cosmic-version", default="3.5")
    parser.add_argument("--genome-build", default="GRCh38")
    return parser.parse_args()


DEFAULT_VARIANTS = {"checkpoint": "A,B", "rq": "D", "subsample": "C1,C2"}


def checkpoint_label(checkpoint_path: str | Path) -> str:
    """Name a checkpoint by the training fraction encoded in its path (run_25pct -> 25pct)."""
    match = re.search(r"run_(\d+)pct", str(checkpoint_path))
    if match:
        return f"{match.group(1)}pct"
    match = re.search(r"rq_([0-9.]+)", str(checkpoint_path))
    if match:
        return f"rq_{match.group(1)}"
    return Path(checkpoint_path).parent.name


def checkpoint_training_rows(checkpoint_path: str | Path) -> int | None:
    """How many rows this VAE was trained on.

    Prefers the run's own summary.json (authoritative, and the only source for the
    retrained rq VAEs, whose row counts are not knowable ahead of time); falls back to
    the pretrained-ladder table.
    """
    summary_path = Path(checkpoint_path).parent / "summary.json"
    if summary_path.exists():
        try:
            rows = json.loads(summary_path.read_text()).get("sample_rows")
            if rows is not None:
                return int(rows)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    return VAE_TRAINING_ROWS.get(checkpoint_label(checkpoint_path))


def resolve_cell(args: argparse.Namespace) -> tuple[str, str, int | None]:
    """Return (cell_label, row_filter, sample_rows) for this cell.

    sample_rows is None (= use every row passing the filter) in every mode except
    ``subsample``: in checkpoint/rq mode the whole deduplicated population flows through
    UMAP/HDBSCAN, and what varies is the VAE or the filter -- not the row count.
    """
    if args.mode == "checkpoint":
        label = args.cell_label or f"vae_{checkpoint_label(args.checkpoint_path)}"
        return label, args.row_filter, None

    if args.mode == "rq":
        if args.rq_threshold is None:
            raise SystemExit("--rq-threshold is required for --mode rq")
        label = args.cell_label or f"rq_{args.rq_threshold}"
        return label, f"{args.row_filter} AND rq < {args.rq_threshold}", None

    # subsample: no VAE and no UMAP, so the subsample IS the clustered data.
    point = (args.sample_rows or "all").strip()
    if point.lower() == "all":
        return args.cell_label or "rows_all", args.row_filter, None
    return args.cell_label or f"rows_{int(point)}", args.row_filter, int(point)


def build_space(variant: str, latent: np.ndarray | None, preproc: np.ndarray | None,
                args: argparse.Namespace) -> np.ndarray:
    if variant in UMAP_VARIANTS:
        assert latent is not None
        log(f"  [{variant}] fitting UMAP on all {latent.shape[0]:,} rows (single-threaded)")
        return core.fit_umap(
            latent,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            seed=args.seed,
        )
    if variant == "B":
        assert latent is not None
        return latent
    if variant == "C1":
        assert preproc is not None
        return preproc
    if variant == "C2":
        assert preproc is not None
        log(f"  [C2] PCA -> {args.pca_components}D on all {preproc.shape[0]:,} rows")
        return core.pca_reduce(preproc, n_components=args.pca_components, seed=args.seed)
    raise ValueError(f"unknown variant {variant!r}")


def write_analysis_parquet(path: Path, context: pl.DataFrame, labels: np.ndarray,
                           probabilities: np.ndarray, space: np.ndarray, variant: str) -> None:
    columns = {name: context[name] for name in context.columns}
    columns["cluster_label"] = pl.Series("cluster_label", labels.astype(np.int32, copy=False))
    columns["cluster_probability"] = pl.Series(
        "cluster_probability", probabilities.astype(np.float32, copy=False)
    )
    if variant in UMAP_VARIANTS and space.shape[1] >= 2:
        columns["umap_1"] = pl.Series("umap_1", space[:, 0].astype(np.float32, copy=False))
        columns["umap_2"] = pl.Series("umap_2", space[:, 1].astype(np.float32, copy=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(columns).write_parquet(path)


def run_sigprofiler(analysis_path: Path, output_root: Path, args: argparse.Namespace) -> dict | None:
    """Reuse the (now-fixed) clustering pipeline's SigProfiler block. Lazy import so this
    module stays importable without SigProfilerAssignment installed."""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import run_variant_cluster_pipeline as rvcp  # noqa: PLC0415  (lazy on purpose)

    cluster_stats = rvcp.query_cluster_stats(analysis_path, threads=args.threads)
    if cluster_stats.height == 0:
        return {"skipped": "no non-noise clusters"}
    trinuc192 = rvcp.query_trinuc192_counts(analysis_path, cluster_stats, threads=args.threads)
    return rvcp.run_sigprofiler_assignment(
        trinuc192_counts=trinuc192,
        output_root=output_root,
        genome_build=args.genome_build,
        cosmic_version=args.cosmic_version,
        cpu=args.sigprofiler_cpu,
    )


def run_variant(variant: str, latent: np.ndarray | None, preproc: np.ndarray | None,
                context: pl.DataFrame, cell_dir: Path, args: argparse.Namespace) -> dict:
    variant_dir = cell_dir / f"variant_{variant}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    space = build_space(variant, latent, preproc, args)
    log(f"  [{variant}] clustering space {space.shape}; HDBSCAN grid {args.min_cluster_sizes}")
    result = core.hdbscan_grid(
        space,
        min_cluster_sizes=parse_int_csv(args.min_cluster_sizes),
        min_samples=args.min_samples,
        core_dist_n_jobs=args.hdbscan_core_jobs,
        gen_min_span_tree=True,
        cache_dir=str(variant_dir / "hdbscan_cache"),
    )

    panel = metrics.internal_metrics(space, result.labels, eval_rows=args.sil_eval_rows, seed=args.seed)
    panel["relative_validity"] = result.dbcv

    analysis_path = variant_dir / "analysis.parquet"
    write_analysis_parquet(analysis_path, context, result.labels, result.probabilities, space, variant)

    sigprofiler = None
    if not args.no_sigprofiler:
        try:
            sigprofiler = run_sigprofiler(analysis_path, variant_dir, args)
        except Exception as exc:  # SigProfiler is optional and heavy; never fail the run on it
            sigprofiler = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"  [{variant}] SigProfiler failed: {exc}")

    metrics_payload = {
        "variant": variant,
        "space_shape": list(space.shape),
        "selected_min_cluster_size": result.min_cluster_size,
        "min_samples": result.min_samples,
        "grid": result.grid,
        "metrics": panel,
        "analysis_path": str(analysis_path),
        "sigprofiler": sigprofiler,
        "seconds": round(perf_counter() - started, 1),
    }
    (variant_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    log(
        f"  [{variant}] done in {metrics_payload['seconds']}s: "
        f"mcs={result.min_cluster_size} clusters={panel['n_clusters']} "
        f"noise={panel['noise_fraction']:.3f} dbcv={result.dbcv} "
        f"silhouette={panel['silhouette']}"
    )
    return {"labels": result.labels, "payload": metrics_payload}


def cross_variant_agreement(labels_by_variant: dict[str, np.ndarray]) -> list[dict]:
    """Pairwise ARI/AMI/NMI among variants in this cell (rows are aligned by construction)."""
    names = sorted(labels_by_variant)
    pairs: list[dict] = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            agreement = metrics.clustering_agreement(labels_by_variant[left], labels_by_variant[right])
            pairs.append({"a": left, "b": right, **agreement})
    return pairs


def main() -> int:
    overall = perf_counter()
    args = parse_args()
    variants = parse_csv(args.variants or DEFAULT_VARIANTS[args.mode])
    cell_label, row_filter, sample_rows = resolve_cell(args)

    if args.mode != "subsample" and args.sample_rows:
        log(f"NOTE: --sample-rows={args.sample_rows} ignored in --mode {args.mode} "
            f"(all rows passing the filter are used)")

    output_root = Path(args.output_root).resolve()
    cell_dir = output_root / cell_label
    cell_dir.mkdir(parents=True, exist_ok=True)
    duckdb_dir = cell_dir / "_duckdb"
    duckdb_dir.mkdir(parents=True, exist_ok=True)  # duckdb.connect() won't create the parent dir

    log(f"Cell {cell_label} [mode={args.mode}]: variants={variants} filter={row_filter!r} "
        f"sample_rows={sample_rows if sample_rows is not None else 'ALL'}")

    checkpoint_path = Path(args.checkpoint_path).resolve()
    feature_names = load_feature_names(checkpoint_path)
    selected_columns = build_selected_columns(feature_names)

    sample_path = cell_dir / "sample.parquet"
    dedup_start = perf_counter()
    dedup_population, sampled_rows = write_deduplicated_sample(
        parquet_path=args.parquet_path,
        output_path=sample_path,
        row_filter=row_filter,
        selected_columns=selected_columns,
        sample_rows=sample_rows,
        seed=args.seed,
        threads=args.threads,
        database_path=duckdb_dir / "pipeline.duckdb",
        temp_directory=duckdb_dir / "tmp",
        memory_limit=args.duckdb_memory_limit,
    )
    log(
        f"Deduplicated population={dedup_population:,}; using {sampled_rows:,} rows "
        f"({perf_counter() - dedup_start:.1f}s)"
    )

    frame = pl.read_parquet(sample_path)
    context_columns = [name for name in CONTEXT_COLUMNS if name in frame.columns]
    context = frame.select(context_columns)

    # In subsample mode this is loaded ONLY for its standardization stats and category
    # vocabulary -- the encoder is never called (C1/C2 are not in LATENT_VARIANTS).
    inference = LatentInference.from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        feature_spec_path=args.feature_spec_path,
        parquet_path=str(sample_path),
        device=args.device,
    )
    trained_on = checkpoint_training_rows(checkpoint_path)
    if args.mode != "subsample":
        log(f"VAE checkpoint {checkpoint_label(checkpoint_path)}"
            + (f" (trained on {trained_on:,} rows)" if trained_on else ""))

    latent = None
    if any(variant in LATENT_VARIANTS for variant in variants):
        embed_start = perf_counter()
        latent = inference.encode_frame(frame.select(inference.feature_names), batch_size=args.embed_batch_size)
        log(f"Embedded {latent.shape[0]:,} rows -> latent {latent.shape} ({perf_counter() - embed_start:.1f}s)")

    preproc = None
    if any(variant in PREPROC_VARIANTS for variant in variants):
        cat_tensor, num_tensor = transform_frame(
            frame=frame,
            categorical_specs=inference.categorical_specs,
            numeric_specs=inference.numeric_specs,
            numeric_means=inference.numeric_means,
            numeric_stds=inference.numeric_stds,
        )
        preproc = core.build_preprocessed_matrix(cat_tensor, num_tensor, inference.categorical_specs)
        log(f"Built preprocessed matrix {preproc.shape}")

    labels_by_variant: dict[str, np.ndarray] = {}
    variant_payloads: dict[str, dict] = {}
    for variant in variants:
        try:
            outcome = run_variant(variant, latent, preproc, context, cell_dir, args)
            labels_by_variant[variant] = outcome["labels"]
            variant_payloads[variant] = outcome["payload"]
        except Exception as exc:
            log(f"  [{variant}] FAILED: {exc}")
            variant_payloads[variant] = {"variant": variant, "error": f"{type(exc).__name__}: {exc}",
                                         "traceback": traceback.format_exc()}

    cross_variant = cross_variant_agreement(labels_by_variant) if len(labels_by_variant) > 1 else []

    summary = {
        "cell_label": cell_label,
        "mode": args.mode,
        "row_filter": row_filter,
        "sample_rows_requested": sample_rows,
        "sample_rows_used": sampled_rows,
        "deduplicated_population": dedup_population,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_label": checkpoint_label(checkpoint_path),
        # The independent variable for this sweep -- what the aggregation plots against.
        "swept_variable": {
            "checkpoint": "vae_training_rows",
            "rq": "rq_threshold",
            "subsample": "clustered_rows",
        }[args.mode],
        "vae_training_rows": trained_on,
        "rq_threshold": args.rq_threshold,
        "seed": args.seed,
        "variants": {name: payload for name, payload in variant_payloads.items()},
        "cross_variant_agreement": cross_variant,
        "seconds": round(perf_counter() - overall, 1),
    }
    (cell_dir / "cell_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Cell {cell_label} complete in {summary['seconds']}s -> {cell_dir / 'cell_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
