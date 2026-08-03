"""Stage 2 -- sweep UMAP x HDBSCAN over the deduplicated latents, SigProfiler at every cell.

Grid, ordered so the expensive axis moves slowest::

    for (n_neighbors, min_dist, n_components):   # one UMAP fit each -- the dominant cost
        fit UMAP
        for (min_samples, min_cluster_size):     # one HDBSCAN fit each
            fit HDBSCAN -> metrics -> analysis.parquet -> SigProfiler

Every cell writes its own ``metrics.json`` and is skipped on re-run if that file exists, so
a 480-cell sweep survives being killed and restarted -- which it will be, because the UMAP
fits at cohort scale run for hours.

**Fit size.** ``--fit-rows all`` (the default) fits on every deduplicated locus, as asked.
Be aware of the wall this runs into: cuML's UMAP holds a kNN graph of roughly
``N x n_neighbors x 8`` bytes, so N=100M at n_neighbors=100 wants ~80 GB on a 48 GB card,
and HDBSCAN's MST is heavier still. When a cell OOMs it is recorded as a failure and the
sweep continues rather than dying -- inspect ``sweep_summary.json`` afterwards and re-run
the failures with a smaller ``--fit-rows``. With a cap set, the remaining rows are labelled
by ``transform`` + ``approximate_predict``, so every locus still reaches SigProfiler.

**Agreement.** Pairwise ARI across 480 cells would be ~115k comparisons. Instead every cell
is compared against the DBCV-best cell on a fixed subsample, which is both affordable and
the more readable question: how far does each configuration sit from the best one?
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, REPO_ROOT / "uv_vae" / "scripts",
                  REPO_ROOT / "clust_regime_sweep", Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import polars as pl

import clustering_metrics as metrics
import sweep_core as core


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UMAP x HDBSCAN sweep with SigProfiler per cell")
    parser.add_argument("--embed-dir", required=True, help="stage 1 output (latent.npy + context.parquet)")
    parser.add_argument("--output-root", required=True)

    parser.add_argument("--umap-n-neighbors", default="15,30,50,100")
    parser.add_argument("--umap-min-dist", default="0.0,0.05,0.1,0.25")
    parser.add_argument("--umap-n-components", default="2,16")
    parser.add_argument("--umap-n-epochs", type=int, default=200,
                        help="pinned so transform() is batch-size invariant; see UmapConfig")
    parser.add_argument("--min-cluster-sizes", default="100,250,500,1000,2500")
    parser.add_argument("--min-samples", default="5,25,50")
    parser.add_argument("--metric", default="euclidean")

    parser.add_argument("--fit-rows", default="all",
                        help="'all' (default) or N: rows UMAP/HDBSCAN are FIT on. "
                             "Rows beyond the cap are labelled by transform/approximate_predict.")
    parser.add_argument("--transform-batch-size", type=int, default=5_000_000)
    parser.add_argument("--sil-eval-rows", type=int, default=50_000)
    parser.add_argument("--agreement-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--hdbscan-core-jobs", type=int, default=1)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--gpu-budget-gb", type=float, default=None,
                        help="GPU ceiling for THIS process. Caps must add up across "
                             "co-tenants -- a trainer capped at 10 GB leaves 6 GB here.")

    parser.add_argument("--no-sigprofiler", action="store_true")
    parser.add_argument("--sigprofiler-cpu", type=int, default=16)
    parser.add_argument("--cosmic-version", default="3.5")
    parser.add_argument("--genome-build", default="GRCh38")

    parser.add_argument("--dry-run", action="store_true", help="print the grid and cost, fit nothing")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def run_sigprofiler(analysis_path: Path, output_root: Path, args: argparse.Namespace) -> dict | None:
    """Reuse the clustering pipeline's SigProfiler block (lazy import -- it is heavy).

    ``UV_VAE_DISABLE_CUML`` is not optional here, and must be set BEFORE the import.
    run_variant_cluster_pipeline installs a GPU budget at module scope with the
    default rmm share (0.25) -- correct for the training path it was written for,
    wrong for this one. Importing it mid-sweep reinitialises the RMM pool that cuML
    UMAP/HDBSCAN is already allocating from, to a quarter of the budget, while the
    live pool is larger than that: every cell after the first then dies with
    "maximum pool size exceeded". The same flag also skips ``cuml.accel.install()``,
    which is equally unwanted -- this stage calls cuML explicitly, and patching the
    estimator stack half way through would change which HDBSCAN later cells get.
    """
    os.environ["UV_VAE_DISABLE_CUML"] = "1"
    import run_variant_cluster_pipeline as rvcp

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


def write_analysis_parquet(
    path: Path, context: pl.DataFrame, labels: np.ndarray,
    probabilities: np.ndarray, coordinates: np.ndarray,
) -> None:
    columns = {name: context[name] for name in context.columns}
    columns["cluster_label"] = pl.Series("cluster_label", labels.astype(np.int32, copy=False))
    columns["cluster_probability"] = pl.Series(
        "cluster_probability", probabilities.astype(np.float32, copy=False)
    )
    # Only 2D embeddings get coordinate columns -- a 16D UMAP would add 16 columns that
    # nothing downstream reads, and SigProfiler never looks at coordinates.
    if coordinates.shape[1] == 2:
        columns["umap_1"] = pl.Series("umap_1", coordinates[:, 0].astype(np.float32, copy=False))
        columns["umap_2"] = pl.Series("umap_2", coordinates[:, 1].astype(np.float32, copy=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(columns).write_parquet(path)


def completed_payload(metrics_path: Path) -> dict | None:
    """A finished cell's payload, or None if it needs (re-)running.

    A cell that recorded an ``error`` is NOT treated as complete: the usual reason to
    re-run a sweep is to retry cells that OOMed with a smaller ``--fit-rows``, and a
    failure that is still a failure fails fast anyway.
    """
    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return None  # killed mid-write
    return None if "error" in payload else payload


def fit_indices(total_rows: int, fit_rows: int | None, seed: int) -> np.ndarray | None:
    """Row indices UMAP/HDBSCAN are fit on; None means every row."""
    if fit_rows is None or fit_rows >= total_rows:
        return None
    return np.sort(np.random.default_rng(seed).choice(total_rows, size=fit_rows, replace=False))


def embed_all_rows(
    fitted: core.FittedUmap, latent: np.ndarray, indices: np.ndarray | None, batch_size: int,
) -> np.ndarray:
    """Full-population embedding: the fit's own output, plus transform() for the rest."""
    if indices is None:
        return fitted.embedding
    embedding = np.empty((latent.shape[0], fitted.embedding.shape[1]), dtype=np.float32)
    embedding[indices] = fitted.embedding
    held_out = np.setdiff1d(np.arange(latent.shape[0]), indices, assume_unique=True)
    for start in range(0, held_out.size, batch_size):
        block = held_out[start:start + batch_size]
        embedding[block] = fitted.transform(latent[block], batch_size=batch_size)
    return embedding


def label_all_rows(
    fitted: core.FittedHdbscan, space: np.ndarray, indices: np.ndarray | None, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if indices is None:
        return fitted.labels, fitted.probabilities
    labels = np.empty(space.shape[0], dtype=np.int32)
    probabilities = np.empty(space.shape[0], dtype=np.float32)
    labels[indices] = fitted.labels
    probabilities[indices] = fitted.probabilities
    held_out = np.setdiff1d(np.arange(space.shape[0]), indices, assume_unique=True)
    for start in range(0, held_out.size, batch_size):
        block = held_out[start:start + batch_size]
        block_labels, block_probabilities = fitted.predict(space[block], batch_size=batch_size)
        labels[block] = block_labels
        probabilities[block] = block_probabilities
    return labels, probabilities


def persist_model(model: object, path: Path) -> str | None:
    """Save a fitted model for stage 3. cuML pickling is version-dependent, so failure is
    reported rather than raised -- stage 3 can refit from the recorded config."""
    try:
        import joblib

        joblib.dump(model, path)
        return str(path)
    except Exception as exc:
        log(f"    could not persist model ({type(exc).__name__}: {exc}); stage 3 must refit")
        return None


def main() -> int:
    overall = perf_counter()
    args = parse_args()

    embed_dir = Path(args.embed_dir).resolve()
    summary = json.loads((embed_dir / "embed_summary.json").read_text())
    latent = np.load(summary["latent_path"], mmap_mode="r")
    context = pl.read_parquet(summary["context_path"])
    total_rows = latent.shape[0]
    if context.height != total_rows:
        raise SystemExit(
            f"latent has {total_rows:,} rows but context has {context.height:,} -- "
            "stage 1 output is inconsistent, re-run it"
        )

    grid = core.build_grid(
        n_neighbors=parse_int_csv(args.umap_n_neighbors),
        min_dist=parse_float_csv(args.umap_min_dist),
        n_components=parse_int_csv(args.umap_n_components),
        min_cluster_size=parse_int_csv(args.min_cluster_sizes),
        min_samples=parse_int_csv(args.min_samples),
        metric=args.metric,
        seed=args.seed,
        n_epochs=args.umap_n_epochs,
    )
    fit_rows = None if str(args.fit_rows).lower() == "all" else int(args.fit_rows)
    indices = fit_indices(total_rows, fit_rows, args.seed)
    use_gpu = False if args.force_cpu else core.gpu_available()

    log(f"Latents: {total_rows:,} x {latent.shape[1]} from {summary['latent_path']}")
    log(f"Grid: {len(grid.umap_configs)} UMAP fits x {len(grid.hdbscan_configs)} HDBSCAN cuts "
        f"= {len(grid)} cells")
    log(f"Backend: {'cuML (GPU)' if use_gpu else 'CPU'}   "
        f"fit rows: {'ALL' if indices is None else f'{indices.size:,} of {total_rows:,}'}")
    log(f"SigProfiler: {'disabled' if args.no_sigprofiler else 'every cell'}")

    if args.dry_run:
        for umap_config in grid.umap_configs:
            log(f"  UMAP {umap_config.label()}")
        for hdbscan_config in grid.hdbscan_configs:
            log(f"  HDBSCAN {hdbscan_config.label()}")
        log(f"Dry run: {len(grid)} cells would run. Nothing was fit.")
        return 0

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Applied AFTER --dry-run returns, so planning the grid never touches the GPU or
    # trips the co-tenancy check, and BEFORE the first cuML call, because RMM has to be
    # bounded before anything allocates from it.
    budget_report = (
        core.apply_gpu_budget("sweep", budget_gb=args.gpu_budget_gb) if use_gpu else
        {"stage": "sweep", "enabled": False, "note": "CPU backend; no GPU budget applied"}
    )

    fit_latent = latent if indices is None else np.asarray(latent[indices])

    cells: list[dict] = []
    agreement_rows = min(args.agreement_rows, total_rows)
    agreement_index = np.sort(
        np.random.default_rng(args.seed).choice(total_rows, size=agreement_rows, replace=False)
    )
    agreement_labels: dict[str, np.ndarray] = {}

    for umap_config in grid.umap_configs:
        umap_dir = output_root / umap_config.label()
        umap_dir.mkdir(parents=True, exist_ok=True)

        done: dict[str, dict] = {}
        pending = []
        for hdbscan_config in grid.hdbscan_configs:
            payload = (
                completed_payload(umap_dir / hdbscan_config.label() / "metrics.json")
                if args.resume else None
            )
            if payload is None:
                pending.append(hdbscan_config)
            else:
                done[hdbscan_config.label()] = payload

        if not pending:
            # Skipping the UMAP fit is the whole point of resuming -- it is the expensive
            # half. The agreement labels still have to be reloaded, or the summary's
            # agreement-vs-best table would come back empty on a fully-resumed run.
            log(f"[{umap_config.label()}] all {len(done)} cells already complete, skipping UMAP fit")
            for payload in done.values():
                cells.append(payload)
                path = payload.get("agreement_labels_path")
                if path and Path(path).exists():
                    agreement_labels[payload["cell"]] = np.load(path)
            continue

        umap_started = perf_counter()
        log(f"[{umap_config.label()}] fitting UMAP on {fit_latent.shape[0]:,} rows "
            f"({len(pending)}/{len(grid.hdbscan_configs)} cells pending)")
        try:
            fitted_umap = core.fit_umap(fit_latent, umap_config, use_gpu=use_gpu)
            space = embed_all_rows(fitted_umap, latent, indices, args.transform_batch_size)
        except Exception as exc:
            log(f"[{umap_config.label()}] UMAP FAILED: {type(exc).__name__}: {exc}")
            (umap_dir / "umap_error.json").write_text(json.dumps(
                {"config": umap_config.as_dict(), "error": f"{type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc()}, indent=2))
            continue
        umap_seconds = perf_counter() - umap_started
        log(f"[{umap_config.label()}] UMAP done in {umap_seconds:.0f}s -> {space.shape}")
        model_path = persist_model(fitted_umap.model, umap_dir / "umap_model.joblib")
        (umap_dir / "umap_config.json").write_text(json.dumps(
            {**umap_config.as_dict(), "seconds": round(umap_seconds, 1),
             "backend": fitted_umap.backend, "model_path": model_path}, indent=2))

        fit_space = space if indices is None else space[indices]

        for hdbscan_config in grid.hdbscan_configs:
            cell_dir = umap_dir / hdbscan_config.label()
            metrics_path = cell_dir / "metrics.json"
            already = done.get(hdbscan_config.label())
            if already is not None:
                cells.append(already)
                path = already.get("agreement_labels_path")
                if path and Path(path).exists():
                    agreement_labels[already["cell"]] = np.load(path)
                continue

            cell_dir.mkdir(parents=True, exist_ok=True)
            cell_started = perf_counter()
            cell_name = f"{umap_config.label()}/{hdbscan_config.label()}"
            fitted_hdbscan = None
            try:
                fitted_hdbscan = core.fit_hdbscan(
                    fit_space, hdbscan_config, use_gpu=use_gpu,
                    core_dist_n_jobs=args.hdbscan_core_jobs,
                    cache_dir=None if use_gpu else str(umap_dir / "hdbscan_cache"),
                )
                labels, probabilities = label_all_rows(
                    fitted_hdbscan, space, indices, args.transform_batch_size
                )

                panel = metrics.internal_metrics(
                    space, labels, eval_rows=args.sil_eval_rows, seed=args.seed
                )
                panel["relative_validity"] = fitted_hdbscan.dbcv

                analysis_path = cell_dir / "analysis.parquet"
                write_analysis_parquet(analysis_path, context, labels, probabilities, space)

                agreement_path = cell_dir / "agreement_labels.npy"
                np.save(agreement_path, labels[agreement_index])
                agreement_labels[cell_name] = labels[agreement_index]

                sigprofiler = None
                if not args.no_sigprofiler:
                    try:
                        sigprofiler = run_sigprofiler(analysis_path, cell_dir, args)
                    except Exception as exc:
                        sigprofiler = {"error": f"{type(exc).__name__}: {exc}"}
                        log(f"    [{hdbscan_config.label()}] SigProfiler failed: {exc}")

                payload = {
                    "cell": cell_name,
                    "umap": umap_config.as_dict(),
                    "hdbscan": hdbscan_config.as_dict(),
                    "space_shape": list(space.shape),
                    "fit_rows": int(fit_space.shape[0]),
                    "metrics": panel,
                    "dbcv": fitted_hdbscan.dbcv,
                    "analysis_path": str(analysis_path),
                    "agreement_labels_path": str(agreement_path),
                    "sigprofiler": sigprofiler,
                    "backend": fitted_hdbscan.backend,
                    "seconds": round(perf_counter() - cell_started, 1),
                }
                metrics_path.write_text(json.dumps(payload, indent=2))
                cells.append(payload)
                log(f"    [{hdbscan_config.label()}] {panel['n_clusters']} clusters  "
                    f"noise={panel['noise_fraction']:.3f}  dbcv={fitted_hdbscan.dbcv}  "
                    f"sil={panel['silhouette']}  ({payload['seconds']}s)")
            except Exception as exc:
                payload = {
                    "cell": cell_name,
                    "umap": umap_config.as_dict(),
                    "hdbscan": hdbscan_config.as_dict(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "seconds": round(perf_counter() - cell_started, 1),
                }
                metrics_path.write_text(json.dumps(payload, indent=2))
                cells.append(payload)
                log(f"    [{hdbscan_config.label()}] FAILED: {type(exc).__name__}: {exc}")
            finally:
                # Drop this cell's clusterer before the next one is fitted. Rebinding
                # the name at the top of the next iteration would only release it AFTER
                # that fit had already allocated, so the pool briefly holds two -- and
                # with prediction_data=True each one keeps a per-row structure on the
                # GPU, which at 157M rows is not a rounding error.
                fitted_hdbscan = None

    ok = [cell for cell in cells if "error" not in cell]
    scored = [cell for cell in ok if cell.get("dbcv") is not None
              and cell["metrics"]["n_clusters"] >= 2]
    best = max(scored, key=lambda cell: cell["dbcv"]) if scored else None

    agreement_vs_best: list[dict] = []
    if best is not None and best["cell"] in agreement_labels:
        reference = agreement_labels[best["cell"]]
        for cell in ok:
            name = cell["cell"]
            if name in agreement_labels:
                agreement_vs_best.append(
                    {"cell": name, **metrics.clustering_agreement(reference, agreement_labels[name])}
                )
        agreement_vs_best.sort(key=lambda entry: entry["ari"], reverse=True)

    sweep_summary = {
        "embed_dir": str(embed_dir),
        "total_rows": total_rows,
        "latent_dim": int(latent.shape[1]),
        "fit_rows": "all" if indices is None else int(indices.size),
        "backend": "cuml" if use_gpu else "cpu",
        # Recorded because UMAP.transform is not batch-invariant (see FittedUmap.transform):
        # comparing two runs is only meaningful when this matches.
        "transform_batch_size": args.transform_batch_size,
        "umap_n_epochs": args.umap_n_epochs,
        "gpu_budget": budget_report,
        "grid": {
            "umap_n_neighbors": parse_int_csv(args.umap_n_neighbors),
            "umap_min_dist": parse_float_csv(args.umap_min_dist),
            "umap_n_components": parse_int_csv(args.umap_n_components),
            "min_cluster_sizes": parse_int_csv(args.min_cluster_sizes),
            "min_samples": parse_int_csv(args.min_samples),
            "cells": len(grid),
        },
        "cells_completed": len(ok),
        "cells_failed": len(cells) - len(ok),
        "best_cell": best["cell"] if best else None,
        "best_dbcv": best["dbcv"] if best else None,
        "agreement_vs_best": agreement_vs_best,
        "results": cells,
        "seconds": round(perf_counter() - overall, 1),
    }
    (output_root / "sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2))

    log("")
    log(f"Sweep complete: {len(ok)}/{len(cells)} cells in {sweep_summary['seconds']}s")
    if best:
        log(f"Best by DBCV: {best['cell']}  dbcv={best['dbcv']}  "
            f"clusters={best['metrics']['n_clusters']}  noise={best['metrics']['noise_fraction']:.3f}")
        log(f"Stage 3: pass --cell '{best['cell']}' to project all reads through it.")
    log(f"Summary -> {output_root / 'sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
