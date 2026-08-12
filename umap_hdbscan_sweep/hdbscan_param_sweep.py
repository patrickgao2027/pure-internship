#!/usr/bin/env python
"""HDBSCAN parameter sweep: fit, label the full cohort, save the labels.

What this does and does not do
------------------------------
It fits HDBSCAN at each grid point, labels all 157.5M cohort rows through
``fast_predict``'s RBC backend, and writes ``cohort_labels.npy``. That is the expensive,
GPU-bound part and the part that has to be swept.

It deliberately stops there. Signature assignment, the SBS96 count matrix, the cluster-size
tables and the profile plots all already exist in
``uv_vae/scripts/run_variant_cluster_pipeline.py`` -- ``annotate_trinuc_counts``,
``write_cluster_sbs96_matrix``, ``run_sigprofiler_assignment``, ``query_cluster_stats``,
``dominant_signature_columns``. Those are the versions every earlier result in this repo was
produced with, so a second implementation here would be a second answer to compare against,
not a shortcut. Point the pipeline at a saved ``cohort_labels.npy`` when signatures are wanted.

The confound this sweep exists to break
---------------------------------------
``scaling_results.json`` swept fit size with ``min_cluster_size = max(50, N * 1e-4)`` --
proportional. Cluster count fell 1338 -> 442 and noise fell 32.6% -> 5.6% from 500K to 25M,
but mcs rose 50x over the same range, so **nothing in that run separates "more data resolves
coarser structure" from "mcs got 50x bigger"**. Here mcs is an independent axis held fixed
across fit sizes.

Why the labelling is affordable at all
--------------------------------------
``approximate_predict`` is brute force -- 109 s per million probe rows against a 25M fit set,
so ~4.8 h per cell. ``fast_predict.py`` swaps in cuML's Random Ball Cover for the neighbour
search and does the same labelling in 5.3 min, 54x. That is what makes a per-cell full-cohort
labelling a sweep step rather than a shortlist step.

Cost model (measured, from scaling_results.json -- fit time is O(N^1.9) and dominates):

    500K  7 s      1M  24 s      5M  470 s     10M  1,870 s
    15M  4,562 s   25M  11,684 s              50M  OOM at 46,450 s

min_samples=15 at 25M OOMed on a 47 GB card, so the 25M row of the grid is min_samples=5 only.

    # print the grid and its projected cost, run nothing
    python umap_hdbscan_sweep/hdbscan_param_sweep.py --dry-run \
        --coords coords.npy --output-dir <out>

    # the sweep (resumable: finished cells are skipped)
    python umap_hdbscan_sweep/hdbscan_param_sweep.py \
        --coords coords.npy --output-dir <out>

    # summarise finished cells
    python umap_hdbscan_sweep/hdbscan_param_sweep.py --aggregate --output-dir <out>

Every cell writes ``<out>/cells/<label>/metrics.json`` and ``cohort_labels.npy`` the moment it
finishes. A crash loses at most the cell in flight.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, REPO_ROOT / "uv_vae" / "scripts",
                  Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import polars as pl


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}", flush=True)


# ── the grid ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Cell:
    fit_rows: int
    min_cluster_size: int
    min_samples: int
    cluster_selection_method: str = "eom"
    cluster_selection_epsilon: float = 0.0

    def label(self) -> str:
        name = (f"fit{self.fit_rows}_mcs{self.min_cluster_size}"
                f"_ms{self.min_samples}_{self.cluster_selection_method}")
        if self.cluster_selection_epsilon:
            name += f"_eps{self.cluster_selection_epsilon}"
        return name


# Measured fit seconds from scaling_results.json, for the cost projection. Interpolated in
# log-log between the measured points (the curve is a clean power law, ~O(N^1.9)).
MEASURED_FIT_SECONDS = {
    500_000: 7.2, 1_000_000: 23.6, 5_000_000: 469.9,
    10_000_000: 1869.6, 15_000_000: 4562.1, 25_000_000: 11684.0,
}


def projected_fit_seconds(fit_rows: int, min_samples: int) -> float:
    sizes = np.array(sorted(MEASURED_FIT_SECONDS))
    seconds = np.array([MEASURED_FIT_SECONDS[size] for size in sizes])
    estimate = float(np.exp(np.interp(np.log(fit_rows), np.log(sizes), np.log(seconds))))
    # min_samples enlarges the kNN graph; ms=15 at 25M OOMed, and the ms=15 probe that did
    # run took 11,714 s against 11,684 s at ms=5, so the time penalty is small even though
    # the memory penalty is not.
    return estimate * (1.0 + 0.02 * (min_samples - 5))


def build_grid(fit_sizes: list[int], min_cluster_sizes: list[int], min_samples: list[int],
               methods: list[str], epsilons: list[float],
               max_min_samples_at: dict[int, int]) -> list[Cell]:
    """Grid ordered so that the cheapest cells run first.

    Cheapest-first is deliberate: it front-loads the information. If the 500K and 1M rows
    show mcs doing nothing (which HDBSCAN_SWEEP_PLAN.md Finding 1 predicts), that is known
    within minutes rather than after the 3.25 h 25M fits.
    """
    cells = []
    for fit_rows in fit_sizes:
        ceiling = max_min_samples_at.get(fit_rows)
        for samples in min_samples:
            if ceiling is not None and samples > ceiling:
                continue
            for mcs in min_cluster_sizes:
                for method in methods:
                    for epsilon in epsilons:
                        cells.append(Cell(fit_rows, mcs, samples, method, epsilon))
    return sorted(cells, key=lambda cell: (cell.fit_rows, cell.min_samples,
                                           cell.min_cluster_size))


# ── one cell ───────────────────────────────────────────────────────────────────

def trinuc_counts_from_labels(context_path: Path, labels: np.ndarray) -> "pl.DataFrame":
    """Group the cohort by (cluster, REF, ALT, X_PREV1, X_NEXT1) -- rvcp's input shape.

    ``run_variant_cluster_pipeline`` normally gets this from DuckDB over an analysis parquet
    carrying a ``cluster_label`` column. Here the labels are an in-memory array, so the
    group-by is done directly against ``context.parquet`` instead of writing a ~2 GB labelled
    parquet per cell. The output columns are exactly what ``annotate_trinuc_counts`` expects,
    so the SBS96 canonicalisation itself is still rvcp's and is not reimplemented.
    """
    frame = pl.read_parquet(context_path, columns=["REF", "ALT", "X_PREV1", "X_NEXT1"])
    if frame.height != labels.shape[0]:
        raise SystemExit(
            f"context.parquet has {frame.height:,} rows but the labelling has "
            f"{labels.shape[0]:,} -- they must come from the same stage-1 run")
    return (
        frame.with_columns(pl.Series("cluster_label", labels))
        .filter(pl.col("cluster_label") >= 0)
        .group_by(["cluster_label", "REF", "ALT", "X_PREV1", "X_NEXT1"])
        .agg(pl.len().alias("count"))
    )


def run_cell(cell: Cell, coords: np.ndarray, cell_dir: Path, args) -> dict:
    """Fit, label the whole cohort, save the labels, then score it. Writes as it goes."""
    import fast_predict

    cell_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"cell": asdict(cell), "label": cell.label(),
                    "started": datetime.now(timezone.utc).isoformat()}
    total_rows = coords.shape[0]

    rng = np.random.default_rng(args.seed)
    fit_indices = np.sort(rng.choice(total_rows, size=cell.fit_rows, replace=False))
    fit_coords = np.asarray(coords[fit_indices], dtype=np.float32)

    log(f"  fitting HDBSCAN on {cell.fit_rows:,} rows "
        f"(mcs={cell.min_cluster_size}, ms={cell.min_samples}, "
        f"{cell.cluster_selection_method}, eps={cell.cluster_selection_epsilon})")
    started = perf_counter()
    clusterer = _fit_hdbscan(fit_coords, cell, args)
    record["fit_seconds"] = round(perf_counter() - started, 1)
    fit_labels = np.asarray(_to_host(clusterer.labels_)).astype(np.int32)
    n_clusters = int(fit_labels.max()) + 1 if (fit_labels >= 0).any() else 0
    record["n_clusters_fit"] = n_clusters
    record["fit_noise_fraction"] = float((fit_labels < 0).mean())
    log(f"    {record['fit_seconds']}s, {n_clusters:,} clusters, "
        f"{record['fit_noise_fraction'] * 100:.2f}% noise on the fit set")
    if n_clusters == 0:
        record["error"] = "no non-noise clusters"
        (cell_dir / "metrics.json").write_text(json.dumps(record, indent=2))
        return record

    # Full-cohort labels: the fit rows keep the labels the fit gave them, held rows come from
    # the RBC path. Re-predicting the fit rows would be both slower and less accurate --
    # approximate_predict is an approximation OF the fit, so where the fit's own answer
    # exists it is the better one.
    log(f"  labelling all {total_rows:,} rows (RBC)")
    started = perf_counter()
    tables = fast_predict.build_tables(clusterer, n_fit=cell.fit_rows,
                                       min_samples=cell.min_samples)
    held_mask = np.ones(total_rows, dtype=bool)
    held_mask[fit_indices] = False
    held_indices = np.nonzero(held_mask)[0]

    labels = np.empty(total_rows, dtype=np.int32)
    labels[fit_indices] = fit_labels
    index = fast_predict.build_index(fit_coords, 2 * tables.min_samples, args.backend)
    held_labels, held_probabilities = fast_predict.predict(
        tables, fit_coords, np.asarray(coords[held_indices], dtype=np.float32),
        backend=args.backend, batch_rows=args.batch_rows, index=index)
    labels[held_indices] = held_labels
    record["label_seconds"] = round(perf_counter() - started, 1)
    record["held_mean_probability"] = float(held_probabilities.mean())
    del held_labels, held_probabilities

    np.save(cell_dir / "cohort_labels.npy", labels)
    record["cohort_labels"] = str(cell_dir / "cohort_labels.npy")
    record["cohort_n_clusters"] = int(labels.max()) + 1 if (labels >= 0).any() else 0
    record["cohort_noise_fraction"] = float((labels < 0).mean())
    log(f"    {record['label_seconds']}s, "
        f"{record['cohort_noise_fraction'] * 100:.2f}% cohort noise")

    if args.context:
        _score_cell(record, cell_dir, Path(args.context), labels, args)

    record["finished"] = datetime.now(timezone.utc).isoformat()
    (cell_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    return record


def _score_cell(record: dict, cell_dir: Path, context_path: Path, labels: np.ndarray,
                args) -> None:
    """SBS96 matrix -> SigProfiler -> metrics, all through run_variant_cluster_pipeline.

    Scoring failures are recorded and swallowed. The labelling above is the expensive part
    (up to 3.25 h of fitting); losing it because SigProfiler tripped over one cell would be
    the worst possible trade, and the saved cohort_labels.npy can always be rescored later.
    """
    import assignment_metrics
    import run_variant_cluster_pipeline as rvcp
    import spectrum_metrics

    try:
        log("  building SBS96 counts")
        started = perf_counter()
        counts = trinuc_counts_from_labels(context_path, labels)
        sizes = (pl.DataFrame({"cluster_label": labels})
                 .filter(pl.col("cluster_label") >= 0)
                 .group_by("cluster_label").agg(pl.len().alias("cluster_size")))
        trinuc = rvcp.annotate_trinuc_counts(counts, sizes)
        record["counts_seconds"] = round(perf_counter() - started, 1)

        log(f"  SigProfiler on {sizes.height:,} clusters")
        started = perf_counter()
        paths = rvcp.run_sigprofiler_assignment(
            trinuc, cell_dir, args.genome_build, args.cosmic_version, args.sigprofiler_cpu)
        record["sigprofiler_seconds"] = round(perf_counter() - started, 1)
        record["sigprofiler_paths"] = paths

        stats_path = (Path(paths["output_dir"]) / "Assignment_Solution" / "Solution_Stats"
                      / "Assignment_Solution_Samples_Stats.txt")
        record["assignment"] = assignment_metrics.summarise(stats_path)
        record["spectrum"] = spectrum_metrics.summarise(Path(paths["matrix_path"]))
        log(f"    cosine mean {record['assignment']['cosine_mean']:.3f}, "
            f"L1% mean {record['assignment'].get('l1_pct_mean', float('nan')):.1f}, "
            f"top-channel median {record['spectrum']['top_channel_share_median']:.3f}")
    except Exception as exc:  # noqa: BLE001 - never lose a labelled cohort over scoring
        record["scoring_error"] = f"{type(exc).__name__}: {exc}"
        record["scoring_traceback"] = traceback.format_exc()
        log(f"  SCORING FAILED (labels kept): {type(exc).__name__}: {str(exc)[:200]}")


def _fit_hdbscan(fit_coords: np.ndarray, cell: Cell, args):
    try:
        from cuml.cluster import HDBSCAN as CumlHDBSCAN
        return CumlHDBSCAN(
            min_cluster_size=cell.min_cluster_size, min_samples=cell.min_samples,
            cluster_selection_method=cell.cluster_selection_method,
            cluster_selection_epsilon=cell.cluster_selection_epsilon,
            metric="euclidean", prediction_data=True,
        ).fit(fit_coords)
    except ImportError:
        import hdbscan
        return hdbscan.HDBSCAN(
            min_cluster_size=cell.min_cluster_size, min_samples=cell.min_samples,
            cluster_selection_method=cell.cluster_selection_method,
            cluster_selection_epsilon=cell.cluster_selection_epsilon,
            metric="euclidean", prediction_data=True, core_dist_n_jobs=args.threads,
        ).fit(fit_coords)


def _to_host(array):
    for attribute in ("to_numpy", "get"):
        method = getattr(array, attribute, None)
        if callable(method):
            return np.asarray(method())
    return np.asarray(array)


# ── aggregate ──────────────────────────────────────────────────────────────────

def aggregate(output_dir: Path) -> pl.DataFrame:
    """Every finished cell: clustering shape, spectrum concentration, SigProfiler fit.

    ``top_chan`` leads because it is the gate. The 2026-08-04 runs measured a median cluster
    at 0.87 of a single trinucleotide channel, and no COSMIC profile is that concentrated --
    so a cell above ~0.5 there has already lost the signature question and its cosine is
    reporting channel identity rather than fit quality.
    """
    rows = []
    for metrics_path in sorted((output_dir / "cells").glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text())
        if "error" in payload:
            continue
        cell = payload["cell"]
        assignment = payload.get("assignment", {})
        spectrum = payload.get("spectrum", {})
        nan = float("nan")
        rows.append({
            "label": payload["label"],
            "fit_rows": cell["fit_rows"],
            "mcs": cell["min_cluster_size"],
            "ms": cell["min_samples"],
            "method": cell["cluster_selection_method"],
            "eps": cell["cluster_selection_epsilon"],
            "n_clusters": payload.get("cohort_n_clusters", 0),
            "noise_pct": round(payload.get("cohort_noise_fraction", 0) * 100, 2),
            "top_chan": round(spectrum.get("top_channel_share_median", nan), 3),
            "tc>0.5": round(spectrum.get("frac_above_05", nan), 3),
            "tc>0.8": round(spectrum.get("frac_above_08", nan), 3),
            "cos_mean": round(assignment.get("cosine_mean", nan), 4),
            "cos_wmean": round(assignment.get("cosine_weighted_mean", nan), 4),
            "l1%_mean": round(assignment.get("l1_pct_mean", nan), 2),
            "l2%_mean": round(assignment.get("l2_pct_mean", nan), 2),
            "mut%>0.8": round(100 * assignment.get("mutation_share_above_08", nan), 1),
            "fit_s": payload.get("fit_seconds", 0),
            "label_s": payload.get("label_seconds", 0),
        })
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("top_chan", nulls_last=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--coords", type=Path, help="coords.npy, 157.5M x 2")
    parser.add_argument("--context", type=Path,
                        help="<embed_dir>/context.parquet, row-aligned with coords.npy. "
                             "Without it the sweep only labels and saves; with it each cell "
                             "also runs SigProfiler and scores itself.")
    parser.add_argument("--genome-build", default="GRCh38")
    parser.add_argument("--cosmic-version", default="3.5")
    parser.add_argument("--sigprofiler-cpu", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the grid and its projected cost, run nothing")
    parser.add_argument("--aggregate", action="store_true",
                        help="summarise finished cells and exit")

    parser.add_argument("--fit-sizes", default="500000,1000000,5000000,25000000")
    parser.add_argument("--min-cluster-sizes", default="250,500,1000,2500",
                        help="held FIXED across fit sizes -- that is the point of this sweep")
    parser.add_argument("--min-samples", default="5,15")
    parser.add_argument("--methods", default="eom")
    parser.add_argument("--epsilons", default="0.0")
    parser.add_argument("--max-ms-at-25m", type=int, default=5,
                        help="min_samples ceiling at 25M rows; ms=15 OOMed on a 47 GB card")

    parser.add_argument("--backend", default="rbc", choices=["rbc", "brute", "sklearn"])
    parser.add_argument("--batch-rows", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-budget-gb", type=float, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--continue-after-failure", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        table = aggregate(output_dir)
        if table.is_empty():
            log("no finished cells yet")
            return 0
        with pl.Config(tbl_rows=200, tbl_cols=30, tbl_width_chars=250):
            print(table)
        table.write_csv(output_dir / "sweep_ranking.csv")
        log(f"wrote {output_dir / 'sweep_ranking.csv'}")
        return 0

    fit_sizes = [int(v) for v in args.fit_sizes.split(",") if v]
    grid = build_grid(
        fit_sizes=fit_sizes,
        min_cluster_sizes=[int(v) for v in args.min_cluster_sizes.split(",") if v],
        min_samples=[int(v) for v in args.min_samples.split(",") if v],
        methods=[v for v in args.methods.split(",") if v],
        epsilons=[float(v) for v in args.epsilons.split(",") if v],
        max_min_samples_at={25_000_000: args.max_ms_at_25m, 50_000_000: 0},
    )

    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    done = {path.parent.name for path in cells_dir.glob("*/metrics.json")} if args.resume else set()
    pending = [cell for cell in grid if cell.label() not in done]

    fit_total = sum(projected_fit_seconds(c.fit_rows, c.min_samples) for c in pending)
    # 5.3 min RBC labelling, plus ~2 min counts + SigProfiler when scoring is on.
    overhead = len(pending) * (5.3 * 60 + (120 if args.context else 0))
    log(f"grid: {len(grid)} cells, {len(done)} already done, {len(pending)} to run")
    log(f"projected: {fit_total / 3600:.1f} h fitting + {overhead / 3600:.1f} h "
        f"labelling{'/scoring' if args.context else ''} "
        f"= {(fit_total + overhead) / 3600:.1f} h total")
    for cell in pending:
        log(f"    {cell.label():48} ~{projected_fit_seconds(cell.fit_rows, cell.min_samples) / 60:6.1f} min fit")
    if args.dry_run:
        return 0

    if not args.coords:
        raise SystemExit("--coords is required to run the sweep")
    coords = np.load(args.coords, mmap_mode="r")
    log(f"coords: {coords.shape[0]:,} x {coords.shape[1]}")
    if args.context:
        log(f"scoring on: {args.context} (SigProfiler per cell)")
    else:
        log("no --context: cells will be labelled and saved but NOT scored")

    if args.gpu_budget_gb:
        try:
            import sweep_core
            sweep_core.apply_gpu_budget("sweep", budget_gb=args.gpu_budget_gb)
        except Exception as exc:  # noqa: BLE001
            log(f"gpu budget not applied: {exc}")

    (output_dir / "sweep_config.json").write_text(json.dumps({
        "grid": [asdict(c) for c in grid], "seed": args.seed, "backend": args.backend,
        "started": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    for position, cell in enumerate(pending, start=1):
        log(f"[{position}/{len(pending)}] {cell.label()}")
        cell_dir = cells_dir / cell.label()
        try:
            run_cell(cell, coords, cell_dir, args)
        except Exception as exc:  # noqa: BLE001 - a failed cell must not lose the sweep
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "failed.json").write_text(json.dumps({
                "cell": asdict(cell), "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }, indent=2))
            log(f"  FAILED: {type(exc).__name__}: {str(exc)[:300]}")
            if not args.continue_after_failure:
                raise

    table = aggregate(output_dir)
    if not table.is_empty():
        table.write_csv(output_dir / "sweep_ranking.csv")
        with pl.Config(tbl_rows=200, tbl_cols=30, tbl_width_chars=250):
            print(table)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
