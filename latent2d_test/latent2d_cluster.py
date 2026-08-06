"""Cluster a 2-D VAE latent DIRECTLY with HDBSCAN -- no UMAP -- and run SigProfiler.

This is the "what if the VAE bottleneck itself is the 2-D embedding" test. The normal
pipeline is ``VAE(16-D) -> UMAP(2-D) -> HDBSCAN``; here the VAE is trained with
``latent_dim=2`` and UMAP is removed entirely, so the clustering space is whatever the
encoder's posterior mean already is.

It is a drop-in replacement for ``umap_hdbscan_sweep/stage2_sweep.py`` and reads the SAME
stage-1 output (``latent.npy`` + ``context.parquet``), so stages 0 and 1 are reused
unchanged -- only the checkpoint differs. Everything downstream of the clustering
(analysis parquet, trinucleotide counts, SigProfilerAssignment) is the same code the
16-D pipeline runs, which is what makes the two results comparable.

Two things this does that stage 2 does not, both because the space is now the raw latent
rather than a UMAP embedding:

* **It reports the latent's scale before clustering.** UMAP output has a conventional
  extent (roughly tens of units) that the ``min_cluster_size`` grid was tuned against; a
  VAE latent's extent depends on ``kl_weight`` and is not knowable in advance. A
  ``cluster_selection_epsilon`` or a density threshold carried over from UMAP space is
  meaningless until you know that number, so it is printed and recorded.
* **It checks for a collapsed dimension.** With only two latent dimensions, one
  posterior-collapsed dimension leaves a 1-D clustering space, and HDBSCAN will happily
  produce confident-looking clusters on a line. ``latent_stats.collapsed_dims`` in the
  summary is the check; a non-empty list invalidates the run, not just weakens it.

The clustering grid is flat -- there is no expensive outer axis to amortise -- so it
sweeps ``min_samples x min_cluster_size x cluster_selection_method x
cluster_selection_epsilon`` and writes one directory per cell, resuming like stage 2.

Usage::

    python latent2d_test/latent2d_cluster.py \
        --embed-dir  <run>/stage1_embed \
        --output-root <run>/latent2d_cluster \
        --fit-rows 5000000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (
    REPO_ROOT / "uv_vae",
    REPO_ROOT,
    REPO_ROOT / "uv_vae" / "scripts",
    REPO_ROOT / "clust_regime_sweep",
    REPO_ROOT / "umap_hdbscan_sweep",
    Path(__file__).resolve().parent,
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import polars as pl

import clustering_metrics as metrics
import sweep_core as core
import umap_metrics as um


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HDBSCAN directly on a 2-D VAE latent (UMAP removed) + SigProfiler per cell"
    )
    parser.add_argument("--embed-dir", required=True,
                        help="stage 1 output dir (latent.npy + context.parquet + embed_summary.json)")
    parser.add_argument("--output-root", required=True)

    parser.add_argument("--min-cluster-sizes", default="100,250,500,1000,2500")
    parser.add_argument("--min-samples", default="5,25,50")
    parser.add_argument("--cluster-selection-methods", default="eom",
                        help="comma-separated: eom, leaf. 'leaf' returns the finest clusters "
                             "in the condensed tree instead of the most persistent ones.")
    parser.add_argument("--cluster-selection-epsilons", default="0",
                        help="comma-separated distances IN LATENT UNITS below which clusters are "
                             "merged. Only meaningful once you have read the latent scale this "
                             "script prints -- a value tuned in UMAP space does not carry over.")
    parser.add_argument("--metric", default="euclidean")

    parser.add_argument("--scale", default="none", choices=["none", "standardize"],
                        help="'standardize' divides each latent dimension by its std (fit on the "
                             "fit subsample). Off by default: it is a real change to the geometry "
                             "HDBSCAN sees, not a formatting step.")

    parser.add_argument("--fit-rows", default="all",
                        help="'all' or N: rows HDBSCAN is FIT on. Rows beyond the cap are "
                             "labelled by approximate_predict, so every locus still reaches "
                             "SigProfiler.")
    parser.add_argument("--predict-batch-size", type=int, default=5_000_000)

    # Metric tiers, named and defaulted to match umap_hdbscan_sweep/size_convergence.py so
    # the numbers here sit in the same table as that sweep's without a footnote.
    #   full  linear in rows          -> everything (Davies-Bouldin, Calinski-Harabasz)
    #   knn   needs neighbour lists   -> --knn-rows  (trustworthiness, continuity, RNX)
    #   pair  needs pairwise distance -> --pair-rows (silhouette, distance correlations)
    parser.add_argument("--knn-rows", type=int, default=250_000)
    parser.add_argument("--pair-rows", type=int, default=20_000)
    parser.add_argument("--distance-pairs", type=int, default=20_000_000)
    parser.add_argument("--metric-k", type=int, default=15,
                        help="neighbourhood size for trustworthiness and continuity")
    parser.add_argument("--metric-k-max", type=int, default=400,
                        help="neighbour-list width and the rank horizon beyond which "
                             "penalties are clamped. 400 is measured, not guessed (see "
                             "size_convergence.py). Check clamped_fraction and raise it "
                             "if that exceeds 0.05.")
    parser.add_argument("--agreement-rows", type=int, default=200_000)
    parser.add_argument("--agreement-reference", default=None,
                        help="Optional agreement_labels.npy from a stage-2 cell. Gives the "
                             "headline number: ARI between this 2-D-latent clustering and the "
                             "16-D+UMAP baseline. Requires the same --seed, --agreement-rows and "
                             "the same stage-0 dedup population.")
    parser.add_argument("--agreement-reference-label", default=None,
                        help="name for the reference in the summary (defaults to its path)")
    parser.add_argument("--reference-latent", default=None,
                        help="The 16-D baseline's stage-1 latent.npy, over the SAME dedup rows. "
                             "Turns on the embedding-quality panel: trustworthiness, continuity, "
                             "RNX AUC and distance correlations of this 2-D space against the "
                             "16-D one. The baseline scores its 2-D UMAP against that same 16-D "
                             "reference, so the two are directly comparable -- this is what makes "
                             "'is the VAE's 2-D as faithful as UMAP's 2-D' an answerable question "
                             "rather than a rhetorical one.")
    parser.add_argument("--reference-coordinates", default=None,
                        help="Optional analysis.parquet from the baseline cell, for Procrustes "
                             "between its 2-D UMAP coordinates and this 2-D latent. Read the "
                             "caveat in umap_metrics.procrustes_disparity: on UMAP output it is "
                             "a genuine signal about global geometry and a misleading one about "
                             "whether the clustering moved. When it and ARI disagree, believe ARI.")

    parser.add_argument("--expect-latent-dim", type=int, default=2)
    parser.add_argument("--allow-any-latent-dim", action="store_true",
                        help="run against a latent whose width is not --expect-latent-dim. This "
                             "is how the 16-D control (latent -> HDBSCAN, no UMAP) is run.")

    parser.add_argument("--plot", dest="plot", action="store_true", default=True,
                        help="scatter the clustered latent per cell (default on -- the space is "
                             "2-D, so it is directly plottable, unlike the 16-D case)")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--plot-rows", type=int, default=200_000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--hdbscan-core-jobs", type=int, default=1)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--gpu-budget-gb", type=float, default=None)

    parser.add_argument("--no-sigprofiler", action="store_true")
    parser.add_argument("--sigprofiler-cpu", type=int, default=16)
    parser.add_argument("--cosmic-version", default="3.5")
    parser.add_argument("--genome-build", default="GRCh38")

    parser.add_argument("--dry-run", action="store_true", help="print the grid, fit nothing")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


# ── Latent diagnostics ─────────────────────────────────────────────────────────

def latent_statistics(space: np.ndarray, collapse_std: float = 1e-3) -> dict:
    """Per-dimension scale of the clustering space, plus a posterior-collapse check.

    ``collapsed_dims`` matters far more here than in the 16-D pipeline. There, one dead
    dimension out of 16 is a capacity note; here it means the "2-D" latent is really 1-D
    and every cluster HDBSCAN reports is a cut of a line. The threshold is on the raw
    std of mu across rows -- a collapsed dimension's posterior mean is constant, so its
    across-row std goes to zero regardless of what the KL term reports.
    """
    sample = np.asarray(space, dtype=np.float64)
    stds = sample.std(axis=0)
    variances = stds ** 2
    total_variance = float(variances.sum())
    return {
        "mean": [round(float(value), 6) for value in sample.mean(axis=0)],
        "std": [round(float(value), 6) for value in stds],
        "min": [round(float(value), 6) for value in sample.min(axis=0)],
        "max": [round(float(value), 6) for value in sample.max(axis=0)],
        # The extent HDBSCAN actually sees. Any distance-valued parameter
        # (cluster_selection_epsilon above all) has to be read against this number.
        "extent": [round(float(hi - lo), 6) for lo, hi in zip(sample.min(axis=0), sample.max(axis=0))],
        "variance_share": [round(float(value / total_variance), 6) if total_variance else None
                           for value in variances],
        "collapsed_dims": [int(index) for index, value in enumerate(stds) if value < collapse_std],
        "collapse_std_threshold": collapse_std,
        "rows_measured": int(sample.shape[0]),
    }


def quality_panel(
    space: np.ndarray,
    labels: np.ndarray,
    dbcv: float | None,
    pair_positions: np.ndarray,
) -> dict:
    """Internal-validity panel for one labelling of the clustering space.

    Davies-Bouldin and Calinski-Harabasz come from ``umap_metrics.cluster_quality`` rather
    than ``clustering_metrics.internal_metrics``, and the difference is not cosmetic: both
    are linear (they only need cluster centroids) so they are computed on **every row**,
    while ``internal_metrics`` evaluates all three on one 50k subsample because silhouette
    forced it to. Silhouette genuinely is O(n^2) and still runs on ``pair_positions``.

    Entropy and mean-WCSS have no equivalent in ``umap_metrics`` and are taken from
    ``clustering_metrics`` directly, rather than by calling ``internal_metrics`` and
    discarding four fifths of its output -- which would recompute silhouette a second time
    on a different subsample and put two disagreeing values in the same JSON.
    """
    panel = um.cluster_quality(
        space, labels,
        silhouette_coordinates=space[pair_positions],
        silhouette_labels=labels[pair_positions],
    )
    panel["dbcv"] = dbcv
    entropy, entropy_normalized = metrics.cluster_size_entropy(labels)
    panel["entropy"] = entropy
    panel["entropy_normalized"] = entropy_normalized
    panel["mean_wcss"] = metrics.mean_wcss(space, labels)
    panel["n_points"] = int(labels.size)
    return panel


def embedding_quality(
    reference_knn: np.ndarray,
    reference_pair: np.ndarray,
    space: np.ndarray,
    knn_positions: np.ndarray,
    pair_positions: np.ndarray,
    args: argparse.Namespace,
    use_gpu: bool,
) -> dict:
    """How faithfully this 2-D space reproduces the 16-D latent's neighbourhood structure.

    This is the comparison the whole test turns on. The baseline pipeline scores its 2-D
    UMAP against the 16-D latent it was built from; scoring this 2-D latent against the
    *same* 16-D reference puts both on one axis, so "did dropping UMAP cost fidelity"
    stops being a matter of opinion.

    Note what is and is not being claimed. The 16-D latent is not ground truth -- it is
    another lossy encoding of the same reads. It is used as the reference because it is the
    richest representation both pipelines share, not because it is correct.

    The two reference blocks arrive pre-materialised and already row-aligned with
    ``space[knn_positions]`` and ``space[pair_positions]`` respectively. They are NOT
    indexed again here: ``knn_positions`` and ``pair_positions`` are offsets into the full
    population, so applying them to an array that has already been subset would silently
    read the wrong rows -- or, when the subset is small enough, raise IndexError.
    """
    local = um.local_quality(
        reference_knn, space[knn_positions],
        k=args.metric_k, k_max=args.metric_k_max, use_gpu=use_gpu,
    )
    global_quality = um.distance_correlations(
        reference_pair, space[pair_positions],
        pairs=args.distance_pairs, seed=args.seed,
    )
    # Both helpers return a key called "rows", counting different tiers. A plain
    # {**local, **global} lets the pair count silently overwrite the knn count and puts one
    # number next to six metrics that were not computed on it, so they are renamed apart.
    return {
        **{key: value for key, value in local.items() if key != "rows"},
        "knn_rows": local["rows"],
        **{key: value for key, value in global_quality.items() if key != "rows"},
        "pair_rows": global_quality["rows"],
    }


def estimate_hdbscan_gpu_gb(n_rows: int, min_samples: int) -> float:
    """Rough device-memory estimate for a cuML HDBSCAN fit, in GB.

    ORDER OF MAGNITUDE, not a guarantee -- it is here so an over-large ``--fit-rows``
    announces itself in seconds rather than as a CUDA OOM part way through the fit.

    Two terms dominate, and both are linear in N:

    * the kNN graph, ``N x min_samples x (float32 distance + int32 index)`` = 8 bytes each
    * the MST edge list, condensed tree, labels, probabilities and the prediction data
      kept for approximate_predict -- a handful of O(N) arrays, taken together as ~32 B/row

    Note min_samples is the multiplier on the dominant term: at ms=25 the graph is five
    times what ms=5 needs. That is exactly why the 16-D sweep OOMed at ms=25 and not ms=5.
    """
    knn_bytes = n_rows * min_samples * 8
    tree_bytes = n_rows * 32
    return (knn_bytes + tree_bytes) / (1024 ** 3)


def scale_stats(fit_space: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(offset, divisor) for ``--scale``, computed on the fit subsample only."""
    if mode == "none":
        return None
    stds = np.asarray(fit_space, dtype=np.float64).std(axis=0)
    # A collapsed dimension would divide by ~0 and blow the scaled space up; leave those
    # dimensions alone and let latent_stats.collapsed_dims be what flags the problem.
    stds[stds < 1e-8] = 1.0
    return np.zeros(fit_space.shape[1], dtype=np.float32), stds.astype(np.float32)


def apply_scale(space: np.ndarray, scaling: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if scaling is None:
        return space
    offset, divisor = scaling
    return ((np.asarray(space, dtype=np.float32) - offset) / divisor).astype(np.float32, copy=False)


# ── Fit / label plumbing (mirrors stage2_sweep so the two are comparable) ──────

def fit_indices(total_rows: int, fit_rows: int | None, seed: int) -> np.ndarray | None:
    """Row indices HDBSCAN is fit on; None means every row."""
    if fit_rows is None or fit_rows >= total_rows:
        return None
    return np.sort(np.random.default_rng(seed).choice(total_rows, size=fit_rows, replace=False))


def label_all_rows(
    fitted: core.FittedHdbscan, space: np.ndarray, indices: np.ndarray | None, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit labels for the fit rows, approximate_predict for everything else."""
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


def write_analysis_parquet(
    path: Path, context: pl.DataFrame, labels: np.ndarray,
    probabilities: np.ndarray, space: np.ndarray,
) -> None:
    """Same schema as stage 2's, with the coordinate columns named for what they are.

    They are the VAE's posterior means, not a UMAP projection -- calling them ``umap_1``
    would make a downstream reader believe a UMAP ran. Nothing in the SigProfiler path
    reads coordinates (it groups on cluster_label and the trinucleotide context), so the
    rename costs nothing there.
    """
    columns = {name: context[name] for name in context.columns}
    columns["cluster_label"] = pl.Series("cluster_label", labels.astype(np.int32, copy=False))
    columns["cluster_probability"] = pl.Series(
        "cluster_probability", probabilities.astype(np.float32, copy=False)
    )
    if space.shape[1] == 2:
        columns["latent_1"] = pl.Series("latent_1", np.asarray(space[:, 0], dtype=np.float32))
        columns["latent_2"] = pl.Series("latent_2", np.asarray(space[:, 1], dtype=np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(columns).write_parquet(path)


def run_sigprofiler(analysis_path: Path, output_root: Path, args: argparse.Namespace) -> dict | None:
    """Reuse the clustering pipeline's SigProfiler block (lazy import -- it is heavy).

    ``UV_VAE_DISABLE_CUML`` must be set BEFORE the import, exactly as in stage 2:
    run_variant_cluster_pipeline installs a GPU budget at module scope with a 0.25 RMM
    share, which would reinitialise the pool cuML HDBSCAN is already allocating from and
    kill every cell after the first.
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


def plot_latent_clusters(
    space: np.ndarray, labels: np.ndarray, output_path: Path,
    plot_rows: int, seed: int, title: str,
) -> str | None:
    """Scatter the clustered latent. Deferred matplotlib import (it can crash on HPC)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception as exc:
        log(f"    plotting unavailable ({type(exc).__name__}: {exc})")
        return None

    total = space.shape[0]
    index = (np.arange(total) if total <= plot_rows else
             np.sort(np.random.default_rng(seed).choice(total, size=plot_rows, replace=False)))
    points = np.asarray(space[index], dtype=np.float32)
    subset = labels[index]
    noise = subset == -1

    figure, axis = plt.subplots(figsize=(9, 8), dpi=150)
    if noise.any():
        axis.scatter(points[noise, 0], points[noise, 1], s=1, c="#d9d9d9",
                     linewidths=0, alpha=0.45, rasterized=True)
    if (~noise).any():
        # Runs produce hundreds to thousands of clusters, so a cyclic 20-colour map keyed
        # on the label id is used: adjacent ids stay distinguishable, and no one can
        # mistake the colours for an ordered scale.
        colours = plt.get_cmap("tab20")(subset[~noise] % 20)
        axis.scatter(points[~noise, 0], points[~noise, 1], s=1, c=colours,
                     linewidths=0, alpha=0.75, rasterized=True)
    axis.set_xlabel("latent 1")
    axis.set_ylabel("latent 2")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


def persist_clusterer(clusterer: object, path: Path) -> str | None:
    """Best-effort dump of the fitted clusterer.

    Nothing in this script reads it back -- it exists so a read-weighted pass over all
    5.08B reads (the stage-3 equivalent) can reuse the fit instead of re-clustering.
    cuML pickling is version-dependent, so a failure is reported and the run continues.
    """
    try:
        import joblib

        joblib.dump(clusterer, path)
        return str(path)
    except Exception as exc:
        log(f"    could not persist clusterer ({type(exc).__name__}: {exc})")
        return None


def completed_payload(metrics_path: Path) -> dict | None:
    """A finished cell's payload, or None if it needs (re-)running."""
    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return None  # killed mid-write
    return None if "error" in payload else payload


def build_grid(args: argparse.Namespace) -> list[core.HdbscanConfig]:
    """min_samples varies slowest: on CPU the joblib MST cache is shared across every
    other axis at fixed min_samples, so a full inner run stays warm."""
    return [
        core.HdbscanConfig(
            min_cluster_size=mcs,
            min_samples=ms,
            metric=args.metric,
            cluster_selection_method=method,
            cluster_selection_epsilon=epsilon,
        )
        for ms, method, epsilon, mcs in product(
            parse_int_csv(args.min_samples),
            parse_csv(args.cluster_selection_methods),
            parse_float_csv(args.cluster_selection_epsilons),
            parse_int_csv(args.min_cluster_sizes),
        )
    ]


def main() -> int:
    overall = perf_counter()
    args = parse_args()

    embed_dir = Path(args.embed_dir).resolve()
    embed_summary = json.loads((embed_dir / "embed_summary.json").read_text())
    latent = np.load(embed_summary["latent_path"], mmap_mode="r")
    context = pl.read_parquet(embed_summary["context_path"])
    total_rows, latent_dim = int(latent.shape[0]), int(latent.shape[1])
    if context.height != total_rows:
        raise SystemExit(
            f"latent has {total_rows:,} rows but context has {context.height:,} -- "
            "stage 1 output is inconsistent, re-run it"
        )

    if latent_dim != args.expect_latent_dim and not args.allow_any_latent_dim:
        raise SystemExit(
            f"latent is {latent_dim}-D but --expect-latent-dim is {args.expect_latent_dim}.\n"
            f"  checkpoint: {embed_summary.get('checkpoint_path')}\n"
            f"  This script drops UMAP, so the latent IS the clustering space -- running it on a "
            f"{latent_dim}-D latent is a different experiment (the 'no-UMAP control'), not a "
            f"mistake to paper over.\n"
            f"  If that control is what you want: --allow-any-latent-dim"
        )

    grid = build_grid(args)
    fit_rows = None if str(args.fit_rows).lower() == "all" else int(args.fit_rows)
    indices = fit_indices(total_rows, fit_rows, args.seed)
    use_gpu = False if args.force_cpu else core.gpu_available()

    log(f"Latents: {total_rows:,} x {latent_dim} from {embed_summary['latent_path']}")
    log(f"Checkpoint: {embed_summary.get('checkpoint_path')}")
    log(f"Grid: {len(grid)} HDBSCAN cells (no UMAP -- the latent is the clustering space)")
    log(f"Backend: {'cuML (GPU)' if use_gpu else 'CPU'}   "
        f"fit rows: {'ALL' if indices is None else f'{indices.size:,} of {total_rows:,}'}")
    log(f"SigProfiler: {'disabled' if args.no_sigprofiler else 'every cell'}")

    if args.dry_run:
        for config in grid:
            log(f"  HDBSCAN {config.label()}")
        log(f"Dry run: {len(grid)} cells would run. Nothing was fit.")
        return 0

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # After --dry-run returns (so planning never touches the GPU) and before the first
    # cuML call (RMM has to be bounded before anything allocates from it). "sweep" is the
    # right stage share here: cuML clusters, torch is never loaded.
    budget_report = (
        core.apply_gpu_budget("sweep", budget_gb=args.gpu_budget_gb) if use_gpu else
        {"stage": "sweep", "enabled": False, "note": "CPU backend; no GPU budget applied"}
    )

    fit_latent = np.asarray(latent if indices is None else latent[indices], dtype=np.float32)
    scaling = scale_stats(fit_latent, args.scale)
    fit_space = apply_scale(fit_latent, scaling)
    space = fit_space if indices is None else apply_scale(np.asarray(latent, dtype=np.float32), scaling)

    stats = latent_statistics(fit_space)
    log(f"Latent scale (on the fit rows, after --scale {args.scale}):")
    for dimension, (mean, std, extent, share) in enumerate(
        zip(stats["mean"], stats["std"], stats["extent"], stats["variance_share"])
    ):
        log(f"  dim {dimension}: mean={mean:+.4f} std={std:.4f} extent={extent:.4f} "
            f"variance_share={share}")
    log("  -> cluster_selection_epsilon is measured in THESE units; a value tuned on a "
        "UMAP embedding does not transfer.")
    if stats["collapsed_dims"]:
        log(f"  !! WARNING: dimension(s) {stats['collapsed_dims']} have collapsed "
            f"(std < {stats['collapse_std_threshold']}). The clustering space is effectively "
            f"{latent_dim - len(stats['collapsed_dims'])}-D. Lower --kl-weight and retrain "
            f"before trusting any cluster below.")

    # Fail-fast footprint check. apply_gpu_budget already refused to start if the slice is
    # not free, but it does not know how big THIS fit will be -- and the whole point of
    # --fit-rows all is to push N as high as the card allows, so being told the number
    # before the fit starts is the difference between a 10-second answer and an OOM an
    # hour in.
    grid_min_samples = max(parse_int_csv(args.min_samples))
    estimated_gb = estimate_hdbscan_gpu_gb(fit_space.shape[0], grid_min_samples)
    budget_gb = budget_report.get("budget_gb") if isinstance(budget_report, dict) else None
    log(f"HDBSCAN fit footprint: ~{estimated_gb:.1f} GB estimated for "
        f"{fit_space.shape[0]:,} rows at min_samples={grid_min_samples}"
        + (f" (budget {budget_gb:.1f} GB)" if isinstance(budget_gb, (int, float)) else ""))
    if use_gpu and isinstance(budget_gb, (int, float)) and estimated_gb > budget_gb:
        affordable = int(budget_gb * (1024 ** 3) / (grid_min_samples * 8 + 32))
        log(f"  !! This is likely to OOM. The estimate is rough, so it is a warning rather "
            f"than a refusal -- but roughly {affordable:,} rows fit in {budget_gb:.1f} GB "
            f"at min_samples={grid_min_samples}.")
        log(f"     Either cap the fit:  --fit-rows {affordable // 1_000_000 * 1_000_000}")
        log(f"     or give it the card: GPU_TOTAL_GB=40 TRAINER_GPU_GB=0 (trainer must be idle)")

    # Metric tiers. pair is drawn FROM knn, not independently, so the two tiers describe
    # the same region of the space and a disagreement between them is about the metric
    # rather than about which rows each happened to see. Same construction as
    # size_convergence.py.
    tier_rng = np.random.default_rng(args.seed + 1)
    knn_positions = np.sort(tier_rng.choice(
        total_rows, size=min(args.knn_rows, total_rows), replace=False))
    pair_positions = np.sort(tier_rng.choice(
        knn_positions, size=min(args.pair_rows, knn_positions.size), replace=False))
    log(f"Metric tiers: full {total_rows:,} / knn {knn_positions.size:,} / "
        f"pair {pair_positions.size:,}")

    reference_latent = None
    if args.reference_latent:
        reference_path = Path(args.reference_latent).resolve()
        reference_latent = np.load(reference_path, mmap_mode="r")
        if reference_latent.shape[0] != total_rows:
            raise SystemExit(
                f"--reference-latent has {reference_latent.shape[0]:,} rows but this run has "
                f"{total_rows:,}. The embedding-quality metrics compare the two spaces ROW BY "
                f"ROW, so they are only meaningful when both stage-1 runs used the same stage-0 "
                f"dedup manifest. Re-run stage 1 against {embed_summary.get('dedup_manifest')}."
            )
        log(f"Embedding-quality reference: {reference_path} "
            f"({reference_latent.shape[0]:,} x {reference_latent.shape[1]})")
        # Materialise only the rows the metrics touch -- the reference is the 16-D latent,
        # which at cohort scale is ~10 GB on disk and there is no reason to hold it all.
        reference_knn = np.asarray(reference_latent[knn_positions], dtype=np.float32)
        reference_pair = np.asarray(reference_latent[pair_positions], dtype=np.float32)
    else:
        reference_knn = reference_pair = None
        log("No --reference-latent: skipping trustworthiness / continuity / RNX / distance "
            "correlations. Pass the 16-D baseline's stage-1 latent.npy to enable them.")

    reference_coordinates = None
    if args.reference_coordinates:
        coordinates_path = Path(args.reference_coordinates).resolve()
        frame = pl.read_parquet(coordinates_path, columns=["umap_1", "umap_2"])
        if frame.height != total_rows:
            raise SystemExit(
                f"--reference-coordinates has {frame.height:,} rows but this run has "
                f"{total_rows:,}; Procrustes is row-aligned and needs the same dedup population."
            )
        reference_coordinates = frame.to_numpy().astype(np.float32, copy=False)
        log(f"Procrustes reference: {coordinates_path}")

    agreement_rows = min(args.agreement_rows, total_rows)
    agreement_index = np.sort(
        np.random.default_rng(args.seed).choice(total_rows, size=agreement_rows, replace=False)
    )
    agreement_labels: dict[str, np.ndarray] = {}

    reference_labels = None
    if args.agreement_reference:
        reference_path = Path(args.agreement_reference).resolve()
        reference_labels = np.load(reference_path)
        if reference_labels.shape[0] != agreement_rows:
            raise SystemExit(
                f"--agreement-reference has {reference_labels.shape[0]:,} labels but this run's "
                f"agreement subsample is {agreement_rows:,}. The two are only aligned when "
                f"--seed, --agreement-rows and the stage-0 dedup population all match the run "
                f"that produced the reference."
            )
        log(f"Agreement reference: {reference_path} ({reference_labels.shape[0]:,} labels)")

    cells: list[dict] = []
    for config in grid:
        cell_dir = output_root / config.label()
        metrics_path = cell_dir / "metrics.json"
        already = completed_payload(metrics_path) if args.resume else None
        if already is not None:
            path = already.get("agreement_labels_path")
            if path and Path(path).exists():
                saved = np.load(path)
                agreement_labels[already["cell"]] = saved
                # Adding --agreement-reference to a run whose cells are already complete is
                # the normal way this gets used (cluster first, compare once the baseline
                # exists). The saved labels are all that comparison needs, so backfill it
                # rather than making the user delete finished cells to get the number.
                if reference_labels is not None and already.get("agreement_vs_reference") is None:
                    already["agreement_vs_reference"] = um.label_agreement(
                        reference_labels, saved
                    )
                    metrics_path.write_text(json.dumps(already, indent=2))
            cells.append(already)
            log(f"  [{config.label()}] already complete, skipping"
                + (f" (backfilled ARI_vs_ref={already['agreement_vs_reference']['adjusted_rand']:.4f})"
                   if already.get("agreement_vs_reference") else ""))
            continue

        cell_dir.mkdir(parents=True, exist_ok=True)
        cell_started = perf_counter()
        fitted = None
        try:
            fitted = core.fit_hdbscan(
                fit_space, config, use_gpu=use_gpu,
                core_dist_n_jobs=args.hdbscan_core_jobs,
                cache_dir=None if use_gpu else str(output_root / "hdbscan_cache"),
            )
            labels, probabilities = label_all_rows(
                fitted, space, indices, args.predict_batch_size
            )

            panel = quality_panel(space, labels, fitted.dbcv, pair_positions)

            fidelity = None
            if reference_knn is not None:
                fidelity_started = perf_counter()
                fidelity = embedding_quality(
                    reference_knn, reference_pair, space,
                    knn_positions, pair_positions, args, use_gpu,
                )
                # Both scores are optimistic by an amount that grows with this, so surface
                # it rather than leaving it in the JSON for someone to find later.
                if fidelity["clamped_fraction"] > 0.05:
                    log(f"    !! {fidelity['clamped_fraction'] * 100:.1f}% of neighbour ranks "
                        f"fell outside k_max={fidelity['k_max']}; trustworthiness and "
                        f"continuity are optimistic. Raise --metric-k-max.")
                fidelity["seconds"] = round(perf_counter() - fidelity_started, 1)

            analysis_path = cell_dir / "analysis.parquet"
            write_analysis_parquet(analysis_path, context, labels, probabilities, space)

            agreement_path = cell_dir / "agreement_labels.npy"
            np.save(agreement_path, labels[agreement_index])
            agreement_labels[config.label()] = labels[agreement_index]

            versus_reference = None
            if reference_labels is not None:
                # label_agreement, not clustering_agreement: it adds pair-counting Jaccard
                # alongside ARI/NMI/AMI. Jaccard is not chance-corrected and so is the more
                # pessimistic of the two -- a high ARI with a low Jaccard means the runs
                # mostly agree on what to keep APART, which at 30-50% noise is exactly the
                # failure mode worth catching.
                versus_reference = um.label_agreement(
                    reference_labels, labels[agreement_index]
                )
            if reference_coordinates is not None:
                versus_reference = versus_reference or {}
                versus_reference["procrustes_disparity"] = round(um.procrustes_disparity(
                    space[knn_positions], reference_coordinates[knn_positions]
                ), 5)

            plot_path = None
            if args.plot and space.shape[1] == 2:
                plot_path = plot_latent_clusters(
                    space, labels, cell_dir / "latent_clusters.png",
                    plot_rows=args.plot_rows, seed=args.seed,
                    title=f"2-D VAE latent, HDBSCAN {config.label()} "
                          f"({panel['n_clusters']} clusters, "
                          f"{panel['noise_fraction']:.1%} noise)",
                )

            sigprofiler = None
            if not args.no_sigprofiler:
                try:
                    sigprofiler = run_sigprofiler(analysis_path, cell_dir, args)
                except Exception as exc:
                    sigprofiler = {"error": f"{type(exc).__name__}: {exc}"}
                    log(f"    [{config.label()}] SigProfiler failed: {exc}")

            payload = {
                "cell": config.label(),
                "hdbscan": config.as_dict(),
                "space": "vae_latent",
                "space_shape": list(space.shape),
                "fit_rows": int(fit_space.shape[0]),
                "scale": args.scale,
                "metrics": panel,
                "embedding_quality": fidelity,
                "dbcv": fitted.dbcv,
                "analysis_path": str(analysis_path),
                "agreement_labels_path": str(agreement_path),
                "clusterer_path": persist_clusterer(
                    fitted.clusterer, cell_dir / "hdbscan_clusterer.joblib"
                ),
                "agreement_vs_reference": versus_reference,
                "plot_path": plot_path,
                "sigprofiler": sigprofiler,
                "backend": fitted.backend,
                "seconds": round(perf_counter() - cell_started, 1),
            }
            metrics_path.write_text(json.dumps(payload, indent=2))
            cells.append(payload)
            log(f"  [{config.label()}] {panel['n_clusters']} clusters  "
                f"noise={panel['noise_fraction']:.3f}  dbcv={fitted.dbcv}  "
                f"sil={panel['silhouette']}"
                + (f"  ARI_vs_ref={versus_reference['adjusted_rand']:.4f}"
                   if versus_reference and "adjusted_rand" in versus_reference else "")
                + (f"  trust={fidelity['trustworthiness']:.4f}"
                   f" cont={fidelity['continuity']:.4f}"
                   f" rnx={fidelity['rnx_auc']:.4f}" if fidelity else "")
                + f"  ({payload['seconds']}s)")
        except Exception as exc:
            payload = {
                "cell": config.label(),
                "hdbscan": config.as_dict(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "seconds": round(perf_counter() - cell_started, 1),
            }
            metrics_path.write_text(json.dumps(payload, indent=2))
            cells.append(payload)
            log(f"  [{config.label()}] FAILED: {type(exc).__name__}: {exc}")
        finally:
            # Release this cell's clusterer before the next fit allocates. With
            # prediction_data=True each one holds a per-row structure on the GPU, which at
            # cohort scale is not a rounding error.
            fitted = None

    ok = [cell for cell in cells if "error" not in cell]

    # cuML does not expose relative_validity_, so on the GPU path DBCV is None for every
    # cell and "best by DBCV" would silently be nothing at all -- which is exactly what
    # happened to the 16-D sweep. Fall back to silhouette and say which was used.
    by_dbcv = [cell for cell in ok
               if cell.get("dbcv") is not None and cell["metrics"]["n_clusters"] >= 2]
    # cluster_quality reports an undefined metric as None, not NaN -- np.isfinite(None)
    # raises, so this must be an explicit None check.
    by_silhouette = [cell for cell in ok
                     if cell["metrics"]["n_clusters"] >= 2
                     and cell["metrics"].get("silhouette") is not None]
    if by_dbcv:
        best, selection_metric = max(by_dbcv, key=lambda cell: cell["dbcv"]), "dbcv"
    elif by_silhouette:
        best, selection_metric = (
            max(by_silhouette, key=lambda cell: cell["metrics"]["silhouette"]), "silhouette"
        )
    else:
        best, selection_metric = None, None

    agreement_vs_best: list[dict] = []
    if best is not None and best["cell"] in agreement_labels:
        reference = agreement_labels[best["cell"]]
        for cell in ok:
            if cell["cell"] in agreement_labels:
                agreement_vs_best.append(
                    {"cell": cell["cell"],
                     **um.label_agreement(reference, agreement_labels[cell["cell"]])}
                )
        agreement_vs_best.sort(key=lambda entry: entry["adjusted_rand"], reverse=True)

    summary = {
        "experiment": "vae_latent_2d_no_umap",
        "embed_dir": str(embed_dir),
        "checkpoint_path": embed_summary.get("checkpoint_path"),
        "total_rows": total_rows,
        "latent_dim": latent_dim,
        "latent_stats": stats,
        "scale": args.scale,
        "fit_rows": "all" if indices is None else int(indices.size),
        "predict_batch_size": args.predict_batch_size,
        "backend": "cuml" if use_gpu else "cpu",
        "gpu_budget": budget_report,
        "grid": {
            "min_cluster_sizes": parse_int_csv(args.min_cluster_sizes),
            "min_samples": parse_int_csv(args.min_samples),
            "cluster_selection_methods": parse_csv(args.cluster_selection_methods),
            "cluster_selection_epsilons": parse_float_csv(args.cluster_selection_epsilons),
            "cells": len(grid),
        },
        "cells_completed": len(ok),
        "cells_failed": len(cells) - len(ok),
        "selection_metric": selection_metric,
        "best_cell": best["cell"] if best else None,
        "best_dbcv": best["dbcv"] if best else None,
        "best_silhouette": best["metrics"]["silhouette"] if best else None,
        "agreement_reference": {
            "path": str(Path(args.agreement_reference).resolve()) if args.agreement_reference else None,
            "label": args.agreement_reference_label,
            "rows": agreement_rows,
            "seed": args.seed,
        },
        "agreement_vs_best": agreement_vs_best,
        "results": cells,
        "seconds": round(perf_counter() - overall, 1),
    }
    (output_root / "latent2d_summary.json").write_text(json.dumps(summary, indent=2))

    log("")
    log(f"Complete: {len(ok)}/{len(cells)} cells in {summary['seconds']}s")
    if best:
        log(f"Best by {selection_metric}: {best['cell']}  "
            f"clusters={best['metrics']['n_clusters']}  "
            f"noise={best['metrics']['noise_fraction']:.3f}  "
            f"silhouette={best['metrics']['silhouette']}")
        if best.get("agreement_vs_reference"):
            log(f"  ARI vs the 16-D+UMAP baseline: "
                f"{best['agreement_vs_reference']['adjusted_rand']:.4f}")
    if stats["collapsed_dims"]:
        log(f"REMINDER: dimension(s) {stats['collapsed_dims']} were collapsed -- these clusters "
            f"live on a line, not a plane.")
    log(f"Summary -> {output_root / 'latent2d_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
