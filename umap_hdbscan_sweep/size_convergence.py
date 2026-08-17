"""How many rows does UMAP actually need to be fit on before the answer stops changing?

Stage 2 currently fits UMAP on every deduplicated locus. Nothing has ever checked whether
that is necessary. This sweeps the fit size -- 1 M, 2.5 M, 5 M, 10 M, 25 M, 50 M -- and
scores each fit on one fixed held-out probe, so the only thing varying between runs is how
much data the fit saw.

Two questions, answered by the two metric families in ``umap_metrics``:

1. **Is the map still improving?** Trustworthiness, continuity, RNX AUC, Spearman and
   Pearson compare the probe's 2-D coordinates against its own 16-D latents. No reference
   run needed -- each size is judged on its own.
2. **Do the clusters still move?** Every size's probe coordinates are clustered with the
   same algorithm and the same seed, then ARI / NMI / AMI / Jaccard measure whether the
   labels agree with the largest run's. This is the number that decides the question,
   because labels are what the pipeline actually consumes.

Both are reported against the largest size *and* against the previous size. The second is
the more honest convergence signal: agreement with the reference is guaranteed to hit 1.0
at the reference itself, whereas consecutive-size agreement plateauing means additional
rows genuinely stopped mattering.

**Design decisions that make the comparison valid**

*The probe is drawn first and excluded from every fit.* One permutation of the row space
supplies the probe from its head and all fit sets from its tail, so disjointness is
structural rather than checked.

*Fit sets are nested by default.* The 2.5 M fit contains the 1 M fit. Independent draws
would confound "more rows" with "different rows"; ``--independent-draws`` opts out when
you want the sampling variance instead.

*The clusterer's seed is fixed across sizes.* Otherwise k-means initialisation noise would
show up as ARI movement and be indistinguishable from the effect being measured.

*k-means is the default clusterer, not HDBSCAN.* The convergence point is a property of
the density structure, so a cheap partitioner finds it in seconds where HDBSCAN would take
minutes per size. Confirm the answer once with ``--clusterer hdbscan`` at the size k-means
picks; do not run the whole sweep that way.

Three evaluation tiers, because the metrics have wildly different costs (see
``umap_metrics`` for why the local ones are not just sklearn calls):

    full probe   1 M    ARI, NMI, AMI, Jaccard, Davies-Bouldin, Calinski-Harabasz, procrustes
    knn subset   250 k  trustworthiness, continuity, RNX AUC
    pair subset  20 k   silhouette, Spearman, Pearson

Both subsets are drawn once, from the probe, and held fixed for every size.

    python umap_hdbscan_sweep/size_convergence.py \\
        --embed-dir <stage1-output> \\
        --output-dir umap_hdbscan_sweep/umap_tests \\
        --sizes 1000000,2500000,5000000,10000000,25000000,50000000 \\
        --gpu-budget-gb 24
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

# Walk up to the directory that holds uv_vae/ rather than counting levels: these scripts
# get reorganised into subfolders (umap/, hdbscan/, tests/), and a hard-coded parents[N]
# silently resolves to the wrong root the moment one moves, failing on `import uv_vae`.
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np

import sweep_core as core
import umap_metrics as metrics

# Bytes per row of the fit set on the GPU, measured against cuML's UMAP at 16 dimensions:
# the float32 input (16 x 4), the kNN graph it builds (n_neighbors x (int32 index +
# float32 distance)), the 2-D embedding, and the optimiser's copy of it. Used only to warn
# before a 40-minute fit dies of OOM.
GPU_BYTES_PER_ROW = 16 * 4 + 15 * 8 + 2 * 4 + 2 * 4


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# ── clustering backends ────────────────────────────────────────────────────────

def cluster(
    coordinates: np.ndarray,
    method: str,
    k: int,
    seed: int,
    min_cluster_size: int,
    min_samples: int,
    use_gpu: bool,
) -> tuple[np.ndarray, float | None]:
    """Label the probe. Returns (labels, dbcv) -- dbcv is None for anything but HDBSCAN."""
    if method == "hdbscan":
        config = core.HdbscanConfig(min_cluster_size=min_cluster_size, min_samples=min_samples)
        fitted = core.fit_hdbscan(coordinates, config, use_gpu=use_gpu)
        return fitted.labels, fitted.dbcv

    if use_gpu:
        from cuml.cluster import KMeans as CuKMeans

        model = CuKMeans(n_clusters=k, random_state=seed, n_init=3, max_iter=300)
        model.fit(np.ascontiguousarray(coordinates, dtype=np.float32))
        return core._to_host(model.labels_).astype(np.int32, copy=False), None

    from sklearn.cluster import MiniBatchKMeans

    # Full KMeans on 1 M rows on CPU is minutes per size; MiniBatch converges to the same
    # partition on a 2-D space this well separated and is the CPU fallback only.
    model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=100_000)
    return model.fit_predict(np.ascontiguousarray(coordinates, dtype=np.float32)).astype(np.int32), None


# ── row selection ──────────────────────────────────────────────────────────────

def draw_indices(
    total_rows: int,
    probe_rows: int,
    sizes: list[int],
    seed: int,
    independent: bool,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """One permutation supplies the probe and every fit set, guaranteeing disjointness.

    The permutation costs 8 bytes a row (1.3 GB at the 157 M-row cohort) but removes an
    entire class of bug: there is no way for a probe row to leak into a fit set, so the
    held-out scores cannot be quietly inflated.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total_rows)
    probe_index = np.sort(permutation[:probe_rows])
    pool = permutation[probe_rows:]

    if max(sizes) > pool.size:
        raise SystemExit(
            f"largest fit size {max(sizes):,} exceeds the {pool.size:,} rows left after "
            f"holding out a {probe_rows:,}-row probe from {total_rows:,}"
        )

    fit_indices: dict[int, np.ndarray] = {}
    for position, size in enumerate(sizes):
        if independent:
            picked = np.random.default_rng(seed + 100 + position).choice(
                pool.size, size=size, replace=False
            )
            fit_indices[size] = np.sort(pool[picked])
        else:
            fit_indices[size] = np.sort(pool[:size])
    return probe_index, fit_indices


# ── one fit size ───────────────────────────────────────────────────────────────

def run_one_size(
    size: int,
    latent: np.ndarray,
    fit_index: np.ndarray,
    probe_latent: np.ndarray,
    knn_positions: np.ndarray,
    pair_positions: np.ndarray,
    args,
    use_gpu: bool,
) -> dict:
    """Fit UMAP on ``size`` rows, project the probe, cluster it, score everything."""
    config = core.UmapConfig(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        seed=args.seed,
        n_epochs=args.umap_epochs,
    )

    log(f"  gathering {size:,} fit rows from the memmap ...")
    started = perf_counter()
    fit_latent = np.ascontiguousarray(latent[fit_index])
    gather_seconds = perf_counter() - started

    log(f"  fitting UMAP on {size:,} rows ...")
    started = perf_counter()
    fitted = core.fit_umap(fit_latent, config)
    fit_seconds = perf_counter() - started
    log(f"    fit in {fit_seconds:.1f}s (backend={fitted.backend})")
    del fit_latent

    log(f"  transform() of the {len(probe_latent):,}-row probe ...")
    started = perf_counter()
    coordinates = fitted.transform(probe_latent, batch_size=args.transform_batch_size)
    transform_seconds = perf_counter() - started
    log(f"    transform in {transform_seconds:.1f}s")
    del fitted

    log(f"  clustering the probe ({args.clusterer}) ...")
    started = perf_counter()
    labels, dbcv = cluster(
        coordinates, args.clusterer, args.kmeans_k, args.cluster_seed,
        args.min_cluster_size, args.min_samples, use_gpu,
    )
    cluster_seconds = perf_counter() - started

    quality = metrics.cluster_quality(
        coordinates, labels,
        silhouette_coordinates=coordinates[pair_positions],
        silhouette_labels=labels[pair_positions],
    )
    quality["dbcv"] = dbcv
    log(f"    {quality['n_clusters']} clusters, "
        f"{quality['noise_fraction'] * 100:.1f}% noise, in {cluster_seconds:.1f}s")

    log(f"  embedding quality: local on {len(knn_positions):,} rows ...")
    started = perf_counter()
    local = metrics.local_quality(
        probe_latent[knn_positions], coordinates[knn_positions],
        k=args.metric_k, k_max=args.metric_k_max, use_gpu=use_gpu,
    )
    local_seconds = perf_counter() - started

    log(f"  embedding quality: global on {len(pair_positions):,} rows ...")
    started = perf_counter()
    global_quality = metrics.distance_correlations(
        probe_latent[pair_positions], coordinates[pair_positions],
        pairs=args.distance_pairs, seed=args.seed,
    )
    global_seconds = perf_counter() - started

    log(f"    trustworthiness {local['trustworthiness']:.4f}  "
        f"continuity {local['continuity']:.4f}  RNX AUC {local['rnx_auc']:.4f}  "
        f"spearman {global_quality['spearman_distance']:.4f}")
    if local["clamped_fraction"] > 0.05:
        log(f"    WARNING: {local['clamped_fraction'] * 100:.1f}% of neighbour ranks fell "
            f"outside the k_max={local['k_max']} window; trustworthiness and continuity "
            f"are optimistic. Raise --metric-k-max.")

    return {
        "fit_rows": size,
        "umap_config": config.as_dict(),
        "timing": {
            "gather_seconds": round(gather_seconds, 1),
            "umap_fit_seconds": round(fit_seconds, 1),
            "probe_transform_seconds": round(transform_seconds, 1),
            "cluster_seconds": round(cluster_seconds, 1),
            "local_metric_seconds": round(local_seconds, 1),
            "global_metric_seconds": round(global_seconds, 1),
        },
        "embedding_quality": {**local, **global_quality},
        "cluster_quality": quality,
        "_coordinates": coordinates,
        "_labels": labels,
    }


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the UMAP fit size at which embedding and clustering stop changing")
    parser.add_argument("--embed-dir", required=True, help="stage 1 output holding latent.npy")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sizes", default="1000000,2500000,5000000,10000000,25000000,50000000",
                        help="fit-row counts, ascending; the last is the reference")
    parser.add_argument("--probe-rows", type=int, default=1_000_000,
                        help="held-out rows every size is scored on")
    parser.add_argument("--knn-rows", type=int, default=250_000,
                        help="subset of the probe used for trustworthiness/continuity/RNX")
    parser.add_argument("--pair-rows", type=int, default=20_000,
                        help="subset of the probe used for silhouette and the distance "
                             "correlations -- these need pairwise distances")
    parser.add_argument("--distance-pairs", type=int, default=20_000_000,
                        help="pairs sampled from --pair-rows for Spearman/Pearson")
    parser.add_argument("--metric-k", type=int, default=15,
                        help="neighbourhood size for trustworthiness and continuity")
    parser.add_argument("--metric-k-max", type=int, default=400,
                        help="neighbour-list width; also the largest K in the RNX AUC and "
                             "the rank horizon beyond which penalties are clamped. 400 is "
                             "measured, not guessed: on real umap-learn output at "
                             "min_dist=0.0, k_max=100 clamped 40%% of ranks and reported "
                             "trustworthiness 0.0323 too high, while 400 clamped 0.1%% and "
                             "landed within 0.0001 of the exact value. Required width "
                             "scales with island size, so check clamped_fraction in the "
                             "output and raise this if it exceeds 0.05")
    parser.add_argument("--clusterer", default="kmeans", choices=["kmeans", "hdbscan"])
    parser.add_argument("--kmeans-k", type=int, default=25,
                        help="held identical across sizes, so ARI reflects the embedding")
    parser.add_argument("--min-cluster-size", type=int, default=500, help="--clusterer hdbscan")
    parser.add_argument("--min-samples", type=int, default=25, help="--clusterer hdbscan")
    parser.add_argument("--cluster-seed", type=int, default=7,
                        help="fixed across sizes so initialisation noise is not read as drift")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--umap-epochs", type=int, default=200)
    parser.add_argument("--transform-batch-size", type=int, default=5_000_000)
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="a metric counts as converged once it and every larger size "
                             "sit within this of the reference")
    parser.add_argument("--independent-draws", action="store_true",
                        help="draw each fit set separately instead of nesting them")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-budget-gb", type=float, default=None)
    parser.add_argument("--save-coords", action="store_true",
                        help="also write each size's probe coordinates and labels as .npy")
    return parser.parse_args()


def main() -> int:
    started = perf_counter()
    args = parse_args()
    sizes = [int(part) for part in args.sizes.split(",") if part.strip()]
    if sizes != sorted(sizes):
        raise SystemExit("--sizes must be ascending; the last entry is used as the reference")

    embed_dir = Path(args.embed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_path = embed_dir / "latent.npy"
    if not latent_path.exists():
        raise SystemExit(f"latent.npy not found in {embed_dir}")

    use_gpu = core.gpu_available()
    if use_gpu and args.gpu_budget_gb is not None:
        core.apply_gpu_budget("sweep", budget_gb=args.gpu_budget_gb)
    if use_gpu:
        needed = max(sizes) * GPU_BYTES_PER_ROW / (1024 ** 3)
        log(f"largest fit ({max(sizes):,} rows) needs roughly {needed:.1f} GB on the GPU")
        if args.gpu_budget_gb is not None and needed > args.gpu_budget_gb - 2.0:
            log(f"WARNING: --gpu-budget-gb {args.gpu_budget_gb} leaves little headroom over "
                f"the {needed:.1f} GB estimate; a mid-fit OOM is likely. "
                f"Suggest --gpu-budget-gb {needed + 4:.0f}")

    latent = np.load(latent_path, mmap_mode="r")
    total_rows, latent_dim = latent.shape
    log(f"{latent_path.name}: {total_rows:,} rows x {latent_dim} dims")

    probe_index, fit_indices = draw_indices(
        total_rows, args.probe_rows, sizes, args.seed, args.independent_draws)
    log(f"probe {len(probe_index):,} held-out rows; fit sizes {', '.join(f'{s:,}' for s in sizes)}"
        f" ({'independent' if args.independent_draws else 'nested'})")

    log("materialising the probe latents ...")
    probe_latent = np.ascontiguousarray(latent[probe_index])

    # Both subsets are positions *into the probe*, drawn once. Every size is therefore
    # scored on identical rows, so differences between sizes are the embedding and nothing
    # else. knn_positions is a superset of pair_positions purely so the two tiers describe
    # the same region of the space.
    subset_rng = np.random.default_rng(args.seed + 1)
    knn_positions = np.sort(subset_rng.choice(
        len(probe_index), size=min(args.knn_rows, len(probe_index)), replace=False))
    pair_positions = np.sort(subset_rng.choice(
        knn_positions, size=min(args.pair_rows, len(knn_positions)), replace=False))
    log(f"metric tiers: full {len(probe_index):,} / knn {len(knn_positions):,} / "
        f"pair {len(pair_positions):,}")

    results = []
    for size in sizes:
        log(f"=== fit size {size:,} ===")
        results.append(run_one_size(
            size, latent, fit_indices[size], probe_latent,
            knn_positions, pair_positions, args, use_gpu,
        ))

    # ── cross-run comparisons, once every size exists ──────────────────────────
    reference = results[-1]
    log(f"=== comparing against the {reference['fit_rows']:,}-row reference ===")
    for position, entry in enumerate(results):
        entry["vs_reference"] = {
            **metrics.label_agreement(entry["_labels"], reference["_labels"]),
            "procrustes_disparity": round(
                metrics.procrustes_disparity(entry["_coordinates"], reference["_coordinates"]), 5),
        }
        if position == 0:
            entry["vs_previous"] = None
        else:
            previous = results[position - 1]
            entry["vs_previous"] = {
                **metrics.label_agreement(entry["_labels"], previous["_labels"]),
                "procrustes_disparity": round(
                    metrics.procrustes_disparity(
                        entry["_coordinates"], previous["_coordinates"]), 5),
                "previous_fit_rows": previous["fit_rows"],
            }

    # ── convergence verdict ───────────────────────────────────────────────────
    verdict = {}
    for name in metrics.EMBEDDING_METRICS:
        if name == "procrustes_disparity":
            values = [entry["vs_reference"]["procrustes_disparity"] for entry in results]
        else:
            values = [entry["embedding_quality"].get(name) for entry in results]
        verdict[name] = metrics.converged_at(sizes, values, args.tolerance)
    for name in ("adjusted_rand", "normalized_mutual_info", "adjusted_mutual_info", "jaccard"):
        values = [entry["vs_reference"][name] for entry in results]
        verdict[name] = metrics.converged_at(sizes, values, args.tolerance)
    for name in ("silhouette", "davies_bouldin", "calinski_harabasz", "dbcv"):
        values = [entry["cluster_quality"].get(name) for entry in results]
        # Calinski-Harabasz is unbounded, so an absolute tolerance is meaningless for it;
        # scale the band to the reference instead.
        tolerance = args.tolerance
        if name == "calinski_harabasz" and values[-1]:
            tolerance = args.tolerance * abs(values[-1])
        verdict[name] = metrics.converged_at(sizes, values, tolerance)

    if args.save_coords:
        for entry in results:
            np.save(output_dir / f"coords_{entry['fit_rows']}.npy", entry["_coordinates"])
            np.save(output_dir / f"labels_{entry['fit_rows']}.npy", entry["_labels"])
        np.save(output_dir / "probe_index.npy", probe_index)

    for entry in results:
        entry.pop("_coordinates", None)
        entry.pop("_labels", None)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "embed_dir": str(embed_dir),
        "total_rows": int(total_rows),
        "latent_dim": int(latent_dim),
        "sizes": sizes,
        "reference_fit_rows": sizes[-1],
        "probe_rows": int(len(probe_index)),
        "knn_rows": int(len(knn_positions)),
        "pair_rows": int(len(pair_positions)),
        "nested_fit_sets": not args.independent_draws,
        "clusterer": args.clusterer,
        "tolerance": args.tolerance,
        "backend": "cuml" if use_gpu else "cpu",
        "metric_tiers": metrics.METRIC_TIERS,
        "results": results,
        "converged_at": verdict,
        "total_seconds": round(perf_counter() - started, 1),
    }
    destination = output_dir / "size_convergence.json"
    destination.write_text(json.dumps(payload, indent=2))

    print_report(payload)
    log(f"wrote {destination}")
    return 0


def print_report(payload: dict) -> None:
    sizes = payload["sizes"]
    results = payload["results"]

    width = 11

    def cell(value):
        if value is None:
            return "--"
        if isinstance(value, float):
            # Calinski-Harabasz runs to five figures while the correlations need four
            # decimals; one format cannot serve both, so pick per magnitude.
            return f"{value:.4f}" if abs(value) < 1000 else f"{value:.4g}"
        return str(value)

    def row(label, values):
        return f"  {label:<26}" + "".join(f"{cell(v):>{width}}" for v in values)

    rule = "=" * (28 + width * len(sizes))
    print("\n" + rule)
    print(f"UMAP fit-size convergence -- probe {payload['probe_rows']:,} held-out rows, "
          f"clusterer={payload['clusterer']}")
    print(rule)
    print(row("fit rows", [f"{s // 1000}k" if s < 1_000_000 else f"{s / 1e6:.4g}M" for s in sizes]))

    print("\n  -- cost --")
    print(row("UMAP fit (s)", [r["timing"]["umap_fit_seconds"] for r in results]))
    print(row("probe transform (s)", [r["timing"]["probe_transform_seconds"] for r in results]))
    print(row("cluster (s)", [r["timing"]["cluster_seconds"] for r in results]))

    # Annotation key used in labels below:
    #   *  hand-coded from scratch (custom kNN / rank-lookup / SVD / contingency math)
    #   ~  custom distance sampling + a single scipy call for the final statistic
    #      (unmarked) thin sklearn / hdbscan API wrapper

    print("\n  -- embedding quality (2-D map vs the 16-D latent) --")
    for name, label in (
        ("trustworthiness",   "trustworthiness *"),
        ("continuity",        "continuity *"),
        ("rnx_auc",           "rnx_auc *"),
        ("spearman_distance", "spearman_distance ~"),
        ("pearson_distance",  "pearson_distance ~"),
    ):
        print(row(label, [r["embedding_quality"].get(name) for r in results]))
    print(row("procrustes vs ref *", [r["vs_reference"]["procrustes_disparity"] for r in results]))

    print(f"\n  -- clustering agreement with the {sizes[-1]:,}-row reference --")
    for name, label in (
        ("adjusted_rand",          "adjusted_rand"),
        ("normalized_mutual_info", "normalized_mutual_info"),
        ("adjusted_mutual_info",   "adjusted_mutual_info"),
        ("jaccard",                "jaccard *"),
    ):
        print(row(label, [r["vs_reference"][name] for r in results]))

    print("\n  -- clustering agreement with the previous size --")
    for name, label in (("adjusted_rand", "adjusted_rand"), ("jaccard", "jaccard *")):
        print(row(label, [None if r["vs_previous"] is None else r["vs_previous"][name]
                          for r in results]))

    print("\n  -- clustering quality --")
    for name, label in (
        ("n_clusters",        "n_clusters"),
        ("noise_fraction",    "noise_fraction"),
        ("silhouette",        "silhouette"),
        ("davies_bouldin",    "davies_bouldin"),
        ("calinski_harabasz", "calinski_harabasz"),
        ("dbcv",              "dbcv"),
    ):
        print(row(label, [r["cluster_quality"].get(name) for r in results]))

    print(f"\n  -- converged at (within {payload['tolerance']} of the reference, and staying there) --")
    for name, size in payload["converged_at"].items():
        verdict = "never" if size is None else f"{size:,} rows"
        print(f"  {name:<28}{verdict}")

    settled = [s for s in payload["converged_at"].values() if s is not None]
    unsettled = [n for n, s in payload["converged_at"].items() if s is None]
    if settled and max(settled) < sizes[-1]:
        print(f"\n  Every metric that converged did so by {max(settled):,} rows; "
              f"fitting above that bought nothing measurable.")
    elif settled:
        print(f"\n  The slowest metric only settles at the reference itself "
              f"({sizes[-1]:,} rows), which is not evidence of convergence -- agreement "
              f"with the reference is 1.0 there by construction. Extend --sizes upward, "
              f"or read the vs-previous rows instead.")
    if unsettled:
        print(f"  Never converged: {', '.join(unsettled)}")
    print(f"\n  * hand-coded from scratch  ~ custom sampling + scipy  "
          f"(unmarked) sklearn / hdbscan API")
    print(rule + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
