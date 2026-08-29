"""Sweep 1: the UMAP parameters that do not depend on how many rows you fit on.

``size_convergence.py`` asked *when does it stop changing* -- a convergence question, which
has a defensible reference (more rows is strictly more information) and is therefore
answered by cross-run agreement. This asks *which setting is best*, which is an
optimisation question with no reference at all: ``min_dist=0.0`` is not a more-correct
version of ``min_dist=0.25``, it is a different choice. Agreement between two configs
tells you they differ, never which one is right.

That one difference reshuffles the whole metric set.

**Primary -- fidelity to the 16-D latent.** RNX AUC, trustworthiness, continuity, Spearman
and Pearson all compare the 2-D output against the *input* space, which does not move when
a parameter moves. They are the only quality signals here that a parameter cannot game.

**Primary -- reproducibility across seeds.** Every config is run at ``--seeds`` independent
replicates, and the headline stability number is the mean pairwise ARI *between replicates
of the same config*. This is the direct test of the failure mode the early subsample sweeps
hit, where good and bad results both turned out to be artefacts of which rows got drawn: a
config scoring 0.95 between replicates produces structure you can rely on, one scoring 0.45
produces noise and any single-seed number from it means nothing.

Each replicate redraws **both** the fit rows and the UMAP seed, because that is what
re-running the pipeline would actually vary. Within a replicate every config shares one fit
set, so differences between configs are never confounded with differences in data.

**Diagnostic only -- silhouette, Davies-Bouldin, Calinski-Harabasz.** These are computed on
the UMAP output coordinates, and UMAP parameters directly control the geometry of those
coordinates: ``min_dist=0.0`` crushes each cluster into a ball, which raises silhouette
mechanically whether or not the structure is real. Ranking configs by them would be
measuring a parameter's effect on itself. The size sweep already showed them pointing the
wrong way -- silhouette fell 0.504 -> 0.406 as fits grew while RNX AUC rose 0.128 -> 0.166.
They are printed, and deliberately excluded from the verdict.

**Dropped -- procrustes and between-config ARI.** No reference config exists, and different
``min_dist``/``spread`` values put the output on a different global scale anyway.

**HDBSCAN, not k-means.** k-means with a pinned k cannot express the thing these parameters
mostly do, which is change how many clusters exist and how much of the data is
unclusterable. A config that collapses to 4 clusters with 70% noise is bad in a way
``k=25`` is structurally incapable of reporting. HDBSCAN costs more than k-means's 0.1s;
the per-cell timing is printed, and ``--cluster-rows`` subsamples the probe if it dominates.

**Clamping.** The size sweep ran the local metrics on 250k rows with ``k_max=1500`` and
clamped ~31% of ranks, which made trustworthiness and continuity useless. Here the knn tier
is 50k rows with ``k_max=5000`` -- a smaller sample with a much wider window, so ``k_max``
comfortably exceeds the typical cluster size and clamping falls to near zero. That matters
more here than it did there: clamping bias *differs between configs* (spreading points out
changes neighbour structure), so it no longer cancels as a constant offset.

**Design.** One-factor-at-a-time around a baseline, not the full cartesian product. The
first value of each list is the baseline. ``min_dist`` and ``spread`` are crossed with each
other because together they fit the (a, b) of the output kernel and neither is interpretable
alone; ``repulsion_strength`` and ``set_op_mix_ratio`` then vary one at a time against the
baseline. The default grid is 9 configs rather than the 36 a full product would give.

    python umap_hdbscan_sweep/parameter_sweep.py \\
        --embed-dir <stage1-output> \\
        --output-dir umap_hdbscan_sweep/umap_tests/param_sweep1 \\
        --fit-rows 5000000 --seeds 3 --gpu-budget-gb 24
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
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

# Ranked on, in order. Everything else in the report is context.
FIDELITY_METRIC = "rnx_auc"
STABILITY_METRIC = "adjusted_rand"

# Printed but never ranked on -- see the module docstring.
DIAGNOSTIC_ONLY = ("silhouette", "davies_bouldin", "calinski_harabasz")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


# ── the grid ───────────────────────────────────────────────────────────────────

def build_configs(args) -> list[core.UmapConfig]:
    """Baseline + one-factor-at-a-time, with min_dist x spread crossed.

    The first value of every list is the baseline. Crossing min_dist with spread is not
    optional: UMAP solves for the output kernel's (a, b) from the pair, so a min_dist that
    looks bad at spread=1.0 can be the best one at spread=2.0. The other two axes have no
    such coupling and are swept against the baseline alone.
    """
    min_dists = parse_floats(args.min_dist)
    spreads = parse_floats(args.spread)
    repulsions = parse_floats(args.repulsion_strength)
    mix_ratios = parse_floats(args.set_op_mix_ratio)

    base_repulsion, base_mix = repulsions[0], mix_ratios[0]
    base_min_dist, base_spread = min_dists[0], spreads[0]

    def make(min_dist, spread, repulsion, mix):
        return core.UmapConfig(
            n_neighbors=args.n_neighbors,
            min_dist=min_dist,
            n_components=2,
            n_epochs=args.n_epochs,
            spread=spread,
            repulsion_strength=repulsion,
            set_op_mix_ratio=mix,
        )

    configs, seen = [], set()

    def add(config):
        if config.label() not in seen:
            seen.add(config.label())
            configs.append(config)

    # min_dist x spread, crossed. The baseline cell falls out of this block.
    for min_dist in min_dists:
        for spread in spreads:
            if spread < min_dist:
                # UMAP requires spread >= min_dist; the (a, b) fit has no solution otherwise.
                log(f"  skipping min_dist={min_dist} spread={spread} (spread must be >= min_dist)")
                continue
            add(make(min_dist, spread, base_repulsion, base_mix))

    # the remaining axes, one at a time against the baseline
    for repulsion in repulsions[1:]:
        add(make(base_min_dist, base_spread, repulsion, base_mix))
    for mix in mix_ratios[1:]:
        add(make(base_min_dist, base_spread, base_repulsion, mix))
    return configs


# ── row selection ──────────────────────────────────────────────────────────────

def draw_probe(total_rows: int, probe_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One fixed probe, held out of every fit, plus the pool the fit sets come from.

    The probe is the measuring stick: identical rows scored for every config and every
    replicate, so a difference in a metric is a difference in the embedding and nothing else.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total_rows)
    return np.sort(permutation[:probe_rows]), permutation[probe_rows:]


def draw_fit_set(pool: np.ndarray, fit_rows: int, seed: int, replicate: int) -> np.ndarray:
    """Replicate ``replicate``'s fit rows.

    Each replicate draws its own rows. That is deliberate: the question is whether a
    config's advantage survives re-running the pipeline, and re-running would redraw the
    data as well as reseeding the optimiser. Holding the rows fixed would measure only
    optimiser noise and would understate the variance that matters.
    """
    rng = np.random.default_rng(seed + 1000 + replicate)
    picked = rng.choice(pool.size, size=fit_rows, replace=False)
    return np.sort(pool[picked])


# ── one (config, replicate) cell ───────────────────────────────────────────────

def run_cell(
    config: core.UmapConfig,
    latent: np.ndarray,
    fit_index: np.ndarray,
    probe_latent: np.ndarray,
    knn_positions: np.ndarray,
    pair_positions: np.ndarray,
    cluster_positions: np.ndarray,
    pair_slots: np.ndarray,
    args,
    use_gpu: bool,
) -> dict:
    """Fit, project the probe, cluster it, score fidelity. Returns metrics plus labels."""
    started = perf_counter()
    fit_latent = np.ascontiguousarray(latent[fit_index])
    gather_seconds = perf_counter() - started

    started = perf_counter()
    fitted = core.fit_umap(fit_latent, config)
    fit_seconds = perf_counter() - started
    del fit_latent

    started = perf_counter()
    coordinates = fitted.transform(probe_latent, batch_size=args.transform_batch_size)
    transform_seconds = perf_counter() - started
    del fitted

    # A config can genuinely break UMAP rather than merely embed badly. Measured on real
    # umap-learn: set_op_mix_ratio=0.0 takes the *intersection* of the fuzzy simplicial
    # sets, which on clustered data keeps roughly a third of the edges (8402 vs 25198 on a
    # 1200-row probe) and disconnects the manifold hard enough that the optimiser returns
    # NaN coordinates. Left unchecked those NaNs flow into every metric and, worse, into the
    # cross-replicate means -- one broken cell would quietly corrupt its config's whole row.
    # Fail the cell loudly instead.
    if not np.isfinite(coordinates).all():
        bad = int((~np.isfinite(coordinates)).any(axis=1).sum())
        log(f"    FAILED: {bad:,}/{len(coordinates):,} probe rows came back non-finite; "
            f"this parameter combination broke the embedding")
        return {
            "failed": f"{bad} of {len(coordinates)} probe rows were non-finite after transform",
            "timing": {
                "gather_seconds": round(gather_seconds, 1),
                "umap_fit_seconds": round(fit_seconds, 1),
                "probe_transform_seconds": round(transform_seconds, 1),
                "cluster_seconds": 0.0,
                "metric_seconds": 0.0,
            },
            "embedding_quality": {},
            "cluster_quality": {},
            "_labels": None,
        }

    started = perf_counter()
    hdbscan_config = core.HdbscanConfig(
        min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
    clustered = core.fit_hdbscan(coordinates[cluster_positions], hdbscan_config, use_gpu=use_gpu)
    labels, dbcv = clustered.labels, clustered.dbcv
    cluster_seconds = perf_counter() - started

    # Davies-Bouldin and Calinski-Harabasz are centroid-based and linear, so they run on the
    # whole cluster tier. Silhouette is O(n^2) in distances and would never finish on 1 M
    # rows, so it goes on the pair tier -- ``pair_slots`` maps those rows to their positions
    # in ``labels``, which is indexed by the cluster tier rather than by the probe.
    quality = metrics.cluster_quality(
        coordinates[cluster_positions], labels,
        silhouette_coordinates=coordinates[pair_positions],
        silhouette_labels=labels[pair_slots],
    )
    quality["dbcv"] = dbcv

    started = perf_counter()
    local = metrics.local_quality(
        probe_latent[knn_positions], coordinates[knn_positions],
        k=args.metric_k, k_max=args.metric_k_max, use_gpu=use_gpu,
    )
    global_quality = metrics.distance_correlations(
        probe_latent[pair_positions], coordinates[pair_positions],
        pairs=args.distance_pairs, seed=args.seed,
    )
    metric_seconds = perf_counter() - started

    if local["clamped_fraction"] > 0.05:
        log(f"    WARNING: {local['clamped_fraction'] * 100:.1f}% of ranks fell outside "
            f"k_max={local['k_max']}; raise --metric-k-max or lower --knn-rows")

    return {
        "timing": {
            "gather_seconds": round(gather_seconds, 1),
            "umap_fit_seconds": round(fit_seconds, 1),
            "probe_transform_seconds": round(transform_seconds, 1),
            "cluster_seconds": round(cluster_seconds, 1),
            "metric_seconds": round(metric_seconds, 1),
        },
        "embedding_quality": {**local, **global_quality},
        "cluster_quality": quality,
        "_labels": labels,
    }


# ── aggregation across replicates ──────────────────────────────────────────────

def summarise(cells: list[dict]) -> dict:
    """Mean/spread of every metric, plus the between-replicate agreement.

    ``seed_stability`` is the headline: mean pairwise ARI/NMI/AMI/Jaccard between replicates
    of this same config. ``adjusted_rand_min`` is reported next to the mean because a config
    that is reproducible twice and wildly off the third time is not reproducible, and a mean
    hides that.
    """
    failed = [c for c in cells if c.get("failed")]
    ok = [c for c in cells if not c.get("failed")]
    summary: dict = {"replicates": len(cells), "failed_replicates": len(failed)}
    if failed:
        summary["failure"] = failed[0]["failed"]
    if not ok:
        # Every replicate broke. Report that rather than a row of nulls that reads like
        # "missing data" -- a config that cannot embed at all is a result.
        summary["seed_stability"] = None
        summary["stability_note"] = "every replicate failed"
        summary["timing_total"] = {key: round(float(np.sum([c["timing"][key] for c in cells])), 1)
                                   for key in cells[0]["timing"]}
        return summary

    def collect(section: str, key: str) -> list[float]:
        return [c[section][key] for c in ok
                if c[section].get(key) is not None and np.isfinite(c[section][key])]

    for key in ("trustworthiness", "continuity", "rnx_auc", "qnx_at_k",
                "spearman_distance", "pearson_distance", "clamped_fraction"):
        values = collect("embedding_quality", key)
        summary[key] = round(float(np.mean(values)), 5) if values else None
        summary[f"{key}_std"] = round(float(np.std(values)), 5) if len(values) > 1 else None

    for key in ("n_clusters", "noise_fraction", "silhouette",
                "davies_bouldin", "calinski_harabasz", "dbcv"):
        values = collect("cluster_quality", key)
        summary[key] = round(float(np.mean(values)), 5) if values else None
        summary[f"{key}_std"] = round(float(np.std(values)), 5) if len(values) > 1 else None

    if len(ok) < 2:
        summary["seed_stability"] = None
        summary["stability_note"] = ("needs --seeds >= 2" if not failed
                                     else f"only {len(ok)} replicate(s) survived")
    else:
        pairwise = [metrics.label_agreement(a["_labels"], b["_labels"])
                    for a, b in combinations(ok, 2)]
        stability = {}
        for key in ("adjusted_rand", "normalized_mutual_info",
                    "adjusted_mutual_info", "jaccard"):
            values = [p[key] for p in pairwise]
            stability[key] = round(float(np.mean(values)), 5)
        stability["adjusted_rand_min"] = round(float(min(p["adjusted_rand"] for p in pairwise)), 5)
        stability["pairs"] = len(pairwise)
        summary["seed_stability"] = stability

    total = {}
    for key in cells[0]["timing"]:
        total[key] = round(float(np.sum([c["timing"][key] for c in cells])), 1)
    summary["timing_total"] = total
    return summary


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep the size-independent UMAP parameters at a fixed fit size")
    parser.add_argument("--embed-dir", required=True, help="stage 1 output holding latent.npy")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-rows", type=int, default=5_000_000,
                        help="held identical across every config and replicate")
    parser.add_argument("--seeds", type=int, default=3,
                        help="replicates per config; each redraws the fit rows AND the UMAP "
                             "seed. 2 is the minimum for a stability number, 3 lets you see "
                             "whether one replicate is an outlier")

    # the grid; first value of each list is the baseline
    parser.add_argument("--min-dist", default="0.0,0.1,0.25")
    parser.add_argument("--spread", default="1.0,2.0",
                        help="crossed with --min-dist, since the two jointly fit the "
                             "output kernel")
    parser.add_argument("--repulsion-strength", default="1.0,0.5,2.0",
                        help="swept against the baseline; first value is the baseline")
    parser.add_argument("--set-op-mix-ratio", default="1.0,0.5",
                        help="swept against the baseline; first value is the baseline")
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="HELD FIXED here. n_neighbors is size-coupled -- the same k "
                             "covers a 25x smaller radius at 25M rows than at 1M -- so it "
                             "belongs in sweep 2 against fit size, not here")
    parser.add_argument("--n-epochs", type=int, default=200)

    # evaluation tiers
    parser.add_argument("--probe-rows", type=int, default=1_000_000)
    parser.add_argument("--cluster-rows", type=int, default=1_000_000,
                        help="rows HDBSCAN is fit on, taken from the probe. Lower this if "
                             "the printed cluster_seconds dominates the cell")
    parser.add_argument("--knn-rows", type=int, default=50_000,
                        help="smaller than the size sweep's 250k on purpose, so k_max can "
                             "be wide enough to stop clamping")
    parser.add_argument("--pair-rows", type=int, default=20_000)
    parser.add_argument("--distance-pairs", type=int, default=20_000_000)
    parser.add_argument("--metric-k", type=int, default=15)
    parser.add_argument("--metric-k-max", type=int, default=5000,
                        help="must exceed the typical cluster size in the knn tier or "
                             "trustworthiness and continuity are optimistic. At 50k rows "
                             "and ~25 clusters that is ~2000, so 5000 has real margin")

    parser.add_argument("--min-cluster-size", type=int, default=500)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--transform-batch-size", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-budget-gb", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    started_all = perf_counter()
    args = parse_args()

    embed_dir = Path(args.embed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_path = embed_dir / "latent.npy"
    if not latent_path.exists():
        raise SystemExit(f"latent.npy not found in {embed_dir}")

    use_gpu = core.gpu_available()
    if use_gpu and args.gpu_budget_gb is not None:
        core.apply_gpu_budget("sweep", budget_gb=args.gpu_budget_gb)

    latent = np.load(latent_path, mmap_mode="r")
    total_rows, latent_dim = latent.shape
    log(f"{latent_path.name}: {total_rows:,} rows x {latent_dim} dims")

    configs = build_configs(args)
    log(f"{len(configs)} configs x {args.seeds} replicates = {len(configs) * args.seeds} fits "
        f"at {args.fit_rows:,} rows")
    for config in configs:
        log(f"    {config.label()}")

    probe_index, pool = draw_probe(total_rows, args.probe_rows, args.seed)
    if args.fit_rows > pool.size:
        raise SystemExit(
            f"--fit-rows {args.fit_rows:,} exceeds the {pool.size:,} rows left after the probe")

    log("materialising the probe latents ...")
    probe_latent = np.ascontiguousarray(latent[probe_index])

    # Every tier is drawn once and reused for every config and replicate, so all comparisons
    # are on identical rows. The tiers are strictly nested (pair < knn < cluster < probe):
    # silhouette needs cluster labels for the pair rows, and nesting makes that a
    # searchsorted rather than an intersection.
    subset_rng = np.random.default_rng(args.seed + 1)
    cluster_positions = np.sort(subset_rng.choice(
        len(probe_index), size=min(args.cluster_rows, len(probe_index)), replace=False))
    knn_positions = np.sort(subset_rng.choice(
        cluster_positions, size=min(args.knn_rows, len(cluster_positions)), replace=False))
    pair_positions = np.sort(subset_rng.choice(
        knn_positions, size=min(args.pair_rows, len(knn_positions)), replace=False))
    # Positions of the pair rows within the cluster tier. Exact because pair ⊆ cluster and
    # both are sorted.
    pair_slots = np.searchsorted(cluster_positions, pair_positions)
    log(f"tiers: probe {len(probe_index):,} / cluster {len(cluster_positions):,} / "
        f"knn {len(knn_positions):,} / pair {len(pair_positions):,}")

    # Replicate outermost so each fit set is gathered once and shared by every config --
    # which is also what guarantees configs are compared on identical data.
    cells: dict[str, list[dict]] = {config.label(): [] for config in configs}
    total_cells = len(configs) * args.seeds
    done = 0
    for replicate in range(args.seeds):
        fit_index = draw_fit_set(pool, args.fit_rows, args.seed, replicate)
        log(f"=== replicate {replicate + 1}/{args.seeds} "
            f"({args.fit_rows:,} rows, umap seed {args.seed + replicate}) ===")
        for config in configs:
            # The UMAP seed moves with the replicate; the config object is otherwise identical
            # across replicates, so label() stays the key.
            seeded = core.UmapConfig(**{**config.as_dict(), "seed": args.seed + replicate})
            log(f"  {config.label()} ...")
            cell = run_cell(
                seeded, latent, fit_index, probe_latent,
                knn_positions, pair_positions, cluster_positions, pair_slots, args, use_gpu,
            )
            cells[config.label()].append(cell)
            done += 1
            timing = cell["timing"]
            elapsed = perf_counter() - started_all
            outcome = ("FAILED" if cell.get("failed") else
                       f"{cell['cluster_quality']['n_clusters']} clusters, "
                       f"{cell['cluster_quality']['noise_fraction'] * 100:.0f}% noise, "
                       f"rnx {cell['embedding_quality']['rnx_auc']:.4f}")
            log(f"    fit {timing['umap_fit_seconds']:.0f}s  "
                f"transform {timing['probe_transform_seconds']:.0f}s  "
                f"hdbscan {timing['cluster_seconds']:.0f}s  "
                f"metrics {timing['metric_seconds']:.0f}s  |  {outcome}")
            log(f"    cell {done}/{total_cells}, elapsed {elapsed / 60:.0f} min, "
                f"eta {(elapsed / done) * (total_cells - done) / 60:.0f} min")

    log("=== aggregating across replicates ===")
    summaries = {label: summarise(cell_list) for label, cell_list in cells.items()}

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sweep": "parameters_independent_of_fit_size",
        "embed_dir": str(embed_dir),
        "total_rows": int(total_rows),
        "latent_dim": int(latent_dim),
        "fit_rows": args.fit_rows,
        "replicates": args.seeds,
        "n_neighbors_held_at": args.n_neighbors,
        "probe_rows": int(len(probe_index)),
        "cluster_rows": int(len(cluster_positions)),
        "knn_rows": int(len(knn_positions)),
        "pair_rows": int(len(pair_positions)),
        "metric_k": args.metric_k,
        "metric_k_max": args.metric_k_max,
        "clusterer": "hdbscan",
        "backend": "cuml" if use_gpu else "cpu",
        "ranked_on": {"fidelity": FIDELITY_METRIC, "stability": STABILITY_METRIC},
        "diagnostic_only": list(DIAGNOSTIC_ONLY),
        "configs": {config.label(): config.as_dict() for config in configs},
        "summaries": summaries,
        "total_seconds": round(perf_counter() - started_all, 1),
    }
    destination = output_dir / "parameter_sweep.json"
    destination.write_text(json.dumps(payload, indent=2))

    print_report(payload)
    log(f"wrote {destination}")
    return 0


def print_report(payload: dict) -> None:
    summaries = payload["summaries"]
    labels = list(summaries)

    def cell(value, width=10):
        if value is None:
            return f"{'--':>{width}}"
        if isinstance(value, float):
            text = f"{value:.4f}" if abs(value) < 1000 else f"{value:.4g}"
        else:
            text = str(value)
        return f"{text:>{width}}"

    def stability_of(label, key="adjusted_rand"):
        block = summaries[label].get("seed_stability")
        return None if block is None else block[key]

    header = (f"  {'config':<26}{'rnx_auc':>10}{'trust':>10}{'spearman':>10}"
              f"{'ARI_seed':>10}{'ARI_min':>10}{'clusters':>10}{'noise%':>10}"
              f"{'dbcv':>10}{'silh':>10}")
    rule = "=" * len(header)
    print("\n" + rule)
    print(f"UMAP size-independent parameter sweep -- fit {payload['fit_rows']:,} rows, "
          f"{payload['replicates']} replicates, n_neighbors pinned at "
          f"{payload['n_neighbors_held_at']}")
    print(rule)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label in labels:
        summary = summaries[label]
        if summary.get("failed_replicates") and summary.get("rnx_auc") is None:
            print(f"  {label:<26}{'FAILED -- ' + summary.get('failure', ''):<70}")
            continue
        flag = "" if not summary.get("failed_replicates") else \
            f"  [{summary['failed_replicates']}/{summary['replicates']} replicates failed]"
        print(f"  {label:<26}"
              + cell(summary["rnx_auc"]) + cell(summary["trustworthiness"])
              + cell(summary["spearman_distance"])
              + cell(stability_of(label)) + cell(stability_of(label, "adjusted_rand_min"))
              + cell(summary["n_clusters"])
              + cell(None if summary["noise_fraction"] is None
                     else summary["noise_fraction"] * 100)
              + cell(summary["dbcv"]) + cell(summary["silhouette"]) + flag)

    print("\n  -- replicate-to-replicate spread (std across seeds) --")
    print(f"  {'config':<26}{'rnx_std':>10}{'trust_std':>10}{'clusters_std':>14}"
          f"{'noise_std':>12}")
    for label in labels:
        summary = summaries[label]
        print(f"  {label:<26}"
              + cell(summary.get("rnx_auc_std")) + cell(summary.get("trustworthiness_std"))
              + cell(summary.get("n_clusters_std"), 14)
              + cell(summary.get("noise_fraction_std"), 12))

    clamped = [summaries[label].get("clamped_fraction") for label in labels]
    worst = max((value for value in clamped if value is not None), default=None)
    if worst is not None:
        verdict = "negligible" if worst < 0.05 else "TOO HIGH -- raise --metric-k-max"
        print(f"\n  worst clamped_fraction {worst * 100:.2f}%  ({verdict})")

    # ── verdict ────────────────────────────────────────────────────────────────
    print("\n  -- verdict --")
    fidelity = [(label, summaries[label].get(FIDELITY_METRIC)) for label in labels]
    fidelity = [(label, value) for label, value in fidelity if value is not None]
    stability = [(label, stability_of(label)) for label in labels]
    stability = [(label, value) for label, value in stability if value is not None]

    if not fidelity:
        print("  no usable fidelity numbers")
        return
    best_fidelity = max(fidelity, key=lambda item: item[1])
    print(f"  best fidelity ({FIDELITY_METRIC}):  {best_fidelity[0]}  "
          f"({best_fidelity[1]:.4f})")

    if not stability:
        print("  no stability numbers -- rerun with --seeds >= 2 before trusting any of this")
        return
    best_stability = max(stability, key=lambda item: item[1])
    print(f"  best stability (seed ARI):   {best_stability[0]}  ({best_stability[1]:.4f})")

    fidelity_values = [value for _, value in fidelity]
    spread = max(fidelity_values) - min(fidelity_values)
    noise_floor = max(
        (summaries[label].get("rnx_auc_std") or 0.0) for label in labels)
    if spread <= noise_floor:
        print(f"\n  The whole grid spans {spread:.4f} in {FIDELITY_METRIC}, which is inside "
              f"the {noise_floor:.4f} replicate-to-replicate spread. None of these "
              f"parameters is doing anything measurable -- keep the defaults and spend the "
              f"time on sweep 2.")
    elif best_fidelity[0] == best_stability[0]:
        print(f"\n  {best_fidelity[0]} wins on both axes; adopt it for sweep 2.")
    else:
        chosen = summaries[best_fidelity[0]]
        chosen_stability = stability_of(best_fidelity[0])
        print(f"\n  Fidelity and stability disagree. Prefer {best_stability[0]} unless "
              f"{best_fidelity[0]} is also stable enough "
              f"(its seed ARI is {chosen_stability:.4f}); a config whose clusters move "
              f"between replicates cannot support a size-convergence claim in sweep 2.")

    weak = [label for label in labels
            if (stability_of(label) or 1.0) < 0.5]
    if weak:
        print(f"\n  Unstable across seeds (ARI < 0.5), treat any single-seed result from "
              f"these as noise: {', '.join(weak)}")
    print(f"\n  Ranked on {FIDELITY_METRIC} and seed ARI only. "
          f"{', '.join(DIAGNOSTIC_ONLY)} are printed for context and deliberately not "
          f"ranked on -- UMAP parameters move them mechanically.")
    print(rule + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
