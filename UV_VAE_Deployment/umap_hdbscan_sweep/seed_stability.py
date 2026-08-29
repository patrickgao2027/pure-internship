"""Is seed noise the reason two cells score differently? Decompose it.

The sweep's family E varies *everything* per replicate -- the fit rows are redrawn and
every seed changes -- so a low ``adjusted_rand_min`` cannot be attributed. It might be
optimiser randomness, or it might be that a different 5 M rows genuinely gives a
different map. With three replicates there is no way to tell, and the difference matters:
optimiser noise is a nuisance to be averaged away, whereas subsample sensitivity *is the
result* this project exists to measure.

This script holds one axis fixed at a time.

``--axis encoder`` (default, cheap)
    Fit UMAP **once**, then train K encoders on that one fitted graph, varying only the
    torch seed. Everything else -- fit rows, UMAP embedding, graph edges -- is identical.
    Whatever spread survives is pure encoder-optimiser noise, and it is the *floor* on
    how reproducible any downstream number can be.

``--axis full`` (expensive)
    Redraw the fit rows and refit UMAP per seed, reproducing what a sweep replicate
    actually varies. Spread here includes the encoder noise above plus subsample and
    UMAP-fit sensitivity.

Run ``encoder`` first. If it already reproduces the spread seen across sweep replicates,
seed noise explains the result and the cell rankings are not measuring the parameters. If
it is much tighter, the variation is coming from the row draw, which is a finding rather
than a nuisance.

**Cost.** ``fit_timing_by_size`` in the sweep JSON puts the UMAP fit at 122 s for 5 M and
2797 s for 25 M. Under ``--axis encoder`` that is paid once per cell; under ``--axis
full`` it is paid once per seed, which at 25 M is ~47 min x K. Encoder training is ~9 s,
and scoring is ~36 s of Spearman plus HDBSCAN on the 500 k cluster tier.

Scored per seed: Spearman distance correlation against the 16-D latent, cluster count and
noise fraction. Scored pairwise across seeds: k-NN overlap at k=15 (frame-invariant --
it compares neighbour identities, not coordinates, so it is valid for ``umap`` mode) plus
ARI and NMI between HDBSCAN labellings. Reported as mean, sd, min.

Rows are the sweep's own probe and tiers, reconstructed from the same seeds, so these
numbers sit on the same scale as the ones in ``parametric_sweep.json``.

Usage::

    python umap_hdbscan_sweep/seed_stability.py \
        --sweep-json  umap_tests/parametric_sweep/runD/parametric_sweep.json \
        --embed-dir   <stage 1 output holding latent.npy> \
        --cells '25000000|nn15_md0.1_nc2|umap' \
        --seeds 5 --output umap_tests/seed_stability_25M.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import parametric_sweep as ps  # noqa: E402
import parametric_umap as pu  # noqa: E402
import sweep_core as core  # noqa: E402
import umap_metrics as metrics  # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def build_tiers(probe_index: np.ndarray, args) -> dict[str, np.ndarray]:
    """The sweep's tier draw, reproduced call-for-call.

    Same generator, same seed offset, same order of ``choice`` calls -- anything else
    would score these seeds on different rows than the sweep used and break the
    comparison the script exists to make.
    """
    rng = np.random.default_rng(args.seed + 1)
    n = len(probe_index)
    cluster = np.sort(rng.choice(n, size=min(args.cluster_rows, n), replace=False))
    overlap = np.sort(rng.choice(n, size=min(args.overlap_rows, n), replace=False))
    knn = np.sort(rng.choice(overlap, size=min(args.knn_rows, len(overlap)), replace=False))
    return {"cluster": cluster, "overlap": overlap, "knn": knn}


def fit_umap_graph(fit_latent: np.ndarray, config, mode: str):
    """UMAP fit plus the pieces the encoder objectives need from it."""
    started = perf_counter()
    fitted = core.fit_umap(fit_latent, config)
    log(f"    UMAP fit on {len(fit_latent):,} rows: {perf_counter() - started:.0f}s")
    embedding = np.ascontiguousarray(fitted.embedding)
    edges = pu.extract_edges(fitted.model.graph_) if mode in ("umap", "hybrid") else None
    negative_rate = int(getattr(fitted.model, "negative_sample_rate", 5))
    del fitted
    return embedding, edges, negative_rate


def train_encoder(fit_latent, embedding, edges, negative_rate, config, mode,
                  torch_seed: int, args, device):
    """One encoder at one torch seed. Mirrors apply_parametric_full's training block."""
    import torch

    torch.manual_seed(torch_seed)
    encoder = pu.ParametricEncoder(
        fit_latent.shape[1], output_dim=2, hidden=tuple(args.hidden)).to(device)
    X = torch.from_numpy(fit_latent).to(device)
    Y = torch.from_numpy(embedding).to(device)

    started = perf_counter()
    if mode in ("regress", "hybrid"):
        pu.train_regression(encoder, X, Y, steps=args.regress_steps,
                            batch_size=args.batch_size, learning_rate=args.learning_rate,
                            device=device)
    if mode in ("umap", "hybrid"):
        head, tail, weight = edges
        if len(head) > args.max_edges:
            keep = np.sort(np.random.default_rng(torch_seed).choice(
                len(head), size=args.max_edges, replace=False))
            head, tail, weight = head[keep], tail[keep], weight[keep]
        a, b = pu.ab_params(config.min_dist, spread=config.spread)
        pu.train_umap_loss(
            encoder, X,
            torch.from_numpy(head).to(device),
            torch.from_numpy(tail).to(device),
            # float64: a float32 running sum saturates near 1.7e7 and most of a large
            # graph would never be drawn. Same reason as in train_umap_loss.
            torch.cumsum(torch.from_numpy(weight).to(device).double(), 0),
            a=a, b=b, steps=args.umap_steps, batch_size=args.batch_size,
            learning_rate=args.umap_learning_rate if mode == "hybrid" else args.learning_rate,
            negative_sample_rate=negative_rate,
            repulsion_strength=config.repulsion_strength, device=device)
    seconds = perf_counter() - started

    del X, Y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pu.ParametricUmap(encoder=encoder, device=device, mode=mode), round(seconds, 1)


def score_seed(model, probe_latent, tiers, pair_rows, args, use_gpu):
    """Per-seed numbers plus the artifacts the pairwise comparison needs."""
    coords = model.transform(probe_latent, batch_size=args.infer_batch_size)
    global_scores = metrics.distance_correlations(
        probe_latent[pair_rows], coords[pair_rows],
        pairs=args.distance_pairs, seed=args.seed)

    hdbscan_config = core.HdbscanConfig(
        min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
    labels = core.fit_hdbscan(coords[tiers["cluster"]], hdbscan_config, use_gpu=use_gpu).labels

    return {
        "spearman_distance": global_scores["spearman_distance"],
        "pearson_distance": global_scores["pearson_distance"],
        "n_clusters": int(len(set(labels.tolist()) - {-1})),
        "noise_fraction": round(float(np.mean(labels == -1)), 5),
    }, coords[tiers["overlap"]], labels


def summarise(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 5),
        "sd": round(float(array.std(ddof=1)), 5) if len(array) > 1 else None,
        "min": round(float(array.min()), 5),
        "max": round(float(array.max()), 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-json", required=True)
    parser.add_argument("--embed-dir", required=True, help="stage 1 output holding latent.npy")
    parser.add_argument("--cells", required=True,
                        help="comma-separated sweep cell keys, e.g. '25000000|nn15_md0.1_nc2|umap'")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seeds", type=int, default=5, help="encoders to train per cell")
    parser.add_argument("--axis", choices=("encoder", "full"), default="encoder",
                        help="'encoder' refits UMAP once and varies only the torch seed; "
                             "'full' redraws fit rows and refits UMAP per seed (see the "
                             "module docstring -- at 25M that is ~47 min PER SEED)")
    parser.add_argument("--seed-base", type=int, default=1000,
                        help="torch seeds are seed-base + 0..K-1, kept clear of the "
                             "sweep's own seed+replicate so these are genuinely new draws")
    parser.add_argument("--replicate", type=int, default=0,
                        help="which replicate's fit rows to hold fixed under --axis encoder")

    parser.add_argument("--seed", type=int, default=42, help="the sweep's --seed")
    parser.add_argument("--cluster-rows", type=int, default=500_000)
    parser.add_argument("--overlap-rows", type=int, default=200_000)
    parser.add_argument("--knn-rows", type=int, default=30_000)
    parser.add_argument("--pair-rows", type=int, default=20_000)
    parser.add_argument("--distance-pairs", type=int, default=20_000_000)
    parser.add_argument("--min-cluster-size", type=int, default=250)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--overlap-k", type=int, default=15)

    parser.add_argument("--hidden", default="256,256,128")
    parser.add_argument("--regress-steps", type=int, default=15_000)
    parser.add_argument("--umap-steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--umap-learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-edges", type=int, default=150_000_000)
    parser.add_argument("--infer-batch-size", type=int, default=2_000_000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    args.hidden = tuple(int(h) for h in args.hidden.split(","))

    import torch

    use_gpu = not args.cpu and core.gpu_available()
    device = torch.device("cuda" if not args.cpu and torch.cuda.is_available() else "cpu")

    payload = json.loads(Path(args.sweep_json).read_text())
    latent = np.load(Path(args.embed_dir) / "latent.npy", mmap_mode="r")
    probe_index, pool = ps.draw_probe(latent.shape[0], payload["probe_rows"], args.seed)
    probe_latent = np.ascontiguousarray(latent[probe_index])
    tiers = build_tiers(probe_index, args)
    pair_rows = np.sort(np.random.default_rng(args.seed).permutation(
        len(probe_index))[:args.pair_rows])

    log(f"latent {latent.shape}; probe {len(probe_index):,}; axis={args.axis}; "
        f"{args.seeds} seeds per cell; device={device}\n")

    results = {}
    for key in [c.strip() for c in args.cells.split(",") if c.strip()]:
        fit_rows_str, label, mode = key.split("|")
        fit_rows = int(fit_rows_str)
        log(f"=== {key} ===")

        per_seed, overlap_coords, label_sets = [], [], []
        shared = None
        for index in range(args.seeds):
            torch_seed = args.seed_base + index
            if args.axis == "full":
                # A different replicate index redraws the fit rows exactly as the sweep
                # would, and the UMAP seed moves with it.
                config = core.UmapConfig(**{**payload["graph_configs"][label],
                                            "seed": torch_seed})
                fit_index = ps.draw_nested_fit_sets(
                    pool, payload["sizes"], args.seed, args.replicate + index)[fit_rows]
                fit_latent = np.ascontiguousarray(latent[fit_index])
                shared = (*fit_umap_graph(fit_latent, config, mode), fit_latent)
            elif shared is None:
                config = core.UmapConfig(**{**payload["graph_configs"][label],
                                            "seed": args.seed + args.replicate})
                fit_index = ps.draw_nested_fit_sets(
                    pool, payload["sizes"], args.seed, args.replicate)[fit_rows]
                fit_latent = np.ascontiguousarray(latent[fit_index])
                shared = (*fit_umap_graph(fit_latent, config, mode), fit_latent)
            else:
                config = core.UmapConfig(**{**payload["graph_configs"][label],
                                            "seed": args.seed + args.replicate})

            embedding, edges, negative_rate, fit_latent = shared
            model, train_seconds = train_encoder(
                fit_latent, embedding, edges, negative_rate, config, mode,
                torch_seed, args, device)

            started = perf_counter()
            scores, coords, labels = score_seed(
                model, probe_latent, tiers, pair_rows, args, use_gpu)
            scores["torch_seed"] = torch_seed
            scores["train_seconds"] = train_seconds
            per_seed.append(scores)
            overlap_coords.append(coords)
            label_sets.append(labels)
            log(f"  seed {torch_seed}: spearman {scores['spearman_distance']:.5f}  "
                f"clusters {scores['n_clusters']}  noise {scores['noise_fraction']:.3f}  "
                f"({train_seconds:.0f}s train, {perf_counter() - started:.0f}s score)")
            del model

        overlaps, aris, nmis = [], [], []
        for left, right in combinations(range(len(per_seed)), 2):
            overlaps.append(ps.overlap_profile(
                overlap_coords[left], overlap_coords[right],
                (args.overlap_k,), use_gpu)[f"knn_overlap_k{args.overlap_k}"])
            agreement = metrics.label_agreement(label_sets[left], label_sets[right])
            aris.append(agreement["adjusted_rand"])
            nmis.append(agreement["normalized_mutual_info"])

        results[key] = {
            "axis": args.axis,
            "seeds": args.seeds,
            "per_seed": per_seed,
            "spearman_distance": summarise([s["spearman_distance"] for s in per_seed]),
            "n_clusters": summarise([s["n_clusters"] for s in per_seed]),
            "noise_fraction": summarise([s["noise_fraction"] for s in per_seed]),
            "pairwise": {
                "pairs": len(overlaps),
                f"knn_overlap_k{args.overlap_k}": summarise(overlaps),
                "adjusted_rand": summarise(aris),
                "normalized_mutual_info": summarise(nmis),
            },
            "sweep_reference": (payload["summaries"].get(key) or {}).get("reproducibility"),
        }
        r = results[key]
        log(f"  -> spearman {r['spearman_distance']['mean']:.5f} "
            f"+- {r['spearman_distance']['sd']}")
        log(f"  -> pairwise ARI {r['pairwise']['adjusted_rand']['mean']:.4f} "
            f"(min {r['pairwise']['adjusted_rand']['min']:.4f}), "
            f"knn{args.overlap_k} {r['pairwise'][f'knn_overlap_k{args.overlap_k}']['mean']:.4f}")
        if r["sweep_reference"]:
            log(f"  -> sweep replicates gave ARI {r['sweep_reference']['adjusted_rand']:.4f} "
                f"(min {r['sweep_reference']['adjusted_rand_min']:.4f}), "
                f"knn15 {r['sweep_reference']['knn_overlap_k15']:.4f}")
        log("")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "sweep_json": args.sweep_json, "axis": args.axis, "seeds": args.seeds,
            "seed_base": args.seed_base, "replicate": args.replicate,
            "cells": results,
        }, indent=2))
        log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
