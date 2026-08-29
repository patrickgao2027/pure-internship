"""Global-structure scoring for the parametric sweep's encoders.

The parametric sweep reports trustworthiness, continuity and RNX AUC. All three are
*local*: they ask where a point's near neighbours went. RNX integrates over k but with
1/k weighting, so it is local-dominated too. None of them can tell you whether two
things that are far apart in the 16-D latent stayed far apart in 2-D -- and that is
exactly the question a visually well-separated map raises. A map can score 0.968
trustworthiness while inventing the gaps between its islands.

``parameter_sweep.py`` already scores this via ``umap_metrics.distance_correlations``
(Spearman rank correlation between 16-D and 2-D pairwise distance over a sampled set of
pairs). ``parametric_sweep.py`` simply never calls it. This script closes that gap
without re-running the sweep: it reloads the encoders the sweep saved under
``--encoder-dir``, embeds one fixed set of probe rows through each, and correlates.

**It never refits.** ``apply_parametric_full.py`` falls back to refitting UMAP when an
encoder is missing, which the sweep's own ``fit_timing_by_size`` puts at 2797 s for the
25 M fit. Here a missing encoder is reported and skipped, because the point of this
script is that it is cheap.

Every cell is scored on the *same* probe rows and, because
``distance_correlations`` draws its pairs from a fixed seed, the *same* pairs. So
differences between cells are differences in the embedding and nothing else -- there is
no sampling noise in the comparison at all.

**Reading the number.** UMAP does not try to preserve global distance and absolute
Spearman on this latent is low: the non-parametric sweep in ``umap_tests/run1`` sits
around 0.18-0.19. Do not read 0.2 as failure. Judge a cell against the other cells, and
against that non-parametric anchor, the way ``rnx_ratio`` is judged against cuML rather
than against 1.0.

The diagnostic this exists for: if a cell has high trustworthiness *and* Spearman in
line with the rest, its visible separation reflects real 16-D structure. If
trustworthiness stays high while Spearman drops, the map is manufacturing separation --
pushing apart points that the latent had close together.

Usage (on miletus, where the encoders and latent.npy live)::

    python umap_hdbscan_sweep/global_structure.py \
        --sweep-json  umap_tests/parametric_sweep/runC/parametric_sweep.json \
        --embed-dir   <stage 1 output holding latent.npy> \
        --encoder-dir <dir the sweep wrote with --encoder-dir> \
        --output      umap_tests/global_structure.json

Add ``--all`` to score every cell instead of the umap-mode default, or
``--cells 'A|B|C,...'`` for an explicit list using the sweep's own cell keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import parametric_sweep as ps  # noqa: E402
import parametric_umap as pu  # noqa: E402
import sweep_core as core  # noqa: E402
import umap_metrics  # noqa: E402


def config_from_payload(payload: dict, label: str, seed: int) -> core.UmapConfig:
    """The sweep's graph config for one label, reseeded for the wanted replicate.

    The stored spec carries the sweep's own seed; the encoder filename is keyed on
    ``seed + replicate``, so it has to be overridden rather than reused.
    """
    spec = dict(payload["graph_configs"][label])
    spec["seed"] = seed
    return core.UmapConfig(**spec)


def log(message: str) -> None:
    print(message, flush=True)


def select_cells(payload: dict, args) -> list[str]:
    """Cell keys to score, in the sweep's own ``fit_rows|label|mode`` form."""
    summaries = payload["summaries"]
    if args.cells:
        wanted = [c.strip() for c in args.cells.split(",") if c.strip()]
        missing = [c for c in wanted if c not in summaries]
        if missing:
            raise SystemExit(f"no such cell(s) in the sweep JSON: {', '.join(missing)}")
        return wanted

    keys = [k for k, s in summaries.items() if s.get("rnx_ratio") is not None]
    if not args.all:
        # umap mode is the default because it is the one whose global behaviour is in
        # question: it optimises the UMAP loss in its own unconstrained frame, so
        # nothing anchors its large-scale geometry to the latent's.
        keys = [k for k in keys if k.split("|")[2] == "umap"]
    keys.sort(key=lambda k: payload["summaries"][k]["rnx_ratio"], reverse=True)
    return keys


def load_encoder(encoder_dir: Path, fit_rows: int, config, mode: str,
                 latent_dim: int, hidden: tuple[int, ...], device):
    """The saved encoder for one cell, or None. Never refits -- see the module docstring."""
    import torch

    path = encoder_dir / f"{fit_rows}__{config.label()}__{mode}__seed{config.seed}.pt"
    if not path.exists():
        return None, path
    blob = torch.load(path, map_location=device)
    encoder = pu.ParametricEncoder(latent_dim, output_dim=2, hidden=hidden).to(device)
    encoder.load_state_dict(blob["state_dict"])
    return pu.ParametricUmap(encoder=encoder, device=device, mode=mode), path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-json", required=True,
                        help="parametric_sweep.json from the run whose cells you want scored")
    parser.add_argument("--embed-dir", required=True, help="stage 1 output holding latent.npy")
    parser.add_argument("--encoder-dir", required=True,
                        help="the dir parametric_sweep.py was run with --encoder-dir; "
                             "cells without a saved encoder are skipped, never refitted")
    parser.add_argument("--output", default=None, help="write the results as JSON here")
    parser.add_argument("--cells", default=None,
                        help="comma-separated sweep cell keys; default is every umap-mode cell")
    parser.add_argument("--all", action="store_true", help="score every mode, not just umap")
    parser.add_argument("--replicate", type=int, default=0,
                        help="which replicate's encoder to load; must match the sweep's seeding")
    parser.add_argument("--seed", type=int, default=42,
                        help="the sweep's --seed, so the probe draw is reproduced exactly")
    parser.add_argument("--pair-rows", type=int, default=20_000,
                        help="rows the pairs are drawn from; parameter_sweep.py's default")
    parser.add_argument("--distance-pairs", type=int, default=20_000_000,
                        help="sampled pairs per cell; 2e7 pins a correlation to well under 0.001")
    parser.add_argument("--hidden", default="256,256,128",
                        help="encoder hidden sizes; must match what the sweep trained")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is present")
    args = parser.parse_args()

    import torch

    hidden = tuple(int(h) for h in args.hidden.split(","))
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    encoder_dir = Path(args.encoder_dir)

    payload = json.loads(Path(args.sweep_json).read_text())
    cells = select_cells(payload, args)

    latent = np.load(Path(args.embed_dir) / "latent.npy", mmap_mode="r")
    log(f"latent {latent.shape} on {device}; {len(cells)} cell(s) to score")

    # The sweep's own probe draw, then a fixed prefix of it. Same rows for every cell,
    # and distance_correlations samples its pairs from a fixed seed, so the comparison
    # between cells carries no sampling noise whatsoever.
    probe, _pool = ps.draw_probe(latent.shape[0], payload["probe_rows"], args.seed)
    rows = np.sort(probe[:args.pair_rows])
    high = np.ascontiguousarray(latent[rows], dtype=np.float32)
    log(f"scoring on {len(rows):,} fixed probe rows, {args.distance_pairs:,} sampled pairs\n")

    results, skipped = {}, []
    for key in cells:
        fit_rows, label, mode = key.split("|")
        fit_rows = int(fit_rows)
        config = config_from_payload(payload, label, args.seed + args.replicate)

        model, path = load_encoder(encoder_dir, fit_rows, config, mode,
                                   latent.shape[1], hidden, device)
        if model is None:
            log(f"  SKIP {key}: no saved encoder at {path.name}")
            skipped.append(key)
            continue

        started = perf_counter()
        low = model.transform(high)
        scores = umap_metrics.distance_correlations(
            high, low, pairs=args.distance_pairs, seed=args.seed)
        scores["seconds"] = round(perf_counter() - started, 1)
        summary = payload["summaries"][key]
        scores["rnx_ratio"] = summary.get("rnx_ratio")
        scores["trustworthiness"] = summary.get("map_quality_net", {}).get("trustworthiness")
        results[key] = scores
        log(f"  {key:36} spearman {scores['spearman_distance']:.4f}  "
            f"pearson {scores['pearson_distance']:.4f}  ({scores['seconds']}s)")

    if not results:
        log("\nnothing scored -- no saved encoders were found under --encoder-dir")
        return 1

    log(f"\n{'cell':38}{'spearman':>10}{'pearson':>10}{'trust':>9}{'rnx_ratio':>11}")
    log("-" * 78)
    for key, s in sorted(results.items(), key=lambda kv: -kv[1]["spearman_distance"]):
        trust = s["trustworthiness"]
        log(f"{key:38}{s['spearman_distance']:>10.4f}{s['pearson_distance']:>10.4f}"
            f"{trust if trust is None else f'{trust:.4f}':>9}"
            f"{s['rnx_ratio']:>11.4f}")
    log("\nAbsolute values are low by design -- UMAP does not preserve global distance. "
        "The non-parametric\nsweep in umap_tests/run1 sits at 0.18-0.19 on this latent; "
        "compare against that and against\neach other, not against 1.0.")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "sweep_json": str(args.sweep_json),
            "probe_rows_scored": int(len(rows)),
            "distance_pairs": int(args.distance_pairs),
            "replicate": args.replicate,
            "seed": args.seed,
            "skipped_no_encoder": skipped,
            "cells": results,
        }, indent=2))
        log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
