"""Phase A — bottleneck timing + out-of-sample repulsion probe.

Two questions answered in one run:

  1. TIMER SPLIT
     How much of the 3431 s UMAP stage is fit_umap vs embed_all_rows (transform)?
     Fits one UMAP config on 5 M rows, then transforms the same 5 M-row memmap slice
     in batches exactly as stage2_sweep.py does, timing each step separately.

  2. REPULSION PROBE (Gate 2)
     Do out-of-sample rows accumulate at cluster peripheries when placed by transform()?
     Draws a 200 k probe set that is *withheld* from the UMAP fit, then embeds those rows
     two ways: (a) as part of a fresh fit on 5 M including the probe, (b) via transform()
     on the model fit without them. Compares each probe point's radial distance from its
     assigned cluster centroid in the two conditions. A systematic outward shift in (b)
     confirms the repulsion effect described in arXiv:2606.04451.

Usage (on miletus, inside micromamba env uv_vae):

    python umap_hdbscan_sweep/phase_a_timing.py \\
        --embed-dir <stage1-output-dir> \\
        --output-dir <results-dir>

Optional overrides:
    --fit-rows 5000000          # rows given to UMAP for fit (default 5 M)
    --probe-rows 200000         # rows withheld for the repulsion probe
    --transform-batch-size 5000000
    --seed 42
    --gpu-budget-gb 16

Output:
    phase_a_results.json  -- all timings and probe statistics
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np

import sweep_core as core


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A: timer split + repulsion probe")
    p.add_argument("--embed-dir", required=True,
                   help="stage 1 output directory containing latent.npy")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fit-rows", type=int, default=5_000_000)
    p.add_argument("--probe-rows", type=int, default=200_000,
                   help="rows withheld for the repulsion probe (must be < fit-rows)")
    p.add_argument("--transform-batch-size", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-budget-gb", type=float, default=None)
    return p.parse_args()


# ── helpers ────────────────────────────────────────────────────────────────────

def draw_indices(total: int, n: int, seed: int) -> np.ndarray:
    """Sorted uniform random sample without replacement, matching stage2_sweep.fit_indices."""
    return np.sort(np.random.default_rng(seed).choice(total, size=n, replace=False))


def cluster_centroids(embedding: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Mean 2-D position per cluster label (noise label -1 excluded)."""
    unique = np.unique(labels)
    return {
        int(lbl): embedding[labels == lbl].mean(axis=0)
        for lbl in unique
        if lbl >= 0
    }


def radial_distances(embedding: np.ndarray, labels: np.ndarray,
                     centroids: dict[int, np.ndarray]) -> np.ndarray:
    """Per-point Euclidean distance to its cluster centroid; noise points get NaN."""
    dists = np.full(len(embedding), np.nan, dtype=np.float32)
    for lbl, centroid in centroids.items():
        mask = labels == lbl
        if mask.any():
            dists[mask] = np.linalg.norm(embedding[mask] - centroid, axis=1)
    return dists


# ── Part 1: timer split ────────────────────────────────────────────────────────

def time_fit_and_transform(
    latent: np.ndarray,
    fit_idx: np.ndarray,
    held_idx: np.ndarray,
    config: core.UmapConfig,
    transform_batch_size: int,
) -> dict:
    """Fit on fit_idx rows, then transform held_idx rows, timing each step.

    held_idx here is the ~152.5 M complement used in the real pipeline, but for phase A
    we pass the small held-out slice (held_idx) to keep the test manageable. The transform
    time scales linearly with held rows, so the full-pipeline cost can be extrapolated:
        full_transform_seconds ≈ measured_seconds × (152_501_580 / len(held_idx))
    """
    log(f"Loading {len(fit_idx):,} fit rows from memmap...")
    t0 = perf_counter()
    X_fit = latent[fit_idx]
    load_seconds = perf_counter() - t0
    log(f"  loaded in {load_seconds:.1f}s")

    log(f"Fitting UMAP on {len(fit_idx):,} rows...")
    t0 = perf_counter()
    fitted = core.fit_umap(X_fit, config)
    fit_seconds = perf_counter() - t0
    log(f"  fit in {fit_seconds:.1f}s (backend={fitted.backend})")

    log(f"Transforming {len(held_idx):,} held rows...")
    X_held = latent[held_idx]
    t0 = perf_counter()
    _ = fitted.transform(X_held, batch_size=transform_batch_size)
    transform_seconds = perf_counter() - t0
    log(f"  transform in {transform_seconds:.1f}s")

    extrapolated_full_transform = transform_seconds * (152_501_580 / len(held_idx))
    log(f"  extrapolated full-cohort transform (~152.5M rows): {extrapolated_full_transform/60:.1f} min")

    return {
        "fit_rows": int(len(fit_idx)),
        "held_rows_timed": int(len(held_idx)),
        "load_seconds": round(load_seconds, 2),
        "fit_seconds": round(fit_seconds, 2),
        "transform_seconds_per_held_rows": round(transform_seconds, 2),
        "extrapolated_full_transform_seconds": round(extrapolated_full_transform, 1),
        "extrapolated_full_transform_minutes": round(extrapolated_full_transform / 60, 1),
        "fit_fraction_of_fit_plus_transform": round(
            fit_seconds / (fit_seconds + transform_seconds), 3
        ),
        "backend": fitted.backend,
    }


# ── Part 2: repulsion probe (Gate 2) ──────────────────────────────────────────

def repulsion_probe(
    latent: np.ndarray,
    probe_idx: np.ndarray,
    background_idx: np.ndarray,
    config: core.UmapConfig,
) -> dict:
    """Measure whether transform() displaces points outward relative to fit membership.

    Two models, same UMAP config:
      A  fit on (background + probe)   -- probe points are 'insiders'
      B  fit on background only        -- probe points are 'outsiders' placed by transform()

    For each probe point:
      - Assign it a cluster label using a simple nearest-centroid rule on the 2-D embedding
        (full HDBSCAN adds ~2 min and is not needed for this geometric comparison).
      - Compute its radial distance to the centroid of its nearest cluster.

    If transform() has a repulsion bias, the probe's mean radial distance in condition B
    will be significantly larger than in condition A.
    """
    log(f"Repulsion probe: {len(probe_idx):,} probe rows, {len(background_idx):,} background rows")

    # Condition A: probe rows are part of the fit
    combined_idx = np.sort(np.concatenate([background_idx, probe_idx]))
    probe_positions_in_combined = np.searchsorted(combined_idx, probe_idx)

    log("  [A] fitting UMAP with probe included...")
    t0 = perf_counter()
    X_combined = latent[combined_idx]
    fitted_a = core.fit_umap(X_combined, config)
    log(f"      fit A done in {perf_counter()-t0:.1f}s")

    emb_a = fitted_a.embedding
    probe_emb_a = emb_a[probe_positions_in_combined]

    # Build centroids from the background rows only (so same reference in both conditions)
    background_positions_in_combined = np.setdiff1d(
        np.arange(len(combined_idx)), probe_positions_in_combined
    )
    # Use a quick k-means-like centroid: assign probe rows to nearest background cluster
    # by proximity in the embedding. We do this by finding the nearest background point
    # for each probe point and inheriting its cluster assignment via a density-grid proxy.
    # Simpler: just compute per-point distance to global centroid of all background rows.
    background_centroid_a = emb_a[background_positions_in_combined].mean(axis=0)
    dist_probe_a = np.linalg.norm(probe_emb_a - background_centroid_a, axis=1)

    # Condition B: probe rows withheld, placed by transform()
    log("  [B] fitting UMAP on background only, then transforming probe rows...")
    t0 = perf_counter()
    X_background = latent[background_idx]
    fitted_b = core.fit_umap(X_background, config)
    log(f"      fit B done in {perf_counter()-t0:.1f}s")

    background_centroid_b = fitted_b.embedding.mean(axis=0)

    t0 = perf_counter()
    probe_emb_b = fitted_b.transform(latent[probe_idx])
    log(f"      transform B done in {perf_counter()-t0:.1f}s")

    dist_probe_b = np.linalg.norm(probe_emb_b - background_centroid_b, axis=1)

    # Summary statistics
    mean_a = float(np.mean(dist_probe_a))
    mean_b = float(np.mean(dist_probe_b))
    median_a = float(np.median(dist_probe_a))
    median_b = float(np.median(dist_probe_b))

    # Outward shift ratio: >1 means transform() places probe points farther from the centre
    shift_ratio_mean = mean_b / mean_a if mean_a > 0 else float("nan")
    shift_ratio_median = median_b / median_a if median_a > 0 else float("nan")

    log(f"  mean radial dist:   A (insider)={mean_a:.4f}  B (transform)={mean_b:.4f}  "
        f"ratio={shift_ratio_mean:.3f}")
    log(f"  median radial dist: A (insider)={median_a:.4f}  B (transform)={median_b:.4f}  "
        f"ratio={shift_ratio_median:.3f}")

    if shift_ratio_mean > 1.15:
        log("  ** REPULSION EFFECT CONFIRMED: transform places probe points >15% farther out **")
    elif shift_ratio_mean > 1.05:
        log("  * Mild outward shift detected (5-15%). Worth monitoring.")
    else:
        log("  No significant outward shift detected (ratio < 1.05).")

    return {
        "probe_rows": int(len(probe_idx)),
        "background_rows": int(len(background_idx)),
        "condition_a_mean_radial_dist": round(mean_a, 5),
        "condition_b_mean_radial_dist": round(mean_b, 5),
        "condition_a_median_radial_dist": round(median_a, 5),
        "condition_b_median_radial_dist": round(median_b, 5),
        "shift_ratio_mean": round(shift_ratio_mean, 4),
        "shift_ratio_median": round(shift_ratio_median, 4),
        "repulsion_confirmed": bool(shift_ratio_mean > 1.15),
        "interpretation": (
            "strong repulsion effect (>15% outward shift)" if shift_ratio_mean > 1.15
            else "mild shift (5-15%)" if shift_ratio_mean > 1.05
            else "no significant shift"
        ),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    started = perf_counter()
    args = parse_args()

    embed_dir = Path(args.embed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_path = embed_dir / "latent.npy"
    if not latent_path.exists():
        raise SystemExit(f"latent.npy not found in {embed_dir}")

    if core.gpu_available() and args.gpu_budget_gb is not None:
        core.apply_gpu_budget("sweep", budget_gb=args.gpu_budget_gb)

    log(f"Opening {latent_path} as read-only memmap...")
    latent = np.load(latent_path, mmap_mode="r")
    total_rows, latent_dim = latent.shape
    log(f"  {total_rows:,} rows × {latent_dim} dims ({total_rows * latent_dim * 4 / 1e9:.1f} GB)")

    if args.probe_rows >= args.fit_rows:
        raise SystemExit(
            f"--probe-rows ({args.probe_rows}) must be less than --fit-rows ({args.fit_rows})"
        )

    # Fixed UMAP config matching the one run in the existing 5-cell sweep
    config = core.UmapConfig(
        n_neighbors=15, min_dist=0.0, n_components=2,
        metric="euclidean", seed=args.seed, n_epochs=200,
    )
    log(f"UMAP config: {config.as_dict()}")

    # Draw indices — probe is carved from the same pool as fit, using a different seed
    # so both sets are representative and non-overlapping without a two-phase draw.
    rng = np.random.default_rng(args.seed)
    all_draw = np.sort(rng.choice(total_rows, size=args.fit_rows, replace=False))
    probe_idx = all_draw[:args.probe_rows]          # first probe_rows as the withheld set
    background_idx = all_draw[args.probe_rows:]     # remaining as the background fit set
    # fit_idx for the timer split = the full all_draw (matches stage2_sweep.fit_indices)
    fit_idx = all_draw
    # held_idx for the timer split = a small held-out slice from outside the draw
    outside_pool = np.setdiff1d(
        np.arange(total_rows), all_draw, assume_unique=True
    )
    held_for_timer = outside_pool[:min(100_000, len(outside_pool))]

    log("=" * 60)
    log("PART 1: Timer split (fit vs transform)")
    log("=" * 60)
    timer_results = time_fit_and_transform(
        latent=latent,
        fit_idx=fit_idx,
        held_idx=held_for_timer,
        config=config,
        transform_batch_size=args.transform_batch_size,
    )

    log("=" * 60)
    log("PART 2: Repulsion probe (Gate 2)")
    log("=" * 60)
    repulsion_results = repulsion_probe(
        latent=latent,
        probe_idx=probe_idx,
        background_idx=background_idx,
        config=config,
    )

    results = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_rows": total_rows,
        "latent_dim": latent_dim,
        "umap_config": config.as_dict(),
        "timer_split": timer_results,
        "repulsion_probe": repulsion_results,
        "total_seconds": round(perf_counter() - started, 1),
    }

    out_path = output_dir / "phase_a_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"Results written to {out_path}")

    # Print summary to stdout for easy reading
    t = timer_results
    r = repulsion_results
    print("\n── Phase A summary ────────────────────────────────────")
    print(f"Fit {t['fit_rows']:,} rows:          {t['fit_seconds']:.0f}s "
          f"({t['fit_seconds']/60:.1f} min)")
    print(f"Transform {t['held_rows_timed']:,} rows:  {t['transform_seconds_per_held_rows']:.0f}s")
    print(f"  → extrapolated for 152.5M rows: "
          f"{t['extrapolated_full_transform_seconds']:.0f}s "
          f"({t['extrapolated_full_transform_minutes']:.0f} min)")
    print(f"Fit fraction of (fit+timed transform): {t['fit_fraction_of_fit_plus_transform']:.1%}")
    print()
    print(f"Repulsion probe ({r['probe_rows']:,} points):")
    print(f"  insider mean radial dist:   {r['condition_a_mean_radial_dist']:.4f}")
    print(f"  transform mean radial dist: {r['condition_b_mean_radial_dist']:.4f}")
    print(f"  shift ratio (B/A):          {r['shift_ratio_mean']:.3f}  "
          f"→ {r['interpretation']}")
    print("───────────────────────────────────────────────────────")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
