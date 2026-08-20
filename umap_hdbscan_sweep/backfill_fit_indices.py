#!/usr/bin/env python
"""Write fit_indices.npy for cells that were fit before the sweep started saving it.

The fit indices are a deterministic draw -- ``np.sort(default_rng(seed).choice(total_rows,
fit_rows, replace=False))``, the same expression ``hdbscan_param_sweep.run_cell`` uses. A
cell missing the file does not need refitting; it needs the draw reproduced. Refitting to
recover a reconstructible array would cost hours and, worse, produce a *different* model,
since nothing guarantees a new fit reproduces the old one bit for bit.

Reconstruction is only safe if it can be checked, and here it can be, because the fit rows
keep the labels the fit gave them:

    cohort_labels[fit_indices] == clusterer.labels_

That has to hold exactly for the true index set. A wrong draw of the right length disagrees
on a large fraction of rows immediately, so this is a real test rather than a plausibility
argument -- it is the same check ``verify_saved_models.py`` runs, applied before writing
instead of after.

Nothing is written unless the check passes. A cell whose indices cannot be reproduced is
reported and skipped, because a confidently wrong fit_indices.npy is worse than an absent
one: absent, downstream scripts warn and re-derive; wrong, they silently label every read
against the wrong neighbours.

Usage::

    python umap_hdbscan_sweep/backfill_fit_indices.py \\
        --cells-root ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/param_sweep_refit_cpu/cells

    # see what it would do
    python umap_hdbscan_sweep/backfill_fit_indices.py --cells-root <...> --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

DEFAULT_COORDS = (Path.home() / "pure-internship" / "umap_hdbscan_sweep" / "hdbscan"
                  / "results" / "hdbscan_scaling" / "coords.npy")


def log(message: str) -> None:
    stream = sys.stdout
    text = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except UnicodeEncodeError:
        text = text.encode(encoding, errors="replace").decode(encoding)
    print(text, flush=True)


def _to_host(array) -> np.ndarray:
    for attribute in ("get", "to_numpy"):
        method = getattr(array, attribute, None)
        if callable(method):
            return np.asarray(method())
    return np.asarray(array)


def derive(total_rows: int, fit_rows: int, seed: int) -> np.ndarray:
    """The exact expression hdbscan_param_sweep.run_cell uses to pick its fit rows."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_rows, size=fit_rows, replace=False))


def backfill_cell(cell_dir: Path, total_rows: int, args) -> dict:
    import joblib

    result = {"cell": cell_dir.name, "dir": str(cell_dir)}

    target = cell_dir / "fit_indices.npy"
    if target.exists() and not args.overwrite:
        result["status"] = "skipped (already present)"
        return result

    metrics_path = cell_dir / "metrics.json"
    labels_path = cell_dir / "cohort_labels.npy"
    model_path = cell_dir / "hdbscan_model.pkl"
    for path in (metrics_path, labels_path):
        if not path.exists():
            result["status"] = f"skipped (no {path.name})"
            return result

    metrics = json.loads(metrics_path.read_text())
    fit_rows = int(metrics.get("cell", {}).get("fit_rows", 0))
    if not fit_rows:
        result["status"] = "skipped (metrics.json records no fit_rows)"
        return result
    result["fit_rows"] = fit_rows
    result["backend"] = metrics.get("cluster_backend", "?")

    # The fit labels come from the model when it is there, and from metrics.json's recorded
    # cluster count only as a shape hint otherwise. Without the model there is nothing to
    # check the draw against, so an unverifiable cell is skipped rather than guessed at.
    if not model_path.exists():
        result["status"] = "skipped (no hdbscan_model.pkl to verify against)"
        return result
    try:
        clusterer = joblib.load(model_path)
        fit_labels = _to_host(clusterer.labels_).ravel().astype(np.int32)
    except Exception as exc:  # noqa: BLE001
        result["status"] = f"skipped (model unloadable: {type(exc).__name__}: {exc})"
        return result

    if len(fit_labels) != fit_rows:
        result["status"] = (f"FAILED: model was fit on {len(fit_labels):,} rows but "
                            f"metrics.json says {fit_rows:,}")
        return result

    cohort_labels = np.load(labels_path, mmap_mode="r")
    if cohort_labels.shape[0] != total_rows:
        result["status"] = (f"FAILED: cohort_labels has {cohort_labels.shape[0]:,} rows, "
                            f"coords has {total_rows:,}")
        return result

    for seed in args.seeds:
        indices = derive(total_rows, fit_rows, seed)
        recorded = np.asarray(cohort_labels[indices])
        differing = int((recorded != fit_labels).sum())
        if differing == 0:
            result.update({"seed": seed, "status": "ok" if not args.dry_run else "would write"})
            if not args.dry_run:
                np.save(target, indices.astype(np.int64))
                result["written"] = str(target)
            return result
        result.setdefault("attempts", []).append(
            {"seed": seed, "differing": differing,
             "pct": round(differing / fit_rows * 100, 3)})

    worst = result.get("attempts", [])
    result["status"] = ("FAILED: no candidate seed reproduces the fit labels ("
                        + ", ".join(f"seed {a['seed']}: {a['pct']}% differ" for a in worst)
                        + ")")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells-root", required=True, type=Path,
                        help="directory containing one subdir per cell")
    parser.add_argument("--coords", default=None,
                        help=f"cohort coords.npy, for the row count (default: {DEFAULT_COORDS})")
    parser.add_argument("--seeds", default="42",
                        help="comma-separated candidate seeds to try, in order")
    parser.add_argument("--overwrite", action="store_true",
                        help="rewrite fit_indices.npy where it already exists")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]

    coords_path = (args.coords or "").strip() or os.environ.get("COORDS", "").strip()
    coords_path = os.path.expanduser(coords_path or str(DEFAULT_COORDS))
    if not os.path.exists(coords_path):
        log(f"ERROR: coords not found: {coords_path}")
        return 1
    total_rows = int(np.load(coords_path, mmap_mode="r").shape[0])
    log(f"coords: {coords_path}  {total_rows:,} rows")

    if not args.cells_root.is_dir():
        log(f"ERROR: no such cells root: {args.cells_root}")
        return 1
    cell_dirs = sorted(p for p in args.cells_root.iterdir() if p.is_dir())
    log(f"{len(cell_dirs)} cell dirs under {args.cells_root}"
        + ("  (dry run)" if args.dry_run else ""))

    results = []
    for cell_dir in cell_dirs:
        result = backfill_cell(cell_dir, total_rows, args)
        results.append(result)
        seed_note = f"  seed={result['seed']}" if "seed" in result else ""
        log(f"  {result['cell']:<34} {result['status']}{seed_note}")

    wrote = [r for r in results if r["status"] in ("ok", "would write")]
    failed = [r for r in results if r["status"].startswith("FAILED")]
    log("")
    log(f"{len(wrote)} reproduced and verified, {len(failed)} failed, "
        f"{len(results) - len(wrote) - len(failed)} skipped")
    if failed:
        log("Failed cells keep no fit_indices.npy. Downstream scripts will warn and "
            "re-derive; do not hand-write one.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
