#!/usr/bin/env python
"""Build a stratified 2D embedding with equal representation from every source parquet.

The existing enriched.parquet was drawn proportionally (files with more reads contributed
more rows), so the 2 M sample is dominated by a handful of high-coverage files.  This
script fixes that by sampling exactly --rows-per-file rows from each of the 95 source
parquets, then embedding them through the already-fitted models so no retraining is needed:

    source parquets  →  DuckDB sample  →  VAE encoder  →  UMAP transform
                                                        →  HDBSCAN predict
                                                        →  output parquet

The output carries: umap_1, umap_2, cluster_label, cluster_probability, source_file (file
stem, e.g. "WS7031"), and every feature column the VAE was trained on.

Timing on miletus (RTX PRO 5000 + /data/lab/ppmseq_parquets/ NFS):
  - DuckDB reservoir sampling, 95 × 100 K rows:  ~30–60 min  (I/O bottleneck)
  - VAE encode 9.5 M rows on GPU:                ~10–20 min
  - UMAP transform 9.5 M rows:
        cuML active  →  ~5–10 min
        CPU umap-learn  →  60–120 min
  - HDBSCAN approximate_predict:                 ~5 min
  Total estimate:  60–120 min with cuML; 2–4 h CPU-only.

Use --parallel-files to overlap DuckDB I/O across multiple files (default 4).

Usage::

    python umap_hdbscan_sweep/build_stratified_embed.py \\
        --source-glob '/data/lab/ppmseq_parquets/*.parquet' \\
        --run-dir    ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/cohort_reports_original/fit25000000_mcs250_ms5_eom \\
        --output     ~/pure-internship/umap_hdbscan_sweep/hdbscan/stratified_9M.parquet \\
        --rows-per-file 100000 --seed 42
"""
from __future__ import annotations

import argparse
import glob as globmod
import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for _c in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import hdbscan as hdbscan_lib
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from uv_vae.data import connect_duckdb
from uv_vae.inference import LatentInference

DEFAULT_ROW_FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-glob", required=True,
                   help="glob for the 95 source parquets, e.g. '/data/lab/ppmseq_parquets/*.parquet'")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="a cohort run directory containing model.pt, umap_model.pkl, "
                        "hdbscan_clusterer.pkl  (typically one of the 28 cell dirs)")
    p.add_argument("--output", type=Path, required=True,
                   help="output parquet path, e.g. stratified_9M.parquet")
    p.add_argument("--rows-per-file", type=int, default=100_000,
                   help="rows to sample from each source parquet (default 100 000)")
    p.add_argument("--row-filter", default=DEFAULT_ROW_FILTER,
                   help="DuckDB WHERE clause applied before sampling")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4096,
                   help="VAE encode batch size (default 4096)")
    p.add_argument("--parallel-files", type=int, default=4,
                   help="files to read concurrently via DuckDB (default 4)")
    p.add_argument("--duckdb-threads", type=int, default=8,
                   help="DuckDB threads per connection (default 8)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--feature-spec-path", type=Path, default=None,
                   help="override; normally read from model.pt's embedded training_config")
    return p.parse_args()


# ── DuckDB sampling ────────────────────────────────────────────────────────────

_local = threading.local()


def sample_file(conn, filepath: str, rows: int, row_filter: str, seed: int,
                feature_names: list[str]) -> pl.DataFrame:
    """Reservoir-sample `rows` filtered rows from one parquet file."""
    if not hasattr(_local, "cursor"):
        _local.cursor = conn.cursor()
    cursor = _local.cursor

    # Pull only the feature columns we need (avoids transferring all 70 columns
    # when only the VAE features are used for encoding).
    col_list = ", ".join(f'"{c}"' for c in feature_names)
    result = cursor.execute(f"""
        SELECT {col_list}
        FROM (
            SELECT {col_list}
            FROM read_parquet('{filepath}')
            WHERE {row_filter}
        )
        USING SAMPLE {rows} ROWS REPEATABLE ({seed})
    """).pl()
    return result


# ── model loading ──────────────────────────────────────────────────────────────

def load_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def find_model_files(run_dir: Path) -> tuple[Path, Path, Path]:
    """Locate model.pt, umap_model.pkl, hdbscan_clusterer.pkl under run_dir."""
    def find(name: str) -> Path:
        matches = sorted(run_dir.rglob(name))
        if not matches:
            raise FileNotFoundError(f"{name} not found under {run_dir}")
        return matches[0]

    return find("model.pt"), find("umap_model.pkl"), find("hdbscan_clusterer.pkl")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    t0 = perf_counter()

    # ── discover source files ──────────────────────────────────────────────────
    source_files = sorted(globmod.glob(args.source_glob))
    if not source_files:
        raise SystemExit(f"no files match {args.source_glob}")
    log(f"{len(source_files)} source parquets  ×  {args.rows_per_file:,} rows each  "
        f"=  {len(source_files) * args.rows_per_file:,} total rows")

    # ── load VAE ───────────────────────────────────────────────────────────────
    model_pt, umap_pkl, hdbscan_pkl = find_model_files(args.run_dir)
    log(f"model.pt         -> {model_pt}")
    log(f"umap_model.pkl   -> {umap_pkl}")
    log(f"hdbscan_model    -> {hdbscan_pkl}")

    inferrer = LatentInference.from_checkpoint(
        checkpoint_path=model_pt,
        feature_spec_path=str(args.feature_spec_path) if args.feature_spec_path else None,
        device=args.device,
    )
    feature_names = inferrer.feature_names
    log(f"VAE loaded: {inferrer.latent_dim}-dim latent, {len(feature_names)} features, "
        f"device={inferrer.device}")

    umap_model   = load_pkl(umap_pkl)
    hdbscan_clust = load_pkl(hdbscan_pkl)
    log("UMAP + HDBSCAN models loaded")

    # ── phase 1: sample + encode ───────────────────────────────────────────────
    log(f"\n=== phase 1: sampling & encoding ({args.parallel_files} files in parallel) ===")
    all_latents: list[np.ndarray] = [None] * len(source_files)  # preserve order
    all_source_names: list[str]   = []
    all_feature_frames: list[pl.DataFrame] = []

    with connect_duckdb(threads=args.duckdb_threads) as conn:
        def process_file(idx_filepath):
            idx, filepath = idx_filepath
            stem = Path(filepath).stem
            frame = sample_file(conn, filepath, args.rows_per_file,
                                args.row_filter, args.seed + idx, feature_names)
            if frame.height == 0:
                log(f"  [{idx+1:>3}/{len(source_files)}] {stem}: 0 rows after filter, skipping")
                return idx, stem, None, None
            latents = inferrer.encode_frame(frame, batch_size=args.batch_size)
            return idx, stem, latents, frame

        parallel = min(args.parallel_files, len(source_files))
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(process_file, (i, fp)): i
                       for i, fp in enumerate(source_files)}
            done = 0
            for fut in as_completed(futures):
                idx, stem, latents, frame = fut.result()
                done += 1
                if latents is None:
                    continue
                all_latents[idx] = latents
                # store stem repeated for every row — we'll concat after
                log(f"  [{done:>3}/{len(source_files)}] {stem}: "
                    f"{latents.shape[0]:,} rows  ({perf_counter()-t0:.0f}s)")
                # save per-file results keyed by order index so we can concat in order
                all_latents[idx] = (stem, latents, frame)

    # flatten in index order, skipping Nones
    ordered: list[tuple[str, np.ndarray, pl.DataFrame]] = [
        x for x in all_latents if x is not None
    ]
    stems   = [s   for s, _, _  in ordered]
    latents = [la  for _, la, _ in ordered]
    frames  = [fr  for _, _, fr in ordered]

    latent_matrix = np.concatenate(latents, axis=0)
    source_col    = np.repeat(stems, [la.shape[0] for la in latents])
    feature_frame = pl.concat(frames, how="diagonal_relaxed")

    log(f"\nphase 1 done: {latent_matrix.shape[0]:,} rows encoded  ({perf_counter()-t0:.0f}s)")

    # ── phase 2: UMAP transform ────────────────────────────────────────────────
    log(f"\n=== phase 2: UMAP transform ({latent_matrix.shape[0]:,} points) ===")
    umap_t0 = perf_counter()
    coords = umap_model.transform(latent_matrix).astype(np.float32)
    log(f"UMAP transform done  ({perf_counter()-umap_t0:.0f}s)")

    # ── phase 3: HDBSCAN predict ───────────────────────────────────────────────
    log(f"\n=== phase 3: HDBSCAN approximate_predict ===")
    hdb_t0 = perf_counter()
    labels, strengths = hdbscan_lib.approximate_predict(hdbscan_clust, coords)
    labels = labels.astype(np.int32)
    strengths = strengths.astype(np.float32)
    n_noise = int((labels == -1).sum())
    log(f"HDBSCAN done: {int((labels >= 0).sum()):,} clustered, {n_noise:,} noise  "
        f"({perf_counter()-hdb_t0:.0f}s)")

    # ── assemble + write ───────────────────────────────────────────────────────
    log(f"\n=== writing output -> {args.output} ===")
    result = pl.concat([
        pl.DataFrame({
            "umap_1":              coords[:, 0],
            "umap_2":              coords[:, 1],
            "cluster_label":       labels,
            "cluster_probability": strengths,
            "source_file":         source_col,
        }),
        feature_frame,
    ], how="horizontal")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(args.output, compression="zstd")

    mb = args.output.stat().st_size / 1e6
    log(f"wrote {result.height:,} rows × {result.width} columns  "
        f"({mb:.0f} MB)  total {perf_counter()-t0:.0f}s")

    # per-sample row count summary
    counts = result["source_file"].value_counts().sort("count", descending=True)
    log(f"\nper-sample row counts (min={counts['count'].min():,}  "
        f"max={counts['count'].max():,}  mean={int(counts['count'].mean()):,}):")
    log(counts.to_pandas().to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
