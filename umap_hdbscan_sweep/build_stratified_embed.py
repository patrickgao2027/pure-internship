#!/usr/bin/env python
"""Build a stratified 2D embedding with equal representation from every source parquet.

The existing enriched.parquet was drawn proportionally (files with more reads contributed
more rows), so the 2 M sample is dominated by a handful of high-coverage files.  This
script fixes that by sampling exactly --rows-per-file rows from each of the 95 source
parquets, then embedding them through the already-fitted models so no retraining is needed:

    source parquets  →  DuckDB sample  →  VAE encoder  →  parametric UMAP  →  output

UMAP is the parametric MLP encoder from parametric_umap.py (a forward pass: very fast,
batch-invariant, no kNN lookup).  No HDBSCAN step -- the output carries umap_1, umap_2,
source_file, and all VAE feature columns.  That is all the contact sheet needs; cluster
labels can be added later by re-fitting HDBSCAN on the stratified coordinates.

Timing on miletus (RTX PRO 5000, /data/lab/ppmseq_parquets/ NFS):
  - DuckDB reservoir sampling 95 × 100 K rows:   ~30–60 min  (I/O bottleneck)
  - VAE encode 9.5 M rows on GPU:                ~10–20 min
  - Parametric UMAP forward pass 9.5 M rows:     ~1–3 min    (just an MLP, no kNN)
  Total estimate:  ~45–90 min

Use --parallel-files to overlap DuckDB I/O across multiple files (default 4).

Usage::

    python umap_hdbscan_sweep/build_stratified_embed.py \\
        --source-glob '/data/lab/ppmseq_parquets/*.parquet' \\
        --vae-checkpoint ~/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/training/run_20260802T192814Z/model.pt \\
        --umap-model     ~/pure-internship/umap_hdbscan_sweep/umap/results/final_models/13_BEST_25M_nn15_md0.1_umap.pt \\
        --output         ~/pure-internship/umap_hdbscan_sweep/hdbscan/stratified_9M.parquet \\
        --rows-per-file 100000 --seed 42
"""
from __future__ import annotations

import argparse
import glob as globmod
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

import numpy as np
import polars as pl
import torch

from uv_vae.data import connect_duckdb
from uv_vae.inference import LatentInference

DEFAULT_ROW_FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-glob", required=True,
                   help="glob for the 95 source parquets")
    p.add_argument("--vae-checkpoint", type=Path, required=True,
                   help="model.pt from any cohort cell run")
    p.add_argument("--umap-model", type=Path, required=True,
                   help="parametric UMAP .pt file, e.g. 13_BEST_25M_nn15_md0.1_umap.pt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows-per-file", type=int, default=100_000)
    p.add_argument("--row-filter", default=DEFAULT_ROW_FILTER)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4096,
                   help="VAE encode batch size (default 4096)")
    p.add_argument("--umap-batch-size", type=int, default=2_000_000,
                   help="parametric UMAP forward-pass batch size (default 2M)")
    p.add_argument("--parallel-files", type=int, default=4,
                   help="files to read concurrently via DuckDB (default 4)")
    p.add_argument("--duckdb-threads", type=int, default=8)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--feature-spec-path", type=Path, default=None)
    return p.parse_args()


# ── DuckDB sampling ────────────────────────────────────────────────────────────

_local = threading.local()


def sample_file(conn, filepath: str, rows: int, row_filter: str,
                seed: int, feature_names: list[str]) -> pl.DataFrame:
    if not hasattr(_local, "cursor"):
        _local.cursor = conn.cursor()
    cursor = _local.cursor
    col_list = ", ".join(f'"{c}"' for c in feature_names)
    return cursor.execute(f"""
        SELECT {col_list}
        FROM (
            SELECT {col_list}
            FROM read_parquet('{filepath}')
            WHERE {row_filter}
        )
        USING SAMPLE {rows} ROWS REPEATABLE ({seed})
    """).pl()


# ── parametric UMAP loader ─────────────────────────────────────────────────────

class _ParametricEncoder(torch.nn.Module):
    """Minimal copy of ParametricEncoder from parametric_umap.py.

    Defined here to avoid importing parametric_umap.py, which pulls in sweep_core
    and aumap at module level — both are training-only dependencies not needed for
    a plain forward-pass inference.  Architecture must stay in sync with the original.
    """
    def __init__(self, input_dim: int, output_dim: int = 2,
                 hidden: tuple = (256, 256, 128)) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for w in hidden:
            layers += [torch.nn.Linear(prev, w), torch.nn.ReLU(inplace=True)]
            prev = w
        layers.append(torch.nn.Linear(prev, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_parametric_umap(path: Path, device: torch.device):
    """Load encoder weights and return a transform(X, batch_size) callable."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state   = payload["state_dict"]
    first_weight = next(v for k, v in state.items() if k.endswith(".weight") and v.ndim == 2)
    input_dim = first_weight.shape[1]

    encoder = _ParametricEncoder(input_dim=input_dim)
    encoder.load_state_dict(state)
    encoder.to(device)
    encoder.eval()

    @torch.no_grad()
    def transform(X: np.ndarray, batch_size: int = 2_000_000) -> np.ndarray:
        out = []
        for i in range(0, len(X), batch_size):
            t = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            out.append(encoder(t).cpu().numpy())
        return np.concatenate(out, axis=0)

    return transform


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    t0 = perf_counter()

    source_files = sorted(globmod.glob(args.source_glob))
    if not source_files:
        raise SystemExit(f"no files match {args.source_glob}")
    log(f"{len(source_files)} source parquets × {args.rows_per_file:,} rows each "
        f"= {len(source_files) * args.rows_per_file:,} total rows")

    # ── load VAE ───────────────────────────────────────────────────────────────
    inferrer = LatentInference.from_checkpoint(
        checkpoint_path=args.vae_checkpoint,
        feature_spec_path=str(args.feature_spec_path) if args.feature_spec_path else None,
        device=args.device,
    )
    feature_names = inferrer.feature_names
    device = inferrer.device
    log(f"VAE: {inferrer.latent_dim}-dim latent, {len(feature_names)} features, device={device}")

    # ── load parametric UMAP ───────────────────────────────────────────────────
    umap_model = load_parametric_umap(args.umap_model, device)
    log(f"parametric UMAP loaded from {args.umap_model.name}")

    # ── phase 1: sample + VAE encode ──────────────────────────────────────────
    log(f"\n=== phase 1: sampling & encoding ({args.parallel_files} files in parallel) ===")

    # index-keyed so we can reconstruct in file order regardless of completion order
    results: dict[int, tuple[str, np.ndarray]] = {}

    with connect_duckdb(threads=args.duckdb_threads) as conn:
        def process(idx_fp):
            idx, fp = idx_fp
            stem  = Path(fp).stem
            frame = sample_file(conn, fp, args.rows_per_file,
                                args.row_filter, args.seed + idx, feature_names)
            if frame.height == 0:
                return idx, stem, None
            latents = inferrer.encode_frame(frame, batch_size=args.batch_size)
            return idx, stem, latents

        with ThreadPoolExecutor(max_workers=min(args.parallel_files, len(source_files))) as pool:
            futures = {pool.submit(process, (i, fp)): i
                       for i, fp in enumerate(source_files)}
            done = 0
            for fut in as_completed(futures):
                idx, stem, latents = fut.result()
                done += 1
                if latents is None:
                    log(f"  [{done:>3}/{len(source_files)}] {stem}: 0 rows after filter, skipping")
                    continue
                results[idx] = (stem, latents)
                log(f"  [{done:>3}/{len(source_files)}] {stem}: "
                    f"{latents.shape[0]:,} rows  ({perf_counter()-t0:.0f}s)")

    # flatten in source-file order
    ordered     = [(stem, lat) for _, (stem, lat) in sorted(results.items())]
    stems       = [s for s, _ in ordered]
    latents_lst = [l for _, l in ordered]
    latent_mat  = np.concatenate(latents_lst, axis=0)
    source_col  = np.repeat(stems, [l.shape[0] for l in latents_lst])

    log(f"\nphase 1 done: {latent_mat.shape[0]:,} rows encoded  ({perf_counter()-t0:.0f}s)")

    # ── phase 2: parametric UMAP (forward pass only — fast) ───────────────────
    log(f"\n=== phase 2: parametric UMAP transform ({latent_mat.shape[0]:,} points) ===")
    umap_t0 = perf_counter()
    coords  = umap_model(latent_mat, batch_size=args.umap_batch_size)
    log(f"UMAP done  ({perf_counter()-umap_t0:.0f}s)")

    # ── write ──────────────────────────────────────────────────────────────────
    log(f"\n=== writing {args.output} ===")
    result = pl.DataFrame({
        "umap_1":      coords[:, 0].astype(np.float32),
        "umap_2":      coords[:, 1].astype(np.float32),
        "source_file": source_col,
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(args.output, compression="zstd")

    mb = args.output.stat().st_size / 1e6
    log(f"wrote {result.height:,} rows × {result.width} columns  "
        f"({mb:.0f} MB)  total {perf_counter()-t0:.0f}s")

    counts = result["source_file"].value_counts().sort("count", descending=True)
    log(f"\nper-sample row counts: min={counts['count'].min():,}  "
        f"max={counts['count'].max():,}  mean={int(counts['count'].mean()):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
