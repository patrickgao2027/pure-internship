#!/usr/bin/env python
"""Stratified sample of the stage 0 dedup set, embedded through the fitted VAE + UMAP.

The 2 M rows in enriched.parquet were drawn proportionally, so files contributing more
loci to the dedup set dominate it and a per-sample picture reads mostly as "which sample
is biggest". This draws --rows-per-file from *each* source sample instead, giving every
one of the 95 equal weight, then pushes the result through the already-fitted models:

    dedup set + source labels  ->  stratified sample  ->  VAE  ->  parametric UMAP

Everything is drawn from the **stage 0 dedup output** -- one row per (CHROM, POS, REF,
ALT) across the whole cohort, the same ~157 M-row population the VAE was trained on and
the same one enriched.parquet samples. Sampling the raw source parquets instead would
draw from ~5.08 B reads at ~32 per locus, a different population entirely.

The dedup output does not record which sample each surviving read came from, so this
needs the ``CHROM, POS, REF, ALT -> source_file`` map that ``label_dedup_source.py``
writes. Run that once first; the map is reusable.

Sampling is deterministic without depending on thread count: rows are ranked within each
source_file by a hash of their identity columns and the seed, and the top --rows-per-file
are taken. DuckDB's reservoir sampling is only reproducible single-threaded, and a
per-group reservoir is awkward to express anyway; hashing gives a uniform draw that is
identical on every re-run regardless of --duckdb-threads.

Timing on miletus, after the labels exist:
  - sample + join dedup for features:  ~15-30 min
  - VAE encode 9.5 M rows on GPU:      ~10-20 min
  - parametric UMAP forward pass:      ~1-3 min   (an MLP, no kNN lookup)

Usage::

    python umap_hdbscan_sweep/build_stratified_embed.py \\
        --dedup-glob  '~/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/stage0_dedup/dedup/*.parquet' \\
        --labels-glob '~/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/stage0_dedup/source_labels/*.parquet' \\
        --vae-checkpoint ~/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/training/run_20260802T192814Z/model.pt \\
        --umap-model     ~/pure-internship/umap_hdbscan_sweep/umap/results/final_models/13_BEST_25M_nn15_md0.1_umap.pt \\
        --output         ~/pure-internship/umap_hdbscan_sweep/hdbscan/stratified_9M.parquet \\
        --rows-per-file 100000 --seed 42
"""
from __future__ import annotations

import argparse
import glob as globmod
import sys
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


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dedup-glob", required=True,
                   help="stage 0 output, e.g. '<run>/stage0_dedup/dedup/*.parquet'")
    p.add_argument("--labels-glob", required=True,
                   help="output of label_dedup_source.py (CHROM/POS/REF/ALT -> source_file)")
    p.add_argument("--vae-checkpoint", type=Path, required=True)
    p.add_argument("--umap-model", type=Path, required=True,
                   help="parametric UMAP .pt, e.g. 13_BEST_25M_nn15_md0.1_umap.pt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows-per-file", type=int, default=100_000,
                   help="rows drawn per source sample; samples with fewer take all they have")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4096,
                   help="VAE encode batch size")
    p.add_argument("--umap-batch-size", type=int, default=2_000_000,
                   help="parametric UMAP forward-pass batch size")
    p.add_argument("--duckdb-threads", type=int, default=16)
    p.add_argument("--memory-limit", default="200GB")
    p.add_argument("--temp-directory", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--feature-spec-path", type=Path, default=None)
    return p.parse_args()


# ── parametric UMAP ────────────────────────────────────────────────────────────

class _ParametricEncoder(torch.nn.Module):
    """Minimal copy of ParametricEncoder from parametric_umap.py.

    Defined here rather than imported because parametric_umap.py pulls in sweep_core and
    aumap at module level -- graph extraction and UMAP-loss machinery that a forward pass
    never touches. The architecture must stay in sync with the original; the weights
    loaded below are the trained ones, nothing is refit.
    """

    def __init__(self, input_dim: int, output_dim: int = 2,
                 hidden: tuple = (256, 256, 128)) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for width in hidden:
            layers += [torch.nn.Linear(prev, width), torch.nn.ReLU(inplace=True)]
            prev = width
        layers.append(torch.nn.Linear(prev, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_parametric_umap(path: Path, device):
    """Load trained encoder weights; return a transform(X, batch_size) callable."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    first_weight = next(v for k, v in state.items() if k.endswith(".weight") and v.ndim == 2)

    encoder = _ParametricEncoder(input_dim=first_weight.shape[1])
    encoder.load_state_dict(state)
    encoder.to(device)
    encoder.eval()

    @torch.no_grad()
    def transform(X: np.ndarray, batch_size: int = 2_000_000) -> np.ndarray:
        chunks = []
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start + batch_size]).float().to(device)
            chunks.append(encoder(batch).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    return transform


# ── sampling ───────────────────────────────────────────────────────────────────

def draw_stratified(conn, dedup_glob: str, labels_glob: str,
                    feature_names: list[str], rows_per_file: int, seed: int) -> pl.DataFrame:
    """Equal-count sample per source_file, joined to the dedup rows for VAE features."""
    dedup_files  = sorted(globmod.glob(str(Path(dedup_glob).expanduser())))
    labels_files = sorted(globmod.glob(str(Path(labels_glob).expanduser())))
    if not dedup_files:
        raise SystemExit(f"no files match {dedup_glob}")
    if not labels_files:
        raise SystemExit(f"no files match {labels_glob} -- run label_dedup_source.py first")

    dedup_list  = "[" + ", ".join(sql_quote(p) for p in dedup_files) + "]"
    labels_list = "[" + ", ".join(sql_quote(p) for p in labels_files) + "]"
    feature_list = ", ".join(f'd."{c}"' for c in feature_names)

    # ROW_NUMBER over a hash of (identity, seed) rather than USING SAMPLE: DuckDB's
    # reservoir sampling is only reproducible at threads=1 (see CLAUDE.md), and it has no
    # per-group form. Hashing the identity columns gives a uniform, thread-independent
    # order, so the same seed selects the same rows on every re-run at any thread count.
    return conn.execute(f"""
        WITH picked AS (
            SELECT "CHROM", "POS", "REF", "ALT", source_file
            FROM read_parquet({labels_list})
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY source_file
                ORDER BY hash(concat_ws('|', "CHROM", "POS", "REF", "ALT", {int(seed)}))
            ) <= {int(rows_per_file)}
        )
        SELECT p.source_file, {feature_list}
        FROM picked AS p
        JOIN read_parquet({dedup_list}) AS d
          ON d."CHROM" = p."CHROM" AND d."POS" = p."POS"
         AND d."REF"   = p."REF"   AND d."ALT" = p."ALT"
    """).pl()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    started = perf_counter()

    inferrer = LatentInference.from_checkpoint(
        checkpoint_path=args.vae_checkpoint,
        feature_spec_path=str(args.feature_spec_path) if args.feature_spec_path else None,
        device=args.device,
    )
    log(f"VAE: {inferrer.latent_dim}-dim latent, {len(inferrer.feature_names)} features, "
        f"device={inferrer.device}")

    umap_transform = load_parametric_umap(args.umap_model, inferrer.device)
    log(f"parametric UMAP loaded from {args.umap_model.name}")

    # ── sample ────────────────────────────────────────────────────────────────
    log(f"\n=== drawing {args.rows_per_file:,} rows per source sample from the dedup set ===")
    temp_directory = Path(args.temp_directory) if args.temp_directory else None
    if temp_directory:
        temp_directory.mkdir(parents=True, exist_ok=True)

    with connect_duckdb(threads=args.duckdb_threads, temp_directory=temp_directory,
                        memory_limit=args.memory_limit) as conn:
        frame = draw_stratified(conn, args.dedup_glob, args.labels_glob,
                                inferrer.feature_names, args.rows_per_file, args.seed)

    if frame.height == 0:
        raise SystemExit("sample is empty -- check that the labels and dedup globs match")

    counts = frame["source_file"].value_counts().sort("count", descending=True)
    short = counts.filter(pl.col("count") < args.rows_per_file)
    log(f"sampled {frame.height:,} rows from {counts.height} samples "
        f"({perf_counter() - started:.0f}s)")
    log(f"  per-sample: min={counts['count'].min():,}  max={counts['count'].max():,}")
    if short.height:
        log(f"  {short.height} sample(s) had fewer than {args.rows_per_file:,} dedup rows "
            f"and contributed all of them")

    # ── encode ────────────────────────────────────────────────────────────────
    log(f"\n=== VAE encode ({frame.height:,} rows) ===")
    encode_started = perf_counter()
    source_col = frame["source_file"].to_numpy()
    latents = inferrer.encode_frame(frame.select(inferrer.feature_names),
                                    batch_size=args.batch_size)
    log(f"encoded to {latents.shape[1]}-dim latent ({perf_counter() - encode_started:.0f}s)")

    # ── parametric UMAP ───────────────────────────────────────────────────────
    log(f"\n=== parametric UMAP transform ===")
    umap_started = perf_counter()
    coords = umap_transform(latents, batch_size=args.umap_batch_size)
    log(f"UMAP done ({perf_counter() - umap_started:.0f}s)")

    # ── write ─────────────────────────────────────────────────────────────────
    result = pl.DataFrame({
        "umap_1":      coords[:, 0].astype(np.float32),
        "umap_2":      coords[:, 1].astype(np.float32),
        "source_file": source_col,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(args.output, compression="zstd")
    log(f"\nwrote {args.output}  ({result.height:,} rows, "
        f"{args.output.stat().st_size / 1e6:.0f} MB, {perf_counter() - started:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
