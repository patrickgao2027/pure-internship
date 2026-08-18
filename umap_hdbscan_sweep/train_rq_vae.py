"""Train one VAE for sweep 2 of the clustering regime -- the rq filter sweep.

Sweeps 1/3/4 reuse the pretrained sweep_results_full ladder. Sweep 2 cannot: filtering on
rq changes the input distribution, so a VAE trained on the unfiltered data would be
encoding rows it was never fit to. This trains a fresh VAE per rq threshold on exactly
the rows that pass ``<base filter> AND rq < <threshold>``.

Hyperparameters mirror scripts/vae_subsample_sweep.py, which produced the ladder, so the
rq VAEs are comparable to the 100pct reference. The only intended difference is the rows.

Rows are FILTERED first, then sampled -- training clamps sample_rows to the number of
rows that actually pass the filter, so the default --max-sample-rows simply means
"train on everything that passes".

  python scripts/train_rq_vae.py \
      --parquet-path /path/to/featuremap.parquet \
      --output-root /path/to/rq_vaes \
      --rq-threshold 0.05

Writes the checkpoint to <output-root>/rq_<threshold>/run_<timestamp>/model.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uv_vae.data import connect_duckdb, get_row_count
from uv_vae.training import DEFAULT_TRAINING_ROW_FILTER, TrainingConfig, train


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one rq-filtered VAE for regime sweep 2")
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--output-root", required=True,
                        help="parent dir; the run lands in <output-root>/rq_<threshold>/")
    parser.add_argument("--rq-threshold", type=float, required=True)
    parser.add_argument("--row-filter", default=DEFAULT_TRAINING_ROW_FILTER,
                        help="base filter; 'AND rq < <threshold>' is appended")
    parser.add_argument("--feature-spec-path", default="ml_features.json")
    # Ladder hyperparameters -- keep these aligned with vae_subsample_sweep.py.
    parser.add_argument("--max-sample-rows", default="all",
                        help="'all' (default) counts the eligible rows and asks for exactly "
                             "that many; an integer caps it lower")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dims", default="256,128")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    row_filter = f"{args.row_filter} AND rq < {args.rq_threshold}"
    cell_dir = Path(args.output_root).resolve() / f"rq_{args.rq_threshold}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(cell_dir.glob("run_*/model.pt"))
    if existing:
        log(f"Checkpoint already exists, skipping training: {existing[-1]}")
        print(existing[-1])
        return 0

    log(f"Training rq VAE: threshold={args.rq_threshold} filter={row_filter!r}")

    # Ask for exactly the number of rows that pass the filter. train() would clamp a
    # too-large request down anyway (training.py: sample_rows = min(sample_rows,
    # total_rows)), but counting first means the request is never the binding constraint
    # and the number is logged rather than inferred after the fact.
    with connect_duckdb(args.threads) as conn:
        eligible_rows = get_row_count(conn, args.parquet_path, where=row_filter)
    if eligible_rows == 0:
        raise SystemExit(f"No rows pass the filter: {row_filter}")

    if str(args.max_sample_rows).strip().lower() == "all":
        requested_rows = eligible_rows
        log(f"Requesting ALL {eligible_rows:,} eligible rows")
    else:
        requested_rows = min(int(args.max_sample_rows), eligible_rows)
        log(f"Requesting {requested_rows:,} of {eligible_rows:,} eligible rows"
            + (" (capped by --max-sample-rows)" if requested_rows < eligible_rows else ""))

    config = TrainingConfig(
        parquet_path=args.parquet_path,
        feature_spec_path=args.feature_spec_path,
        output_dir=str(cell_dir),
        row_filter=row_filter,
        sample_rows=requested_rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dims=[int(part) for part in args.hidden_dims.split(",") if part.strip()],
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        train_fraction=args.train_fraction,
        seed=args.seed,
        threads=args.threads,
        data_seed=args.data_seed,
    )

    started = perf_counter()
    run_dir = train(config)
    elapsed = perf_counter() - started

    summary = json.loads((run_dir / "summary.json").read_text())
    trained_rows = summary.get("sample_rows")
    eligible = summary.get("eligible_rows_in_parquet")
    log(f"Trained on {trained_rows:,} rows of {eligible:,} eligible "
        f"in {elapsed:.1f}s -> {run_dir / 'model.pt'}")
    if trained_rows is not None and eligible is not None and trained_rows < eligible:
        log(f"WARNING: trained on fewer rows than are eligible "
            f"({trained_rows:,} < {eligible:,}) -- check --max-sample-rows")

    # A stable, greppable record of what this threshold cost in rows -- the rq sweep
    # confounds filter stringency with training-set size, so both get reported.
    (cell_dir / "rq_train_summary.json").write_text(json.dumps({
        "rq_threshold": args.rq_threshold,
        "row_filter": row_filter,
        "run_dir": str(run_dir),
        "checkpoint_path": str(run_dir / "model.pt"),
        "requested_rows": requested_rows,
        "training_rows": trained_rows,
        "eligible_rows": eligible,
        "seconds": round(elapsed, 1),
        "seed": args.seed,
    }, indent=2))

    print(run_dir / "model.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
