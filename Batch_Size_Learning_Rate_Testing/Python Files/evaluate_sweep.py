"""Post-training evaluation for a hyperparameter sweep.

Loads each checkpoint produced by run_batch_lr_sweep_gpu.sh and computes the full
latent-metrics report (activation, entanglement, structure, reconstruction).
Writes per-config JSON reports and a summary CSV.

Usage:
    python "Batch_Size_Learning_Rate_Testing/Python Files/evaluate_sweep.py" \
        --sweep-root /path/to/sweep_<jobid> \
        --parquet-path /path/to/combined.parquet \
        --feature-spec-path ml_features.json \
        --eval-rows 10000

Or evaluate a single checkpoint:
    python "Batch_Size_Learning_Rate_Testing/Python Files/evaluate_sweep.py" \
        --checkpoint /path/to/run_dir/model.pt \
        --parquet-path /path/to/combined.parquet \
        --feature-spec-path ml_features.json \
        --eval-rows 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2] / "uv_vae"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

if torch.cuda.is_available():
    try:
        import cuml.accel
        cuml.accel.install()
    except ImportError:
        pass


def find_checkpoints(sweep_root: Path) -> list[tuple[str, Path]]:
    """Find all model.pt checkpoints under a sweep root directory."""
    results = []
    for model_pt in sorted(sweep_root.rglob("model.pt")):
        run_dir = model_pt.parent
        config_dir = run_dir.parent
        config_name = config_dir.name
        if config_name == sweep_root.name:
            config_name = run_dir.name
        results.append((config_name, model_pt))
    return results


def evaluate_one(
    config_name: str,
    checkpoint_path: Path,
    parquet_path: str,
    feature_spec_path: str,
    row_filter: str,
    eval_rows: int,
    seed: int,
    threads: int | None,
    output_dir: Path | None,
) -> dict:
    """Evaluate one checkpoint and optionally write a JSON report."""
    from uv_vae.latent_metrics import evaluate_checkpoint

    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"Evaluating: {config_name}", file=sys.stderr, flush=True)
    print(f"  checkpoint: {checkpoint_path}", file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    report = evaluate_checkpoint(
        checkpoint_path=str(checkpoint_path),
        parquet_path=parquet_path,
        feature_spec_path=feature_spec_path,
        row_filter=row_filter,
        sample_rows=eval_rows,
        seed=seed,
        threads=threads,
    )
    elapsed = time.perf_counter() - t0
    report["config_name"] = config_name
    report["eval_seconds"] = round(elapsed, 2)

    training_report_path = checkpoint_path.parent / "training_report.json"
    if training_report_path.exists():
        tr = json.loads(training_report_path.read_text())
        history = tr.get("history", [])
        if history:
            report["training_summary"] = {
                "epochs_run": len(history),
                "best_val_loss": round(min(e["val_total_loss"] for e in history), 6),
                "final_val_loss": round(history[-1]["val_total_loss"], 6),
                "final_train_loss": round(history[-1]["train_total_loss"], 6),
                "au_epoch1": history[0].get("active_units", "?"),
                "au_final": history[-1].get("active_units", "?"),
            }
        config = tr.get("config", {})
        report["hyperparams"] = {
            "batch_size": config.get("batch_size"),
            "learning_rate": config.get("learning_rate"),
            "kl_weight": config.get("kl_weight"),
            "latent_dim": config.get("latent_dim"),
            "hidden_dims": config.get("hidden_dims"),
        }
        report["dropout"] = {
            "input_dropout": tr.get("input_dropout"),
            "hidden_dropout": tr.get("hidden_dropout"),
        }
        early = tr.get("early_stopping", {})
        report["early_stopping"] = {
            "stopped_early": early.get("stopped_early"),
            "best_epoch": early.get("best_epoch"),
        }

    if output_dir:
        out_path = output_dir / f"{config_name}_eval.json"
        out_path.write_text(json.dumps(report, indent=2))
        print(f"  wrote: {out_path}", file=sys.stderr, flush=True)

    print(
        f"  done in {elapsed:.1f}s — "
        f"au={report['activation']['active_units']}, "
        f"eff_dim={report['activation']['effective_dimensionality']}, "
        f"trust_k10={report['structure']['trustworthiness_k10']:.4f}, "
        f"mse={report['reconstruction']['numeric_mse_total']:.6f}",
        file=sys.stderr,
        flush=True,
    )
    return report


def write_summary_csv(reports: list[dict], output_path: Path) -> None:
    """Write a flat CSV with one row per config and key metrics as columns."""
    fieldnames = [
        "config_name",
        "epochs_run",
        "batch_size",
        "learning_rate",
        "best_val_loss",
        "final_val_loss",
        "au_epoch1",
        "au_final",
        "effective_dim",
        "off_diag_cov_mean",
        "off_diag_cov_max",
        "kl_total",
        "numeric_mse",
        "trust_k10",
        "trust_k50",
        "eval_rows",
        "eval_seconds",
    ]

    rows = []
    for r in reports:
        act = r.get("activation", {})
        struct = r.get("structure", {})
        recon = r.get("reconstruction", {})
        ts = r.get("training_summary", {})
        hp = r.get("hyperparams", {})

        rows.append({
            "config_name": r.get("config_name", "?"),
            "epochs_run": ts.get("epochs_run", "?"),
            "batch_size": hp.get("batch_size", "?"),
            "learning_rate": hp.get("learning_rate", "?"),
            "best_val_loss": ts.get("best_val_loss", "?"),
            "final_val_loss": ts.get("final_val_loss", "?"),
            "au_epoch1": ts.get("au_epoch1", "?"),
            "au_final": ts.get("au_final", "?"),
            "effective_dim": act.get("effective_dimensionality", "?"),
            "off_diag_cov_mean": act.get("off_diagonal_cov_mean", "?"),
            "off_diag_cov_max": act.get("off_diagonal_cov_max", "?"),
            "kl_total": act.get("kl_total", "?"),
            "numeric_mse": recon.get("numeric_mse_total", "?"),
            "trust_k10": struct.get("trustworthiness_k10", "?"),
            "trust_k50": struct.get("trustworthiness_k50", "?"),
            "eval_rows": r.get("sample_rows", "?"),
            "eval_seconds": r.get("eval_seconds", "?"),
        })

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\nSummary CSV: {output_path}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate sweep checkpoints")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sweep-root", help="Root directory of a sweep (evaluates all checkpoints)")
    group.add_argument("--checkpoint", help="Single checkpoint path to evaluate")

    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--feature-spec-path", required=True)
    parser.add_argument("--row-filter", default="st = 'MIXED' AND et = 'MIXED' AND FILT = 1")
    parser.add_argument("--eval-rows", type=int, default=10000,
                        help="Rows to sample for evaluation (10K default; trades accuracy for speed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output-dir", default=None,
                        help="Directory for per-config JSON reports (default: sweep-root or checkpoint parent)")
    args = parser.parse_args()

    if args.sweep_root:
        sweep_root = Path(args.sweep_root)
        if not sweep_root.exists():
            print(f"ERROR: sweep root not found: {sweep_root}", file=sys.stderr)
            return 1
        checkpoints = find_checkpoints(sweep_root)
        if not checkpoints:
            print(f"ERROR: no model.pt found under {sweep_root}", file=sys.stderr)
            return 1
        output_dir = Path(args.output_dir) if args.output_dir else sweep_root
        print(f"Found {len(checkpoints)} checkpoints under {sweep_root}", file=sys.stderr, flush=True)
    else:
        cp = Path(args.checkpoint)
        if not cp.exists():
            print(f"ERROR: checkpoint not found: {cp}", file=sys.stderr)
            return 1
        checkpoints = [(cp.parent.name, cp)]
        output_dir = Path(args.output_dir) if args.output_dir else cp.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for config_name, checkpoint_path in checkpoints:
        try:
            report = evaluate_one(
                config_name=config_name,
                checkpoint_path=checkpoint_path,
                parquet_path=args.parquet_path,
                feature_spec_path=args.feature_spec_path,
                row_filter=args.row_filter,
                eval_rows=args.eval_rows,
                seed=args.seed,
                threads=args.threads,
                output_dir=output_dir,
            )
            reports.append(report)
        except Exception as exc:
            print(f"  ERROR evaluating {config_name}: {exc}", file=sys.stderr, flush=True)
            reports.append({"config_name": config_name, "error": str(exc)})

    if reports:
        write_summary_csv(reports, output_dir / "sweep_eval_summary.csv")

    print("\n" + "=" * 90)
    header = (
        f"{'config':<28}  {'au':>4}  {'eff_d':>6}  "
        f"{'off_cov':>8}  {'t_k10':>6}  {'t_k50':>6}  {'mse':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        if "error" in r:
            print(f"{r['config_name']:<28}  ERROR: {r['error']}")
            continue
        act = r.get("activation", {})
        struct = r.get("structure", {})
        recon = r.get("reconstruction", {})
        print(
            f"{r['config_name']:<28}  "
            f"{act.get('active_units', '?'):>4}  "
            f"{act.get('effective_dimensionality', 0):>6.2f}  "
            f"{act.get('off_diagonal_cov_mean', 0):>8.6f}  "
            f"{struct.get('trustworthiness_k10', 0):>6.4f}  "
            f"{struct.get('trustworthiness_k50', 0):>6.4f}  "
            f"{recon.get('numeric_mse_total', 0):>8.6f}"
        )
    print("=" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
