"""
Compute Procrustes and Linear CKA for every epoch sweep checkpoint
against the full 52M reference model.

Directory structure expected:

epoch_sweep/
    rows_750000_epochs_10/
        sweep_results.json
        run_100pct/
            run_xxxxx/
                model.pt
    rows_750000_epochs_20/
    ...
    rows_26000000_epochs_50/

Usage:

export UV_VAE_ROOT=~/uv_vae
cd ~/uv_vae

python VAE_Stability_Testing/scripts/epoch_sweep_vs_52M.py \
    --test-set-path /cta/users/patrickgao765/uv_vae/test_set.parquet \
    --ref-json VAE_Stability_Testing/sweep_results_full/sweep_results.json \
    --epoch-sweep-dir VAE_Stability_Testing/epoch_sweep \
    --output-json VAE_Stability_Testing/epoch_sweep/vs_52M_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import procrustes

UV_VAE_ROOT = Path(
    os.environ.get("UV_VAE_ROOT", Path.home() / "uv_vae")
).resolve()

if str(UV_VAE_ROOT) not in sys.path:
    sys.path.insert(0, str(UV_VAE_ROOT))

from uv_vae.inference import LatentInference


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    numerator = np.linalg.norm(X.T @ Y, "fro") ** 2
    denom_x = np.linalg.norm(X.T @ X, "fro")
    denom_y = np.linalg.norm(Y.T @ Y, "fro")

    if denom_x < 1e-10 or denom_y < 1e-10:
        return 0.0

    return float(numerator / (denom_x * denom_y))


def encode(checkpoint_path, test_df, batch_size=4096):
    inference = LatentInference.from_checkpoint(
        str(checkpoint_path),
        device="auto",
    )

    embedding = inference.encode_frame(
        test_df.select(inference.feature_names),
        batch_size=batch_size,
    )

    return inference, embedding


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--test-set-path", required=True)

    parser.add_argument(
        "--ref-json",
        required=True,
        help="sweep_results.json from the full 52M sweep",
    )

    parser.add_argument(
        "--epoch-sweep-dir",
        required=True,
        help="Root directory containing rows_*_epochs_* folders",
    )

    parser.add_argument("--output-json", required=True)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    args = parser.parse_args()

    ####################################################################
    # Load test set
    ####################################################################

    log("Loading test set")
    test_df = pl.read_parquet(args.test_set_path)

    ####################################################################
    # Find 52M reference checkpoint
    ####################################################################

    ref_json = json.loads(Path(args.ref_json).read_text())

    ref_entry = max(ref_json, key=lambda x: x["sample_rows"])

    ref_dir = Path(ref_entry["run_dir"])
    ref_checkpoint = ref_dir / "model.pt"

    if not ref_checkpoint.exists():

        ref_base = Path(args.ref_json).resolve().parent

        for depth in (2, 3):

            parts = ref_dir.parts

            if len(parts) >= depth:

                candidate = (
                    ref_base
                    / Path(*parts[-depth:])
                    / "model.pt"
                )

                if candidate.exists():
                    ref_checkpoint = candidate
                    break

    if not ref_checkpoint.exists():
        raise FileNotFoundError(
            f"Reference checkpoint not found:\n{ref_dir}"
        )

    ####################################################################
    # Encode reference
    ####################################################################

    log(
        f"Encoding reference model ({ref_entry['sample_rows']:,} rows)"
    )

    _, ref_embedding = encode(
        ref_checkpoint,
        test_df,
        args.batch_size,
    )

    log(f"Reference embedding shape: {ref_embedding.shape}")

    ####################################################################
    # Iterate through epoch sweep
    ####################################################################

    sweep_root = Path(args.epoch_sweep_dir)

    results = []

    experiment_dirs = sorted(
        sweep_root.glob("rows_*_epochs_*")
    )

    log(f"Found {len(experiment_dirs)} experiments")

    for experiment_dir in experiment_dirs:

        parts = experiment_dir.name.split("_")

        rows = int(parts[1])
        epochs = int(parts[3])

        sweep_json = experiment_dir / "sweep_results.json"

        if not sweep_json.exists():
            log(
                f"SKIP rows={rows:,} epochs={epochs}: "
                "missing sweep_results.json"
            )
            continue

        sweep = json.loads(sweep_json.read_text())

        if len(sweep) == 0:
            log(
                f"SKIP rows={rows:,} epochs={epochs}: "
                "empty sweep_results.json"
            )
            continue

        entry = sweep[0]

        ###############################################################
        # Locate checkpoint
        ###############################################################

        run_dir = Path(entry["run_dir"])

        checkpoint = run_dir / "model.pt"

        if not checkpoint.exists():

            checkpoint = (
                experiment_dir
                / "run_100pct"
                / run_dir.name
                / "model.pt"
            )

        if not checkpoint.exists():

            candidates = list(experiment_dir.rglob("model.pt"))

            if candidates:
                checkpoint = candidates[0]

        if not checkpoint.exists():
            log(
                f"SKIP rows={rows:,} epochs={epochs}: "
                "checkpoint not found"
            )
            continue

        ###############################################################
        # Encode checkpoint
        ###############################################################

        log(
            f"Encoding rows={rows:,} epochs={epochs}"
        )

        _, embedding = encode(
            checkpoint,
            test_df,
            args.batch_size,
        )

        ###############################################################
        # Compute metrics
        ###############################################################

        _, _, disparity = procrustes(
            ref_embedding,
            embedding,
        )

        cka = linear_cka(
            ref_embedding,
            embedding,
        )

        results.append(
            {
                "rows": rows,
                "epochs": epochs,
                "procrustes_disparity": round(float(disparity), 6),
                "linear_cka": round(float(cka), 6),
                "trustworthiness": entry.get("trustworthiness"),
                "latent_collapse": entry.get("latent_collapse"),
                "val_total_loss": entry.get("val_total_loss"),
                "val_numeric_loss": entry.get("val_numeric_loss"),
                "val_kl_loss": entry.get("val_kl_loss"),
            }
        )

        log(
            f"    Procrustes={disparity:.6f}    "
            f"CKA={cka:.6f}"
        )

    ####################################################################
    # Save
    ####################################################################

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(results, indent=2)
    )

    log(f"Saved {len(results)} results to {output_path}")

    ####################################################################
    # Summary
    ####################################################################

    print("\n" + "═" * 90)
    print(" Epoch Sweep vs Full 52M Reference")
    print("═" * 90)

    print(
        f"{'Rows':>12}  "
        f"{'Epochs':>6}  "
        f"{'Procrustes':>12}  "
        f"{'CKA':>8}  "
        f"{'Val Loss':>10}"
    )

    print("─" * 90)

    for r in sorted(
        results,
        key=lambda x: (x["rows"], x["epochs"]),
    ):

        val_loss = (
            f"{r['val_total_loss']:.4f}"
            if r["val_total_loss"] is not None
            else "n/a"
        )

        print(
            f"{r['rows']:>12,}  "
            f"{r['epochs']:>6}  "
            f"{r['procrustes_disparity']:>12.6f}  "
            f"{r['linear_cka']:>8.4f}  "
            f"{val_loss:>10}"
        )

    print("─" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())