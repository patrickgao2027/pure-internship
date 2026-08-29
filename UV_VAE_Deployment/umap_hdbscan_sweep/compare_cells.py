#!/usr/bin/env python
"""Is a re-fitted sweep cell the same clustering as the original?

The reason this is not just ``np.array_equal``: HDBSCAN's cluster **ids** carry no meaning.
Two runs can partition all 157.5M rows identically and still disagree on every label, because
ids are assigned in condensed-tree traversal order and any reordering renumbers them. So this
reports two different things and they answer different questions:

``identical``      byte-for-byte equal labels. The strong claim -- the run is bit-reproducible.
``ARI``            agreement on the PARTITION, ignoring ids. 1.0 with ``identical`` false means
                   the same clustering was found and merely numbered differently, which is a
                   pass. Below 1.0 means the runs genuinely disagree about which rows group
                   together, which is not.

Noise is the one label that *is* meaningful (-1 is not a cluster id), so it is also reported
separately: two runs can score a high ARI while disagreeing about what counts as noise.

What determinism to expect. The fit row draw is ``default_rng(seed).choice`` over the cohort,
so it is exactly reproducible given the same ``--seed``, ``--coords`` and ``fit_rows``. The
HDBSCAN fit and the RBC labelling both run on the GPU, where reductions are not order-stable
and bitwise equality is not guaranteed. Treat ARI 1.0 with a handful of differing rows as a
pass; treat a materially lower ARI as a real difference worth chasing.

Usage::

    python umap_hdbscan_sweep/compare_cells.py \\
        --a umap_tests/param_sweep/cells --b umap_tests/param_sweep_refit/cells --all

    # one cell, exact comparison over every row
    python umap_hdbscan_sweep/compare_cells.py \\
        --a <old>/cells/fit5000000_mcs250_ms5_eom \\
        --b <new>/cells/fit5000000_mcs250_ms5_eom --ari-rows 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHUNK = 20_000_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=Path, required=True, help="original cell dir, or a cells root")
    p.add_argument("--b", type=Path, required=True, help="re-fitted counterpart")
    p.add_argument("--all", action="store_true",
                   help="treat --a/--b as cells roots and compare every cell in both")
    p.add_argument("--ari-rows", type=int, default=5_000_000,
                   help="rows sampled for ARI (0 = every row; O(n) memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None, help="write the report as JSON")
    return p.parse_args()


def compare_labels(path_a: Path, path_b: Path, ari_rows: int, seed: int) -> dict:
    a = np.load(path_a, mmap_mode="r")
    b = np.load(path_b, mmap_mode="r")
    if a.shape != b.shape:
        return {"error": f"shape {a.shape} vs {b.shape}"}

    total = a.shape[0]
    # Chunked so a 157.5M-row pair never lands in memory at once just to be compared.
    differing = 0
    noise_a = noise_b = noise_agree = 0
    for start in range(0, total, CHUNK):
        stop = min(start + CHUNK, total)
        block_a = np.asarray(a[start:stop])
        block_b = np.asarray(b[start:stop])
        differing += int((block_a != block_b).sum())
        is_noise_a = block_a < 0
        is_noise_b = block_b < 0
        noise_a += int(is_noise_a.sum())
        noise_b += int(is_noise_b.sum())
        noise_agree += int((is_noise_a == is_noise_b).sum())

    report = {
        "rows": int(total),
        "identical": differing == 0,
        "differing_rows": differing,
        "differing_fraction": differing / total,
        "noise_fraction_a": noise_a / total,
        "noise_fraction_b": noise_b / total,
        "noise_agreement": noise_agree / total,
        "clusters_a": int(np.asarray(a[:]).max()) + 1 if total else 0,
        "clusters_b": int(np.asarray(b[:]).max()) + 1 if total else 0,
    }

    if ari_rows and ari_rows < total:
        index = np.sort(np.random.default_rng(seed).choice(total, ari_rows, replace=False))
        sample_a, sample_b = np.asarray(a[index]), np.asarray(b[index])
        report["ari_rows"] = int(ari_rows)
    else:
        sample_a, sample_b = np.asarray(a[:]), np.asarray(b[:])
        report["ari_rows"] = int(total)

    from sklearn.metrics import adjusted_rand_score

    report["adjusted_rand"] = float(adjusted_rand_score(sample_a, sample_b))
    return report


def compare_probabilities(dir_a: Path, dir_b: Path) -> dict | None:
    path_a, path_b = dir_a / "cohort_probabilities.npy", dir_b / "cohort_probabilities.npy"
    if not (path_a.exists() and path_b.exists()):
        return None
    a = np.load(path_a, mmap_mode="r")
    b = np.load(path_b, mmap_mode="r")
    if a.shape != b.shape:
        return {"error": f"shape {a.shape} vs {b.shape}"}
    worst = 0.0
    for start in range(0, a.shape[0], CHUNK):
        stop = min(start + CHUNK, a.shape[0])
        worst = max(worst, float(np.abs(np.asarray(a[start:stop])
                                        - np.asarray(b[start:stop])).max()))
    return {"max_abs_difference": worst, "identical": worst == 0.0}


def verdict(report: dict) -> str:
    if "error" in report:
        return "ERROR"
    if report["identical"]:
        return "IDENTICAL"
    if report["adjusted_rand"] >= 0.9999:
        return "SAME PARTITION (renumbered)"
    if report["adjusted_rand"] >= 0.99:
        return "NEAR-IDENTICAL"
    return "DIFFERENT"


def main() -> int:
    args = parse_args()

    if args.all:
        names = sorted({d.name for d in args.a.iterdir() if (d / "cohort_labels.npy").exists()}
                       & {d.name for d in args.b.iterdir() if (d / "cohort_labels.npy").exists()})
        pairs = [(name, args.a / name, args.b / name) for name in names]
        if not pairs:
            raise SystemExit("no cells with cohort_labels.npy in both roots")
    else:
        pairs = [(args.a.name, args.a, args.b)]

    results: list[dict] = []
    for name, dir_a, dir_b in pairs:
        report = compare_labels(dir_a / "cohort_labels.npy", dir_b / "cohort_labels.npy",
                                args.ari_rows, args.seed)
        report["cell"] = name
        report["verdict"] = verdict(report)
        probabilities = compare_probabilities(dir_a, dir_b)
        if probabilities is not None:
            report["probabilities"] = probabilities
        results.append(report)

        if "error" in report:
            print(f"{name:42} ERROR: {report['error']}")
            continue
        print(f"{name:42} {report['verdict']:28} "
              f"ARI {report['adjusted_rand']:.6f}  "
              f"diff {report['differing_fraction'] * 100:.4f}%  "
              f"clusters {report['clusters_a']}/{report['clusters_b']}  "
              f"noise {report['noise_fraction_a'] * 100:.2f}/"
              f"{report['noise_fraction_b'] * 100:.2f}%"
              + (f"  |dp|max {report['probabilities']['max_abs_difference']:.2e}"
                 if probabilities and "error" not in probabilities else ""))

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nreport -> {args.output}")

    return 0 if all(r.get("verdict") in {"IDENTICAL", "SAME PARTITION (renumbered)"}
                    for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
