"""How HDBSCAN fit cost, GPU memory and cluster structure scale with fit-set size.

`phase_a_hdbscan_timing.py` decomposed the cell at one fit size (5 M): fit 457 s,
``approximate_predict`` 3031 s (87%), parquet write 7 s. It left the more consequential
question untouched -- **what happens when the fit set grows**. There is exactly one
measured point (5 M, safe) and one catastrophic failure (70 M at ``min_samples=25``:
56 GB requested on a 47 GB card, OOM after 26 h). Nothing in between. This script fills
that gap for 500 K -> 50 M.

Three things are measured per size, and the third is the one nobody has data on:

    1. fit seconds          how the MST construction actually scales
    2. peak GPU memory      where the ceiling is, measured rather than extrapolated
                            from the single 70 M data point
    3. predict rate         s per million rows. The 19.9 s/M measured at 5 M was against
                            a 5 M reference set; ``approximate_predict`` searches the fit
                            set, so the rate is expected to grow with it. Whether it grows
                            linearly, logarithmically or not at all decides whether a
                            larger fit set is affordable at labelling time.

**Coordinates are precomputed, and that is not incidental.** The timing run capped the
process at 40 GB and `gpu_budget` split it ``torch 20.0 GB + rmm pool 20.0 GB``, because
the encoder held torch on the card throughout. cuML therefore had 20 GB, not 40 -- roughly
a 25 M-row ceiling. Running ``--embed-only`` first writes ``coords.npy`` and exits, so the
sweep process never imports torch and ``STAGE_RMM_SHARE["sweep"] = 0.9`` gives RMM the
budget it was always supposed to have. This alone is what puts 50 M within reach.

**min_cluster_size is proportional** (``N x 1e-4``, floored at 50): 50 / 100 / 500 / 1000 /
1500 / 2500 / 5000. Holding it constant in absolute terms would let granularity drift with
N -- mcs=500 is 0.1% of a 500 K fit and 0.001% of a 50 M one -- and the cross-size label
comparison this script exists to enable would then be confounded by the drift rather than
measuring it. Fit time is insensitive to mcs (457.1 / 458.5 / 457.8 s at mcs 100 / 500 /
2500), so nothing in the timing result is traded away for this.

**What gets saved, and why.** Every size writes its fit-set labels *and* labels for a fixed
2 M evaluation probe that no fit set contains. The probe is the same rows every time, which
is what makes ARI/AMI between fit sizes computable later -- comparing two clusterings needs
them to have labelled the same points. Without it the saved labels describe different row
sets and cannot be compared at all. Because the proportional ``min_cluster_size`` keeps the
smallest admissible cluster at a constant 0.01% of the fit set, the probe covers even that
cluster with ~200 rows at *every* size, so coverage is comparable across the sweep rather
than degrading at one end.

A ``draw_signature.json`` pins (seed, probe sizes, total rows) and the run refuses to
continue into a directory drawn with different ones -- resume would otherwise leave finished
cells on the old probe and new cells on the new one, which no downstream metric could detect.

    python umap_hdbscan_sweep/hdbscan_scaling_sweep.py --embed-dir <stage1> \\
        --encoder-model <final_models/13_BEST_...pt> --output-dir <out> --embed-only
    python umap_hdbscan_sweep/hdbscan_scaling_sweep.py --embed-dir <stage1> \\
        --output-dir <out>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np

import sweep_core as core

TOTAL_COHORT_ROWS = 157_501_580

# N x 1e-4, floored. See module docstring for why this is proportional and not absolute.
MCS_FRACTION = 1e-4
MCS_FLOOR = 50


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def min_cluster_size_for(fit_rows: int) -> int:
    return max(MCS_FLOOR, int(round(fit_rows * MCS_FRACTION)))


# ── GPU memory ─────────────────────────────────────────────────────────────────

class GpuMemoryProbe:
    """Peak *device* memory over a window, sampled from a background thread.

    ``torch.cuda.max_memory_allocated`` is useless here: it reports only torch's own
    allocator, and cuML allocates through RMM, which torch cannot see. The 70 M OOM was
    invisible by that measure right up until the driver refused the allocation. Polling
    what the driver reports catches every allocator on the card, which is the only number
    that predicts an OOM.

    Sampling means the true peak can be missed if it lives for less than one interval.
    That biases the result *low*, so a size reported as fitting comfortably may be closer
    to the edge than it looks -- treat these as lower bounds on the real peak.
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.peak_used_gb = 0.0
        self.baseline_used_gb = 0.0
        self.samples = 0
        self.backend = "none"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml = None
        self._handle = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.backend = "pynvml"
            return
        except Exception:
            pass
        try:
            self._read_nvidia_smi()
            self.backend = "nvidia-smi"
        except Exception:
            self.backend = "none"

    @staticmethod
    def _read_nvidia_smi() -> float:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        return float(out.strip().splitlines()[0]) / 1024.0

    def read_used_gb(self) -> float | None:
        try:
            if self.backend == "pynvml":
                info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                return info.used / (1024 ** 3)
            if self.backend == "nvidia-smi":
                return self._read_nvidia_smi()
        except Exception:
            return None
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            used = self.read_used_gb()
            if used is not None:
                self.peak_used_gb = max(self.peak_used_gb, used)
                self.samples += 1
            self._stop.wait(self.interval)

    def start(self) -> None:
        baseline = self.read_used_gb()
        self.baseline_used_gb = baseline if baseline is not None else 0.0
        self.peak_used_gb = self.baseline_used_gb
        self.samples = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return {
            "backend": self.backend,
            "baseline_used_gb": round(self.baseline_used_gb, 3),
            "peak_used_gb": round(self.peak_used_gb, 3),
            "delta_gb": round(self.peak_used_gb - self.baseline_used_gb, 3),
            "samples": self.samples,
        }


def free_gpu_caches() -> None:
    """Drop cached device memory so the next size starts from a clean baseline.

    Without this the peak of size N-1 stays resident and is charged to size N, which
    would make the memory column cumulative rather than per-fit.
    """
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import gc

        gc.collect()
    except Exception:
        pass


# ── coordinates ────────────────────────────────────────────────────────────────

def embed_coordinates(args) -> int:
    """Embed every cohort row through the parametric encoder and save the 2-D result.

    Runs as its own process on purpose. See the module docstring: sharing a process with
    the cuML sweep is what halved RMM's budget on the previous run.
    """
    import torch

    import parametric_umap as pu

    embed_dir = Path(args.embed_dir).resolve()
    latent_path = embed_dir / "latent.npy"
    if not latent_path.exists():
        raise SystemExit(f"latent.npy not found in {embed_dir}")
    coords_path = Path(args.coords or (Path(args.output_dir) / "coords.npy")).resolve()
    coords_path.parent.mkdir(parents=True, exist_ok=True)

    if coords_path.exists() and not args.force_embed:
        existing = np.load(coords_path, mmap_mode="r")
        log(f"{coords_path.name} already exists ({existing.shape[0]:,} x {existing.shape[1]}); "
            "pass --force-embed to rebuild")
        return 0

    latent = np.load(latent_path, mmap_mode="r")
    total_rows, latent_dim = latent.shape
    log(f"latent.npy: {total_rows:,} rows x {latent_dim} dims")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(Path(args.encoder_model).resolve(), map_location=device,
                      weights_only=False)
    encoder = pu.ParametricEncoder(latent_dim, output_dim=2, hidden=(256, 256, 128)).to(device)
    encoder.load_state_dict(blob["state_dict"])
    mode = blob.get("mode", "umap")
    model = pu.ParametricUmap(encoder=encoder, device=device, mode=mode)
    log(f"encoder loaded (mode={mode}, device={device})")

    # Written straight into a memmap rather than accumulated and concatenated: the
    # concatenate would need the whole 1.26 GB result plus its chunks live at once.
    out = np.lib.format.open_memmap(
        coords_path, mode="w+", dtype=np.float32, shape=(total_rows, 2)
    )
    started = perf_counter()
    for start in range(0, total_rows, args.embed_batch_rows):
        stop = min(start + args.embed_batch_rows, total_rows)
        # np.array, not ascontiguousarray: the latter returns the input untouched when it
        # is already contiguous float32, and a mmap_mode="r" slice is read-only, so
        # torch.from_numpy downstream warns about a non-writable tensor. Harmless (nothing
        # writes to it) but it masks the same warning if it ever means something.
        chunk = np.array(latent[start:stop], dtype=np.float32)
        out[start:stop] = model.transform(chunk, batch_size=args.infer_batch_size)
        if (start // args.embed_batch_rows) % 5 == 0:
            done = stop / total_rows
            elapsed = perf_counter() - started
            log(f"  {stop:,}/{total_rows:,} ({done * 100:.0f}%) "
                f"eta {elapsed / max(done, 1e-9) * (1 - done):.0f}s")
    out.flush()
    seconds = perf_counter() - started
    finite = int(np.isfinite(np.asarray(out[:1_000_000])).all(axis=1).sum())
    log(f"embedded {total_rows:,} rows in {seconds:.1f}s -> {coords_path}")
    log(f"  sanity: {finite:,}/1,000,000 of the first rows are finite")
    return 0


# ── row draws ──────────────────────────────────────────────────────────────────

def draw_indices(total_rows: int, fit_sizes: list[int], eval_probe_rows: int,
                 timing_probe_rows: int, seed: int) -> dict:
    """Nested fit sets plus two probes that no fit set contains.

    Nesting matters: the 1 M fit set is a prefix of the 5 M one, so a difference between
    them is the effect of the extra rows and not of having drawn a different sample. The
    probes are carved off first so every fit size is scored on identical held-out rows --
    without that, cross-size agreement metrics compare clusterings of different points and
    mean nothing.
    """
    largest = max(fit_sizes)
    needed = largest + eval_probe_rows + timing_probe_rows
    if needed > total_rows:
        raise SystemExit(
            f"largest fit set ({largest:,}) plus probes ({eval_probe_rows:,} + "
            f"{timing_probe_rows:,}) needs {needed:,} rows but the cohort has {total_rows:,}"
        )

    permutation = np.random.default_rng(seed).permutation(total_rows)
    eval_probe = np.sort(permutation[:eval_probe_rows])
    timing_probe = np.sort(permutation[eval_probe_rows:eval_probe_rows + timing_probe_rows])
    pool = permutation[eval_probe_rows + timing_probe_rows:]
    fit_sets = {size: np.sort(pool[:size]) for size in fit_sizes}
    return {"eval_probe": eval_probe, "timing_probe": timing_probe, "fit_sets": fit_sets}


# ── one size ───────────────────────────────────────────────────────────────────

def run_size(fit_rows: int, coordinates, draws: dict, args, output_dir: Path,
             probe_sizes: list[int], min_samples: int, tag: str = "") -> dict:
    """Fit at one size, time predict, save labels and the model."""
    label = f"fit{fit_rows}{tag}"
    min_cluster_size = min_cluster_size_for(fit_rows)
    fit_idx = draws["fit_sets"][fit_rows]
    eval_idx = draws["eval_probe"]
    timing_idx = draws["timing_probe"]

    record: dict = {
        "fit_rows": int(fit_rows),
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "cluster_selection_method": "eom",
        "label": label,
    }

    log(f"[{label}] loading {fit_rows:,} coordinate rows (mcs={min_cluster_size}, "
        f"ms={min_samples}) ...")
    t0 = perf_counter()
    fit_coordinates = np.array(coordinates[fit_idx], dtype=np.float32)
    record["coordinate_load_seconds"] = round(perf_counter() - t0, 2)

    config = core.HdbscanConfig(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,
    )

    probe = GpuMemoryProbe(interval=args.memory_poll_seconds)
    probe.start()
    log(f"[{label}] fitting HDBSCAN ...")
    t0 = perf_counter()
    try:
        fitted = core.fit_hdbscan(fit_coordinates, config)
    except Exception as exc:
        memory = probe.stop()
        record.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "fit_seconds": round(perf_counter() - t0, 1),
            "memory": memory,
        })
        log(f"[{label}] FAILED after {record['fit_seconds']:.0f}s: "
            f"{type(exc).__name__}: {str(exc)[:200]}")
        free_gpu_caches()
        return record
    fit_seconds = perf_counter() - t0
    memory = probe.stop()

    n_clusters = int(fitted.labels.max()) + 1 if fitted.labels.size else 0
    noise_fraction = float((fitted.labels == -1).mean())
    record.update({
        "status": "ok",
        "backend": fitted.backend,
        "fit_seconds": round(fit_seconds, 1),
        "seconds_per_million_fit_rows": round(fit_seconds / (fit_rows / 1e6), 2),
        "n_clusters": n_clusters,
        "fit_noise_fraction": round(noise_fraction, 5),
        "dbcv": fitted.dbcv,
        "memory": memory,
    })
    log(f"[{label}]   fit in {fit_seconds:.1f}s (backend={fitted.backend}, "
        f"{n_clusters:,} clusters, {noise_fraction:.1%} noise, "
        f"peak {memory['peak_used_gb']:.1f} GB)")

    # Bank the fit output BEFORE predicting. The fit is the expensive artefact -- hours at
    # the large end -- and predict allocates again on a card that has just held a peak. If
    # an OOM lands there, saving afterwards would throw away a completed fit to a failure
    # in a step that only measures it.
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    np.save(labels_dir / f"{label}_labels.npy", fitted.labels)
    np.save(labels_dir / f"{label}_probabilities.npy", fitted.probabilities)
    np.save(labels_dir / f"{label}_indices.npy", fit_idx)
    record["labels_saved"] = True

    held_rows = TOTAL_COHORT_ROWS - fit_rows
    record["held_rows"] = int(held_rows)
    try:
        # One warm-up call: the first predict pays allocation and cuML kernel setup, and
        # charging that to the smallest probe would fake a superlinear curve.
        fitted.predict(np.array(coordinates[timing_idx[:2000]], dtype=np.float32))

        predictions = []
        for size in probe_sizes:
            block = np.array(coordinates[timing_idx[:size]], dtype=np.float32)
            t0 = perf_counter()
            labels, _ = fitted.predict(block)
            seconds = perf_counter() - t0
            per_million = seconds / (size / 1e6)
            predictions.append({
                "probe_rows": int(size),
                "seconds": round(seconds, 3),
                "seconds_per_million": round(per_million, 3),
                "extrapolated_held_seconds": round(per_million * held_rows / 1e6, 1),
                "extrapolated_held_minutes": round(per_million * held_rows / 1e6 / 60, 2),
                "extrapolated_cohort_minutes": round(
                    per_million * TOTAL_COHORT_ROWS / 1e6 / 60, 2),
                "noise_fraction": round(float((labels == -1).mean()), 5),
            })
            log(f"[{label}]   predict {size:>7,} rows in {seconds:6.2f}s = "
                f"{per_million:7.2f} s/M -> "
                f"{predictions[-1]['extrapolated_held_minutes']:6.1f} min "
                f"for {held_rows:,} held rows")

        rates = [entry["seconds_per_million"] for entry in predictions]
        record["predict"] = predictions
        record["predict_rate_spread"] = (
            round(max(rates) / min(rates), 3) if min(rates) > 0 else None
        )

        log(f"[{label}]   labelling the {len(eval_idx):,}-row eval probe ...")
        t0 = perf_counter()
        eval_labels, eval_probabilities = fitted.predict(
            np.array(coordinates[eval_idx], dtype=np.float32)
        )
        record["eval_probe_seconds"] = round(perf_counter() - t0, 2)
        record["eval_probe_noise_fraction"] = round(float((eval_labels == -1).mean()), 5)
        record["eval_probe_clusters_hit"] = int(np.unique(eval_labels[eval_labels >= 0]).size)
        np.save(labels_dir / f"{label}_evalprobe_labels.npy", eval_labels)
        np.save(labels_dir / f"{label}_evalprobe_probabilities.npy", eval_probabilities)
        record["eval_probe_saved"] = True
    except Exception as exc:
        # The fit succeeded and is on disk; only the measurement of it failed. Recorded as
        # a distinct status so the summary can say "fit fine, predict OOMed at this size",
        # which is itself the finding -- it is where full-cohort labelling stops being
        # possible, and it is not the same wall as the fit running out of memory.
        record["status"] = "predict_failed"
        record["predict_error_type"] = type(exc).__name__
        record["predict_error"] = str(exc)[:2000]
        record.setdefault("predict", [])
        record["eval_probe_saved"] = False
        log(f"[{label}]   predict FAILED ({type(exc).__name__}: {str(exc)[:200]}); "
            "fit labels are saved")

    if args.save_models:
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"hdbscan_{label}.joblib"
        try:
            import joblib

            joblib.dump(fitted.clusterer, model_path)
            size_mb = model_path.stat().st_size / 1e6
            record["model_path"] = str(model_path)
            record["model_size_mb"] = round(size_mb, 1)
            log(f"[{label}]   model saved ({size_mb:.0f} MB)")
        except Exception as exc:
            # sweep_core.persist_model documents this: cuML pickling is version-dependent.
            # The labels are the artefact that matters and they are already on disk, so a
            # failure here is recorded and the sweep continues.
            record["model_path"] = None
            record["model_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            log(f"[{label}]   could not save model ({type(exc).__name__}); labels are saved")

    del fitted, fit_coordinates
    free_gpu_caches()
    return record


# ── cli ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HDBSCAN fit-size scaling sweep: time, memory, cluster structure"
    )
    parser.add_argument("--embed-dir", required=True,
                        help="stage 1 output holding latent.npy")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--coords", default=None,
                        help="precomputed 2-D coordinates .npy (default <output-dir>/coords.npy)")

    parser.add_argument("--embed-only", action="store_true",
                        help="embed the cohort through --encoder-model, write coords.npy, "
                             "exit. Run this first, in its own process, so the sweep never "
                             "imports torch and RMM gets the whole GPU budget.")
    parser.add_argument("--encoder-model", default=None,
                        help="parametric encoder .pt, required with --embed-only")
    parser.add_argument("--force-embed", action="store_true")
    parser.add_argument("--embed-batch-rows", type=int, default=10_000_000)
    parser.add_argument("--infer-batch-size", type=int, default=2_000_000)

    parser.add_argument("--fit-sizes",
                        default="500000,1000000,5000000,10000000,15000000,25000000,50000000")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="held fixed across sizes. 5 is the low end of the 0-15 range "
                             "under consideration and the memory-safe end; the kNN graph "
                             "grows with it, and ms=25 is what OOMed at 70M.")
    parser.add_argument("--extra-min-samples", type=int, default=15,
                        help="after the main loop, refit the largest successful size at this "
                             "min_samples to measure the memory delta. 0 disables.")
    parser.add_argument("--probe-sizes", default="50000,200000,500000")
    parser.add_argument("--eval-probe-rows", type=int, default=2_000_000,
                        help="fixed rows labelled by every size, excluded from all fit sets. "
                             "This is what makes cross-size ARI computable later. Must not "
                             "change within a sweep -- see the draw-signature guard in main.")
    parser.add_argument("--timing-probe-rows", type=int, default=500_000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-budget-gb", type=float, default=None)
    parser.add_argument("--memory-poll-seconds", type=float, default=0.5)
    parser.add_argument("--save-models", action="store_true", default=True)
    parser.add_argument("--no-save-models", dest="save_models", action="store_false")
    parser.add_argument("--continue-after-failure", action="store_true",
                        help="by default the sweep stops at the first failure, because sizes "
                             "run ascending and a larger fit cannot succeed where a smaller "
                             "one ran out of memory")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    return parser.parse_args()


def main() -> int:
    overall = perf_counter()
    args = parse_args()

    if args.embed_only:
        if not args.encoder_model:
            raise SystemExit("--embed-only requires --encoder-model")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        return embed_coordinates(args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    coords_path = Path(args.coords or (output_dir / "coords.npy")).resolve()
    if not coords_path.exists():
        raise SystemExit(
            f"{coords_path} not found. Run --embed-only first:\n"
            f"  python {Path(__file__).name} --embed-dir {args.embed_dir} "
            f"--output-dir {args.output_dir} --encoder-model <encoder.pt> --embed-only"
        )

    fit_sizes = sorted({int(v) for v in args.fit_sizes.split(",") if v.strip()})
    probe_sizes = sorted({int(v) for v in args.probe_sizes.split(",") if v.strip()})

    coordinates = np.load(coords_path, mmap_mode="r")
    total_rows = coordinates.shape[0]
    log(f"coords: {total_rows:,} x {coordinates.shape[1]} from {coords_path.name}")

    budget_report = None
    if core.gpu_available():
        # Stage "sweep", not "apply": torch is never imported in this process, so RMM takes
        # 0.9 of the budget instead of the 0.5 split that capped the previous run at 20 GB.
        budget_report = core.apply_gpu_budget("sweep", budget_gb=args.gpu_budget_gb)
    else:
        log("WARNING: cuML not available -- this will run on the CPU backend and the "
            "timings will not describe the GPU path")

    # Guard against the one change that corrupts a sweep silently. Resume skips completed
    # cells, so re-running with a different probe size or seed leaves the finished cells
    # holding labels for the OLD probe while new cells get the NEW one -- and a cross-size
    # ARI computed over that mixture compares clusterings of different rows without any
    # error surfacing. Fit sizes are deliberately NOT part of the signature: the
    # permutation depends only on (total_rows, seed) and the probes are its first slices,
    # so appending a larger fit size later leaves every existing draw untouched.
    signature = {
        "seed": int(args.seed),
        "eval_probe_rows": int(args.eval_probe_rows),
        "timing_probe_rows": int(args.timing_probe_rows),
        "total_rows": int(total_rows),
    }
    signature_path = output_dir / "draw_signature.json"
    if signature_path.exists():
        previous = json.loads(signature_path.read_text())
        if previous != signature:
            changed = [k for k in signature if previous.get(k) != signature[k]]
            raise SystemExit(
                f"{output_dir} holds a sweep drawn with different settings.\n"
                f"  changed: {', '.join(f'{k} {previous.get(k)} -> {signature[k]}' for k in changed)}\n"
                f"\nCompleted cells would keep the old probe while new ones get the new one, "
                f"and nothing downstream could tell them apart.\n"
                f"To start over with the new settings (coords.npy is reused, it does not "
                f"depend on the probes):\n"
                f"  rm -rf {output_dir}/cells {output_dir}/labels {output_dir}/models \\\n"
                f"         {output_dir}/draw_signature.json {output_dir}/scaling_results.json"
            )
    else:
        signature_path.write_text(json.dumps(signature, indent=2))

    draws = draw_indices(total_rows, fit_sizes, args.eval_probe_rows,
                         args.timing_probe_rows, args.seed)
    log(f"eval probe {len(draws['eval_probe']):,} rows, "
        f"timing probe {len(draws['timing_probe']):,} rows, "
        f"fit sizes {', '.join(f'{s:,}' for s in fit_sizes)}")
    log("min_cluster_size per size: " + ", ".join(
        f"{s:,}->{min_cluster_size_for(s)}" for s in fit_sizes))

    np.save(output_dir / "eval_probe_indices.npy", draws["eval_probe"])
    np.save(output_dir / "timing_probe_indices.npy", draws["timing_probe"])

    records: list[dict] = []
    stopped_early = False
    for fit_rows in fit_sizes:
        cell_path = cells_dir / f"fit{fit_rows}.json"
        if args.resume and cell_path.exists():
            payload = json.loads(cell_path.read_text())
            if payload.get("status") == "ok":
                log(f"[fit{fit_rows}] already done ({payload['fit_seconds']}s) -- skipping")
                records.append(payload)
                continue

        record = run_size(fit_rows, coordinates, draws, args, output_dir,
                          probe_sizes, args.min_samples)
        cell_path.write_text(json.dumps(record, indent=2))
        records.append(record)

        if record["status"] != "ok" and not args.continue_after_failure:
            reason = ("the fit ran out of memory" if record["status"] == "failed"
                      else "the fit succeeded but predict failed")
            log(f"stopping: fit{fit_rows} -- {reason}, and sizes ascend so nothing larger "
                "can do better. Pass --continue-after-failure to try them anyway.")
            stopped_early = True
            break

    # "ok" means fit AND predict; "predict_failed" still has a usable fit on disk. The fit
    # scaling curve and the min_samples probe care only about the fit, so they use the
    # wider set; anything reading predict rates uses the narrower one.
    fit_ok = [r for r in records if r["status"] in ("ok", "predict_failed")]
    successful = [r for r in records if r["status"] == "ok"]

    # The memory question the main loop cannot answer: min_samples drives the kNN graph
    # size, and the 0-15 range under consideration tops out well above the 5 used here.
    extra = None
    if args.extra_min_samples and fit_ok:
        largest = max(r["fit_rows"] for r in fit_ok)
        log(f"extra probe: refitting {largest:,} rows at min_samples={args.extra_min_samples} "
            "to measure the memory delta")
        extra_path = cells_dir / f"fit{largest}_ms{args.extra_min_samples}.json"
        if args.resume and extra_path.exists():
            extra = json.loads(extra_path.read_text())
            log("  already done -- skipping")
        else:
            extra = run_size(largest, coordinates, draws, args, output_dir, probe_sizes,
                             args.extra_min_samples, tag=f"_ms{args.extra_min_samples}")
            extra_path.write_text(json.dumps(extra, indent=2))

    results = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "coords_path": str(coords_path),
        "total_rows": int(total_rows),
        "seed": args.seed,
        "hdbscan": {
            "cluster_selection_method": "eom",
            "min_samples": args.min_samples,
            "metric": "euclidean",
            "cluster_selection_epsilon": 0.0,
            "min_cluster_size_policy": f"max({MCS_FLOOR}, round(fit_rows * {MCS_FRACTION}))",
        },
        "eval_probe_rows": int(args.eval_probe_rows),
        "timing_probe_rows": int(args.timing_probe_rows),
        "probe_sizes": probe_sizes,
        "gpu_budget": budget_report,
        "cells": records,
        "extra_min_samples_probe": extra,
        "stopped_early": stopped_early,
        "total_seconds": round(perf_counter() - overall, 1),
    }
    out_path = output_dir / "scaling_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"wrote {out_path}")

    print("\n-- HDBSCAN fit-size scaling ---------------------------------------------")
    print("   fit rows    mcs   clusters   noise      fit s    s/M fit   peak GB   "
          "predict s/M")
    for record in records:
        if record["status"] == "failed":
            print(f"{record['fit_rows']:>11,}  {record['min_cluster_size']:>5}   "
                  f"FIT FAILED: {record.get('error_type', '?')}")
            continue
        rate = (f"{record['predict'][-1]['seconds_per_million']:>11.2f}"
                if record.get("predict") else f"{'predict OOM':>11}")
        print(f"{record['fit_rows']:>11,}  {record['min_cluster_size']:>5}  "
              f"{record['n_clusters']:>9,}  {record['fit_noise_fraction']:>6.1%}  "
              f"{record['fit_seconds']:>9.1f}  {record['seconds_per_million_fit_rows']:>8.1f}  "
              f"{record['memory']['peak_used_gb']:>7.1f}  {rate}")
    if extra and extra["status"] in ("ok", "predict_failed"):
        base = next((r for r in fit_ok if r["fit_rows"] == extra["fit_rows"]), None)
        extra_peak = extra["memory"]["peak_used_gb"]
        if base is not None:
            base_peak = base["memory"]["peak_used_gb"]
            print(f"\nmin_samples probe at {extra['fit_rows']:,} rows: "
                  f"ms={args.min_samples} peak {base_peak:.1f} GB -> "
                  f"ms={extra['min_samples']} peak {extra_peak:.1f} GB "
                  f"({extra_peak - base_peak:+.1f} GB, "
                  f"{extra['fit_seconds'] / max(base['fit_seconds'], 1e-9):.2f}x the fit time)")
        else:
            print(f"\nmin_samples probe at {extra['fit_rows']:,} rows, "
                  f"ms={extra['min_samples']}: peak {extra_peak:.1f} GB")
    elif extra:
        print(f"\nmin_samples probe at ms={args.extra_min_samples}: "
              f"FAILED ({extra.get('error_type', '?')}) -- ms={args.extra_min_samples} is out "
              f"of reach at {extra['fit_rows']:,} rows, so cap the later sweep below it")
    print("-------------------------------------------------------------------------")

    if len(fit_ok) >= 2:
        first, last = fit_ok[0], fit_ok[-1]
        size_ratio = last["fit_rows"] / first["fit_rows"]
        time_ratio = last["fit_seconds"] / max(first["fit_seconds"], 1e-9)
        exponent = np.log(time_ratio) / np.log(size_ratio) if size_ratio > 1 else float("nan")
        print(f"\nfit cost scales as N^{exponent:.2f} between {first['fit_rows']:,} and "
              f"{last['fit_rows']:,} rows")

    if len(successful) >= 2:
        first, last = successful[0], successful[-1]
        size_ratio = last["fit_rows"] / first["fit_rows"]
        rate_first = first["predict"][-1]["seconds_per_million"]
        rate_last = last["predict"][-1]["seconds_per_million"]
        print(f"predict rate {rate_first:.1f} -> {rate_last:.1f} s/M over the same range "
              f"({rate_last / max(rate_first, 1e-9):.2f}x for a {size_ratio:.0f}x larger fit set)")
        if rate_last / max(rate_first, 1e-9) < 1.2:
            print("  -> predict cost is essentially independent of fit-set size, so a larger")
            print("     fit does not make the labelling pass more expensive.")
        else:
            print("  -> predict cost grows with fit-set size; budget the labelling pass from")
            print("     the rate at the size you actually choose, not from the 5M measurement.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
