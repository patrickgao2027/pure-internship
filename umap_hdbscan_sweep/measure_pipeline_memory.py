#!/usr/bin/env python
"""Measure peak GPU VRAM and host RAM for the UMAP and HDBSCAN stages, phase by phase.

The sweep artifacts record timing but never memory, so the report can only quote the
*budget caps* (``torch 6.75 GB + rmm pool 38.25 GB`` for pUMAP, ``torch 4 GB + rmm pool
36 GB`` for the HDBSCAN sweep) rather than what was actually used. This script re-runs
each stage's real code path with a sampler thread watching the card and the process.

Two stages, selected with ``--stage``:

``umap``
    ``core.fit_umap`` -> ``transform`` -> ``pu.extract_edges`` -> ``pu.train_umap_loss``,
    the same calls ``parametric_sweep.run_cell`` makes, in the same order.

``hdbscan``
    ``core.fit_hdbscan`` -> ``fast_predict.build_tables`` -> ``fast_predict.build_index``
    -> ``fast_predict.predict``, the same calls ``hdbscan_param_sweep`` makes to fit a
    cell and label the full cohort.

``both`` (the default) runs them back to back in one process. Note that a shared process
means the HDBSCAN device peak inherits whatever the UMAP stage failed to hand back to the
driver -- for a clean per-stage number, run each stage in its own process, which is what
``tmux_measure_memory.sh`` does.

**Run it with the RMM pool off** (the default, ``--rmm-share 0``). A pool pre-allocates a
quarter of itself up front and then never returns memory to the driver, so with a pool the
device reading is the pool size and tells you nothing about demand. With no pool, cuML
allocates on demand and ``nvidia-smi`` tracks the real high-water mark. Pass
``--rmm-share 0.85`` to reproduce the sweep's own configuration instead -- that answers
"did the run stay inside its cap", not "how much did it need".

Peaks are attributed to whichever phase was open when they were observed, so a phase's
number includes everything still resident from earlier phases (the fit set stays on the
host for the whole run, the edge list stays on the device through training). They are
high-water marks, not per-phase deltas.

What is already known without running this
------------------------------------------
* The HDBSCAN *fit* is the stage that OOMs, and it is superlinear in fit rows. From
  ``hdbscan_param_sweep``'s cost model: 25M fit succeeded inside a 40 GB cap; 50M OOMed on
  a 47 GB card, as did ``min_samples=15`` at 25M. So the sweep's ceiling sits between those.
* The *deployed* HDBSCAN fits only 1M rows (``mcs=2500, ms=1, epsilon=0.05``, 86 s) and
  labels the cohort through the RBC index in 46 s. It is nowhere near the sweep's ceiling.
This script exists to put a measured number on both instead of bracketing them.

Usage on miletus::

    bash umap_hdbscan_sweep/tmux_measure_memory.sh

or directly::

    micromamba run -n uv_vae python umap_hdbscan_sweep/measure_pipeline_memory.py \\
        --stage hdbscan --coords <coords.npy> --hdbscan-fit-rows 1000000 \\
        --min-cluster-size 2500 --min-samples 1 --cluster-selection-epsilon 0.05 \\
        --out umap_hdbscan_sweep/umap_tests/memory_profile/hdbscan_deployed.json

Start small (``--fit-rows 2000000`` / ``--hdbscan-fit-rows 500000``) to confirm the harness
works before spending the hours the 25M fits cost.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np

import sweep_core as core

MB = 1024 ** 2
GB = 1024 ** 3


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# ── host RAM ──────────────────────────────────────────────────────────────────

def _proc_status_kb(field: str) -> float:
    """Read one VmXXX field from /proc/self/status, in KB. 0.0 off Linux."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith(field):
                    return float(line.split()[1])
    except OSError:
        pass
    return 0.0


def host_rss_gb() -> float:
    return _proc_status_kb("VmRSS:") / MB


def host_peak_rss_gb() -> float:
    """VmHWM -- the kernel's own high-water mark, immune to sampler aliasing."""
    return _proc_status_kb("VmHWM:") / MB


# ── device VRAM ───────────────────────────────────────────────────────────────

class DeviceProbe:
    """Device-wide and this-process VRAM, via NVML when available, else nvidia-smi.

    The device-wide number includes co-tenants; ``used_by_pid`` is the honest one to
    quote when the card is shared, but NVML only reports it for compute processes it can
    see, so it can come back as 0 in a container. Both are recorded.
    """

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.handle = None
        self.nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml = None

    def sample(self) -> tuple[float, float, float]:
        """(device_used_gb, device_total_gb, this_process_gb)."""
        if self.nvml is not None and self.handle is not None:
            try:
                info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                mine = 0.0
                try:
                    procs = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                    for proc in procs:
                        if proc.pid == self.pid and proc.usedGpuMemory:
                            mine = proc.usedGpuMemory / GB
                except Exception:
                    pass
                return info.used / GB, info.total / GB, mine
            except Exception:
                pass
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True).stdout
            used, total = (float(v) for v in out.strip().splitlines()[0].split(","))
            return used / 1024.0, total / 1024.0, 0.0
        except Exception:
            return 0.0, 0.0, 0.0


class MemoryProbe:
    """Background sampler that attributes high-water marks to the open phase."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.device = DeviceProbe()
        self.phase = "startup"
        self.phases: dict[str, dict] = {}
        self.trace: list[dict] = []
        self.device_total_gb = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_at = perf_counter()

    # -- lifecycle --
    def start(self) -> "MemoryProbe":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._observe()
            self._stop.wait(self.interval)

    def _observe(self) -> None:
        device_used, device_total, mine = self.device.sample()
        rss = host_rss_gb()
        now = perf_counter() - self.started_at
        with self._lock:
            entry = self.phases.setdefault(self.phase, {
                "device_used_peak_gb": 0.0,
                "process_vram_peak_gb": 0.0,
                "host_rss_peak_gb": 0.0,
                "samples": 0,
            })
            entry["device_used_peak_gb"] = max(entry["device_used_peak_gb"], device_used)
            entry["process_vram_peak_gb"] = max(entry["process_vram_peak_gb"], mine)
            entry["host_rss_peak_gb"] = max(entry["host_rss_peak_gb"], rss)
            entry["samples"] += 1
            self.device_total_gb = device_total
            self.trace.append({
                "t": round(now, 1),
                "phase": self.phase,
                "device_used_gb": round(device_used, 3),
                "process_vram_gb": round(mine, 3),
                "host_rss_gb": round(rss, 3),
            })

    # -- phases --
    class _Phase:
        def __init__(self, probe: "MemoryProbe", name: str) -> None:
            self.probe, self.name = probe, name

        def __enter__(self):
            self.started = perf_counter()
            with self.probe._lock:
                self.probe.phase = self.name
            self.probe._observe()          # bracket the phase so a fast one is never empty
            return self

        def __exit__(self, *exc):
            self.probe._observe()
            elapsed = round(perf_counter() - self.started, 1)
            with self.probe._lock:
                self.probe.phases.setdefault(self.name, {})["seconds"] = elapsed
                self.probe.phase = "between"
            log(f"    {self.name}: {elapsed}s")
            return False

    def phase_scope(self, name: str) -> "MemoryProbe._Phase":
        return MemoryProbe._Phase(self, name)

    def peaks_for(self, prefix: str) -> dict:
        """Max over every phase whose name starts with ``prefix``."""
        selected = [entry for name, entry in self.phases.items() if name.startswith(prefix)]
        return {
            "device_used_gb": round(
                max((e.get("device_used_peak_gb", 0.0) for e in selected), default=0.0), 3),
            "process_vram_gb": round(
                max((e.get("process_vram_peak_gb", 0.0) for e in selected), default=0.0), 3),
            "host_rss_gb": round(
                max((e.get("host_rss_peak_gb", 0.0) for e in selected), default=0.0), 3),
        }


# ── shared setup ──────────────────────────────────────────────────────────────

def _free_device_caches(use_gpu: bool) -> None:
    """Hand back what cupy and torch are caching so the next stage starts clean(er).

    This is best-effort. RMM never returns pool memory to the driver, which is exactly why
    the honest per-stage measurement is one process per stage.
    """
    if not use_gpu:
        return
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _open_latent(path: Path, what: str):
    if not path.exists():
        raise SystemExit(f"{what} not found: {path}")
    array = np.load(path, mmap_mode="r")
    log(f"    {what} {array.shape[0]:,} x "
        f"{array.shape[1] if array.ndim > 1 else 1} ({array.dtype})")
    return array


def _gather(array, rows: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """A sorted random subset, materialised contiguously (mmap reads are page-ordered)."""
    total = array.shape[0]
    rows = min(rows, total)
    index = np.sort(rng.choice(total, size=rows, replace=False))
    return np.ascontiguousarray(array[index]), index


# ── the UMAP stage ────────────────────────────────────────────────────────────

def measure_umap(args, probe: MemoryProbe, use_gpu: bool) -> dict:
    import torch

    import parametric_umap as pu

    device = torch.device("cuda" if use_gpu else "cpu")

    with probe.phase_scope("umap.load_latent"):
        latent = _open_latent(Path(args.embed_dir) / "latent.npy", "latent")
        total_rows, latent_dim = latent.shape

    fit_rows = min(args.fit_rows, total_rows)
    rng = np.random.default_rng(args.seed)

    with probe.phase_scope("umap.gather_fit_set"):
        fit_latent, _ = _gather(latent, fit_rows, rng)
        log(f"    fit set {fit_latent.shape} = {fit_latent.nbytes / GB:.2f} GB on host")

    with probe.phase_scope("umap.gather_probe"):
        probe_latent, _ = _gather(latent, args.probe_rows, rng)

    config = core.UmapConfig(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        seed=args.seed,
    )

    with probe.phase_scope("umap.fit"):
        fitted = core.fit_umap(fit_latent, config, use_gpu=use_gpu)

    with probe.phase_scope("umap.probe_transform"):
        fitted.transform(probe_latent, batch_size=args.transform_batch_size)

    with probe.phase_scope("umap.extract_edges"):
        fit_embedding = np.ascontiguousarray(fitted.embedding)
        edges = pu.extract_edges(fitted.model.graph_)
        negative_sample_rate = int(getattr(fitted.model, "negative_sample_rate", 5))
        log(f"    {len(edges[0]):,} edges = {edges[0].nbytes * 3 / GB:.2f} GB on host")

    with probe.phase_scope("umap.free_cuml"):
        del fitted
        _free_device_caches(use_gpu)

    # ── torch side, mirroring run_cell's ordering ─────────────────────────────
    if use_gpu:
        torch.cuda.reset_peak_memory_stats()

    with probe.phase_scope("umap.upload_tensors"):
        X = torch.from_numpy(fit_latent).to(device)
        # Y is unused by the umap objective (only regress/hybrid read it) but run_cell
        # uploads it unconditionally, so it is uploaded here too -- leaving it out would
        # under-report the footprint by the size of the embedding.
        Y = torch.from_numpy(fit_embedding).to(device)  # noqa: F841
        del fit_latent, fit_embedding

        head_np, tail_np, weight_np = edges
        n_edges = len(head_np)
        if n_edges > args.max_edges:
            keep = np.random.default_rng(config.seed).choice(
                n_edges, size=args.max_edges, replace=False)
            keep.sort()
            head_np, tail_np, weight_np = head_np[keep], tail_np[keep], weight_np[keep]
            log(f"    edge list capped {n_edges:,} -> {len(head_np):,}")
        edge_head = torch.from_numpy(head_np).to(device)
        edge_tail = torch.from_numpy(tail_np).to(device)
        edge_cumsum = torch.cumsum(torch.from_numpy(weight_np).to(device).double(), 0)
        del edges, head_np, tail_np, weight_np

    a, b = pu.ab_params(config.min_dist, spread=config.spread)
    encoder = pu.ParametricEncoder(latent_dim, output_dim=2, hidden=(256, 256, 128)).to(device)

    with probe.phase_scope("umap.encoder_train"):
        pu.train_umap_loss(
            encoder=encoder,
            X=X,
            edge_head=edge_head,
            edge_tail=edge_tail,
            edge_cumsum=edge_cumsum,
            a=a, b=b,
            steps=args.umap_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            negative_sample_rate=negative_sample_rate,
            repulsion_strength=config.repulsion_strength,
            device=device,
        )

    torch_peak_gb = (torch.cuda.max_memory_allocated() / GB) if use_gpu else 0.0
    torch_reserved_gb = (torch.cuda.max_memory_reserved() / GB) if use_gpu else 0.0

    with probe.phase_scope("umap.full_transform"):
        wrapper = pu.ParametricUmap(encoder, device, mode="umap")
        wrapper.transform(probe_latent, batch_size=args.transform_batch_size)

    peaks = probe.peaks_for("umap.")
    peaks["torch_allocated_gb"] = round(torch_peak_gb, 3)
    peaks["torch_reserved_gb"] = round(torch_reserved_gb, 3)

    del X, edge_head, edge_tail, edge_cumsum, encoder, probe_latent
    _free_device_caches(use_gpu)

    return {
        "config": {
            "fit_rows": fit_rows,
            "probe_rows": args.probe_rows,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "umap_steps": args.umap_steps,
            "batch_size": args.batch_size,
            "max_edges": args.max_edges,
            "latent_rows": int(total_rows),
            "latent_dim": int(latent_dim),
        },
        "peaks": peaks,
    }


# ── the HDBSCAN stage ─────────────────────────────────────────────────────────

def measure_hdbscan(args, probe: MemoryProbe, use_gpu: bool) -> dict:
    """Fit HDBSCAN on a coordinate subset, then label rows through the RBC index.

    The fit is the phase that OOMs -- it is superlinear in fit rows and holds the mutual
    reachability structure on the device. Labelling is batched by design (``predict``
    holds ``batch_rows * 2 * min_samples`` neighbours at a time), so it is bounded by
    ``--predict-batch-rows`` and by the RBC index over the fit set, not by cohort size.
    """
    import fast_predict

    coords_path = Path(args.coords) if args.coords else \
        Path(args.embed_dir) / "umap_coords_2d.npy"

    with probe.phase_scope("hdbscan.load_coords"):
        coords = _open_latent(coords_path, "coords")
        total_rows = coords.shape[0]

    fit_rows = min(args.hdbscan_fit_rows, total_rows)
    rng = np.random.default_rng(args.seed)

    with probe.phase_scope("hdbscan.gather_fit_set"):
        fit_coords, _ = _gather(coords, fit_rows, rng)
        fit_coords = np.ascontiguousarray(fit_coords, dtype=np.float32)
        log(f"    fit set {fit_coords.shape} = {fit_coords.nbytes / GB:.2f} GB on host")

    config = core.HdbscanConfig(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
    )
    log(f"    hdbscan config {config.label()} "
        f"eps={args.cluster_selection_epsilon} on {fit_rows:,} rows")

    with probe.phase_scope("hdbscan.fit"):
        fitted = core.fit_hdbscan(fit_coords, config, use_gpu=use_gpu)
        n_clusters = int(fitted.labels.max()) + 1 if fitted.labels.size else 0
        noise = float((fitted.labels < 0).mean()) if fitted.labels.size else 0.0
        log(f"    {n_clusters} clusters, {noise * 100:.2f}% noise ({fitted.backend})")

    # Steps that turn the fitted model into something that can label unseen rows. cuML does
    # not always keep core distances, so the recompute path is not an edge case here.
    with probe.phase_scope("hdbscan.build_tables"):
        min_samples = fast_predict.resolve_min_samples(fitted.clusterer, config.min_samples)
        try:
            tables = fast_predict.build_tables(fitted.clusterer, fit_rows,
                                               min_samples=min_samples)
            recomputed = False
        except Exception as exc:
            log(f"    core distances unavailable ({exc}); recomputing from the fit set")
            core_distances = fast_predict.recompute_core_distances(
                fit_coords, min_samples, backend=args.predict_backend,
                batch_rows=args.predict_batch_rows)
            tables = fast_predict.build_tables(fitted.clusterer, fit_rows,
                                               min_samples=min_samples,
                                               core_distances=core_distances)
            recomputed = True

    with probe.phase_scope("hdbscan.build_index"):
        index = fast_predict.build_index(fit_coords, 2 * tables.min_samples,
                                         args.predict_backend)

    label_rows = total_rows if args.label_rows <= 0 else min(args.label_rows, total_rows)
    with probe.phase_scope("hdbscan.label_rows"):
        # Slicing the mmap rather than gathering keeps the query side off the heap: predict
        # batches internally, so only batch_rows of coordinates are ever faulted in.
        query = coords[:label_rows]
        labels, probabilities = fast_predict.predict(
            tables, fit_coords, query,
            backend=args.predict_backend,
            batch_rows=args.predict_batch_rows,
            index=index,
        )
        log(f"    labelled {labels.shape[0]:,} rows, "
            f"{float((labels < 0).mean()) * 100:.2f}% noise")

    peaks = probe.peaks_for("hdbscan.")

    del fitted, tables, index, fit_coords, labels, probabilities
    _free_device_caches(use_gpu)

    return {
        "config": {
            "coords_path": str(coords_path),
            "fit_rows": fit_rows,
            "label_rows": label_rows,
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "cluster_selection_epsilon": args.cluster_selection_epsilon,
            "predict_backend": args.predict_backend,
            "predict_batch_rows": args.predict_batch_rows,
            "core_distances_recomputed": recomputed,
            "coords_rows": int(total_rows),
        },
        "result": {
            "n_clusters": n_clusters,
            "fit_noise_fraction": round(noise, 6),
        },
        "peaks": peaks,
    }


# ── driver ────────────────────────────────────────────────────────────────────

def build_payload(args, probe: MemoryProbe, stages: dict, budget_report, use_gpu: bool,
                  complete: bool) -> dict:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "note": ("Peaks are high-water marks with everything earlier still resident, not "
                 "per-phase deltas. Run with --rmm-share 0 for demand; a pool masks it. "
                 "With --stage both the hdbscan peaks inherit the umap stage's residue -- "
                 "run one stage per process for clean per-stage numbers."),
        "stage": args.stage,
        "backend": "cuml" if use_gpu else "cpu",
        "seed": args.seed,
        "gpu_budget_gb": args.gpu_budget_gb,
        "rmm_share": args.rmm_share,
        "gpu_budget_report": budget_report,
        "device_total_gb": round(probe.device_total_gb, 2),
        "peaks": {
            "device_used_gb": round(
                max((p.get("device_used_peak_gb", 0.0) for p in probe.phases.values()),
                    default=0.0), 3),
            "process_vram_gb": round(
                max((p.get("process_vram_peak_gb", 0.0) for p in probe.phases.values()),
                    default=0.0), 3),
            "host_rss_gb": round(host_peak_rss_gb(), 3),
        },
        "stages": stages,
        "phases": {
            name: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in entry.items()}
            for name, entry in probe.phases.items()
        },
    }
    if not complete:
        payload["open_phase"] = probe.phase
        payload["elapsed_seconds"] = round(perf_counter() - probe.started_at, 1)
        payload["note"] = ("PARTIAL -- the run did not finish. The open phase's peak is "
                           "a lower bound on what that phase needed. " + payload["note"])
    if args.trace:
        payload["trace"] = probe.trace
    return payload


def _write_json(path: Path, payload: dict) -> None:
    """Write via a temp file + rename so a kill mid-write cannot truncate the last good one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def measure(args) -> dict:
    probe = MemoryProbe(interval=args.sample_interval).start()

    budget_report = None
    if args.gpu_budget_gb is not None:
        with probe.phase_scope("gpu_budget"):
            budget_report = core.apply_gpu_budget(
                "apply", budget_gb=args.gpu_budget_gb, rmm_share=args.rmm_share)
            log(f"    budget: {json.dumps(budget_report)}")

    use_gpu = core.gpu_available() and not args.force_cpu
    log(f"backend: {'cuML/GPU' if use_gpu else 'CPU'}   stage={args.stage}")

    stages: dict[str, dict] = {}

    # A 25M fit runs for ~50 minutes inside a single library call, and systemd-oomd on
    # miletus kills on memory *pressure* rather than exhaustion -- so the run that most
    # needs measuring is exactly the one most likely to be killed before it can report.
    # Checkpointing the peaks turns that kill from "51 minutes and nothing" into "51
    # minutes and a lower bound", which is the number the report actually needs.
    def snapshot(complete: bool) -> dict:
        return build_payload(args, probe, stages, budget_report, use_gpu, complete)

    if args.out:
        out_path = Path(args.out)

        def checkpoint_loop() -> None:
            while not checkpoint_stop.wait(args.checkpoint_seconds):
                try:
                    _write_json(out_path, snapshot(complete=False))
                except Exception:
                    pass                       # a failed checkpoint must never kill the run

        checkpoint_stop = threading.Event()
        threading.Thread(target=checkpoint_loop, daemon=True).start()

        def on_signal(signum, _frame):
            log(f"signal {signum} -- writing the partial profile before exiting")
            probe.stop()
            try:
                _write_json(out_path, snapshot(complete=False))
                log(f"wrote partial {out_path}")
            except Exception as exc:
                log(f"could not write the partial profile: {exc}")
            os._exit(143 if signum == signal.SIGTERM else 130)

        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)

    if args.stage in ("umap", "both"):
        log("=== UMAP stage ===")
        stages["umap"] = measure_umap(args, probe, use_gpu)
    if args.stage in ("hdbscan", "both"):
        log("=== HDBSCAN stage ===")
        stages["hdbscan"] = measure_hdbscan(args, probe, use_gpu)

    probe.stop()
    return snapshot(complete=True)


def print_report(payload: dict) -> None:
    print()
    print("=" * 78)
    print(f"  Memory profile -- stage={payload['stage']} ({payload['backend']})")
    if not payload.get("complete", True):
        print(f"  PARTIAL -- killed during '{payload.get('open_phase')}' after "
              f"{payload.get('elapsed_seconds')}s. Peaks are LOWER BOUNDS.")
    print("=" * 78)
    print(f"  device total                 {payload['device_total_gb']:>8.2f} GB")
    print(f"  PEAK device VRAM used        {payload['peaks']['device_used_gb']:>8.2f} GB   "
          f"(all processes on the card)")
    print(f"  PEAK this-process VRAM       {payload['peaks']['process_vram_gb']:>8.2f} GB   "
          f"(NVML per-pid; 0.00 = unavailable)")
    print(f"  PEAK host RSS (VmHWM)        {payload['peaks']['host_rss_gb']:>8.2f} GB")

    for name, stage in payload["stages"].items():
        peaks, cfg = stage["peaks"], stage["config"]
        print()
        print(f"  -- {name} --")
        if name == "umap":
            print(f"     {cfg['fit_rows']:,} fit rows, nn={cfg['n_neighbors']} "
                  f"min_dist={cfg['min_dist']}")
        else:
            print(f"     {cfg['fit_rows']:,} fit rows, mcs={cfg['min_cluster_size']} "
                  f"ms={cfg['min_samples']} eps={cfg['cluster_selection_epsilon']}, "
                  f"labelling {cfg['label_rows']:,}")
            print(f"     -> {stage['result']['n_clusters']} clusters")
        print(f"     peak device VRAM  {peaks['device_used_gb']:>8.2f} GB")
        print(f"     peak process VRAM {peaks['process_vram_gb']:>8.2f} GB")
        print(f"     peak host RSS     {peaks['host_rss_gb']:>8.2f} GB")
        if "torch_allocated_gb" in peaks:
            print(f"     peak torch alloc  {peaks['torch_allocated_gb']:>8.2f} GB")
            print(f"     peak torch resvd  {peaks['torch_reserved_gb']:>8.2f} GB")

    print()
    print(f"  {'phase':<28} {'secs':>8} {'devVRAM':>9} {'procVRAM':>9} {'hostRSS':>9}")
    print("  " + "-" * 68)
    for name, entry in payload["phases"].items():
        if name in ("between", "startup"):
            continue
        print(f"  {name:<28} {entry.get('seconds', 0):>8.1f} "
              f"{entry.get('device_used_peak_gb', 0):>9.2f} "
              f"{entry.get('process_vram_peak_gb', 0):>9.2f} "
              f"{entry.get('host_rss_peak_gb', 0):>9.2f}")
    print("  " + "-" * 68)
    print("  Columns are high-water marks (GB) with earlier allocations still resident.")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["umap", "hdbscan", "both"], default="both")
    parser.add_argument("--embed-dir",
                        help="stage1_embed dir holding latent.npy (required for --stage umap)")
    parser.add_argument("--out", default=None, help="write the JSON payload here")

    umap_group = parser.add_argument_group("umap stage")
    umap_group.add_argument("--fit-rows", type=int, default=25_000_000)
    umap_group.add_argument("--probe-rows", type=int, default=500_000)
    umap_group.add_argument("--n-neighbors", type=int, default=15)
    umap_group.add_argument("--min-dist", type=float, default=0.1)
    umap_group.add_argument("--umap-steps", type=int, default=30_000,
                            help="encoder training steps (sweep used 30000)")
    umap_group.add_argument("--batch-size", type=int, default=16_384)
    umap_group.add_argument("--learning-rate", type=float, default=1e-3)
    umap_group.add_argument("--max-edges", type=int, default=150_000_000)
    umap_group.add_argument("--transform-batch-size", type=int, default=2_000_000)

    hdb_group = parser.add_argument_group("hdbscan stage")
    hdb_group.add_argument("--coords",
                           help="2-D coords .npy; defaults to <embed-dir>/umap_coords_2d.npy")
    hdb_group.add_argument("--hdbscan-fit-rows", type=int, default=1_000_000,
                           help="1000000 is the deployed model; the sweep went to 25M")
    hdb_group.add_argument("--min-cluster-size", type=int, default=2500)
    hdb_group.add_argument("--min-samples", type=int, default=1)
    hdb_group.add_argument("--cluster-selection-epsilon", type=float, default=0.05)
    hdb_group.add_argument("--label-rows", type=int, default=-1,
                           help="rows to label through the index; -1 (default) = all of them")
    hdb_group.add_argument("--predict-backend", default="rbc",
                           choices=["rbc", "brute", "sklearn"])
    hdb_group.add_argument("--predict-batch-rows", type=int, default=5_000_000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-budget-gb", type=float, default=None,
                        help="omit to measure uncapped demand; 40 reproduces the sweep")
    parser.add_argument("--rmm-share", type=float, default=0.0,
                        help="0 (default) = no RMM pool, so the device reading tracks real "
                             "demand. 0.85 reproduces the sweep's pooled configuration.")
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--checkpoint-seconds", type=float, default=30.0,
                        help="rewrite --out this often with the peaks so far, so a run "
                             "killed mid-fit still leaves a lower bound behind")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--trace", action="store_true",
                        help="include the full per-sample trace in the JSON")

    args = parser.parse_args()
    if args.stage in ("umap", "both") and not args.embed_dir:
        parser.error("--embed-dir is required for --stage umap/both")
    if args.stage == "hdbscan" and not (args.coords or args.embed_dir):
        parser.error("--stage hdbscan needs --coords (or --embed-dir to derive it)")
    return args


def main() -> int:
    args = parse_args()
    payload = measure(args)
    print_report(payload)
    if args.out:
        _write_json(Path(args.out), payload)
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
