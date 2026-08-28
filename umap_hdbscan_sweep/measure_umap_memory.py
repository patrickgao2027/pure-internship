#!/usr/bin/env python
"""Measure peak GPU VRAM and host RAM for one parametric-UMAP cell, phase by phase.

The sweep artifacts record timing but never memory, so the report can only quote the
*budget caps* (``torch 6.75 GB + rmm pool 38.25 GB``) rather than what was actually used.
This script re-runs the deployed cell's code path -- ``core.fit_umap`` -> ``transform`` ->
``pu.extract_edges`` -> ``pu.train_umap_loss``, the same calls ``parametric_sweep.run_cell``
makes, in the same order -- with a sampler thread watching the card and the process.

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

Usage on miletus::

    bash umap_hdbscan_sweep/tmux_measure_umap_memory.sh

or directly::

    micromamba run -n uv_vae python umap_hdbscan_sweep/measure_umap_memory.py \\
        --embed-dir ~/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed \\
        --fit-rows 25000000 --n-neighbors 15 --min-dist 0.1 \\
        --out umap_hdbscan_sweep/umap_tests/memory_profile.json

Start small (``--fit-rows 2000000``) to confirm the harness works before spending the
~47 minutes the 25M fit costs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np

import sweep_core as core
import parametric_umap as pu

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


# ── the measured run ──────────────────────────────────────────────────────────

def measure(args) -> dict:
    import torch

    probe = MemoryProbe(interval=args.sample_interval).start()

    budget_report = None
    if args.gpu_budget_gb is not None:
        with probe.phase_scope("gpu_budget"):
            budget_report = core.apply_gpu_budget(
                "apply", budget_gb=args.gpu_budget_gb, rmm_share=args.rmm_share)
            log(f"    budget: {json.dumps(budget_report)}")

    use_gpu = core.gpu_available() and not args.force_cpu
    device = torch.device("cuda" if use_gpu else "cpu")
    log(f"backend: {'cuML/GPU' if use_gpu else 'CPU'}   device={device}")

    latent_path = Path(args.embed_dir) / "latent.npy"
    if not latent_path.exists():
        raise SystemExit(f"latent not found: {latent_path}")

    with probe.phase_scope("load_latent"):
        latent = np.load(latent_path, mmap_mode="r")
        total_rows, latent_dim = latent.shape
        log(f"    latent {total_rows:,} x {latent_dim} ({latent.dtype})")

    fit_rows = min(args.fit_rows, total_rows)
    rng = np.random.default_rng(args.seed)

    with probe.phase_scope("gather_fit_set"):
        fit_index = np.sort(rng.choice(total_rows, size=fit_rows, replace=False))
        fit_latent = np.ascontiguousarray(latent[fit_index])
        log(f"    fit set {fit_latent.shape} = {fit_latent.nbytes / GB:.2f} GB on host")

    with probe.phase_scope("gather_probe"):
        probe_index = np.sort(rng.choice(total_rows, size=args.probe_rows, replace=False))
        probe_latent = np.ascontiguousarray(latent[probe_index])

    config = core.UmapConfig(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        seed=args.seed,
    )

    with probe.phase_scope("umap_fit"):
        fitted = core.fit_umap(fit_latent, config, use_gpu=use_gpu)

    with probe.phase_scope("probe_transform"):
        cuml_probe = fitted.transform(probe_latent, batch_size=args.transform_batch_size)

    with probe.phase_scope("extract_edges"):
        fit_embedding = np.ascontiguousarray(fitted.embedding)
        edges = pu.extract_edges(fitted.model.graph_)
        negative_sample_rate = int(getattr(fitted.model, "negative_sample_rate", 5))
        log(f"    {len(edges[0]):,} edges = {edges[0].nbytes * 3 / GB:.2f} GB on host")

    with probe.phase_scope("free_cuml"):
        del fitted
        if use_gpu:
            try:
                import cupy

                cupy.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    # ── torch side, mirroring run_cell's ordering ─────────────────────────────
    if use_gpu:
        torch.cuda.reset_peak_memory_stats()

    with probe.phase_scope("upload_tensors"):
        X = torch.from_numpy(fit_latent).to(device)
        # Y is unused by the umap objective (only regress/hybrid read it) but run_cell
        # uploads it unconditionally, so it is uploaded here too -- leaving it out would
        # under-report the footprint by the size of the embedding.
        Y = torch.from_numpy(fit_embedding).to(device)
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

    with probe.phase_scope("encoder_train_umap_loss"):
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

    with probe.phase_scope("full_transform"):
        wrapper = pu.ParametricUmap(encoder, device, mode="umap")
        _ = wrapper.transform(probe_latent, batch_size=args.transform_batch_size)

    probe.stop()

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": ("Peaks are high-water marks with everything earlier still resident, not "
                 "per-phase deltas. Run with --rmm-share 0 for demand; a pool masks it."),
        "config": {
            "fit_rows": fit_rows,
            "probe_rows": args.probe_rows,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "seed": args.seed,
            "umap_steps": args.umap_steps,
            "batch_size": args.batch_size,
            "max_edges": args.max_edges,
            "gpu_budget_gb": args.gpu_budget_gb,
            "rmm_share": args.rmm_share,
            "backend": "cuml" if use_gpu else "cpu",
            "latent_rows": int(total_rows),
            "latent_dim": int(latent_dim),
        },
        "gpu_budget_report": budget_report,
        "device_total_gb": round(getattr(probe, "device_total_gb", 0.0), 2),
        "peaks": {
            "device_used_gb": round(
                max((p.get("device_used_peak_gb", 0.0) for p in probe.phases.values()),
                    default=0.0), 3),
            "process_vram_gb": round(
                max((p.get("process_vram_peak_gb", 0.0) for p in probe.phases.values()),
                    default=0.0), 3),
            "host_rss_gb": round(host_peak_rss_gb(), 3),
            "torch_allocated_gb": round(torch_peak_gb, 3),
            "torch_reserved_gb": round(torch_reserved_gb, 3),
        },
        "phases": {
            name: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in entry.items()}
            for name, entry in probe.phases.items()
        },
    }
    if args.trace:
        payload["trace"] = probe.trace
    return payload


def print_report(payload: dict) -> None:
    peaks = payload["peaks"]
    cfg = payload["config"]
    print()
    print("=" * 78)
    print(f"  UMAP memory profile -- {cfg['fit_rows']:,} fit rows, "
          f"nn={cfg['n_neighbors']} min_dist={cfg['min_dist']} ({cfg['backend']})")
    print("=" * 78)
    print(f"  device total                 {payload['device_total_gb']:>8.2f} GB")
    print(f"  PEAK device VRAM used        {peaks['device_used_gb']:>8.2f} GB   "
          f"(all processes on the card)")
    print(f"  PEAK this-process VRAM       {peaks['process_vram_gb']:>8.2f} GB   "
          f"(NVML per-pid; 0.00 = unavailable)")
    print(f"  PEAK torch allocated         {peaks['torch_allocated_gb']:>8.2f} GB")
    print(f"  PEAK torch reserved          {peaks['torch_reserved_gb']:>8.2f} GB")
    print(f"  PEAK host RSS (VmHWM)        {peaks['host_rss_gb']:>8.2f} GB")
    print()
    print(f"  {'phase':<26} {'secs':>8} {'devVRAM':>9} {'procVRAM':>9} {'hostRSS':>9}")
    print("  " + "-" * 66)
    for name, entry in payload["phases"].items():
        if name in ("between", "startup"):
            continue
        print(f"  {name:<26} {entry.get('seconds', 0):>8.1f} "
              f"{entry.get('device_used_peak_gb', 0):>9.2f} "
              f"{entry.get('process_vram_peak_gb', 0):>9.2f} "
              f"{entry.get('host_rss_peak_gb', 0):>9.2f}")
    print("  " + "-" * 66)
    print("  Columns are high-water marks (GB) with earlier allocations still resident.")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embed-dir", required=True,
                        help="stage1_embed dir holding latent.npy")
    parser.add_argument("--out", default=None, help="write the JSON payload here")
    parser.add_argument("--fit-rows", type=int, default=25_000_000)
    parser.add_argument("--probe-rows", type=int, default=500_000)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap-steps", type=int, default=30_000,
                        help="encoder training steps (sweep used 30000)")
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-edges", type=int, default=150_000_000)
    parser.add_argument("--transform-batch-size", type=int, default=2_000_000)
    parser.add_argument("--gpu-budget-gb", type=float, default=None,
                        help="omit to measure uncapped demand; 40 reproduces the sweep")
    parser.add_argument("--rmm-share", type=float, default=0.0,
                        help="0 (default) = no RMM pool, so the device reading tracks real "
                             "demand. 0.85 reproduces the sweep's pooled configuration.")
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--trace", action="store_true",
                        help="include the full per-sample trace in the JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = measure(args)
    print_report(payload)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
