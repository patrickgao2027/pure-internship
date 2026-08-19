#!/usr/bin/env python
"""Cohort-level HDBSCAN with noise-minimising parameters.

Fits on 1 M rows of the existing coords.npy, labels all 157.5 M reads via the
fast_predict RBC backend, runs SigProfiler (uv_only), saves the model, and
generates a full suite of analysis plots.

Key parameters vs the existing sweep:
  min_samples=1          most aggressive: every reachable point is absorbed
  cluster_selection_epsilon=0.05  merges micro-clusters near each other
  min_cluster_size=2500  same granularity as the recommended sweep cell

Usage (set env vars or pass flags directly):
    python umap_hdbscan_sweep/low_noise_hdbscan.py \\
        --coords   $HOME/pure-internship/umap_hdbscan_sweep/umap_tests/hdbscan_scaling/coords.npy \\
        --context  $HOME/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed/context.parquet \\
        --output-dir $HOME/pure-internship/umap_hdbscan_sweep/low_noise_hdbscan
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
_SCRIPT_DIR = Path(__file__).resolve().parent
for _c in (REPO_ROOT / "uv_vae", REPO_ROOT, REPO_ROOT / "uv_vae" / "scripts",
           REPO_ROOT / "umap_hdbscan_sweep", _SCRIPT_DIR):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import fast_predict
from hdbscan_param_sweep import (
    sbs96_counts, load_or_build_index, _to_host, Cell,
)

# ── palette (COSMIC standard) ───────────────────────────────────────────────────
SUBSTITUTIONS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
SUB_COLOURS = {"C>A": "#03BCEE", "C>G": "#000000", "C>T": "#E32926",
               "T>A": "#CAC9C9", "T>C": "#A1CE63", "T>G": "#EBC6C4"}
UV_ONLY_SIGS = ["SBS7A", "SBS7B", "SBS7C", "SBS7D", "SBS38"]
UV_ONLY_COLOURS = {"SBS7A": "#E32926", "SBS7B": "#03BCEE", "SBS7C": "#A1CE63",
                   "SBS7D": "#EBC6C4", "SBS38": "#7B4FA3"}
NOISE_COLOUR = "#cccccc"
TAB_COLOURS = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


# ── SigProfiler helpers ─────────────────────────────────────────────────────────

def load_sbs96_matrix(matrix_path: Path) -> tuple[list[str], np.ndarray]:
    """Returns (channel_names, counts) where counts is (96, n_clusters)."""
    frame = pl.read_csv(matrix_path, separator="\t")
    channels = frame[:, 0].to_list()
    counts = frame.select(frame.columns[1:]).to_numpy()  # (96, n_clusters)
    return channels, counts.astype(np.float32)


def load_sigprofiler_stats(stats_path: Path) -> dict[int, float]:
    """cluster_id -> cosine similarity from Assignment_Solution_Samples_Stats.txt."""
    frame = pl.read_csv(stats_path, separator="\t")
    # SigProfiler names this column "Sample Names" in Samples_Stats but "Samples" in
    # Activities -- take whichever is present rather than assuming.
    name_col = next((c for c in ("Sample Names", "Samples") if c in frame.columns), None)
    if name_col is None:
        return {}
    out = {}
    for row in frame.iter_rows(named=True):
        m = re.search(r"(\d+)$", str(row.get(name_col, "")))
        if m:
            cos = row.get("Cosine Similarity", None)
            if cos is not None:
                out[int(m.group(1))] = float(cos)
    return out


def load_activities(act_path: Path) -> tuple[dict[int, str], list[str]]:
    """cluster_id -> dominant signature name; also returns the ordered sig columns."""
    frame = pl.read_csv(act_path, separator="\t")
    sig_cols = frame.columns[1:]
    out = {}
    for row in frame.iter_rows(named=True):
        m = re.search(r"(\d+)$", str(row.get("Samples", "")))
        if m is None:
            continue
        counts = np.array([float(row[s]) for s in sig_cols])
        if counts.sum() > 0:
            out[int(m.group(1))] = sig_cols[int(np.argmax(counts))]
    return out, list(sig_cols)


def write_summary_row(record: dict, path: Path) -> None:
    """One row in the sweep spreadsheet's column vocabulary.

    Same names and units as ``hdbscan_param_sweep.aggregate`` produces so this run can be
    pasted straight into the comparison sheet next to the swept cells.
    """
    g = record.get("geometry", {})
    a = record.get("assignment", {})
    s = record.get("spectrum", {})

    def pct(value):
        return None if value is None else round(100 * value, 2)

    row = {
        "label": f"low_noise_mcs{record['mcs']}_ms{record['ms']}_eps{record['epsilon']}",
        "fit_rows": record["fit_rows"],
        "mcs": record["mcs"],
        "ms": record["ms"],
        "eps": record["epsilon"],
        "n_clusters": record.get("cohort_n_clusters"),
        "noise_pct": pct(record.get("cohort_noise_fraction")),
        "dbcv": g.get("dbcv"),
        "rel_val": g.get("relative_validity"),
        "neg_dbcv_pct": pct(g.get("dbcv_frac_clusters_negative")),
        "cos_mean": a.get("cosine_mean"),
        "mut_gt08_pct": pct(a.get("mutation_share_above_08")),
        "top_chan": s.get("top_channel_share_median"),
        "l1_res_pct": a.get("l1_pct_mean"),
        "fit_s": record.get("fit_seconds"),
        "label_s": record.get("label_seconds"),
    }
    pl.DataFrame({k: [v] for k, v in row.items()}).write_csv(path)
    log(f"summary_row.csv written")


def dominant_substitution(channels: list[str], counts: np.ndarray) -> dict[int, str]:
    """cluster_id -> dominant substitution type."""
    subs = [c[c.index("[") + 1:c.index("]")] for c in channels]
    out = {}
    for ci in range(counts.shape[1]):
        col = counts[:, ci]
        if col.sum() > 0:
            out[ci] = subs[int(np.argmax(col))]
    return out


# ── plotting helpers ────────────────────────────────────────────────────────────

def scatter_sample(xy: np.ndarray, labels: np.ndarray,
                   rng: np.random.Generator, max_pts: int = 2_000_000) -> tuple[np.ndarray, np.ndarray]:
    total = xy.shape[0]
    if total <= max_pts:
        return xy, labels
    idx = np.sort(rng.choice(total, size=max_pts, replace=False))
    return xy[idx], labels[idx]


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {path.name}")


def plot_cluster(xy: np.ndarray, labels: np.ndarray,
                 n_clusters: int, out_path: Path, title: str = "HDBSCAN clusters") -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    noise = labels < 0
    ax.scatter(xy[noise, 0], xy[noise, 1], c=NOISE_COLOUR, s=0.3, alpha=0.3, rasterized=True, label="noise")
    for ci in range(n_clusters):
        mask = labels == ci
        if not mask.any():
            continue
        c = TAB_COLOURS[ci % len(TAB_COLOURS)]
        ax.scatter(xy[mask, 0], xy[mask, 1], c=[c], s=0.4, alpha=0.5, rasterized=True)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.axis("off")
    save_fig(fig, out_path)


def plot_by_colour_map(xy: np.ndarray, labels: np.ndarray,
                       colour_map: dict[int, str], out_path: Path,
                       title: str, legend_items: list[tuple[str, str]] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    colours = np.array([colour_map.get(l, NOISE_COLOUR) for l in labels])
    order = np.argsort(colours == NOISE_COLOUR)[::-1]  # draw noise first (behind)
    ax.scatter(xy[order, 0], xy[order, 1], c=colours[order], s=0.4, alpha=0.5, rasterized=True)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.axis("off")
    if legend_items:
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                               markersize=8, label=lbl) for lbl, c in legend_items]
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)
    save_fig(fig, out_path)


def plot_cosine_umap(xy: np.ndarray, labels: np.ndarray,
                     cosine_map: dict[int, float], out_path: Path) -> None:
    values = np.array([cosine_map.get(l, np.nan) for l in labels])
    fig, ax = plt.subplots(figsize=(10, 8))
    noise = np.isnan(values)
    ax.scatter(xy[noise, 0], xy[noise, 1], c=NOISE_COLOUR, s=0.3, alpha=0.3, rasterized=True)
    sc = ax.scatter(xy[~noise, 0], xy[~noise, 1], c=values[~noise],
                    cmap="viridis", vmin=0, vmax=1, s=0.4, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Cosine similarity")
    ax.set_title("UMAP coloured by SigProfiler cosine similarity", fontsize=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.axis("off")
    save_fig(fig, out_path)


def plot_noise_map(xy: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    noise = labels < 0
    ax.scatter(xy[~noise, 0], xy[~noise, 1], c="#4393c3", s=0.3, alpha=0.4, rasterized=True, label="clustered")
    ax.scatter(xy[noise, 0], xy[noise, 1], c="#d6604d", s=0.4, alpha=0.6, rasterized=True, label="noise")
    ax.set_title(f"Noise map  ({noise.mean()*100:.1f}% noise)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, markerscale=6)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.axis("off")
    save_fig(fig, out_path)


def plot_centroids(xy: np.ndarray, labels: np.ndarray, n_clusters: int,
                   out_path: Path, top_n: int = 40) -> None:
    sizes = Counter(int(l) for l in labels if l >= 0)
    top = sorted(sizes, key=sizes.get, reverse=True)[:top_n]
    fig, ax = plt.subplots(figsize=(12, 10))
    noise = labels < 0
    ax.scatter(xy[noise, 0], xy[noise, 1], c=NOISE_COLOUR, s=0.2, alpha=0.2, rasterized=True)
    for ci in range(n_clusters):
        mask = labels == ci
        if not mask.any():
            continue
        c = TAB_COLOURS[ci % len(TAB_COLOURS)]
        ax.scatter(xy[mask, 0], xy[mask, 1], c=[c], s=0.3, alpha=0.4, rasterized=True)
    for ci in top:
        mask = labels == ci
        if not mask.any():
            continue
        cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
        ax.plot(cx, cy, "*", color="white", markersize=7, markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate(str(ci), (cx, cy), fontsize=5.5, ha="center", va="bottom",
                    color="black", fontweight="bold",
                    xytext=(0, 4), textcoords="offset points")
    ax.set_title(f"Cluster centroids (top {top_n} by size labelled)", fontsize=12)
    ax.axis("off")
    save_fig(fig, out_path)


def plot_cosine_contact_sheet(cosine_map: dict[int, float], n_clusters: int,
                               out_path: Path) -> None:
    ids = sorted(cosine_map)
    values = [cosine_map[i] for i in ids]
    colours = ["#2ca02c" if v >= 0.8 else "#ff7f0e" if v >= 0.5 else "#d62728" for v in values]
    fig, ax = plt.subplots(figsize=(max(12, len(ids) * 0.12), 4))
    ax.bar(range(len(ids)), values, color=colours, width=1.0, linewidth=0)
    ax.axhline(0.8, color="black", lw=0.8, ls="--", label="0.8 threshold")
    ax.axhline(0.5, color="grey", lw=0.6, ls="--", label="0.5 threshold")
    ax.set_xlabel("Cluster ID"); ax.set_ylabel("Cosine similarity")
    ax.set_title(f"SigProfiler cosine similarity per cluster  (n={len(ids)})", fontsize=12)
    ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_xticks(range(len(ids))); ax.set_xticklabels([str(i) for i in ids], fontsize=4, rotation=90)
    save_fig(fig, out_path)


def plot_sbs96_contact_sheet(channels: list[str], counts: np.ndarray,
                              n_clusters: int, out_path: Path,
                              max_clusters: int = 60) -> None:
    sub_of = [c[c.index("[") + 1:c.index("]")] for c in channels]
    colours_per_channel = [SUB_COLOURS.get(s, "#888888") for s in sub_of]
    cols_to_plot = min(n_clusters, max_clusters)
    ncols = 10
    nrows = (cols_to_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.0))
    axes_flat = np.array(axes).flatten()
    for idx in range(cols_to_plot):
        ax = axes_flat[idx]
        col = counts[:, idx]
        total = col.sum()
        if total > 0:
            col = col / total
        ax.bar(range(96), col, color=colours_per_channel, linewidth=0, width=1.0)
        ax.set_title(f"cluster {idx}", fontsize=6)
        ax.set_xticks([]); ax.tick_params(labelsize=4)
    for idx in range(cols_to_plot, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.suptitle(f"SBS96 spectra (first {cols_to_plot} clusters)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, out_path, dpi=120)


def plot_cluster_size_hist(labels: np.ndarray, out_path: Path) -> None:
    sizes = np.bincount(labels[labels >= 0].astype(np.int64))
    sizes = sizes[sizes > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.log10(sizes + 1), bins=40, color="#4393c3", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("log10(reads per cluster + 1)"); ax.set_ylabel("Number of clusters")
    ax.set_title(f"Cluster size distribution  (n={len(sizes)} clusters)", fontsize=12)
    save_fig(fig, out_path)


def plot_condensed_tree(clusterer, out_path: Path) -> None:
    try:
        from hdbscan.plots import CondensedTree
        ct = CondensedTree(clusterer.condensed_tree_, cluster_selection_method="eom")
        ax = ct.plot(select_clusters=True)
        ax.set_title("HDBSCAN condensed tree")
        save_fig(ax.figure, out_path)
    except Exception as exc:
        log(f"  condensed tree skipped: {exc}")


# ── main ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coords", required=True)
    p.add_argument("--context", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fit-rows", type=int, default=1_000_000)
    p.add_argument("--mcs", type=int, default=2500, help="min_cluster_size")
    p.add_argument("--ms", type=int, default=1, help="min_samples")
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-rows", type=int, default=2_000_000, help="rows to plot")
    p.add_argument("--sigprofiler-cpu", type=int, default=16)
    p.add_argument("--genome-build", default="GRCh38")
    p.add_argument("--cosmic-version", default="3.5")
    p.add_argument("--sbs96-index", default=None, help="path to existing sbs96_index.npy to skip rebuild")
    p.add_argument("--cluster-backend", choices=["auto", "gpu", "cpu"], default="auto")
    p.add_argument("--predict-backend", choices=["rbc", "brute", "sklearn"], default="rbc",
                   help="fast_predict kNN backend: rbc=cuML RBC (GPU, default), sklearn=CPU KDTree")
    p.add_argument("--batch-rows", type=int, default=5_000_000)
    p.add_argument("--threads", type=int, default=1, help="CPU threads for CPU hdbscan")
    p.add_argument("--skip-sigprofiler", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--skip-metrics", action="store_true",
                   help="skip the geometry/assignment/spectrum summarisers")
    # DBCV / connectivity budgets -- same defaults as hdbscan_param_sweep so the numbers
    # are comparable against the swept cells.
    p.add_argument("--connectivity-rows", type=int, default=200_000)
    p.add_argument("--dbcv-rows", type=int, default=25_000)
    p.add_argument("--dbcv-per-cluster", type=int, default=400,
                   help="points drawn from EVERY cluster; 0 falls back to --dbcv-rows total")
    p.add_argument("--dbcv-backend", default="auto", choices=["auto", "kdbcv", "hdbscan"])
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── load coords ─────────────────────────────────────────────────────────────
    log("Loading UMAP coords …")
    coords = np.load(args.coords, mmap_mode="r")
    total_rows = coords.shape[0]
    log(f"  {total_rows:,} rows × {coords.shape[1]} dims")

    # ── sample fit indices ───────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    fit_indices = np.sort(rng.choice(total_rows, size=min(args.fit_rows, total_rows), replace=False))
    fit_coords = np.asarray(coords[fit_indices], dtype=np.float32)
    log(f"  fit set: {fit_coords.shape[0]:,} rows")

    # ── fit HDBSCAN ──────────────────────────────────────────────────────────────
    cell = Cell(fit_rows=args.fit_rows, min_cluster_size=args.mcs, min_samples=args.ms,
                cluster_selection_method="eom", cluster_selection_epsilon=args.epsilon)

    class _HdbArgs:
        cluster_backend = args.cluster_backend
        threads = args.threads
        no_min_span_tree = False
    from hdbscan_param_sweep import _fit_hdbscan

    log(f"Fitting HDBSCAN  mcs={args.mcs}  ms={args.ms}  eps={args.epsilon} …")
    t0 = perf_counter()
    clusterer = _fit_hdbscan(fit_coords, cell, _HdbArgs())
    fit_seconds = perf_counter() - t0
    fit_labels = np.asarray(_to_host(clusterer.labels_)).astype(np.int32)
    n_clusters_fit = int(fit_labels.max()) + 1 if (fit_labels >= 0).any() else 0
    log(f"  {fit_seconds:.1f}s  {n_clusters_fit} clusters  "
        f"{(fit_labels < 0).mean()*100:.2f}% noise on fit set")

    # ── save model ───────────────────────────────────────────────────────────────
    import joblib
    model_path = out / "hdbscan_model.pkl"
    try:
        joblib.dump(clusterer, model_path)
        # The fit indices travel with the model: fast_predict's tables index into the fit
        # set positionally, so any consumer (per_parquet_inference) needs these exact rows
        # in this exact order rather than re-deriving them and hoping the seed matched.
        np.save(out / "fit_indices.npy", fit_indices)
        log(f"  model + fit_indices saved → {model_path.name}")
    except Exception as exc:
        log(f"  WARNING: model save failed ({exc})")

    # ── label all 157.5M rows ───────────────────────────────────────────────────
    log(f"Labelling {total_rows:,} rows (RBC fast_predict) …")
    t0 = perf_counter()
    tables = fast_predict.build_tables(clusterer, n_fit=fit_coords.shape[0],
                                       min_samples=args.ms)
    held_mask = np.ones(total_rows, dtype=bool)
    held_mask[fit_indices] = False
    held_indices = np.nonzero(held_mask)[0]

    labels = np.empty(total_rows, dtype=np.int32)
    probabilities = np.empty(total_rows, dtype=np.float32)
    labels[fit_indices] = fit_labels
    probabilities[fit_indices] = np.asarray(_to_host(
        getattr(clusterer, "probabilities_", np.ones(len(fit_indices), np.float32))
    ), dtype=np.float32)

    predict_backend = args.predict_backend
    if predict_backend == "rbc":
        try:
            import cuml  # noqa: F401
        except ImportError:
            log("  cuML not found, falling back to sklearn KDTree for fast_predict")
            predict_backend = "sklearn"
    index = fast_predict.build_index(fit_coords, 2 * args.ms, predict_backend)
    held_labels, held_probs = fast_predict.predict(
        tables, fit_coords,
        np.asarray(coords[held_indices], dtype=np.float32),
        backend=predict_backend, batch_rows=args.batch_rows, index=index)
    labels[held_indices] = held_labels
    probabilities[held_indices] = held_probs
    label_seconds = perf_counter() - t0

    cohort_n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
    cohort_noise = float((labels < 0).mean())
    log(f"  {label_seconds:.1f}s  {cohort_n_clusters} clusters  {cohort_noise*100:.2f}% cohort noise")

    np.save(out / "cohort_labels.npy", labels)
    np.save(out / "cohort_probabilities.npy", probabilities)

    # plot_feature_atlas.py --cluster-labels wants a parquet with a `cluster_label` column
    # in full-cohort row order (it does its own positional sampling), not the .npy.
    cluster_labels_parquet = out / "cluster_labels.parquet"
    pl.DataFrame({"cluster_label": labels}).write_parquet(cluster_labels_parquet)
    log(f"  cluster_labels.parquet written for the feature atlas")

    # ── model artefacts ─────────────────────────────────────────────────────────
    for name, attr, dtype in [("cluster_persistence", "cluster_persistence_", np.float64),
                               ("outlier_scores", "outlier_scores_", np.float32)]:
        try:
            val = getattr(clusterer, attr, None)
            if val is not None:
                np.save(out / f"{name}.npy", np.asarray(_to_host(val), dtype=dtype))
        except Exception:
            pass
    try:
        ex = getattr(clusterer, "exemplars_", None)
        if ex is not None:
            blocks = {f"cluster_{i}": np.asarray(_to_host(b), dtype=np.float32)
                      for i, b in enumerate(ex)}
            np.savez_compressed(out / "exemplars.npz", **blocks)
    except Exception:
        pass

    # ── SBS96 / SigProfiler ─────────────────────────────────────────────────────
    record: dict = {
        "mcs": args.mcs, "ms": args.ms, "epsilon": args.epsilon,
        "fit_rows": args.fit_rows, "fit_seconds": round(fit_seconds, 1),
        "n_clusters_fit": n_clusters_fit,
        "label_seconds": round(label_seconds, 1),
        "cohort_n_clusters": cohort_n_clusters,
        "cohort_noise_fraction": cohort_noise,
        "coords_path": args.coords,
        "cohort_labels": str(out / "cohort_labels.npy"),
    }

    sigprof_root = None
    if not args.skip_sigprofiler:
        import run_variant_cluster_pipeline as rvcp

        sig_db_dir = out / "signature_db"
        sig_db_dir.mkdir(exist_ok=True)
        sig_db = sig_db_dir / f"uv_only_SBS_{args.genome_build}.tsv"
        row_order = rvcp.write_uv_only_signature_database(
            args.genome_build, args.cosmic_version, sig_db)

        # Use existing sbs96_index if provided, else build/cache in output dir
        index_src = Path(args.sbs96_index) if args.sbs96_index and Path(args.sbs96_index).exists() else None
        if index_src:
            import shutil
            cache = out / "sbs96_index.npy"
            if not cache.exists():
                shutil.copy2(index_src, cache)
            sbs96_index = np.load(cache, mmap_mode="r")
            log(f"SBS96 index loaded from {index_src.name}")
        else:
            sbs96_index = load_or_build_index(args.context, row_order, out)

        log(f"Building SBS96 counts ({cohort_n_clusters} clusters) …")
        t0 = perf_counter()
        counts_matrix = sbs96_counts(sbs96_index, labels, cohort_n_clusters)
        present = np.nonzero(counts_matrix.sum(axis=0) > 0)[0]
        cluster_cols = [f"cluster_{int(c)}" for c in present]
        counts_present = counts_matrix[:, present]
        log(f"  {perf_counter()-t0:.1f}s  {counts_present.sum():,} total mutations")

        sigprof_root = out / f"sigprofilerassignment_uv_only_{args.genome_build.lower()}_v{args.cosmic_version}"
        input_dir = sigprof_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = input_dir / "cluster_sbs96_matrix.tsv"
        pl.DataFrame({"Type": row_order, **{c: counts_present[:, i].tolist()
                                             for i, c in enumerate(cluster_cols)}}
                     ).write_csv(matrix_path, separator="\t")
        record["sbs96_matrix"] = str(matrix_path)

        log(f"Running SigProfiler ({args.sigprofiler_cpu} CPUs) …")
        t0 = perf_counter()
        rvcp.Analyzer.cosmic_fit(
            samples=str(matrix_path),
            output=str(sigprof_root / "output"),
            signature_database=str(sig_db),
            genome_build=args.genome_build, cosmic_version=float(args.cosmic_version),
            make_plots=False, collapse_to_SBS96=True, connected_sigs=False, verbose=False,
            input_type="matrix", context_type="96", export_probabilities=True,
            sample_reconstruction_plots=False, cpu=args.sigprofiler_cpu,
            add_background_signatures=False,
        )
        record["sigprofiler_seconds"] = round(perf_counter() - t0, 1)
        log(f"  SigProfiler done in {record['sigprofiler_seconds']}s")

    # ── geometry / assignment / spectrum metrics ────────────────────────────────
    # The same three summarisers hdbscan_param_sweep uses, so this run lands in the
    # sweep spreadsheet with identical column semantics. Each is diagnostic-only:
    # a failure is recorded and swallowed rather than losing the fit above.
    if not args.skip_metrics:
        # Lazy import: cluster_quality lives in umap_hdbscan_sweep/ (same dir as this
        # script). If sys.path doesn't reach it for any reason, record the error and
        # continue rather than crashing the whole run.
        sys.path.insert(0, str(_SCRIPT_DIR / "hdbscan"))
        try:
            import cluster_quality
        except ImportError as _e:
            log(f"  cluster_quality not found on sys.path: {_e}")
            log(f"  sys.path = {sys.path[:8]}")
            record["geometry_error"] = f"ImportError: {_e}"
            args.skip_metrics = True
        if not args.skip_metrics:
            pass  # continues below
    if not args.skip_metrics:
        log("Scoring geometry (DBCV, relative validity, connectivity) …")
        t0 = perf_counter()
        try:
            probs_fit = getattr(clusterer, "probabilities_", None)
            record["geometry"] = cluster_quality.summarise(
                fit_coords, fit_labels,
                probabilities=None if probs_fit is None else _to_host(probs_fit),
                clusterer=clusterer,
                connectivity_rows=args.connectivity_rows,
                dbcv_rows=args.dbcv_rows,
                dbcv_max_clusters=None,
                dbcv_stratified=True,
                dbcv_per_cluster=args.dbcv_per_cluster or None,
                dbcv_backend=args.dbcv_backend,
                seed=args.seed)
            record["geometry_seconds"] = round(perf_counter() - t0, 1)
            g = record["geometry"]
            log(f"  DBCV {g.get('dbcv')}  rel_validity {g.get('relative_validity')}  "
                f"neg-DBCV {100 * g.get('dbcv_frac_clusters_negative', float('nan')):.1f}%")
        except Exception as exc:  # noqa: BLE001
            record["geometry_error"] = f"{type(exc).__name__}: {exc}"
            log(f"  geometry metrics failed: {type(exc).__name__}: {str(exc)[:200]}")

        if sigprof_root is not None:
            import assignment_metrics
            import spectrum_metrics
            stats_path = (sigprof_root / "output" / "Assignment_Solution" / "Solution_Stats"
                          / "Assignment_Solution_Samples_Stats.txt")
            try:
                if stats_path.exists():
                    record["assignment"] = assignment_metrics.summarise(stats_path)
                    a = record["assignment"]
                    log(f"  cos_mean {a.get('cosine_mean'):.4f}  "
                        f"mut>0.8 {100 * a.get('mutation_share_above_08', 0):.2f}%  "
                        f"L1res {a.get('l1_pct_mean'):.1f}%")
            except Exception as exc:  # noqa: BLE001
                record["assignment_error"] = f"{type(exc).__name__}: {exc}"
                log(f"  assignment metrics failed: {type(exc).__name__}: {str(exc)[:200]}")
            try:
                record["spectrum"] = spectrum_metrics.summarise(matrix_path, seed=args.seed)
                log(f"  top_chan {record['spectrum'].get('top_channel_share_median'):.4f}")
            except Exception as exc:  # noqa: BLE001
                record["spectrum_error"] = f"{type(exc).__name__}: {exc}"
                log(f"  spectrum metrics failed: {type(exc).__name__}: {str(exc)[:200]}")

        write_summary_row(record, out / "summary_row.csv")

    (out / "metrics.json").write_text(json.dumps(record, indent=2))
    log(f"metrics.json written")

    # ── plots ───────────────────────────────────────────────────────────────────
    if not args.skip_plots:
        log("Generating plots …")
        plot_rng = np.random.default_rng(args.seed)
        xy_s, lbl_s = scatter_sample(
            np.asarray(coords, dtype=np.float32), labels, plot_rng, args.sample_rows)

        plot_cluster(xy_s, lbl_s, cohort_n_clusters,
                     plots_dir / "umap_cluster.png",
                     f"HDBSCAN clusters (mcs={args.mcs}, ms={args.ms}, ε={args.epsilon})")

        plot_noise_map(xy_s, lbl_s, plots_dir / "umap_noise.png")

        plot_centroids(xy_s, lbl_s, cohort_n_clusters,
                       plots_dir / "cluster_centroid_annotation.png")

        plot_cluster_size_hist(labels, plots_dir / "cluster_size_hist.png")

        if sigprof_root is not None:
            matrix_path = sigprof_root / "input" / "cluster_sbs96_matrix.tsv"
            act_path = (sigprof_root / "output" / "Assignment_Solution" / "Activities"
                        / "Assignment_Solution_Activities.txt")
            stats_path = (sigprof_root / "output" / "Assignment_Solution" / "Solution_Stats"
                          / "Assignment_Solution_Samples_Stats.txt")

            if matrix_path.exists():
                channels, counts_mat = load_sbs96_matrix(matrix_path)
                dom_sub = dominant_substitution(channels, counts_mat)
                sub_colour_map = {ci: SUB_COLOURS.get(s, NOISE_COLOUR) for ci, s in dom_sub.items()}
                plot_by_colour_map(
                    xy_s, lbl_s, sub_colour_map, plots_dir / "umap_substitution.png",
                    "UMAP coloured by dominant SBS96 substitution",
                    [(s, SUB_COLOURS[s]) for s in SUBSTITUTIONS])

                plot_sbs96_contact_sheet(channels, counts_mat, counts_mat.shape[1],
                                         plots_dir / "sbs96_contact_sheet.png")

            if act_path.exists():
                dom_sig, _ = load_activities(act_path)
                sig_colour_map = {ci: UV_ONLY_COLOURS.get(s, NOISE_COLOUR) for ci, s in dom_sig.items()}
                plot_by_colour_map(
                    xy_s, lbl_s, sig_colour_map, plots_dir / "umap_sigprofiler.png",
                    "UMAP coloured by dominant SigProfiler signature (uv_only)",
                    [(s, UV_ONLY_COLOURS[s]) for s in UV_ONLY_SIGS])

            if stats_path.exists():
                cosine_map = load_sigprofiler_stats(stats_path)
                plot_cosine_umap(xy_s, lbl_s, cosine_map, plots_dir / "umap_cosine.png")
                plot_cosine_contact_sheet(cosine_map, cohort_n_clusters,
                                          plots_dir / "sigprofiler_cosine_contact_sheet.png")

        plot_condensed_tree(clusterer, plots_dir / "condensed_tree.png")

    log(f"\nDone → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
