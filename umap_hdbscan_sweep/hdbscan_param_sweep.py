#!/usr/bin/env python
"""HDBSCAN parameter sweep with full-cohort SigProfiler on every cell.

What changed to make this possible
----------------------------------
``HDBSCAN_SWEEP_PLAN.md`` Phase 3 says to carry only 2-3 candidates to SigProfiler, because
labelling the cohort cost ~4.8 h per cell (``approximate_predict`` is brute force: 109 s per
million probe rows against a 25M fit set). ``fast_predict.py`` replaces the neighbour search
with cuML's Random Ball Cover and does the same labelling in 5.3 min -- 54x. SigProfiler is
therefore no longer a shortlist step, and every cell in this sweep gets a real full-cohort
signature assignment rather than a proxy metric.

Two more costs are removed by precomputation:

* **SBS96 context.** ``stage1_embed`` already wrote ``context.parquet`` row-aligned with
  ``latent.npy`` (and so with ``coords.npy``). Its REF/ALT/X_PREV1/X_NEXT1 collapse to one
  int8 per row, cached once as ``sbs96_index.npy`` (157 MB). Per cell the 96 x n_clusters
  matrix is then a single ``np.add.at`` -- seconds, no parquet re-read. ``stage3_apply_full``
  needs a 5B-read pass for the read-weighted version; ``locus_reads`` in the same parquet
  gives the locus-weighted answer here for free.
* **Fit-set draws.** A given ``fit_rows`` always draws the same rows (seeded), so cells that
  differ only in ``min_cluster_size`` are comparable by construction.

The confound this sweep exists to break
---------------------------------------
``scaling_results.json`` swept fit size with ``min_cluster_size = max(50, N * 1e-4)`` --
proportional. Cluster count fell 1338 -> 442 and noise fell 32.6% -> 5.6% from 500K to 25M,
but mcs rose 50x over the same range, so **nothing in that run separates "more data resolves
coarser structure" from "mcs got 50x bigger"**. Here mcs is an independent axis held fixed
across fit sizes.

Cost model (measured, from scaling_results.json -- fit time is O(N^1.9) and dominates):

    500K  7 s      1M  24 s      5M  470 s     10M  1,870 s
    15M  4,562 s   25M  11,684 s              50M  OOM at 46,450 s

Per cell add ~5.3 min RBC labelling + ~1-2 min SigProfiler. min_samples=15 at 25M OOMed on a
47 GB card, so the 25M row of the grid is min_samples=5 only.

    # one-time: build the SBS96 index cache (~10 min over 157.5M rows), then exit
    python umap_hdbscan_sweep/hdbscan_param_sweep.py --build-index-only \
        --context <embed_dir>/context.parquet --output-dir <out>

    # print the grid and its projected cost, run nothing
    python umap_hdbscan_sweep/hdbscan_param_sweep.py --dry-run \
        --coords coords.npy --context <embed_dir>/context.parquet --output-dir <out>

    # the sweep (resumable: finished cells are skipped)
    python umap_hdbscan_sweep/hdbscan_param_sweep.py \
        --coords coords.npy --context <embed_dir>/context.parquet --output-dir <out>

    # rank finished cells
    python umap_hdbscan_sweep/hdbscan_param_sweep.py --aggregate --output-dir <out>

Every cell writes ``<out>/cells/<label>/metrics.json`` the moment it finishes, plus its
labels, count matrix and SigProfiler tree. A crash loses at most the cell in flight.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, REPO_ROOT / "uv_vae" / "scripts",
                  Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import polars as pl

# The SBS96 target band, from HDBSCAN_SWEEP_PLAN.md Finding 2. Clusters below the floor fit
# badly in every cell that has any (2-7% above cosine 0.7, vs 11-14% for the band); clusters
# above the ceiling are signature MIXTURES no single reference profile fits (0-10%, median
# cosine 0.04-0.12). Both were established by de-confounding within mcs cells, so they are
# properties of cluster size, not of the parameter that produced it.
MUTATION_FLOOR = 3_000
MUTATION_CEILING = 300_000
COSINE_THRESHOLDS = (0.7, 0.8)

# SBS7A-D are the UV signatures; whether they land in DISTINCT clusters is the biological
# read-out the plan asks for (Phase 3), not a generic quality score.
UV_SIGNATURES = ("SBS7a", "SBS7b", "SBS7c", "SBS7d")


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}", flush=True)


# ── SBS96 context, precomputed once ────────────────────────────────────────────

def sbs96_expr() -> pl.Expr:
    """Canonical SBS96 label. Identical to stage3_apply_full.sbs96_expr and to
    run_variant_cluster_pipeline.annotate_trinuc_counts -- purine references are reverse
    complemented onto the pyrimidine strand, which SWAPS the flanking bases as well as
    complementing them."""
    def complement(expr: pl.Expr) -> pl.Expr:
        result = pl.when(expr == pl.lit("A")).then(pl.lit("T"))
        for base, other in [("C", "G"), ("G", "C"), ("T", "A")]:
            result = result.when(expr == pl.lit(base)).then(pl.lit(other))
        return result.otherwise(pl.lit(None))

    ref, alt = pl.col("REF"), pl.col("ALT")
    previous, following = pl.col("X_PREV1"), pl.col("X_NEXT1")
    is_pyrimidine = ref.is_in(["C", "T"])

    canonical_ref = pl.when(is_pyrimidine).then(ref).otherwise(complement(ref))
    canonical_alt = pl.when(is_pyrimidine).then(alt).otherwise(complement(alt))
    canonical_prev = pl.when(is_pyrimidine).then(previous).otherwise(complement(following))
    canonical_next = pl.when(is_pyrimidine).then(following).otherwise(complement(previous))

    substitution = canonical_ref + pl.lit(">") + canonical_alt
    return (
        pl.when(substitution.is_in(["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]))
        .then(canonical_prev + pl.lit("[") + substitution + pl.lit("]") + canonical_next)
        .otherwise(pl.lit(None))
    )


def canonical_sbs96_channels() -> list[str]:
    """The 96 pyrimidine-reference channels, generated rather than typed out."""
    bases = "ACGT"
    return [f"{previous}[{reference}>{alternate}]{following}"
            for reference, alternates in (("C", "AGT"), ("T", "ACG"))
            for alternate in alternates
            for previous in bases
            for following in bases]


def validate_row_order(row_order: list[str]) -> None:
    """Every channel the encoder can emit must exist in the reference's row order.

    A missing channel does not raise anywhere downstream -- ``position.get(label, -1)``
    turns it into a dropped row, so the matrix stays well-formed and the mutation counts are
    simply short. Catching it here is the only place it is visible.
    """
    missing = sorted(set(canonical_sbs96_channels()) - set(row_order))
    if missing:
        raise SystemExit(
            f"the signature reference is missing {len(missing)} of the 96 SBS channels "
            f"(e.g. {missing[:3]}). Counts for those contexts would be silently dropped."
        )


def build_sbs96_index(context_path: Path, row_order: list[str], cache_path: Path,
                      reads_cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """One int8 per cohort row: its position in the 96-channel order, or -1.

    Streamed by row group so peak memory is one group rather than 157.5M strings. The result
    is what turns each cell's count matrix into a scatter-add instead of a parquet scan.
    """
    import pyarrow.parquet as pq

    validate_row_order(row_order)
    position = {label: index for index, label in enumerate(row_order)}
    parquet_file = pq.ParquetFile(context_path)
    total = parquet_file.metadata.num_rows
    log(f"building SBS96 index over {total:,} rows from {context_path.name}")

    index = np.empty(total, dtype=np.int8)
    reads = np.empty(total, dtype=np.int32)
    columns = ["REF", "ALT", "X_PREV1", "X_NEXT1", "locus_reads"]
    available = set(parquet_file.schema_arrow.names)
    if "locus_reads" not in available:
        columns.remove("locus_reads")

    written = 0
    for group in range(parquet_file.num_row_groups):
        frame = pl.from_arrow(parquet_file.read_row_group(group, columns=columns))
        labels = frame.select(sbs96_expr().alias("sbs96"))["sbs96"].to_numpy()
        codes = np.fromiter((position.get(label, -1) for label in labels),
                            dtype=np.int8, count=len(labels))
        index[written:written + codes.size] = codes
        if "locus_reads" in frame.columns:
            reads[written:written + codes.size] = frame["locus_reads"].to_numpy()
        else:
            reads[written:written + codes.size] = 1
        written += codes.size
        if group % 50 == 0:
            log(f"  {written:,} / {total:,}")

    if written != total:
        raise RuntimeError(f"read {written:,} rows from {context_path} but metadata said {total:,}")
    np.save(cache_path, index)
    np.save(reads_cache_path, reads)
    log(f"  {float((index >= 0).mean()) * 100:.2f}% of rows carry an SBS96 context "
        f"-> {cache_path.name}")
    return index, reads


def load_or_build_index(context_path: Path, row_order: list[str],
                        output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    cache_path = output_dir / "sbs96_index.npy"
    reads_cache_path = output_dir / "locus_reads.npy"
    if cache_path.exists() and reads_cache_path.exists():
        index = np.load(cache_path, mmap_mode="r")
        reads = np.load(reads_cache_path, mmap_mode="r")
        log(f"SBS96 index cache hit: {index.shape[0]:,} rows")
        return index, reads
    if context_path is None or not Path(context_path).exists():
        raise SystemExit(
            f"no SBS96 index cache at {cache_path} and --context was not given or does not "
            f"exist. Build it once with --build-index-only --context <embed_dir>/context.parquet"
        )
    return build_sbs96_index(Path(context_path), row_order, cache_path, reads_cache_path)


# ── the grid ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Cell:
    fit_rows: int
    min_cluster_size: int
    min_samples: int
    cluster_selection_method: str = "eom"
    cluster_selection_epsilon: float = 0.0

    def label(self) -> str:
        name = (f"fit{self.fit_rows}_mcs{self.min_cluster_size}"
                f"_ms{self.min_samples}_{self.cluster_selection_method}")
        if self.cluster_selection_epsilon:
            name += f"_eps{self.cluster_selection_epsilon}"
        return name


# Measured fit seconds from scaling_results.json, for the cost projection. Interpolated in
# log-log between the measured points (the curve is a clean power law, ~O(N^1.9)).
MEASURED_FIT_SECONDS = {
    500_000: 7.2, 1_000_000: 23.6, 5_000_000: 469.9,
    10_000_000: 1869.6, 15_000_000: 4562.1, 25_000_000: 11684.0,
}


def projected_fit_seconds(fit_rows: int, min_samples: int) -> float:
    sizes = np.array(sorted(MEASURED_FIT_SECONDS))
    seconds = np.array([MEASURED_FIT_SECONDS[size] for size in sizes])
    estimate = float(np.exp(np.interp(np.log(fit_rows), np.log(sizes), np.log(seconds))))
    # min_samples enlarges the kNN graph; ms=15 at 25M OOMed, and the ms=15 probe that did
    # run took 11,714 s against 11,684 s at ms=5, so the time penalty is small even though
    # the memory penalty is not.
    return estimate * (1.0 + 0.02 * (min_samples - 5))


def build_grid(fit_sizes: list[int], min_cluster_sizes: list[int], min_samples: list[int],
               methods: list[str], epsilons: list[float],
               max_min_samples_at: dict[int, int]) -> list[Cell]:
    """Grid ordered so that the cheapest cells run first.

    Cheapest-first is deliberate: it front-loads the information. If the 500K and 1M rows
    show mcs doing nothing (which HDBSCAN_SWEEP_PLAN.md Finding 1 predicts), that is known
    within minutes rather than after the 3.25 h 25M fits.
    """
    cells = []
    for fit_rows in fit_sizes:
        ceiling = max_min_samples_at.get(fit_rows)
        for samples in min_samples:
            if ceiling is not None and samples > ceiling:
                continue
            for mcs in min_cluster_sizes:
                for method in methods:
                    for epsilon in epsilons:
                        cells.append(Cell(fit_rows, mcs, samples, method, epsilon))
    return sorted(cells, key=lambda cell: (cell.fit_rows, cell.min_samples,
                                           cell.min_cluster_size))


# ── metrics ────────────────────────────────────────────────────────────────────

def cluster_size_stats(labels: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Realised cluster sizes over the FULL cohort, which is what the decision rule uses.

    HDBSCAN_SWEEP_PLAN.md Finding 3: fit-set cluster sizes cannot be scaled up to predict
    these. The smallest mcs=100 cluster held 174 cohort mutations, not the 3,150 a 31.5x
    scale factor implies, because approximate_predict sends boundary rows to noise and small
    clusters attract disproportionately few held-out rows. So this is measured, never derived.
    """
    clustered = labels >= 0
    if not clustered.any():
        return {"n_clusters": 0, "noise_fraction": 1.0}

    if weights is None:
        sizes = np.bincount(labels[clustered])
    else:
        sizes = np.bincount(labels[clustered], weights=weights[clustered].astype(np.float64))
    sizes = sizes[sizes > 0]

    total = float(sizes.sum())
    in_band = sizes[(sizes >= MUTATION_FLOOR) & (sizes <= MUTATION_CEILING)]
    proportions = sizes / total
    return {
        "n_clusters": int(sizes.size),
        "noise_fraction": float((~clustered).mean()),
        "min": float(sizes.min()),
        "p10": float(np.percentile(sizes, 10)),
        "median": float(np.median(sizes)),
        "p90": float(np.percentile(sizes, 90)),
        "max": float(sizes.max()),
        "total_assigned": total,
        "clusters_below_floor": int((sizes < MUTATION_FLOOR).sum()),
        "clusters_above_ceiling": int((sizes > MUTATION_CEILING).sum()),
        # The headline: how much of the cohort sits in the band that fits well.
        "mutation_share_in_band": float(in_band.sum() / total) if total else 0.0,
        # Balance. A cell whose clusters are one giant plus a tail is worse than an even
        # split at the same count, and cluster count alone does not show that.
        "size_entropy_normalised": float(
            -(proportions * np.log(proportions)).sum() / np.log(sizes.size)
        ) if sizes.size > 1 else 0.0,
    }


def sbs96_counts(sbs96_index: np.ndarray, labels: np.ndarray, n_clusters: int,
                 weights: np.ndarray | None = None, chunk_rows: int = 20_000_000) -> np.ndarray:
    """96 x n_clusters counts over the whole cohort.

    ``bincount`` on a flattened (channel, cluster) index rather than ``np.add.at``: the
    latter is unbuffered and runs at roughly a million updates a second, which is minutes
    per cell over 157.5M rows and is paid once per cell. Same arithmetic, ~50x faster.

    Chunked because the flattened index is int64 -- one 157.5M-row array would be 1.3 GB on
    its own, and it is scratch space nothing reads back.
    """
    counts = np.zeros(96 * n_clusters, dtype=np.int64)
    for start in range(0, labels.shape[0], chunk_rows):
        stop = min(start + chunk_rows, labels.shape[0])
        channels = np.asarray(sbs96_index[start:stop]).astype(np.int64)
        clusters = np.asarray(labels[start:stop]).astype(np.int64)
        keep = (channels >= 0) & (clusters >= 0) & (clusters < n_clusters)
        if not keep.any():
            continue
        flat = channels[keep] * n_clusters + clusters[keep]
        if weights is None:
            counts += np.bincount(flat, minlength=counts.size)
        else:
            counts += np.bincount(
                flat, weights=np.asarray(weights[start:stop])[keep].astype(np.float64),
                minlength=counts.size).astype(np.int64)
    return counts.reshape(96, n_clusters)


def sigprofiler_metrics(output_dir: Path, cluster_columns: list[str],
                        cluster_mutations: dict[str, float]) -> dict:
    """Turn SigProfiler's per-sample stats into the numbers the decision rule needs.

    The metric that matters is MUTATION share above a cosine threshold, not cluster count
    above it: HDBSCAN_SWEEP_PLAN.md Finding 1 measured cluster counts of 544 vs 68 across a
    25x mcs range carrying the same ~10.5% of mutations. Counting clusters would have called
    those cells very different; counting mutations correctly calls them identical.
    """
    stats_path = (output_dir / "Assignment_Solution" / "Solution_Stats"
                  / "Assignment_Solution_Samples_Stats.txt")
    if not stats_path.exists():
        return {"error": f"no stats at {stats_path}"}

    stats = pl.read_csv(stats_path, separator="\t")
    cosine_column = next((c for c in stats.columns if "Cosine" in c), None)
    sample_column = stats.columns[0]
    if cosine_column is None:
        return {"error": f"no cosine column in {stats.columns}"}

    cosines = stats[cosine_column].to_numpy().astype(np.float64)
    samples = stats[sample_column].to_list()
    mutations = np.array([cluster_mutations.get(str(s), 0.0) for s in samples], dtype=np.float64)
    total_mutations = mutations.sum()

    metrics = {
        "n_clusters_assigned": int(len(samples)),
        "cosine_median": float(np.median(cosines)),
        "cosine_mean": float(cosines.mean()),
        "cosine_p10": float(np.percentile(cosines, 10)),
        "cosine_p90": float(np.percentile(cosines, 90)),
        # Size-weighted: the cohort-level quality, not the per-cluster average, so a cell is
        # not rewarded for a long tail of tiny well-fit clusters.
        "cosine_mutation_weighted_mean": float((cosines * mutations).sum() / total_mutations)
        if total_mutations else 0.0,
    }
    for threshold in COSINE_THRESHOLDS:
        above = cosines > threshold
        key = str(threshold).replace(".", "")
        metrics[f"clusters_above_{key}"] = int(above.sum())
        metrics[f"mutation_share_above_{key}"] = (
            float(mutations[above].sum() / total_mutations) if total_mutations else 0.0)

    activities_path = (output_dir / "Assignment_Solution" / "Activities"
                       / "Assignment_Solution_Activities.txt")
    if activities_path.exists():
        activities = pl.read_csv(activities_path, separator="\t")
        signature_columns = [c for c in activities.columns if c != activities.columns[0]]
        matrix = activities.select(signature_columns).to_numpy().astype(np.float64)
        used = matrix.sum(axis=0) > 0
        metrics["signatures_used"] = int(used.sum())
        metrics["signatures"] = [c for c, u in zip(signature_columns, used) if u]
        # Do the four UV subtypes end up dominating DIFFERENT clusters? The plan asks this
        # directly -- SBS7a-d separating is evidence the clustering resolves UV biology
        # rather than smearing it across one blob.
        dominant = {}
        for row_index, sample in enumerate(activities[activities.columns[0]].to_list()):
            row = matrix[row_index]
            if row.sum() > 0:
                dominant[str(sample)] = signature_columns[int(row.argmax())]
        metrics["uv_subtypes_dominant_somewhere"] = sorted(
            {s for s in dominant.values() if s in UV_SIGNATURES})
        metrics["n_uv_subtypes_separated"] = len(metrics["uv_subtypes_dominant_somewhere"])
    return metrics


def plot_sbs96_profiles(counts: np.ndarray, row_order: list[str], cluster_columns: list[str],
                        output_path: Path, cosines: dict[str, float] | None = None,
                        top_n: int = 12) -> Path:
    """The 96-channel spectrum for the largest clusters, in the conventional 6-block layout.

    matplotlib is imported here rather than at module scope: it can abort on the HPC when
    style files are missing, and that must not take down a sweep that has already spent
    hours fitting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    substitutions = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
    block_colors = ["#02bced", "#010101", "#e32926", "#cac9c9", "#a1ce63", "#ebc6c4"]
    order = np.argsort([label[2:5] for label in row_order], kind="stable")
    substitution_of = [label[2:5] for label in row_order]
    color = [block_colors[substitutions.index(s)] for s in substitution_of]

    totals = counts.sum(axis=0)
    chosen = np.argsort(totals)[::-1][:top_n]
    rows = int(np.ceil(len(chosen) / 2)) or 1
    figure, axes = plt.subplots(rows, 2, figsize=(18, 2.1 * rows), squeeze=False)

    for position, cluster_index in enumerate(chosen):
        axis = axes[position // 2][position % 2]
        values = counts[:, cluster_index].astype(np.float64)
        total = values.sum()
        axis.bar(range(96), values / total if total else values, color=color, width=0.85)
        name = cluster_columns[cluster_index]
        title = f"{name} — {int(total):,} mutations"
        if cosines and name in cosines:
            title += f", cosine {cosines[name]:.3f}"
        axis.set_title(title, fontsize=9, loc="left")
        axis.set_xticks([])
        axis.set_xlim(-0.5, 95.5)
        axis.tick_params(labelsize=7)

    for leftover in range(len(chosen), rows * 2):
        axes[leftover // 2][leftover % 2].axis("off")

    figure.suptitle(f"SBS96 profiles — {top_n} largest clusters", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=110)
    plt.close(figure)
    return output_path


# ── one cell ───────────────────────────────────────────────────────────────────

def run_cell(cell: Cell, coords: np.ndarray, sbs96_index: np.ndarray, locus_reads: np.ndarray,
             row_order: list[str], signature_database: Path, cell_dir: Path,
             args) -> dict:
    """Fit, label the whole cohort, count SBS96, assign signatures, score. Writes as it goes."""
    import fast_predict
    import run_variant_cluster_pipeline as rvcp

    cell_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"cell": asdict(cell), "label": cell.label(),
                    "started": datetime.now(timezone.utc).isoformat()}
    total_rows = coords.shape[0]

    rng = np.random.default_rng(args.seed)
    fit_indices = np.sort(rng.choice(total_rows, size=cell.fit_rows, replace=False))
    fit_coords = np.asarray(coords[fit_indices], dtype=np.float32)

    log(f"  fitting HDBSCAN on {cell.fit_rows:,} rows "
        f"(mcs={cell.min_cluster_size}, ms={cell.min_samples}, "
        f"{cell.cluster_selection_method}, eps={cell.cluster_selection_epsilon})")
    started = perf_counter()
    clusterer = _fit_hdbscan(fit_coords, cell, args)
    record["fit_seconds"] = round(perf_counter() - started, 1)
    fit_labels = np.asarray(_to_host(clusterer.labels_)).astype(np.int32)
    n_clusters = int(fit_labels.max()) + 1 if (fit_labels >= 0).any() else 0
    record["n_clusters_fit"] = n_clusters
    record["fit_noise_fraction"] = float((fit_labels < 0).mean())
    log(f"    {record['fit_seconds']}s, {n_clusters:,} clusters, "
        f"{record['fit_noise_fraction'] * 100:.2f}% noise on the fit set")
    if n_clusters == 0:
        record["error"] = "no non-noise clusters; SigProfiler cannot run"
        (cell_dir / "metrics.json").write_text(json.dumps(record, indent=2))
        return record

    # Full-cohort labels: the fit rows keep the labels the fit gave them, held rows come from
    # the RBC path. Re-predicting the fit rows would be both slower and less accurate --
    # approximate_predict is an approximation OF the fit, so where the fit's own answer
    # exists it is the better one.
    log(f"  labelling all {total_rows:,} rows (RBC)")
    started = perf_counter()
    tables = fast_predict.build_tables(clusterer, n_fit=cell.fit_rows,
                                       min_samples=cell.min_samples)
    held_mask = np.ones(total_rows, dtype=bool)
    held_mask[fit_indices] = False
    held_indices = np.nonzero(held_mask)[0]

    labels = np.empty(total_rows, dtype=np.int32)
    labels[fit_indices] = fit_labels
    index = fast_predict.build_index(fit_coords, 2 * tables.min_samples, args.backend)
    held_labels, held_probabilities = fast_predict.predict(
        tables, fit_coords, np.asarray(coords[held_indices], dtype=np.float32),
        backend=args.backend, batch_rows=args.batch_rows, index=index)
    labels[held_indices] = held_labels
    record["label_seconds"] = round(perf_counter() - started, 1)
    record["held_mean_probability"] = float(held_probabilities.mean())
    log(f"    {record['label_seconds']}s, "
        f"{float((labels < 0).mean()) * 100:.2f}% cohort noise")
    del held_labels, held_probabilities
    np.save(cell_dir / "cohort_labels.npy", labels)

    record["cohort"] = cluster_size_stats(labels)
    record["cohort_read_weighted"] = cluster_size_stats(labels, weights=locus_reads)

    counts = sbs96_counts(sbs96_index, labels, n_clusters)
    present = np.nonzero(counts.sum(axis=0) > 0)[0]
    cluster_columns = [f"cluster_{int(c)}" for c in present]
    counts = counts[:, present]

    input_dir = cell_dir / "sigprofiler" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = input_dir / "cluster_sbs96_matrix.tsv"
    matrix = {"Type": row_order}
    for position, name in enumerate(cluster_columns):
        matrix[name] = counts[:, position].tolist()
    pl.DataFrame(matrix).write_csv(matrix_path, separator="\t")
    record["sbs96_matrix"] = str(matrix_path)
    record["sbs96_total_mutations"] = int(counts.sum())

    log(f"  SigProfiler on {len(cluster_columns):,} clusters "
        f"({int(counts.sum()):,} mutations)")
    started = perf_counter()
    sigprofiler_out = cell_dir / "sigprofiler" / "output"
    sigprofiler_out.mkdir(parents=True, exist_ok=True)
    rvcp.Analyzer.cosmic_fit(
        samples=str(matrix_path), output=str(sigprofiler_out),
        signature_database=str(signature_database) if signature_database else None,
        genome_build=args.genome_build, cosmic_version=float(args.cosmic_version),
        make_plots=False, collapse_to_SBS96=True, connected_sigs=False, verbose=False,
        input_type="matrix", context_type="96", export_probabilities=True,
        sample_reconstruction_plots=False, cpu=args.sigprofiler_cpu,
        add_background_signatures=False,
    )
    record["sigprofiler_seconds"] = round(perf_counter() - started, 1)

    cluster_mutations = {name: float(counts[:, i].sum())
                         for i, name in enumerate(cluster_columns)}
    record["sigprofiler"] = sigprofiler_metrics(sigprofiler_out, cluster_columns,
                                                cluster_mutations)

    cosines = _cosine_lookup(sigprofiler_out)
    try:
        plot_path = plot_sbs96_profiles(counts, row_order, cluster_columns,
                                        cell_dir / "sbs96_profiles.png", cosines,
                                        top_n=args.plot_clusters)
        record["profile_plot"] = str(plot_path)
    except Exception as exc:  # noqa: BLE001 - a failed plot must not lose the cell
        record["profile_plot_error"] = f"{type(exc).__name__}: {exc}"

    record["finished"] = datetime.now(timezone.utc).isoformat()
    (cell_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    return record


def _fit_hdbscan(fit_coords: np.ndarray, cell: Cell, args):
    try:
        from cuml.cluster import HDBSCAN as CumlHDBSCAN
        return CumlHDBSCAN(
            min_cluster_size=cell.min_cluster_size, min_samples=cell.min_samples,
            cluster_selection_method=cell.cluster_selection_method,
            cluster_selection_epsilon=cell.cluster_selection_epsilon,
            metric="euclidean", prediction_data=True,
        ).fit(fit_coords)
    except ImportError:
        import hdbscan
        return hdbscan.HDBSCAN(
            min_cluster_size=cell.min_cluster_size, min_samples=cell.min_samples,
            cluster_selection_method=cell.cluster_selection_method,
            cluster_selection_epsilon=cell.cluster_selection_epsilon,
            metric="euclidean", prediction_data=True, core_dist_n_jobs=args.threads,
        ).fit(fit_coords)


def _to_host(array):
    for attribute in ("to_numpy", "get"):
        method = getattr(array, attribute, None)
        if callable(method):
            return np.asarray(method())
    return np.asarray(array)


def _cosine_lookup(sigprofiler_out: Path) -> dict[str, float]:
    stats_path = (sigprofiler_out / "Assignment_Solution" / "Solution_Stats"
                  / "Assignment_Solution_Samples_Stats.txt")
    if not stats_path.exists():
        return {}
    stats = pl.read_csv(stats_path, separator="\t")
    cosine_column = next((c for c in stats.columns if "Cosine" in c), None)
    if cosine_column is None:
        return {}
    return {str(sample): float(value) for sample, value
            in zip(stats[stats.columns[0]].to_list(), stats[cosine_column].to_list())}


# ── aggregate ──────────────────────────────────────────────────────────────────

def aggregate(output_dir: Path) -> pl.DataFrame:
    """Rank finished cells by the plan's rule, with the raw numbers alongside.

    The rule (HDBSCAN_SWEEP_PLAN.md Phase 1, fixed in advance so it cannot be tuned to the
    answer): the SMALLEST min_cluster_size whose p10 cluster clears 3,000 cohort mutations
    and whose p90 stays under 300,000, with noise no more than 10 points above the best
    cell. Smallest, because Finding 1 showed resolution is free -- aggregate signature
    quality did not move across a 25x mcs range -- so mcs only has to buy the floor.
    """
    rows = []
    for metrics_path in sorted((output_dir / "cells").glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text())
        if "error" in payload or "cohort" not in payload:
            continue
        cell, cohort = payload["cell"], payload["cohort"]
        signatures = payload.get("sigprofiler", {})
        rows.append({
            "label": payload["label"],
            "fit_rows": cell["fit_rows"],
            "mcs": cell["min_cluster_size"],
            "ms": cell["min_samples"],
            "method": cell["cluster_selection_method"],
            "eps": cell["cluster_selection_epsilon"],
            "n_clusters": cohort["n_clusters"],
            "noise_pct": round(cohort["noise_fraction"] * 100, 2),
            "p10": int(cohort.get("p10", 0)),
            "median": int(cohort.get("median", 0)),
            "p90": int(cohort.get("p90", 0)),
            "share_in_band": round(cohort.get("mutation_share_in_band", 0) * 100, 1),
            "cos_median": round(signatures.get("cosine_median", float("nan")), 4),
            "cos_wmean": round(signatures.get("cosine_mutation_weighted_mean", float("nan")), 4),
            "mut_share_cos07": round(signatures.get("mutation_share_above_07", 0) * 100, 2),
            "mut_share_cos08": round(signatures.get("mutation_share_above_08", 0) * 100, 2),
            "n_signatures": signatures.get("signatures_used", 0),
            "uv_subtypes": signatures.get("n_uv_subtypes_separated", 0),
            "fit_s": payload.get("fit_seconds", 0),
            "label_s": payload.get("label_seconds", 0),
        })
    if not rows:
        return pl.DataFrame()

    table = pl.DataFrame(rows)
    best_noise = table["noise_pct"].min()
    table = table.with_columns([
        (pl.col("p10") >= MUTATION_FLOOR).alias("passes_floor"),
        (pl.col("p90") <= MUTATION_CEILING).alias("passes_ceiling"),
        (pl.col("noise_pct") <= best_noise + 10).alias("passes_noise"),
    ])
    return table.with_columns(
        (pl.col("passes_floor") & pl.col("passes_ceiling") & pl.col("passes_noise")).alias("eligible")
    ).sort(["eligible", "mut_share_cos07"], descending=[True, True])


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--coords", type=Path, help="coords.npy, 157.5M x 2")
    parser.add_argument("--context", type=Path,
                        help="<embed_dir>/context.parquet, row-aligned with coords.npy")
    parser.add_argument("--build-index-only", action="store_true",
                        help="build the SBS96 index cache and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the grid and its projected cost, run nothing")
    parser.add_argument("--aggregate", action="store_true", help="rank finished cells and exit")

    parser.add_argument("--fit-sizes", default="500000,1000000,5000000,25000000")
    parser.add_argument("--min-cluster-sizes", default="250,500,1000,2500",
                        help="held FIXED across fit sizes -- that is the point of this sweep")
    parser.add_argument("--min-samples", default="5,15")
    parser.add_argument("--methods", default="eom")
    parser.add_argument("--epsilons", default="0.0")
    parser.add_argument("--max-ms-at-25m", type=int, default=5,
                        help="min_samples ceiling at 25M rows; ms=15 OOMed on a 47 GB card")

    parser.add_argument("--signature-set", default="uv_only", choices=["uv_only", "full"],
                        help="uv_only restricts to the UV reference set; full uses all of "
                             "COSMIC. HDBSCAN_SWEEP_PLAN.md Phase 0 step 1 says settle this "
                             "FIRST -- median cosine was 0.296 at every mcs, which may mean "
                             "most reads are simply not UV, in which case no HDBSCAN "
                             "parameter can improve a uv_only fit.")
    parser.add_argument("--genome-build", default="GRCh38")
    parser.add_argument("--cosmic-version", default="3.5")
    parser.add_argument("--sigprofiler-cpu", type=int, default=16)
    parser.add_argument("--plot-clusters", type=int, default=12)

    parser.add_argument("--backend", default="rbc", choices=["rbc", "brute", "sklearn"])
    parser.add_argument("--batch-rows", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-budget-gb", type=float, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--continue-after-failure", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        table = aggregate(output_dir)
        if table.is_empty():
            log("no finished cells yet")
            return 0
        with pl.Config(tbl_rows=200, tbl_cols=30, tbl_width_chars=250):
            print(table)
        table.write_csv(output_dir / "sweep_ranking.csv")
        log(f"wrote {output_dir / 'sweep_ranking.csv'}")
        return 0

    import run_variant_cluster_pipeline as rvcp

    signature_dir = output_dir / "signature_db"
    signature_dir.mkdir(parents=True, exist_ok=True)
    if args.signature_set == "uv_only":
        signature_database = signature_dir / f"uv_only_SBS_{args.genome_build}.tsv"
        row_order = rvcp.write_uv_only_signature_database(
            args.genome_build, args.cosmic_version, signature_database)
    else:
        # Full COSMIC: pass no database and let SigProfiler use its packaged reference, but
        # still read the row order from that file so the matrix rows line up.
        signature_database = None
        reference = pl.read_csv(
            rvcp.bundled_reference_signature_path(args.genome_build, args.cosmic_version),
            separator="\t")
        row_order = reference["Type"].to_list()
    log(f"signature set: {args.signature_set} ({len(row_order)} SBS96 channels)")

    if args.build_index_only:
        load_or_build_index(args.context, row_order, output_dir)
        return 0

    fit_sizes = [int(v) for v in args.fit_sizes.split(",") if v]
    grid = build_grid(
        fit_sizes=fit_sizes,
        min_cluster_sizes=[int(v) for v in args.min_cluster_sizes.split(",") if v],
        min_samples=[int(v) for v in args.min_samples.split(",") if v],
        methods=[v for v in args.methods.split(",") if v],
        epsilons=[float(v) for v in args.epsilons.split(",") if v],
        max_min_samples_at={25_000_000: args.max_ms_at_25m, 50_000_000: 0},
    )

    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    done = {path.parent.name for path in cells_dir.glob("*/metrics.json")} if args.resume else set()
    pending = [cell for cell in grid if cell.label() not in done]

    fit_total = sum(projected_fit_seconds(c.fit_rows, c.min_samples) for c in pending)
    overhead = len(pending) * (5.3 * 60 + 120)
    log(f"grid: {len(grid)} cells, {len(done)} already done, {len(pending)} to run")
    log(f"projected: {fit_total / 3600:.1f} h fitting + {overhead / 3600:.1f} h "
        f"labelling/SigProfiler = {(fit_total + overhead) / 3600:.1f} h total")
    for cell in pending:
        log(f"    {cell.label():48} ~{projected_fit_seconds(cell.fit_rows, cell.min_samples) / 60:6.1f} min fit")
    if args.dry_run:
        return 0

    if not args.coords:
        raise SystemExit("--coords is required to run the sweep")
    coords = np.load(args.coords, mmap_mode="r")
    log(f"coords: {coords.shape[0]:,} x {coords.shape[1]}")
    sbs96_index, locus_reads = load_or_build_index(args.context, row_order, output_dir)
    if sbs96_index.shape[0] != coords.shape[0]:
        raise SystemExit(
            f"SBS96 index has {sbs96_index.shape[0]:,} rows but coords has "
            f"{coords.shape[0]:,} -- they must come from the same stage-1 run")

    if args.gpu_budget_gb:
        try:
            import sweep_core
            sweep_core.apply_gpu_budget("sweep", budget_gb=args.gpu_budget_gb)
        except Exception as exc:  # noqa: BLE001
            log(f"gpu budget not applied: {exc}")

    (output_dir / "sweep_config.json").write_text(json.dumps({
        "grid": [asdict(c) for c in grid], "seed": args.seed,
        "signature_set": args.signature_set, "backend": args.backend,
        "mutation_floor": MUTATION_FLOOR, "mutation_ceiling": MUTATION_CEILING,
        "started": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    for position, cell in enumerate(pending, start=1):
        log(f"[{position}/{len(pending)}] {cell.label()}")
        cell_dir = cells_dir / cell.label()
        try:
            run_cell(cell, coords, sbs96_index, locus_reads, row_order,
                     signature_database, cell_dir, args)
        except Exception as exc:  # noqa: BLE001 - a failed cell must not lose the sweep
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "failed.json").write_text(json.dumps({
                "cell": asdict(cell), "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }, indent=2))
            log(f"  FAILED: {type(exc).__name__}: {str(exc)[:300]}")
            if not args.continue_after_failure:
                raise

    table = aggregate(output_dir)
    if not table.is_empty():
        table.write_csv(output_dir / "sweep_ranking.csv")
        with pl.Config(tbl_rows=200, tbl_cols=30, tbl_width_chars=250):
            print(table)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
