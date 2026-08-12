# What the sweep produces, and what scores it

Companion to `hdbscan_param_sweep.py`.

---

## What the sweep itself measures

Very little, on purpose. Each cell writes `cells/<label>/metrics.json` the moment it finishes:

| Metric | |
|---|---|
| `n_clusters_fit`, `fit_noise_fraction` | Shape of the fit, before any held-out rows |
| `cohort_n_clusters`, `cohort_noise_fraction` | Shape of the full 157.5M labelling |
| `held_mean_probability` | Mean membership probability of the held-out rows |
| `fit_seconds`, `label_seconds` | Cost, so a marginal gain can be weighed against 3.25 h of fitting |
| `cohort_labels.npy` | **The actual product.** int32, row-aligned with `coords.npy` |

Everything that needs the mutation contexts — the SBS96 count matrix, SigProfiler, per-cluster
cosine, the size distribution — comes from `uv_vae/scripts/run_variant_cluster_pipeline.py`
run against a saved `cohort_labels.npy`. That pipeline is what produced every earlier result
in this repo; a second implementation inside the sweep would be a second answer to reconcile,
not a shortcut. The relevant entry points are `annotate_trinuc_counts`,
`write_cluster_sbs96_matrix`, `run_sigprofiler_assignment`, `query_cluster_stats` and
`dominant_signature_columns`.

---

## The ranking rule, fixed in advance

From `HDBSCAN_SWEEP_PLAN.md` Phase 1. It is applied **after** the pipeline has produced
cluster sizes and cosines for the finished cells, not inside the sweep:

> Choose the **smallest** `min_cluster_size` such that
> 1. the p10 cluster clears **3,000** full-cohort mutations,
> 2. the p90 cluster stays under **300,000**, and
> 3. noise is within **10 points** of the lowest-noise cell.
>
> Tie-break on **mutation share in clusters with cosine > 0.7**.

Smallest, not largest. That follows from Finding 1: aggregate signature quality does not move
across a 25× `mcs` range. If coarsening buys no accuracy, resolution is free and `mcs` only
has to buy the floor.

**Sizes must be measured, never derived.** Finding 3: at `mcs=100` a 31.5× scale factor
predicts a 3,150-mutation floor and the smallest cluster actually held **174**. Held-out rows
near cluster boundaries go to noise, so small fit-set clusters attract disproportionately few
of them and shrink.

**Mutation share, not cluster count.** Finding 1 measured, across `mcs` 100 → 2500:

| mcs | clusters | clusters cos>0.7 | **muts in cos>0.7** |
|---|---|---|---|
| 100 | 5,709 | 544 | **10.7 %** |
| 2500 | 613 | 68 | **10.2 %** |

Counting clusters says these differ 8-fold. Counting mutations says they are the same cell
packaged differently — which is the truth. Spearman correlation between cluster size and
cosine, pooled across all five cells, was **0.073**.

---

## Stability, run separately on the saved labels

- **Seed ARI** — refit at seed 43, ARI between the two labellings of the same rows. The
  baseline to beat is **0.261** (model 13, from the final_models README).
- **Cross-size ARI + Meilă VI** — `cross_size_ari.py`. Distinguishes *a larger fit set found
  new structure* from *a larger fit set merged what the smaller one already had*.
  `H(coarse|fine) = 0` is an exact identity for a clean coarsening, not a threshold.

---

## Deliberately not used

- **Silhouette / Davies–Bouldin / Calinski–Harabasz.** They assume convex, isotropic
  clusters. HDBSCAN on a UMAP embedding produces neither, and they would rank a wrong answer
  above a right one.
- **DBCV.** Density-aware and the right family, but `LIMITATIONS.md` caps it at a 25k-row
  sample and skips cells above 500 clusters — most of this grid. Available via
  `rescore_dbcv.py` if wanted for the finalists.
- **Cluster count as a target.** Seed-sensitive by ±5 % at 25M fit rows (final_models caveat
  1). A count difference between two cells is not by itself a real effect.

---

## Two things to settle before trusting any of it

**1. `uv_only` versus full COSMIC — this decides what is being optimised.**
Median cosine was **0.296 at every single `mcs`** in the earlier sweep, and the distribution
is bimodal. The row filter (`st='MIXED' AND et='MIXED' AND FILT=1`) admits non-UV variants,
but assignment used the restricted `uv_only` set. If most reads are not UV, no UV-only
reference can fit them and no HDBSCAN parameter will change that.

Re-running the pipeline against all of COSMIC costs seconds on a matrix that already exists.
Do it on one finished cell before committing GPU-days:

- Low-cosine clusters fit *other* signatures → real non-UV structure, and the objective
  becomes "maximise clusters above threshold across the full reference".
- They still fit nothing → unstructured reads, and the objective is "maximise mutation share
  in well-fit clusters while keeping noise low".

**2. The RBC labelling is not bit-identical to `approximate_predict`.** Measured on the 2M
eval probe against the 25M model: **97.67 %** exact agreement. The disagreement is almost
entirely the noise gate (2.27 % assigned-by-us / noise-by-cuML); genuine cluster-identity
disagreement is **0.001 %**. Brute force through the same code path shows the same 2.20 %
gap, so this is cuML's `approximate_predict` differing from the CPU reference, not RBC. Every
cell in the sweep is labelled the same way, so cross-cell comparisons are unaffected.
