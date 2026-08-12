# What the sweep measures, and how a cell wins

Companion to `hdbscan_param_sweep.py`. Every number below is written to
`cells/<label>/metrics.json` the moment a cell finishes.

---

## The ranking rule, fixed in advance

From `HDBSCAN_SWEEP_PLAN.md` Phase 1, restated here because the code implements it literally
(`aggregate()`):

> Choose the **smallest** `min_cluster_size` such that
> 1. the p10 cluster clears **3,000** full-cohort mutations,
> 2. the p90 cluster stays under **300,000**, and
> 3. noise is within **10 points** of the lowest-noise cell.
>
> Tie-break on **mutation share in clusters with cosine > 0.7**.

Smallest, not largest. That is not a preference — it follows from Finding 1 (below), which
measured that aggregate signature quality does not move across a 25× `mcs` range. If
coarsening buys no accuracy, resolution is free, and `mcs` only has to buy the floor.

`aggregate()` emits `eligible` as a boolean column so cells that fail a criterion are still
visible with their numbers rather than being dropped.

---

## Tier A — cluster geometry (no signatures involved)

Computed over the **full 157.5M-locus labelling**, never the fit set.

| Metric | Why it is here |
|---|---|
| `n_clusters`, `noise_fraction` | Baseline shape. Neither is an objective on its own. |
| `min / p10 / median / p90 / max` | The decision variables. Cluster **size in cohort mutations**. |
| `mutation_share_in_band` | Share of assigned mutations in clusters sized 3k–300k — the band that fits well. |
| `clusters_below_floor`, `clusters_above_ceiling` | Where a failing cell is failing. |
| `size_entropy_normalised` | Balance, 0–1. One giant cluster plus a tail scores far below an even split at the same count — cluster count alone hides that. |
| `cohort_read_weighted.*` | Same statistics weighted by `locus_reads`, i.e. what the answer looks like if every read votes instead of every locus. |

**Sizes are measured, never derived.** `HDBSCAN_SWEEP_PLAN.md` Finding 3: at `mcs=100` a
31.5× scale factor predicts a 3,150-mutation floor and the smallest cluster actually held
**174**. Held-out rows near cluster boundaries go to noise, so small fit-set clusters attract
disproportionately few of them and shrink. Any metric computed on the fit set and scaled up
is wrong.

## Tier B — SigProfiler (the actual objective)

| Metric | Why it is here |
|---|---|
| `mutation_share_above_07` / `_08` | **The headline.** Share of cohort mutations in confidently-fit clusters. |
| `clusters_above_07` / `_08` | Reported, but not ranked on — see below. |
| `cosine_median`, `_mean`, `_p10`, `_p90` | Distribution shape. It is bimodal, so the median alone misleads. |
| `cosine_mutation_weighted_mean` | Cohort-level quality. Stops a long tail of tiny well-fit clusters from flattering a cell. |
| `signatures_used`, `signatures` | How much of the reference the clustering actually exercises. |
| `n_uv_subtypes_separated` | Do SBS7a/b/c/d dominate **different** clusters? Evidence the clustering resolves UV biology instead of smearing it. |

**Why mutation share and not cluster count.** Finding 1 measured, across `mcs` 100 → 2500:

| mcs | clusters | clusters cos>0.7 | **muts in cos>0.7** |
|---|---|---|---|
| 100 | 5,709 | 544 | **10.7 %** |
| 2500 | 613 | 68 | **10.2 %** |

Counting clusters says these cells differ 8-fold. Counting mutations says they are the same
cell packaged differently — which is the truth. Spearman correlation between cluster size and
cosine, pooled across all five cells, was **0.073**.

## Tier C — stability

Not computed inside the sweep; run separately on the saved `cohort_labels.npy`:

- **Seed ARI** — refit at seed 43, ARI between the two labellings of the same rows. The
  baseline to beat is **0.261** (model 13, from the final_models README).
- **Cross-size ARI + Meilă VI** — `cross_size_ari.py`. Distinguishes *a larger fit set found
  new structure* from *a larger fit set merged what the smaller one already had*.
  `H(coarse|fine) = 0` is an exact identity for a clean coarsening, not a threshold.

## Tier D — cost

`fit_seconds`, `label_seconds`, `sigprofiler_seconds`. Reported so a marginal quality gain
can be weighed against 3.25 h of fitting, not to rank cells.

---

## Deliberately not used

- **Silhouette / Davies–Bouldin / Calinski–Harabasz.** They assume convex, isotropic
  clusters. HDBSCAN on a UMAP embedding produces neither, and they would rank a wrong answer
  above a right one.
- **DBCV.** Density-aware and the right family, but `LIMITATIONS.md` caps it at a 25k-row
  sample and skips cells above 500 clusters — most of this grid. It is a ranking score on a
  subsample, not a cohort measurement. Available via `rescore_dbcv.py` if wanted for the
  finalists.
- **Cluster count as a target.** Seed-sensitive by ±5 % at 25M fit rows (final_models caveat
  1). A count difference between two cells is not by itself a real effect.

---

## Two things to settle before trusting any of it

**1. `uv_only` versus full COSMIC — this decides what the sweep optimises.**
Median cosine was **0.296 at every single `mcs`** in the earlier sweep, and the distribution
is bimodal. The row filter (`st='MIXED' AND et='MIXED' AND FILT=1`) admits non-UV variants,
but assignment used the restricted `uv_only` set. If most reads are not UV, no UV-only
reference can fit them and no HDBSCAN parameter will change that — the sweep would be
optimising a quantity that is capped by the reference, not by the clustering.

`--signature-set full` runs the same cell against all of COSMIC. It costs seconds on a matrix
that already exists. Run it on one finished cell before committing GPU-days:

- Low-cosine clusters fit *other* signatures → they are real non-UV structure, and the
  objective becomes "maximise clusters above threshold across the full reference".
- They still fit nothing → they are unstructured reads, and the objective is "maximise
  mutation share in well-fit clusters while keeping noise low".

**2. The RBC labelling is not bit-identical to `approximate_predict`.** Measured on the 2M
eval probe against the 25M model: **97.67 %** exact agreement. The disagreement is almost
entirely the noise gate (2.27 % assigned-by-us / noise-by-cuML); genuine cluster-identity
disagreement is **0.001 %**. Brute force through the same code path shows the same 2.20 %
gap, so this is cuML's `approximate_predict` differing from the CPU reference, not RBC. Every
cell in the sweep is labelled the same way, so cross-cell comparisons are unaffected.
