# What the sweep produces, and what scores it

Companion to `hdbscan_param_sweep.py`.

---

## What each cell writes

`cells/<label>/metrics.json`, the moment it finishes, plus `cohort_labels.npy` (int32,
row-aligned with `coords.npy` — the actual product).

**Shape** — `n_clusters_fit`, `fit_noise_fraction`, `cohort_n_clusters`,
`cohort_noise_fraction`, `fit_seconds`, `label_seconds`.

**Geometry** (`cluster_quality.py`, on the fit set) — DBCV, connectivity, membership
probabilities, cluster persistence. See the table below.

**Spectrum** (`spectrum_metrics.py`, no signature reference) — top-channel share median and
fractions above 0.5 / 0.8, distinct dominant channels, clusters per dominant channel,
pairwise cosine / L1 / L2 between normalised cluster spectra.

**Signatures** (`assignment_metrics.py`, aggregating SigProfiler's own output) — cosine mean
/ median / mutation-weighted, `L1_Norm_%`, `L2_Norm_%`, `l2_over_total_pct`, KL, correlation,
and cluster + mutation shares above cosine 0.7 / 0.8.

**No ranking rule is applied.** The sweep emits numbers side by side and stops there.

The SBS96 canonicalisation is imported from `stage3_apply_full.sbs96_expr`, itself documented
and tested as identical to `run_variant_cluster_pipeline.annotate_trinuc_counts`. SigProfiler
is `rvcp.Analyzer.cosmic_fit` — the same CPU call every other stage in this repo makes.
**There is no GPU SigProfiler and it would not help:** `cosmic_fit` only ever sees a
96 × n_clusters matrix (~110k numbers at 1,150 clusters). What is expensive is *building*
that matrix from 157.5M rows, which is why the context is collapsed to one `int8` per row
**once** (157 MB, cached as `sbs96_index.npy`) and each cell's matrix is then a `bincount`.

```bash
BUILD_INDEX_ONLY=1 bash umap_hdbscan_sweep/tmux_param_sweep.sh
```

## Geometry metrics, and what each one hides

| Metric | Reads as | The caveat that must travel with it |
|---|---|---|
| `dbcv` | Density separation, −1…1 | Scored on a **25k subsample**, and declined above 500 clusters. HDBSCAN's own `relative_validity_` is a cheaper MST approximation, **not** true DBCV. |
| `connectivity_mean` | Fraction of each point's k-NN sharing its label | **Monotone in cluster count** — merge everything into one cluster and it is exactly 1.0. Meaningless without `n_clusters`. |
| `prob_mean`, `prob_frac_above_*` | Membership strength | `probabilities_` is renormalised *per cluster* (λ_p / λ_max), so many small tight clusters inflate the mean. The threshold fractions are the honest summary. |
| `persistence_*` | HDBSCAN's own stability | Partly circular under `eom` — the sum is what EOM maximises. Independent evidence only under `leaf`. |

**CDbw is deliberately absent.** No maintained, validated Python implementation exists, it
needs multiple representatives per cluster (O(n²)-ish), and it would land as a third
subsample-based density score beside DBCV without asking an independent question.

**The caveat above all of these:** DBCV, connectivity, persistence and the excluded
silhouette family all measure geometry *in the embedding*. The 2026-08-04 UMAP cells separate
cleanly by every density criterion and are still 87 % single-trinucleotide-context — they
score well *because* UMAP manufactured the separation the metric then rewards. Read them
beside `top_channel_share`, never instead of it.

---

## Reading the numbers

There is no ranking rule and no combined score. Two facts inform how to read them:

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
