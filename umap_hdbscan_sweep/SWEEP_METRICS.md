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
| `dbcv` | Density separation, −1…1 | Scored on a **25k subsample**, now **stratified** so there is no cluster-count ceiling — see below. HDBSCAN's own `relative_validity_` is a cheaper MST approximation, **not** true DBCV. DBCV validates an EXISTING label array — it is not a clustering algorithm — so it always runs on `clusterer.labels_` from the HDBSCAN fit; there is no unlabelled variant. |
| `connectivity_mean` | Fraction of each point's k-NN sharing its label | **Monotone in cluster count** — merge everything into one cluster and it is exactly 1.0. Meaningless without `n_clusters`. |
| `prob_mean`, `prob_frac_above_*` | Membership strength | `probabilities_` is renormalised *per cluster* (λ_p / λ_max), so many small tight clusters inflate the mean. The threshold fractions are the honest summary. |
| `persistence_*` | HDBSCAN's own stability | Partly circular under `eom` — the sum is what EOM maximises. Independent evidence only under `leaf`. |

### Why DBCV no longer refuses above 500 clusters

The old ceiling was a sampling artefact, not a property of DBCV. Uniform sampling draws from
a cluster in proportion to its size, so at 1,330 clusters the small ones arrived under
`MIN_SAMPLED_POINTS_PER_CLUSTER = 4` and `validity_index` dropped or raised on them. The
2026-08-12 sweep scored only **11 of 28 cells** for this reason, excluding both extremes.

`cluster_quality.stratified_sample` spends the *same total budget* with an equal-share
allocation (water-filling, so a cluster smaller than its quota returns the surplus to the
rest). Measured on 600 clusters over 25k rows: **stratified 63.7 s / 600 clusters scored,
uniform 72.0 s / 593 scored** — no extra cost, because runtime is driven by the total point
count, which is held fixed.

One correction comes with it. DBCV's aggregate is `Σᵢ (|Cᵢ|/n)·Vᵢ`, weighted by cluster size,
and `validity_index` takes those weights from the **sample** — which stratification
deliberately distorts. So the per-cluster scores are pulled out and reweighted by the **true**
cluster sizes. `dbcv` is that corrected figure and stays comparable to the uniform numbers
already recorded (they agree to 0.005 on a clustering where both work);
`dbcv_sample_weighted` is what `validity_index` returned.

The per-cluster scores also give resolution no aggregate can: `dbcv_frac_clusters_negative`
and **`dbcv_point_share_negative`** say how much of the data sits in clusters that are not
density-separated at all. Read the point share — a hundred bad micro-clusters matter less
than one bad cluster holding a third of the cohort.

### Why DBCV is sampled at all — the memory wall, not just the time cost

DBCV's intracluster step computes the all-points-core-distance, which materialises an
**n_i × n_i distance matrix for every cluster**. So the binding constraint is the size of the
LARGEST single cluster, not the total row count. k-DBCV states the formula it refuses on:

```
predicted_GB = ((max_cluster_size ** 2) * 8) / 1024**3 * 8
```

At its 25 GB default that caps the largest scoreable cluster at 20,480 points.
`hdbscan.validity_index` builds the same matrix, so the same wall applies to it too — this is
not a k-DBCV-specific limit.

Applying the real cluster-size distribution from `fit1000000_mcs2500_ms5_eom` (largest
cluster = 3.39% of assigned points) to full DBCV on each cell's WHOLE fit set:

| fit rows | largest cluster | RAM for that one cluster | |
|---|---|---|---|
| 500,000 | 16,946 | 17 GB | OK, barely |
| 1,000,000 | 33,891 | **68 GB** | refused |
| 5,000,000 | 169,456 | **1,712 GB** | refused |
| 25,000,000 | 847,282 | **42,789 GB** | refused |

Full-fit-set DBCV is impossible from 1M fit rows upward on memory alone, before the O(k²)
cluster-pair time cost is even considered. (This is the *fit set* — the 157.5M-row cohort was
never a candidate.)

**This is why stratified `per_cluster` capping is the right fix, not uniform sampling.**
Memory is quadratic in `max_cluster_size`, so that is the variable worth controlling directly
— `per_cluster=N` caps it absolutely regardless of how skewed the clustering is:

| points/cluster | RAM for the largest cluster |
|---|---|
| 400 (the default) | 0.0095 GB |
| 5,000 | 1.49 GB |
| 20,000 | 23.8 GB |

Uniform sampling can't do this cleanly, because it preserves every cluster's *share* — the
largest cluster stays proportionally largest and only shrinks with the total budget. Getting
it under the wall by shrinking the total budget alone pushes small clusters below the 4-point
floor and they get dropped — precisely the failure that left 17 of 28 cells unscored under
the old uniform ceiling.

`cluster_quality.score_kdbcv` predicts this same quantity before calling `DBCV_score` and
refuses with a stated reason (`dbcv_note`) rather than trusting the return value: k-DBCV
signals *all-noise*, *fewer than two clusters*, **and** *memory cutoff exceeded* all with the
same `-1` sentinel, which is also a legal DBCV score for a bad clustering. Silently accepting
that `-1` as a real score would have recorded some memory-refused cells as "worst possible
clustering" instead of "not scored."

**Comparability is the other reason to sample, even where full DBCV would technically fit**
(the 500k cells, at 17 GB). A full-data score and a sampled score are not the same
estimator — sampled DBCV runs optimistic (see the accuracy measurement above). Scoring small
cells in full and large cells by sample would make their scores incomparable, which defeats
the purpose of a sweep. Every cell is scored at the same `per_cluster` resolution so the
numbers can be read against each other.

**CDbw is deliberately absent.** No maintained, validated Python implementation exists, it
needs multiple representatives per cluster (O(n²)-ish), and it would land as a third
subsample-based density score beside DBCV without asking an independent question.

**Dunn is deliberately absent, and there is no "robust Dunn index".** The real method is the
family of *generalised Dunn indices* — Bezdek & Pal (1998) identify two deficiencies that
make the original "overly sensitive to noisy clusters" and propose variants "not as brittle
to outliers". Not adopted here because that paper's own framing is about clusters "expected
to form volumetric clouds", and HDBSCAN clusters are arbitrarily shaped by construction —
exactly the case DBCV was introduced for. A Dunn variant would also compress 1,000+ clusters
into one scalar driven by the single closest pair out of ~660,000, which cannot say *which*
clusters are poor.

### Sources

| | |
|---|---|
| HDBSCAN stability, EOM selection | Campello, Moulavi & Sander (2013), *Density-Based Clustering Based on Hierarchical Density Estimates*, PAKDD, LNCS 7819:160–172; extended in ACM TKDD 10(1):5 (2015) |
| `cluster_persistence_`, `relative_validity_` | McInnes, Healy & Astels (2017), *hdbscan: Hierarchical density based clustering*, JOSS 2(11):205 |
| DBCV | Moulavi, Jaskowiak, Campello, Zimek & Sander (2014), *Density-Based Clustering Validation*, SDM:839–847, doi:10.1137/1.9781611973440.96 |
| Connectivity | Handl & Knowles (2005), *Computational cluster validation in post-genomic data analysis*, Bioinformatics 21(15):3201–3212 |
| Generalised Dunn | Bezdek & Pal (1998), *Some new indexes of cluster validity*, IEEE Trans. SMC-B 28(3):301–315; original Dunn (1974), J. Cybernetics 4(1):95–104 |
| CDbw | Halkidi & Vazirgiannis (2008), Pattern Recognition Letters 29(6):773–786 |

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

## Stability — the metric family that is not circular

Every index in the geometry table asks whether clusters are compact and separated **in the
UMAP embedding**, and manufacturing compact, separated neighbourhoods is UMAP's objective.
That is why connectivity sat at 0.998–1.000 across a 15× change in cluster count: it is not
broken, it is answering a question the embedding already guaranteed. Stability asks whether
the structure is *reproducible*, assumes nothing about cluster shape, and for this project is
the thesis question rather than a diagnostic for it.

`stability_sweep.py` runs it: reserve a probe, refit each cell on independent subsamples
drawn from the remaining rows, label the probe with each, compare pairwise. The probe is held
out of **every** fit set, so no replicate grades its own training rows.

- **Per-cluster Jaccard** (Hennig 2007) — the one to lead with. For each cluster, the best
  match in the other replicate: **≥0.75 stable, <0.5 dissolved**. Reported by cluster count
  *and* by point share, because a hundred unstable micro-clusters and one unstable cluster
  holding a third of the data give the same ARI and very different point shares.
- **ARI**, with and without noise rows. Baseline to beat: **0.261** (model 13, final_models).
- **Meilă VI** — a true metric on partitions; `H(b|a) = 0` is an exact identity for a clean
  coarsening, not a threshold. Distinguishes *found new structure* from *merged what was
  already there*.
- **Noise agreement** — all four cells of clustered/noise against clustered/noise.

**The caveat that must travel with it** (von Luxburg 2010): stability rises as clusterings
get coarser — a 2-cluster solution is nearly always stable. Read it against cluster count,
exactly like connectivity, and report the stability-vs-granularity curve rather than picking
the single highest number.

**Cost.** Fit time × replicates; probe labelling is ~3% of a cohort pass and rounds to
nothing. At 3 replicates: **1.7 h** for the 12 cells up to 5M, **40.6 h** if the 25M row is
included — so add 25M deliberately, on one cell, not across the block.

`cross_size_ari.py` holds the comparison maths and is also usable standalone on any two saved
labellings of the same rows.

### Sources for the stability family

| | |
|---|---|
| Per-cluster Jaccard, the 0.75 / 0.5 thresholds | Hennig (2007), *Cluster-wise assessment of cluster stability*, Comput. Stat. Data Anal. 52(1):258–271 |
| Stability as a validation principle | Ben-Hur, Elisseeff & Guyon (2002), PSB 7:6–17; Lange, Roth, Braun & Buhmann (2004), Neural Computation 16(6):1299–1323 |
| Prediction strength | Tibshirani & Walther (2005), *Cluster Validation by Prediction Strength*, JCGS 14(3):511–528 |
| The caveat, and when stability misleads | von Luxburg (2010), *Clustering Stability: An Overview*, Found. Trends ML 2(3):235–274 |
| Variation of information | Meilă (2007), *Comparing clusterings — an information based distance*, J. Multivariate Analysis 98(5):873–895 |

---

## Deliberately not used

- **Silhouette / Davies–Bouldin / Calinski–Harabasz.** They assume convex, isotropic
  clusters. HDBSCAN on a UMAP embedding produces neither, and they would rank a wrong answer
  above a right one.
- ~~**DBCV.**~~ Was skipped above 500 clusters — most of this grid. **Stratified sampling
  removed that ceiling**, so DBCV is now scored on every cell. See above.
- **Cluster count as a target.** Seed-sensitive by ±5 % at 25M fit rows (final_models caveat
  1). A count difference between two cells is not by itself a real effect.

---

## Two things to settle before trusting any of it

**1. ~~`uv_only` versus full COSMIC~~ — SETTLED 2026-08-13. `uv_only` was the wrong reference.**
On the same matrix, full COSMIC v3.5 lifts median cosine **0.25 → 0.73** and mutation share
above 0.7 from **5.6% → 51.8%**. Every `uv_only` number in the sweep is measured through a
reference that cannot fit the data. Re-score before drawing conclusions from any of them.

What full COSMIC shows, across four cells spanning 89 → 1,330 clusters: **UV 8.75–9.38%,
artefact-suspect (SBS45–60) 20.2–22.6%, unknown-etiology (SBS96+) 10.4–12.1%** — a 15× change
in granularity moves UV by 0.63 points. The composition is a property of the cohort, not of
the clustering.

A shuffled null (same cluster sizes, permuted membership) recovers **UV 0.00%** and collapses
onto unknown-etiology signatures at 47%, at a *higher* median cosine (0.80) than the real
clustering. Two conclusions: the clustering is doing real work — UV is undetectable in the
pooled cohort and only becomes assignable after clustering — and **"maximise cosine" is a
broken objective**, because a meaningless clustering wins on it. Smooth, averaged spectra fit
easily; the real, spiky ones do not.

**Cosine is also weaker than it looks.** SigProfiler's reconstruction preserves total counts
exactly (verified: recon/obs = 1.0000), so `L1_Norm_% = 133` means **~67% of a cluster's
mutations sit in a channel the fit did not predict** — at a reported cosine of 0.753. One
cluster reports cosine 0.911 with L1 120.6%, i.e. ~60% of its mass misplaced. L1 is the
honest fit statistic, and it is 168–177% in every cell of the grid.

**2. The RBC labelling is not bit-identical to `approximate_predict`.** Measured on the 2M
eval probe against the 25M model: **97.67 %** exact agreement. The disagreement is almost
entirely the noise gate (2.27 % assigned-by-us / noise-by-cuML); genuine cluster-identity
disagreement is **0.001 %**. Brute force through the same code path shows the same 2.20 %
gap, so this is cuML's `approximate_predict` differing from the CPU reference, not RBC. Every
cell in the sweep is labelled the same way, so cross-cell comparisons are unaffected.
