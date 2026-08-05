# Plan — rank UMAP embeddings first, defer HDBSCAN

Written 2026-08-04, following `SESSION_HANDOFF_2026-08-04.md`. This is a **plan only**;
no code in this repository has been changed for it.

The handoff's §6 proposes settling the UMAP embedding before finalising HDBSCAN, evaluated
with a clusterer cheap enough to run once per candidate embedding. This document works that
idea out: what the recorded artifacts actually prove about cost, which metric the ranking
must use (and why the obvious one is wrong), which clusterer to use, how large the fit set
needs to be, and what has to be verified before any of it can be trusted.

Constraint carried over from the handoff and reaffirmed: **`n_components = 2` throughout**,
because HDBSCAN's density estimate degrades in higher dimensions. This halves the old grid
and is compatible with the clusterer chosen in §4, which is 2-D by construction.

---

## 1. What the recorded artifacts prove about cost — and what they don't

From `uv_vae/runs/nn15_md0.0_nc2/`:

| | contents | recorded |
|---|---|---|
| UMAP stage, once per UMAP config | fit on 5 M + `transform` of 152.5 M | **3431 s** |
| HDBSCAN cell, once per HDBSCAN config | fit on 5 M + `approximate_predict` of 152.5 M + metrics + 157.5 M-row parquet write + SigProfiler (~2 s) | **3827–4155 s** |

The killed 5-cell run therefore cost `3431 + 5 × 3861 ≈ 22,736 s ≈ 6.3 h`, of which the UMAP
stage was ~15% — not because UMAP is intrinsically cheaper, but because it ran once while
HDBSCAN ran five times.

**The handoff's claim that `approximate_predict` dominates the cell is not established by
these numbers.** Both timers wrap a full pass over 152.5 M rows, and neither was decomposed.
Two observations argue against the claim:

- Cell time spans only 4155 s → 3827 s (**8%**) while `min_cluster_size` changes 25× and
  cluster count changes 9× (5709 → 613). `approximate_predict` costs scale with the cluster
  structure being looked up against; if it dominated, the spread should be far larger.
- Every cell calls `write_analysis_parquet` ([stage2_sweep.py:130](stage2_sweep.py)), writing
  a fresh 157.5 M-row, 11-column parquet. That is a multi-GB write, genuinely fixed across
  `min_cluster_size`, and unaccounted for in the handoff.

The `hdbscan` library's own docs describe `approximate_predict` as "very fast, even with
large datasets" given `prediction_data=True` — it is a condensed-tree descent, not a refit.
That further weakens the claim, and matters because the handoff's proposed fix
(nearest-centroid assignment, §7 item 1) would then buy much less than expected.

### Where the 3431 s UMAP timer actually goes

**Phase A measured this directly (2026-08-04).**

| Step | Time | Share |
|---|---|---|
| `fit_umap` on 5 M rows | **119 s (2 min)** | **1.8%** |
| `embed_all_rows` — transform of 152.5 M rows (extrapolated) | **3291 s (55 min)** | **98.2%** |

The fit is essentially free. The 3431 s recorded timer was almost entirely `embed_all_rows`.
Every proposed fix that targets `approximate_predict` or `fit_umap` misses the bottleneck.

**Consequence for this plan.** Phase D (25 UMAP configs, fit + score 5 M rows each, no
transform) costs **25 × 2 min = ~50 min total**, not days. The saving from a UMAP-only
ranking sweep is even larger than the handoff anticipated.

For the final full-cohort stage, the unavoidable cost of the current architecture is ~55 min
of `transform()` per UMAP config chosen. The only way to eliminate it entirely is to fit UMAP
on all 157.5 M loci — see §1 addendum below.

### §1 addendum — fitting UMAP on all 157.5 M loci

Phase A establishes that the fit itself is cheap (119 s for 5 M rows). The only reason
152.5 M rows go through `transform()` is that fitting on all 157.5 M was assumed to be
infeasible. The MLSys 2026 memory formula says it is not:

- Global kNN graph at k=15: 157.5 M × 15 × 8 bytes = **18.9 GB**
- Fits within the 48 GB RTX PRO 5000 with ~29 GB headroom for UMAP optimisation state

If cuML can fit on all 157.5 M rows, `embed_all_rows` disappears entirely — every point
gets its embedding from the fit, not from `transform()`. The kNN graph construction time
at 157.5 M × 16 extrapolates via O(N^1.14): 119 s × (157.5/5.0)^1.14 ≈ **~5,000 s
(~83 min)** for the graph alone, plus optimisation. That is slower than fit + transform
combined (~57 min total), so it is not a speed win. But it removes transform bias entirely
and simplifies the pipeline. Decision analysis in §10.

---

## 2. How the current 5 M fit set is drawn

[stage2_sweep.py:164](stage2_sweep.py):

```python
def fit_indices(total_rows, fit_rows, seed):
    if fit_rows is None or fit_rows >= total_rows:
        return None
    return np.sort(np.random.default_rng(seed).choice(total_rows, size=fit_rows, replace=False))
```

Concretely `np.random.default_rng(42).choice(157_501_580, size=5_000_000, replace=False)`,
sorted ascending. A plain uniform simple random sample without replacement, **3.17%** of rows,
no stratification. The sort gives sequential reads against the 10 GB `latent.npy` memmap and
lets `embed_all_rows` / `label_all_rows` use `np.setdiff1d(..., assume_unique=True)`.

Properties that matter downstream:

- **One draw, both fits.** `fit_latent = latent[indices]` feeds UMAP; `fit_space =
  space[indices]` feeds HDBSCAN. Identical 5 M loci, every cell.
- **Uniform over loci, not reads.** Stage 0 collapsed 5,078,201,907 reads into 157,501,580
  loci with a `locus_reads` weight, which this sample ignores — a locus with 500 supporting
  reads is as likely to be drawn as a singleton. Clustering happens in locus space while
  SigProfiler's catalogs are read-weighted. Defensible, but undocumented until now.
- **Small real islands vanish first.** Uniform sampling preserves density *ratios*, so an
  island holding 0.001% of loci contributes ~50 points at 5 M and ~1 at 100 k. This sets the
  floor for how far §5's scaling study can push N before it stops measuring sampling and
  starts measuring cluster disappearance.
- **The 50 k silhouette sample is drawn from all 157.5 M non-noise points**, so ~97% of the
  scored points were placed by `transform` / `approximate_predict`, not by the fit. See §6.
- `fit_rows = 5_000_000` was a CLI default, never justified against anything.

---

## 3. The metric trap: score in latent space, not embedding space

**This is the change that makes the whole plan valid.**

[stage2_sweep.py:356](stage2_sweep.py) currently does:

```python
panel = metrics.internal_metrics(space, labels, eval_rows=args.sil_eval_rows, seed=args.seed)
```

`space` is the 2-D UMAP embedding. So `silhouette = 0.578` means "in the 2-D picture UMAP
drew, these clusters look well separated." That is fine for stage 2 as built, where UMAP was
fixed and only `min_cluster_size` varied.

It becomes invalid the moment **UMAP itself** is the thing being swept:

- `min_dist` is literally a knob controlling how tightly UMAP packs points within a cluster.
  Silhouette measures within-cluster tightness against between-cluster separation. Ranking
  `min_dist` by embedding-space silhouette approximates measuring the knob's setting.
- Larger `n_neighbors` changes inter-cluster distances in the layout, with the same effect.

This is not only a theoretical worry. A microbiome UMAP study (mSystems, source below) reports
silhouette-measured separation improving at low `n_neighbors` but not high, and attributes it
explicitly to reduced inter-cluster distance in UMAP coordinates at higher `n_neighbors` — the
score tracked the hyperparameter, not the biology. A broader benchmark of internal validation
measures finds DBSCAN scoring best on silhouette and Davies–Bouldin while *not* having the
highest agreement with ground truth.

### The fix

Keep deriving labels in the 2-D embedding — that is where the clusterer runs — but compute the
score against the **16-D latent coordinates of those same points**. Stage 1 guarantees the
positional contract (row *i* of `latent.npy` ↔ row *i* of `space` ↔ `labels[i]`), so the
correspondence is exact. A configuration then scores well only if the groups it bounded are
also separated in the VAE's own space, which no UMAP hyperparameter can inflate.

**Implementation gotcha.** You cannot simply pass `latent` in place of `space`.
[clustering_metrics.py:102](../clust_regime_sweep/clustering_metrics.py) does
`np.asarray(X, dtype=np.float64)` over the *whole* array before subsampling, and `latent` is a
157.5 M × 16 memmap — that cast would try to materialise ~20 GB. `mean_wcss(X, labels)` has
the same problem. The correct shape of the change: draw a probe index up front (~200 k rows),
slice both `space[probe]` and `latent[probe]`, and call `internal_metrics` on the sliced
latent with the sliced labels.

### Clustering-independent companions

Add two metrics that need no clusterer at all, computed on the probe set:

- **Trustworthiness** — of the k points that are neighbours in the **2-D embedding**, how many
  were *not* neighbours in the 16-D latent? Those are **false neighbours**: UMAP invented
  proximity and pulled unrelated loci into one island.
- **Continuity** — the mirror: of the k points that were neighbours in the **16-D latent**, how
  many are *no longer* neighbours in the embedding? Those are **missing neighbours**: UMAP tore
  a genuine group apart.

Both in [0, 1], 1 perfect. sklearn ships `trustworthiness` only; continuity is the same
function with the two spaces swapped — a few lines.

Both are needed because the sweep's extremes fail in opposite directions: low `n_neighbors`
with `min_dist=0` fragments (continuity drops, trustworthiness stays high); large
`n_neighbors` over-merges (trustworthiness drops, continuity stays high). Existing runs
recorded trustworthiness only — and that was VAE-vs-input (handoff §5), not UMAP-vs-latent —
so the fragmentation mode is currently invisible.

**Known limitation, do not over-trust these.** Trustworthiness and continuity are purely
*local*. Published evaluations report embeddings scoring >0.95 on trustworthiness while global
geodesic distance correlation collapsed to ~0.3. They are necessary, not sufficient — use them
alongside the latent-space silhouette, not instead of it.

Also add **kNN-preservation rate**: fraction of each point's 15 latent-space neighbours still
among its 15 embedding neighbours. Rotation-invariant, no free parameters.

---

## 4. The cheap clusterer

### Primary — 2-D density-grid connected components

Bin the embedding into a 1024 × 1024 histogram over its bounding box, threshold occupied bins
at a fixed quantile, label 8-connected components with `scipy.ndimage.label`, map points back
by bin index.

- One `histogram2d` pass: **O(N)**, ~1–2 s for 5 M points, CPU-only, zero GPU memory.
- Fully deterministic — no RNG anywhere.
- **Density-based**, so it is in HDBSCAN's model class: it finds the archipelago rather than
  imposing convex cells on it. The handoff's `density_heatmap.png` shows ~20–30 high-density
  islands, which is exactly what this method is built to recover.
- Emits a noise fraction (points in sub-threshold bins) directly comparable to HDBSCAN's.
- Bounding-box binning auto-normalises the spatial-extent differences between embeddings; a
  quantile threshold is scale-free. The two obvious cross-config biases are handled.
- Inherently 2-D, which the `n_components = 2` constraint makes free rather than limiting.

Free parameters are `bins` and the density `quantile`. Fix both across the whole sweep and
record them; a sensitivity check at ±1 setting on the top 3 embeddings is cheap insurance.

### Cross-check — cuML KMeans

Fixed `k` across all embeddings, seconds on GPU. Its only job is to confirm the ranking is not
an artifact of the grid method. If grid-CC and KMeans disagree on the ordering, the ranking is
not robust and the tiebreaker is required.

### Tiebreaker — real HDBSCAN on the top 3

Worth stating plainly: **HDBSCAN was never the expensive part.** The per-cell cost is
dominated by full-cohort work (§1), not by the fit. A cuML HDBSCAN *fit* on a 250 k–500 k 2-D
subsample is seconds. So the most faithful proxy is affordable too — provided `predict` is
never called during ranking.

### Deferred blockers

Neither handoff §7 item 1 (nearest-centroid assignment) nor item 2 (RMM pool reset) is a
prerequisite here. No full-cohort labelling happens anywhere in a UMAP-only sweep, and the RMM
pool never approaches the ms=5 high-water mark that caused the ms=25/50 OOMs. Both defer to
the final HDBSCAN stage.

---

## 5. How much data does UMAP need?

Measurable directly with code already in the repo, and a scaled-down replica of the project's
headline question — which makes it a result rather than just tuning.

### What the literature already establishes

The original UMAP paper runs essentially this experiment on a comparably-shaped dataset (flow
cytometry, 1 M × 17; yours is 157.5 M × 16). Its metric is **normalized Procrustes distance**:
optimal translation / rotation / uniform scaling alignment, squared error, divided by the
average norm of the embedding. Subsample embeddings are compared against the *corresponding
points* of the full-data embedding.

Their result: at **5% subsampling (~50 k points)** UMAP's per-point error was already below
anything t-SNE achieved on the *full* dataset, and subsample embeddings stayed very close to
full embeddings. UMAP is unusually well-behaved under subsampling. Mild evidence that 5 M
(3.17%) is not obviously too small; stronger evidence that well below 5 M may suffice.

Stated complexity: kNN construction empirically **O(N^1.14)** (NN-Descent), optimization
**O(kN)** with k = `n_neighbors`. Both near-linear — halving N roughly halves the fit. Real
payoff, but linear, not dramatic.

### The study

Fit UMAP at **N ∈ {100 k, 250 k, 500 k, 1 M, 2 M, 5 M}** at one fixed config. Hold out a
**fixed 200 k probe set excluded from every fit**, `transform()` it through each model, compare
consecutive N.

Comparison metrics, in priority order:

1. **kNN-overlap of the probe set between successive embeddings** — rotation / reflection /
   translation-invariant by construction, which matters because UMAP embeddings are only
   defined up to those transforms.
2. **ARI / AMI of grid-CC labels on the probe set** —
   [`clustering_agreement`](../clust_regime_sweep/clustering_metrics.py) works unchanged.
3. **Normalized Procrustes disparity**, as the UMAP paper defines it. Normalise as they do, or
   values are not comparable across configs with different spatial extents.
   `compare_latent_spaces` in [evaluation.py:466](../uv_vae/uv_vae/evaluation.py) has the raw
   form. Its linear CKA is nearly uninformative on 2-D data — do not lean on it.

Where the curves plateau is the N to use.

### Two caveats to build in

- **N and `n_neighbors` are coupled.** At fixed `n_neighbors = 15`, a 5 M-row fit uses a
  neighbourhood covering a far smaller fraction of the data than a 100 k-row fit does, so the
  effective smoothing changes with N. The scaling study and the `n_neighbors` sweep cannot be
  cleanly separated. Simplest resolution: run the scaling study, fix N, sweep `n_neighbors`
  only at that N — and report the coupling rather than pretending it away.
- **The small-island floor from §2.** Below some N, a declining curve means clusters are
  disappearing, not that the embedding is unstable. Track cluster count alongside the
  agreement metrics so the two causes stay distinguishable.

---

## 6. Two verification gates that must pass first

### Gate 1 — determinism (likely a real problem, not a precaution)

`fit_umap` sets `random_state=int(config.seed)` ([sweep_core.py:236](sweep_core.py)). The cuML
UMAP documentation says:

- `build_algo` defaults to `'auto'`, which uses brute-force kNN at **≤ 50,000 rows and
  NN-Descent above it**.
- "Explicitly setting `build_algo='nn_descent'` will break reproducibility, as NN Descent
  produces non-deterministic KNN graphs."
- `random_state` gives reproducible *optimization* at the cost of slower training and more
  memory, because parallel float addition ordering is otherwise non-deterministic.

Fits here are on 5,000,000 rows, so `'auto'` resolves to NN-Descent. `random_state` pins the
optimizer, but **the graph it optimizes is built by the non-deterministic path.** The doc's
wording covers the *explicit* setting and I will not overclaim about `'auto'` — but it is the
same code path, and there are open reports of non-determinism under a fixed `random_state`
(cuML issue #5099).

**Action.** Fit one config twice at the chosen N; compute kNN-overlap and ARI between the two
runs. That is the **noise floor**. Any difference in the sweep smaller than it is not a
difference. Run this before reading any ranking.

**Possible fix to test:** forcing `build_algo='brute_force_knn'` restores a deterministic
graph. Brute force is quadratic in N, so it may only be viable at the low end of the scaling
study — but if the study lands on 500 k–1 M, it could be affordable, and a fully deterministic
sweep is worth real compute.

**Note on the IVF+spilling path.** The MLSys 2026 paper's IVF+spilling algorithm (what
`build_algo='auto'` uses at N > 50 k) runs hierarchical k-means clustering on a subsample,
then NN-Descent within each cluster. The k-means itself is approximately seeded but not exact
across runs. So even though individual cluster-local graphs are reproducible given a seed,
the global graph composition varies. Gate 1's fit-twice comparison will capture this combined
noise.

### Gate 2 — the out-of-sample repulsion effect

**Measured and cleared (phase A, 2026-08-04).**

200 k probe rows were embedded as insiders (part of the fit) and as outsiders (via
`transform()` on a model fit without them). Shift ratio = **0.994** (mean), **0.979**
(median) — no outward displacement detected. The repulsion effect described in
arXiv:2606.04451 is not present at this config and scale.

The 29.6% noise fraction at mcs=1000 has a different cause: genuine data noise or HDBSCAN
parameter sensitivity, not transform bias. Gate 2 is clear for the ranking sweep and for
the final full-cohort stage.

---

## 7. Phases

| Phase | What | Gates on | Est. cost |
|---|---|---|---|
| **A** | ~~Split the `fit_umap` / `embed_all_rows` timers; time a bare 5 M fit. Run the repulsion probe (Gate 2).~~ **DONE** — fit=119s (2%), transform=3291s (98%). Gate 2 cleared (shift ratio 0.994). | — | ✓ |
| **B** | Scaling study, 6 values of N, fixed config, 200 k probe set (§5). | A (for budgeting) | small N cheap; ~2–4 h |
| **B2** | Determinism floor (Gate 1): same config fitted twice, kNN-overlap + ARI. Optionally test `build_algo='brute_force_knn'`. | B (needs chosen N) | 2 fits |
| **C** | Build grid-CC clusterer, latent-space scoring, trustworthiness / continuity / kNN-preservation. | — | code only |
| **D** | UMAP sweep at chosen N, `n_components=2` fixed. | B, B2, C | 25 fits × measured fit time |
| **E** | Real HDBSCAN on top 3 embeddings; then handoff §7 items 1–2, 6, 7. | D | — |

Phase B before D is deliberate: if the geometry plateaus at 1 M, phase D gets ~5× cheaper, and
phase A says whether that matters.

**Grid for phase D:** `n_neighbors ∈ {5, 15, 30, 50, 100}` × `min_dist ∈ {0.0, 0.05, 0.1,
0.25, 0.5}`, `n_components = 2` — 25 embeddings. `min_dist` is widened past the old 0.25
ceiling *precisely because* latent-space scoring can now evaluate the high end fairly, which
embedding-space silhouette could not.

---

## 8. Where the code goes

Additive, matching the existing convention. Nothing in `uv_vae/uv_vae/` changes.

- **`sweep_core.py`** — add `grid_density_clusters(embedding, bins, quantile)`,
  `knn_overlap(a, b, k)`, `continuity(X_high, X_low, k)`, and a `UmapOnlyGrid`.
  `UmapConfig` / `fit_umap` are reused as-is (plus an optional `build_algo` passthrough for
  Gate 1).
- **`stage2a_umap_rank.py`** *(new)* — loads the stage-1 memmap, fits each UMAP config on the
  chosen N, clusters with grid-CC, scores with `internal_metrics(latent_probe, labels_probe)`,
  writes one `metrics.json` per embedding. Mirrors stage 2's resume-on-existing-file behaviour.
- **`stage2a_scaling.py`** *(new)* — phases B and B2.

Reused unchanged: `clustering_metrics.internal_metrics` / `clustering_agreement`,
`evaluation.compare_latent_spaces`, `stage1_embed.py` output.

---

## 9. Open questions

- **SigProfiler as a ranking criterion.** It is the only *external* validity signal available —
  everything in §3 is internal geometry — and it runs in ~2 s. The handoff rejects capped
  labelling because per-*sample* catalogs go sparse, but per-*cluster* catalogs at ~30 islands
  from 5 M rows would be dense. If that holds, mean cosine-to-SBS7A–D becomes a tier-3 ranking
  signal reflecting actual biology. **Check per-cluster read counts before committing.**
- **`n_components = 2`.** The reasoning is sound and the grid-CC clusterer is 2-D by
  construction, so it locks in cleanly. Standard practice often sits at 3–5 D; cheap to revisit
  if all 2-D embeddings score poorly in latent space. Not a reason to change the plan now.
- **Locus vs read weighting (§2).** Should the fit set be drawn ∝ `locus_reads` so the
  clustering sees read space, matching SigProfiler's weighting? Currently uniform over loci.
  Not urgent, but it should be a decision rather than a default.
- **Whether `write_analysis_parquet` is a major share of the per-cell cost (§1).** Phase A
  answers this, and the answer determines whether handoff §7 item 1 is worth building.

---

## 10. What the MLSys 2026 out-of-core UMAP paper adds

Park, Nolet, Naruse, Raff, Oates (NVIDIA/UMBC). *Massive-Scale Out-of-Core UMAP on the GPU*.
MLSys 2026. This is the academic description of what shipped in cuML 24.10 — the
`build_algo='auto'` path your pipeline already uses.

### The algorithm

1. Hierarchical balanced k-means on a subsample to get `c` cluster centroids.
2. Assign each vector to its `s` nearest centroids — **spilling** puts boundary points into
   multiple clusters so cross-cluster neighbours are not missed.
3. Build a local all-neighbors graph per cluster on GPU (only that cluster's points in VRAM
   at once).
4. Merge local graphs globally, keeping top-k by distance, deduplicating spilled points.

The two tunable parameters are **`c`** (number of clusters, controls per-cluster memory)
and **`s`** (spill factor, controls recall at the cost of more compute).

### Memory formula

Per-cluster GPU memory (Equation 1 from the paper):

```
M = (N·s/c) × D × 4 bytes  [data]
  + (N·s/c) × k × 8 bytes  [graph edges]
```

For your 5 M × 16 fit with k=15, c=16, s=2:

- Points per cluster: 5M × 2 / 16 ≈ 625,000
- Data: 625k × 16 × 4 = 40 MB
- Graph: 625k × 15 × 8 = 75 MB
- **Total: ~115 MB per cluster** — trivially within 48 GB.

For the full 157.5 M loci (if you ever fit UMAP on all of them), k=15, c=16, s=2:
- Per-cluster: ~19.6 M × 16 × 4 + 19.6 M × 15 × 8 ≈ 1.26 GB + 2.35 GB = **3.6 GB**
- Global kNN graph only: 157.5 M × 15 × 8 / 1e9 = **~18.9 GB** — fits on 48 GB.

### What it changes in the plan

**It doesn't change the sequencing or the phases** — but it validates two assumptions:

1. **Phase D is cheap.** The Food benchmark (5 M × 384, H100) took 21.16 s just for graph
   construction. Your 5 M × 16 is 24× lower dimension; the fit including optimisation should
   be well under 10 minutes. Phase D's 25-config sweep is a 2–4 hour job, not days.

2. **The full 157.5 M-row fit is not memory-impossible.** The global graph at 18.9 GB fits.
   The paper's bottleneck — graph construction up to 99% of UMAP time — still applies, so the
   fit cost would be large, but it is not ruled out by VRAM alone. This matters for the
   long-term pipeline: fitting UMAP on all loci and scoring all of them *during* the fit (no
   `transform` bias) becomes an option.

### Tuning `c` and `s`

These map to `build_kwds` in the cuML UMAP constructor:

```python
cuml.UMAP(build_kwds={"nnd_n_clusters": c, "nnd_n_clusters_spill_factor": s})
```

For 5 M × 16, the default auto-selection of `c` is almost certainly fine — memory is not
the constraint. `s=2` (default) is a reasonable starting point. Increasing `s` to 4
improves boundary recall at roughly 2× more graph-construction time. Worth one comparison
during phase B's scaling study, but not a priority: at 16 dimensions, cross-cluster
neighbours are easy to recover because the space is not high-dimensional enough for boundary
points to be deeply buried.

### What it does not solve

- **Determinism.** NN-Descent within clusters plus approximate k-means seeding means the
  IVF+spilling path remains non-deterministic. Gate 1 still required.
- **`transform()` bias (Gate 2).** Cleared by phase A — not present at this scale/config.
- **`transform()` of 152.5 M rows.** Not discussed. Confirmed as the dominant cost (98.2%
  of UMAP stage time) by phase A.

### Should you fit UMAP on all 157.5 M loci?

Phase A and the memory formula together make this a real option. Here is the honest
trade-off:

**Arguments for:**
- Eliminates `embed_all_rows` (55 min) entirely — every point is an insider, transform
  bias is impossible by construction regardless of the phase A finding.
- Simpler pipeline: no fit/held split, no `transform()` call, no batching artefacts.
- VRAM is not the constraint: global kNN graph = 18.9 GB, fits with ~29 GB headroom.
- Gives HDBSCAN the full density field to work with, not 3.17% of it — the condensed
  tree sees every locus, which is what HDBSCAN was designed for.

**Arguments against:**
- Estimated wall time: ~83 min for kNN graph construction alone (O(N^1.14) from phase A's
  119 s at 5 M) + optimisation passes on 157.5 M rows. Total likely **2–3 hours per fit**.
  That is slower than fit (2 min) + transform (55 min) = 57 min.
- Not measured — the extrapolation could be wrong. NN-Descent's empirical exponent varies
  with data dimensionality and cluster structure; 16D may be faster or slower than the
  flow-cytometry data the UMAP paper measured.
- Phase D (25 UMAP configs) becomes 25 × 2–3 h = 50–75 h instead of ~50 min. The whole
  point of the ranking sweep is to be cheap. **Do not attempt a full-cohort fit during
  phase D.**
- cuML has not been tested at 157.5 M rows on this card. At N=157.5 M the IVF+spilling
  path needs `c` tuned so each cluster fits in VRAM alongside the graph-merge workspace.
  This needs one test run, not 25.

**Recommendation:** finish the ranking sweep (phases B–D) on 5 M rows as planned. After
choosing the best UMAP config, run **one** full-cohort fit as a separate experiment to
measure actual wall time and compare embedding quality (kNN-overlap, Procrustes) against
the 5 M fit + transform result. If it is comparable in quality and wall time is acceptable
for a one-off run, adopt it for the final labelling. If it takes 3 h and quality is the
same, keep fit + transform — 57 min is already known to work.

---

## Sources

- [UMAP: Uniform Manifold Approximation and Projection (McInnes, Healy, Melville)](https://arxiv.org/pdf/1802.03426)
  — normalized Procrustes stability under subsampling; 5% of 1 M flow-cytometry points;
  kNN O(N^1.14), optimization O(kN).
- [On Out-of-sample Embedding in UMAP](https://arxiv.org/html/2606.04451)
  — the repulsion effect: out-of-sample points accumulate at cluster peripheries; accumulation
  statistic; mitigations.
- [cuML UMAP API documentation](https://docs.rapids.ai/api/cuml/nightly/api/generated/cuml.manifold.umap/)
  — `build_algo='auto'` switches to NN-Descent above 50,000 rows; NN-Descent breaks
  reproducibility; `random_state` cost.
- [cuML issue #5099 — Deterministic UMAP is not deterministic](https://github.com/rapidsai/cuml/issues/5099)
- [umap-learn reproducibility docs](https://umap-learn.readthedocs.io/en/latest/reproducibility.html)
- [umap-learn basic parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html)
- [TopOMetry — evaluating embeddings](https://topometry.readthedocs.io/en/latest/e_evaluations.html)
  — trustworthiness / continuity definitions.
- [On the Validating UMAP Embeddings](https://medium.com/data-science/on-the-validating-umap-embeddings-2c8907588175)
  — local metrics >0.95 alongside collapsed global geodesic correlation (~0.3).
- [UMAP reveals composite patterns and resolves visualization artifacts in microbiome data (mSystems)](https://journals.asm.org/doi/10.1128/msystems.00691-21)
  — silhouette tracking `n_neighbors` rather than biology.
- [Supervised application of internal validation measures to benchmark DR methods in scRNA-seq (bioRxiv)](https://www.biorxiv.org/content/10.1101/2020.10.29.361451.full.pdf)
  — best silhouette / Davies–Bouldin without best ground-truth agreement.
- [Why Can't I See My Clusters? A Precision-Recall Approach to Dimensionality Reduction Validation](https://arxiv.org/abs/2509.04222)
  — DR quality metrics assess projection reliability but do not explain missing structure.
- [hdbscan — predicting clusters for new points](https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html)
  — `approximate_predict` as condensed-tree descent given `prediction_data=True`.
- [hdbscan — performance and scalability](https://github.com/scikit-learn-contrib/hdbscan/blob/master/docs/performance_and_scalability.rst)
  — fit is sub-O(n²) with no clean closed form.
- [Massive-Scale Out-of-Core UMAP on the GPU (Park, Nolet, Naruse, Raff, Oates — MLSys 2026)](https://openreview.net/forum?id=CR35IJQD2J)
  — IVF+spilling algorithm: k-means partitioning + NN-Descent within clusters; shipped in
  cuML 24.10 as `build_algo='auto'`. Benchmark: 5.08 M × 384 kNN graph in 21.16 s on 1× H100.
  Confirms kNN construction is 75–99% of UMAP *fit* time. Memory formula (§10) shows global
  kNN graph for 157.5 M loci at k=15 ≈ 18.9 GB, within the 48 GB budget. Tuning levers:
  `nnd_n_clusters` (c) and `nnd_n_clusters_spill_factor` (s).
