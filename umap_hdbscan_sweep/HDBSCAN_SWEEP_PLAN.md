# HDBSCAN sweep plan — settling parameters for SigProfiler

Written 2026-08-11, after UMAP was finalised (`umap_tests/final_models/README.md`).
Scope: pick `min_cluster_size` and `min_samples` (EOM selection) on the chosen parametric
encoder, then hand finalised labels to SigProfiler.

---

## 0. What the existing sweep already tells us — read this first

The 5-cell `min_cluster_size` sweep from 2026-08-04
(`uv_vae/runs/nn15_md0.0_nc2/mcs{100,250,500,1000,2500}_ms5/`) already ran SigProfiler on
every cell. Re-reading its per-cluster stats
(`Solution_Stats/Assignment_Solution_Samples_Stats.txt`) changes the premise of this phase.

### Finding 1 — aggregate fit quality is flat across a 25× `min_cluster_size` range

Share of assigned mutations sitting in clusters SigProfiler fits well:

| mcs | clusters | median cluster (muts) | clusters cos>0.7 | **muts in cos>0.7** | clusters cos>0.8 | muts in cos>0.8 |
|---|---|---|---|---|---|---|
| 100 | 5,709 | 4,220 | 544 | **10.7 %** | 185 | 3.2 % |
| 250 | 2,763 | 15,229 | 313 | **10.4 %** | 98 | 2.8 % |
| 500 | 1,731 | 36,324 | 197 | **10.4 %** | 55 | 3.0 % |
| 1000 | 1,150 | 73,996 | 127 | **10.6 %** | 35 | 3.3 % |
| 2500 | 613 | 160,073 | 68 | **10.2 %** | 21 | 3.5 % |

`min_cluster_size` does **not** change how much of the cohort SigProfiler explains
confidently. It only changes how that fixed ~10.5 % is packaged — 544 small clusters or 68
large ones. Spearman correlation between cluster size and cosine similarity, pooled across
all five cells, is **0.073**: essentially none.

So "larger clusters give better SigProfiler assignment" is true only in a narrow sense
(below), and it is not a reason to push `min_cluster_size` up. Pushing it up costs
resolution and buys no aggregate accuracy.

### Finding 2 — there is a real floor and a real ceiling, both narrow

Fraction of clusters with cosine > 0.7, by cluster mutation count, computed *within* each
`min_cluster_size` cell so the comparison is not confounded by the cells' differing size
distributions:

| mcs | <3k | 3–10k | 10–30k | 30–100k | 100–300k | >300k |
|---|---|---|---|---|---|---|
| 100 | 6 % | 12 % | 12 % | 14 % | 9 % | 0 % |
| 250 | 2 % | 11 % | 13 % | 12 % | 10 % | 0 % |
| 500 | 7 % | 5 % | 12 % | 13 % | 10 % | 0 % |
| 1000 | — | 0 % | 4 % | 13 % | 10 % | 7 % |
| 2500 | — | — | 0 % | 10 % | 12 % | 10 % |

Two effects survive de-confounding:

- **Floor ≈ 3,000 mutations.** Clusters below it fit badly in every cell that has any
  (2–7 % vs 11–14 %). Your instinct is right, but it saturates at 3k — not at 30k or 300k.
- **Ceiling ≈ 300,000 mutations.** Clusters above it are uniformly bad (0–10 %, median
  cosine 0.04–0.12) in every cell. Very large clusters are signature *mixtures* that no
  single reference profile fits. This is the effect that caps how far up `mcs` can help.

The comfortable window is roughly **3k–300k mutations per cluster**, with 30k–100k the
best-performing band. That is the target to hit — via *any* combination of parameters, not
by `min_cluster_size` alone.

### Finding 3 — do not derive `min_cluster_size` from a scale factor

Tempting arithmetic: fit on 5 M of 157.5 M loci → 31.5× scale → `mcs=100` guarantees
~3,150-mutation clusters. **The data says otherwise** — the smallest cluster at `mcs=100`
held 174 mutations, not 3,150. `approximate_predict` sends held-out rows near cluster
boundaries to noise, so small fit-set clusters attract disproportionately few of them and
shrink. Measure the realised size distribution; do not predict it.

### Finding 4 — the 89 % low-cosine population may not be a clustering problem at all

Median cosine is **0.296 at every single `mcs`**, and the distribution is bimodal. The row
filter (`st='MIXED' AND et='MIXED' AND FILT=1`) admits non-UV variants, and the assignment
used the restricted `uv_only` signature set. If most reads are not UV, no UV-only reference
can fit them, and no HDBSCAN parameter will change that.

**This has to be settled before the sweep, not after** — it decides what the sweep is even
optimising. See Phase 0, step 1. It is a ~2 s re-run on data already on disk.

---

## Phase 0 — settle the objective and unblock the code (do before any sweeping)

**1. Re-run assignment with the full COSMIC v3.5 SBS set on one existing cell.**
Take `mcs1000_ms5`'s existing `input/cluster_sbs96_matrix.tsv` and re-run
SigProfilerAssignment against full COSMIC instead of `uv_only_SBS_GRCh38.tsv`. Cost: ~2 s,
no clustering, no GPU.

- If the low-cosine population fits *other* signatures → those clusters are real
  non-UV structure, the pipeline is working, and the objective becomes "maximise clusters
  above the cosine threshold across the full reference set."
- If they still fit nothing → they are genuinely unstructured reads, and the objective
  becomes "maximise the mutation share in well-fit clusters while keeping noise low."

Everything downstream ranks against whichever objective this produces. Do not skip it.

**2. Pin the embedding once.** Embed all 157,501,580 loci through
`final_models/13_BEST_25M_nn15_md0.1_umap.pt` → save `coords.npy` (float32, 157.5M × 2,
≈1.26 GB). ~22 s per the final_models README. Every cell below reads this file; nothing
re-embeds.

**3. Pin the fit set.** 5,000,000 rows drawn with seed 42, index saved. 5 M is the measured
cuML HDBSCAN safe limit — 70 M rows at `ms=25` ran 26 h and then OOMed at 56 GB on a 47 GB
card. Do not raise it. Save a second 5 M draw at seed 43 now; Phase 2 needs it.

**4. Run `phase_a_hdbscan_timing.py`. It has never been executed** — there is no
`umap_tests/phase_a_hdbscan_timing.json`. The whole cost model of this phase is unverified:
the 2026-08-04 handoff blames `approximate_predict` for the ~65 min/cell, `UMAP_RANKING_PLAN.md`
disputes it and points at the multi-GB per-cell parquet write. One ~30 min run decides
whether the plan below costs 3 hours or 3 weeks, and whether step 5 is worth building.

```bash
python umap_hdbscan_sweep/phase_a_hdbscan_timing.py \
    --embed-dir <stage1-output> --output-dir umap_hdbscan_sweep/umap_tests \
    --coords-parquet <a finished cell's analysis parquet> --min-samples 5
```

**5. Implement nearest-centroid assignment in `sweep_core.py`** — *only if step 4 confirms
predict is the bottleneck.* Open item 1 from the handoff; `grep centroid sweep_core.py`
returns nothing, so it was never built. Every unlabelled row goes to its nearest cluster
centroid: seconds instead of an hour, still labels every row (which SigProfiler requires —
a label-rows cap does not work, it starves the per-cluster catalogs). Accuracy loss is
confined to boundaries.

**6. RMM pool reset between `min_samples` groups in `stage2_sweep.py`.** Open item 2. Lower
priority for you: it bit at `ms=25`/`ms=50`, and your range tops out at 15. Add it anyway —
it is a few lines and prevents a silent OOM at the top of the range.

---

## Phase 1 — settle `min_cluster_size`

Fixed: EOM selection, `min_samples=5`, the pinned 5 M fit set, the pinned coords. No
full-cohort labelling, no SigProfiler in this phase.

**Ladder:** `mcs ∈ {100, 250, 500, 1000, 2000, 4000, 8000}` — 7 fits, ~5–10 min each on
GPU, ≈1 hour total. It brackets the old sweep on the high side because the new embedding
produces far coarser structure: model 13 gave 505 clusters at `mcs=250` on a 500 k probe,
which is equivalent to `mcs≈2500` at 5 M scale. **The old cell-to-cluster-count mapping does
not transfer** — different encoder, `min_dist` 0.1 not 0.0, parametric not non-parametric.

**Record per cell:** `n_clusters`, noise %, and the full realised cluster-size distribution
(min, p10, median, p90, max) expressed in *full-cohort mutations*, not fit-set rows. Plus
silhouette / DB / CH on a 50 k subsample and DBCV via `rescore_dbcv.py` (note its limits in
`LIMITATIONS.md`: 25 k-row sample, ranking score only, cells above 500 clusters skipped).

**Decision rule, fixed in advance:** choose the smallest `mcs` such that

1. the **p10 cluster clears 3,000 full-cohort mutations** (Finding 2's floor), and
2. the **p90 cluster stays under 300,000** (Finding 2's ceiling), and
3. noise % is at or below the lowest cell's noise + 10 points.

Smallest, not largest — resolution is free given Finding 1, and the floor is the only thing
`mcs` needs to buy. If no `mcs` satisfies both 1 and 2 simultaneously, the distribution is
too heavy-tailed for `mcs` alone; go to the epsilon option below.

**If you need larger clusters, `cluster_selection_epsilon` is the better lever than `mcs`.**
`HdbscanConfig` already supports it (`sweep_core.py:302`) and it is currently 0.0. It merges
clusters closer than epsilon, which pulls up the *small* end without raising the floor under
genuinely distinct structure the way `mcs` does. Worth 3 cells (`eps ∈ {0.05, 0.1, 0.25}` at
the chosen `mcs`) if Phase 1 leaves the p10 short of 3k. Note `label()` appends `eps…` so
these land in their own directories and will not collide with the resume logic.

---

## Phase 2 — settle `min_samples`

Fixed: the `mcs` from Phase 1, EOM.

**Ladder:** `ms ∈ {1, 2, 3, 5, 8, 12, 15}` — 7 fits. (`min_samples` must be ≥ 1; 0 is not
valid. `ms=1` is the maximally permissive extreme and is worth including as the boundary
case even though it will be noisy.)

`min_samples` in EOM mostly moves the noise fraction and boundary conservativeness, not the
cluster count. So rank it on **stability, not on internal metrics** — this is the one place
the README's caveat 2 bites:

> rep ARI is the relevant one for the clustering phase — 0.261 for model 13 is the baseline
> to improve on.

**Protocol:** fit each `ms` on both pinned 5 M draws (seeds 42 and 43), label the *same*
held-out 500 k probe set from each, and compute ARI between the two labellings. Pick the
`ms` maximising that ARI subject to noise % staying under the Phase 1 cap. Report the ARI
against the 0.261 baseline — if none of the seven beats it, say so; that is a finding about
the embedding, not a reason to pick the least-bad `ms` silently.

Memory grows with `min_samples` (larger kNN graphs). At 5 M rows and `ms ≤ 15` this is well
inside budget — the `ms=5` cells peaked at 13.6 GB of a 14.4 GB pool at the old scale, and
`ms=25` needed ~1 GB more. Use `GPU_TOTAL_GB=40 TRAINER_GPU_GB=0` for HDBSCAN-only jobs.

---

## Phase 3 — SigProfiler on a shortlist

Carry **2–3** `(mcs, ms[, eps])` candidates, not one: the Phase 1 winner, one deliberately
finer, one deliberately coarser. For each: full-cohort labelling → SBS96 matrix →
SigProfiler under the Phase-0 objective.

Compare on: cosine distribution, count and mutation-share of clusters above threshold,
whether all four SBS7 subtypes (7A/7B/7C/7D) separate into distinct clusters as they did at
`mcs=1000`, and whether SBS38 still appears.

---

## Phase 4 — the acceptance test

From the final_models README, caveat 4, and it is the caveat that decides whether any of
this holds:

> Cluster two independently trained encoders and compare aggregated SigProfiler profiles
> rather than point assignments — boundaries can shuffle while the aggregate profile holds.

Run the chosen `(mcs, ms)` unchanged through **model 14** (`min_dist` control) and **model
11** (mode control), and compare aggregated signature profiles against model 13's. Agreement
means the parameters describe the data. Disagreement means they describe the encoder, and
the choice does not generalise.

Cluster *counts* are seed-sensitive by ±5 % at 25 M fit rows (caveat 1) — do not read a
count difference between encoders as a real effect.

---

## Cost summary

| phase | cells | est. cost | gates the next phase? |
|---|---|---|---|
| 0.1 full-COSMIC re-assignment | 1 | ~2 s | **yes — sets the objective** |
| 0.4 timing decomposition | 1 | ~30 min | **yes — sets the budget** |
| 1 `mcs` ladder | 7 | ~1 h | yes |
| 1b epsilon (conditional) | 3 | ~30 min | no |
| 2 `ms` ladder × 2 seeds | 14 | ~2 h | yes |
| 3 SigProfiler shortlist | 2–3 | depends on 0.4 | — |
| 4 cross-encoder acceptance | 2 | depends on 0.4 | — |

Phases 1 and 2 together are ~24 cheap cells and no full-cohort labelling. The expensive
work is confined to Phase 3 onward, and how expensive it is depends entirely on what step
0.4 measures.
