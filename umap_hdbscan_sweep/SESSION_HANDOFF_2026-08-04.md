# Session handoff — 2026-08-04

Record of what the stage-2 UMAP/HDBSCAN sweep actually produced, what it cost, and
why the next round of experimentation changes shape. Written at the point where the
sweep was killed after 5 of 480 planned cells.

---

## 1. What was fixed before the run

Two GPU-memory bugs were found and fixed in the previous session; both were verified
on real hardware during this one.

**Bug A — SigProfiler stomping the RMM pool.**
`run_variant_cluster_pipeline.py` calls `gpu_budget.apply()` at *import* time with a
0.25 share (4 GB). Importing it from inside `run_sigprofiler()` therefore reinitialised
the RMM pool while ~3.7 GB of cuML state was still live. Fixed by setting
`os.environ["UV_VAE_DISABLE_CUML"] = "1"` before the import in `stage2_sweep.py`.

**Bug B — `initial_pool_size` not 256-byte aligned.**
`gpu_budget.py` computed `initial_bytes = pool_bytes // 4`, which is not always a
multiple of 256. RMM rejects that. For the 16 GB × 0.9 case the initial size came out
at 3,865,470,528 and `rmm.reinitialize` failed silently into a CPU fallback. Fixed
(commit `deaaa35`) by `initial_bytes -= initial_bytes % 256`.

**Smoke-test confirmation on miletus:**

```
rmm_pool_gb: 14.4
rmm_note:    None
installed:   15461882112
```

`rmm_note: None` is the signal that matters — the pool installed without falling back.

A no-shrink guard (`_INSTALLED_POOL_BYTES`) was also added: once a pool is installed it
is never replaced by a smaller one. This turns out to matter later (§4).

---

## 2. What was run

One UMAP configuration, five HDBSCAN cells.

**UMAP (fixed for all five cells):**

| parameter | value |
|---|---|
| `n_neighbors` | 15 |
| `min_dist` | 0.0 |
| `n_components` | 2 |
| `metric` | euclidean |
| `n_epochs` | 200 (pinned for determinism) |
| `seed` | 42 |

**HDBSCAN (swept over `min_cluster_size`, `min_samples` fixed at 5):**

| parameter | value |
|---|---|
| `min_cluster_size` | 100 / 250 / 500 / 1000 / 2500 |
| `min_samples` | 5 |
| `metric` | euclidean |
| `cluster_selection_method` | eom |
| `prediction_data` | True |
| `gen_min_span_tree` | True |

**Scale — three different row counts are involved, and conflating them is easy:**

- HDBSCAN was **fitted** on a 5,000,000-row subsample of the UMAP embedding (`fit_rows`).
- Labels were then **extended to all 157,501,580 reads** via `approximate_predict`.
  So `n_clusters` and `noise_fraction` are full-cohort numbers.
- Silhouette / Davies–Bouldin / Calinski–Harabasz were computed on a further
  **50,000-point** subsample (`internal_eval_points`) — silhouette is O(n²) and cannot
  run on 157.5 M points. These quality scores are *not* full-cohort numbers.
- SigProfiler input (`cluster_sbs96_matrix.tsv`) was built from the full 157.5 M labels.

---

## 3. Results

### Clustering metrics

| mcs | n_clusters | noise % | silhouette | Davies–Bouldin | Calinski–Harabasz | seconds |
|---|---|---|---|---|---|---|
| 100 | 5,709 | 50.5 % | 0.473 | 0.487 | 112,161 | 4155 |
| 250 | 2,763 | 43.4 % | 0.531 | 0.538 | 181,525 | 3962 |
| 500 | 1,731 | 36.7 % | 0.565 | 0.560 | 284,553 | 3899 |
| **1000** | **1,150** | **29.6 %** | **0.578** | 0.553 | 286,460 | 3861 |
| 2500 | 613 | 20.8 % | 0.562 | 0.574 | 173,562 | 3827 |

`min_cluster_size = 1000` is the interior optimum. Below it the manifold is
over-split into small noisy fragments; above it real sub-structure merges away.
Silhouette > 0.5 on four of five cells indicates genuinely separated clusters.

`dbcv` is `null` on every cell — the HDBSCAN-native validity metric was never computed.

### Visual structure

`plots/density_heatmap.png` shows an **archipelago**: roughly 20–30 high-density islands
with sparse space between them. Not a uniform blob — the VAE learned discrete structure.

`plots/comparison_grid.png` shows the same archipelago at all five `min_cluster_size`
values. The islands are stable; `min_cluster_size` only changes how finely each island
is subdivided. This is the strongest evidence that the structure is real rather than a
clustering artefact.

### SigProfiler (uv_only signature set, GRCh38, COSMIC v3.5)

Best-fitting clusters at mcs=1000:

| cluster | signature | cosine similarity |
|---|---|---|
| cluster_720 | SBS7C | 0.903 |
| cluster_729 / 898 / 1045 | SBS7D | 0.902 |
| cluster_583 | SBS7C | 0.895 |
| cluster_592 | SBS7C | 0.891 |
| cluster_178 | SBS7C | 0.890 |
| cluster_354 / 353 | SBS7C | 0.850 / 0.847 |

All four UV subtypes (SBS7A, 7B, 7C, 7D) plus SBS38 appear cleanly separated across
different clusters. SBS7C at cosine 0.903 is the headline: it is the CC→TT tandem
signature at dipyrimidine sites, the most specific UV-damage fingerprint, and it was
recovered without supervision.

The cosine distribution is **bimodal**. A large population sits above ~0.70 and another
below ~0.30. The low-cosine clusters are not failures — the row filter
(`st='MIXED' AND et='MIXED' AND FILT=1`) admits non-UV variants, so those clusters are
the pipeline reporting "these reads group together, but not by a UV signature." That is
information, not error.

### Where the files are

Per-cell SigProfiler output (5 cells, all at `min_samples=5`):

```
uv_vae/runs/nn15_md0.0_nc2/mcs{100,250,500,1000,2500}_ms5/sigprofilerassignment_uv_only_grch38_v3.5/
```

Inside each: `input/` (matrices fed in) and `output/Assignment_Solution/` with
`Activities/`, `Signatures/`, `Solution_Stats/`. The two files that carry the result are
`Solution_Stats/Assignment_Solution_Samples_Stats.sorted_by_cosine_similarity_desc.txt`
and `Activities/Assignment_Solution_Activities.txt`.

Plots: `plots/{comparison_grid,metrics_vs_param,density_heatmap,cluster_size_hist}.png`,
generated by `plot_sweep_comparison.py` and `plot_umap_clusters.py`.

---

## 4. Why the sweep was killed

Two independent problems, both structural rather than incidental.

**Cost.** Each cell took ~3,900 s (≈65 min), and the time is dominated by
`approximate_predict` labelling all 157.5 M rows through a kNN lookup against the fitted
condensed tree. The code already uses the cuML GPU implementation — 65 min *is* the GPU
speed, there is no faster switch to flip. SigProfiler itself takes ~2 s per cell and is
not the bottleneck. At 480 planned cells this is ≈520 GPU-hours, over three weeks of
continuous occupancy on a shared node.

A `--label-rows` cap was considered as a workaround (cap prediction at 5 M rows, reducing
per-cell time to ~2 min). It was rejected because SigProfiler requires a label for *every*
read to build dense per-cluster mutation catalogs — a 5 M-row cap would give only ~52 k
reads per sample, making per-cluster counts too sparse for reliable NMF decomposition.
Capping fit rows is fine (UMAP and HDBSCAN already do this); capping label rows is not.

**OOM on `min_samples` > 5.** The RMM pool grows to its high-water mark and never
shrinks. The `ms=5` cells peaked at 13.637 GB of a 14.4 GB pool, leaving 0.763 GB. The
`ms=25` and `ms=50` cells need roughly 991 MB more for their larger kNN graphs and fail
immediately. Every `ms=25` and `ms=50` cell in the run has only a `metrics.json` with an
error recorded. Note this is the no-shrink guard from §1 working as designed and having
an unintended consequence — the guard is correct, but the sweep needs an explicit pool
reset between `min_samples` groups.

---

## 5. Known weaknesses in the VAE feeding this

Diagnosed during this session from `latent_metrics.evaluate_checkpoint()`:

- `active_units`: 16 / 16 — no dead dimensions.
- `effective_dimensionality` (participation ratio): **11.33 / 16** — the model uses
  about 11 dimensions meaningfully, 5 are near-redundant.
- `off_diagonal_cov_max`: 0.878 — noticeable correlation between latent dimensions.
- `trustworthiness`: 0.993 at k=10, 0.984 at k=50 — local neighbourhoods survive the
  encoding well.
- Highest per-feature reconstruction MSE: **BCSQ at 1.007** — variant consequence type
  is the hardest feature for the model.

**Two features are pure noise.** The row filter pins `st = 'MIXED'` and `et = 'MIXED'`,
so both are constant across every training row while still consuming embedding
parameters.

**Fourteen numerics are entirely null** (`sd`, `ed`, `l1`–`l7`, `q2`–`q6`), MSE 0.0.
They occupy input width and decoder head capacity while contributing nothing.

Dropping these 16 features and retraining is expected to improve effective
dimensionality, but requires a fresh training run on a free GPU and has not been done.

**SNVQ (base quality score) was considered as an additional feature** and rejected.
The row filter `FILT=1` already gates reads on quality, so SNVQ has low variance within
the filtered set. More importantly, SNVQ varies *within* a locus (different reads at the
same position have different quality scores) — adding it would inject sequencer noise
rather than mutational signal, and would not help with the locus-clustering problem since
reads at the same locus would still share the same genomic context features.

---

## 6. The new plan — settle UMAP first, defer HDBSCAN

The sweep as designed couples two decisions that do not need to be coupled. Every
HDBSCAN cell re-pays the 65-minute full-cohort labelling cost, so a 16 × 30 grid over
(UMAP config × HDBSCAN config) multiplies a cheap decision by an expensive one.

**New sequencing:**

1. **Choose the UMAP embedding first**, evaluated with a clustering algorithm cheap
   enough to run once per candidate embedding. The point is to rank embeddings, not to
   produce final labels — so the evaluation clusterer only has to be *consistent*, not
   final. Candidates worth considering: k-means / MiniBatchKMeans (cuML, near-linear,
   trivially fast on 5 M rows), or Leiden/Louvain on the kNN graph UMAP already builds.
   Evaluate on the 5 M fit subsample only — no full-cohort labelling at this stage.

2. **Fix the two engineering blockers** before any full-cohort run:
   - Replace `approximate_predict` with **nearest-centroid assignment** for the
     unlabelled rows. Each point goes to its nearest cluster centroid — seconds instead
     of 65 minutes, and it still labels *every* row, which SigProfiler requires (it
     needs a complete per-read assignment to build per-cluster catalogs; a label-rows
     cap will not do). Accuracy loss is confined to cluster boundaries and is acceptable
     for SigProfiler's aggregate statistics.
   - Add an explicit **RMM pool reset between `min_samples` groups** so `ms=25` and
     `ms=50` stop OOMing.

3. **Then finalise HDBSCAN** on the single chosen embedding, sweeping
   `min_cluster_size` and `min_samples` properly, with DBCV computed (currently `null`
   everywhere — `rescore_dbcv.py` exists for this).

4. **Then run SigProfiler** on the finalised labels only.

The core scientific question — whether a VAE trained on a ~5 M-row representative
subsample reproduces the latent space of a VAE trained on the full 157.5 M rows — is
still untouched. Everything above is infrastructure for measuring it.

---

## 7. Open items, ordered

1. Implement nearest-centroid assignment in `sweep_core.py`.
2. Implement RMM pool reset between `min_samples` groups in `stage2_sweep.py`.
3. Pick and wire up the cheap evaluation clusterer for UMAP-only ranking.
4. Sweep UMAP (`n_neighbors`, `min_dist`, `n_components`) and settle on one embedding.
5. Drop `st`, `et`, and the 14 null numerics from `ml_features.json`; retrain the VAE.
6. Finalise HDBSCAN on the chosen embedding; run `rescore_dbcv.py`.
7. Train the 5 M-row subsample VAE and compare its latent space against the full-data
   VAE (Procrustes / CKA / trustworthiness) — the actual deliverable.
