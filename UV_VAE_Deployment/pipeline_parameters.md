# Pipeline Parameters Master Sheet

---

## 1. VAE — Tabular Variational Autoencoder

### Architecture

| Parameter | Value |
|---|---|
| Model type | VAE |
| Latent dimension | 16 |
| Encoder hidden dims | [256, 128] |
| Decoder hidden dims | [128, 256] (symmetric) |
| Categorical input | Learned embedding per feature, dim = min(16, 2·⌈card^0.5⌉) |
| Numeric input | z-score standardised; NULL → training mean (= 0 post-standardisation) |
| Reconstruction heads | Separate numeric MSE head + per-categorical cross-entropy head |

**Reconstruction heads explained:** The decoder has one output head per feature type. Numeric features (e.g. read quality, length) use MSE loss — the head predicts a single continuous value and is penalised by squared error. Categorical features (e.g. REF base A/C/G/T) use cross-entropy loss — the head predicts a probability distribution over the feature's vocabulary and is penalised by log-loss. Separate heads are required because numeric and categorical outputs need fundamentally different loss functions; a shared head cannot optimise both simultaneously.

### Training — Cohort Run (final model)

| Parameter | Value |
|---|---|
| Dataset | 95 per-sample parquet files, 5.08 billion rows post-filter |
| Row filter | `st = 'MIXED' AND et = 'MIXED' AND FILT = 1` |
| Epochs (requested / run / best) | 40 / 38 / **30** |
| Batch size | 1,048,576 rows |
| Learning rate | 0.001 (Adam) |
| KL weight (β) | 0.005 |
| Input dropout | 0.1 |
| Hidden dropout | 0.1 |
| patience | 8 |
| Train / val split | 90 / 10 (content-hash split on genomic site, seed 42) |
| Seed | 42 |
| Best val ELBO | 0.2209 |
| Final active units | 16 / 16 (no collapsed dimensions) |
| Hardware | NVIDIA RTX PRO 5000 Blackwell (48 GB) |
| Mixed precision | AMP (float16 forward, float32 gradients) |

**Training time:** Total wall-clock time 4 h 25 m 24 s (38 epochs on 5.08 B rows). Torch VRAM budget capped at 16 GB; projected per-batch allocation ~5.63 GB (batch size 1,048,576 × 5,765 bytes/row); peak VRAM not separately recorded.

### Early Stopping

Stopped when **both** conditions held for `patience = 8` consecutive epochs:
- Relative val ELBO improvement < 0.001
- Active unit count (Burda et al. 2016 criterion, threshold 0.01) stable

### Features Used

**Categorical (11):** REF, ALT, X_PREV1, X_NEXT1, X_PREV2, X_NEXT2, X_PREV3, X_NEXT3, tm, st, et

**Numeric (29):** X_HMER_REF, X_HMER_ALT, BCSQ, BCSQCSS, RL, INDEX, REV, SCST, SCED, SMQ_BEFORE, SMQ_AFTER, rq, sd, ed, l1–l7, q2–q6, EDIST, HAMDIST, HAMDIST_FILT

*(l1–l7, sd, ed, q2–q6 non-null in 63.8 M / 5.08 B rows, sourced exclusively from sample ddb0-2-ppm0028; no features were dropped — `dropped_all_null_features` and `dropped_sample_null_features` are both empty. All 5.08 B rows passed through training; NULLs in sparse features were imputed to the training mean, which equals 0.0 post-standardisation, with the mask tensor distinguishing real values from imputed ones.)*

---

## 2. UMAP — Parametric UMAP

### Selected Configuration (25M-row fit, applied to full cohort)

| Parameter | Value |
|---|---|
| Mode | Parametric UMAP (neural encoder, trained once, applied to all rows) |
| Fit rows | 25,000,000 |
| n_neighbors | 15 |
| min_dist | 0.1 |
| n_components | 2 |
| metric | Euclidean |
| n_epochs | 200 |
| spread | 1.0 |
| repulsion_strength | 1.0 |
| set_op_mix_ratio | 1.0 |
| Seed | 42 |
| Encoder fit time (25M rows) | 102 s |
| Full-cohort apply time (157.5M rows) | 22 s |
| GPU VRAM | Not separately recorded in sweep artifacts |

---

## 3. HDBSCAN — Cohort Clustering

### Selected Configuration (sweep cell `fit1000000_mcs2500_ms15_eom`)

| Parameter | Value |
|---|---|
| min_cluster_size (mcs) | 2,500 |
| min_samples (ms) | 15 |
| cluster_selection_method | eom (excess of mass) |
| cluster_selection_epsilon | 0.0 |
| Fit rows | 1,000,000 (random subsample of 2D UMAP coordinates, seed 42) |
| Input | 2D UMAP coordinates (157,501,580 × 2, from the rank-13 parametric encoder) |
| **Clustering backend** | **cuML (RAPIDS GPU)** — the production model |
| Clusters found | 175 |
| Noise fraction (cohort) | 7.36 % |
| Noise fraction (fit set) | 7.2375 % |
| Fit time | 18.7 s |
| Label time (full cohort, 157.5 M rows, RBC fast-predict) | 127.0 s |

### Cluster Quality (cuML)

| Metric | Value |
|---|---|
| DBCV (stratified, 400 pts/cluster, `hdbscan` backend) | 0.4074 |
| DBCV, per-cluster median / min / max | 0.4729 / −0.3186 / 0.9052 |
| Points in negative-DBCV clusters | 11.85 % |
| Clusters with negative DBCV | 10.86 % (19 / 175) |
| Relative validity (`relative_validity_`) | *not available* — cuML does not expose it |
| Connectivity (k = 10, mean / fraction pure) | 0.999964 / 0.999855 |
| Mean membership probability | 0.8284 |
| Fraction of points with probability > 0.8 | 0.7216 |
| Mean probability, held-out rows | 0.8339 |
| Cluster persistence (median) | 1.0000 — *degenerate, see note* |

> **DBCV changed backend.** An earlier version of this sheet reported **0.4313**, scored with
> the `kdbcv` backend. The deployed run pins `DBCV_BACKEND=hdbscan` so the cuML and CPU
> figures are computed by the same implementation and are therefore comparable — which they
> were not before. 0.4074 and 0.4313 describe the *same partition* scored two ways; neither is
> wrong, but only the `hdbscan` figure belongs beside the CPU column below.

### Signature Fit — SigProfilerAssignment (`uv_only`, GRCh38, v3.5)

| Metric | Value |
|---|---|
| Total mutations in SBS96 matrix | 145,916,931 |
| Cosine similarity, mean / median | 0.2804 / 0.2390 |
| Cosine similarity, mutation-weighted mean | 0.2774 |
| Clusters with cosine > 0.7 | 11 (6.29 %), carrying 5.65 % of mutations |
| Clusters with cosine > 0.8 | 3 (1.71 %), carrying 1.39 % of mutations |
| L1 residual, mean % | 169.73 |
| Top-channel share, median | 0.6526 |
| Fraction of clusters with top-channel share > 0.5 | 0.6629 |
| Distinct dominant channels | 38 across 175 clusters |

### CPU Cross-Check (reference implementation, *not* the deployed model)

The same cell refit with CPU `hdbscan` 0.8.44 to obtain the diagnostics cuML does not expose:

| Metric | cuML (deployed) | CPU `hdbscan` |
|---|---|---|
| Clusters | 175 | 181 |
| Noise (cohort) | 7.36 % | 8.00 % |
| DBCV (`hdbscan` backend, both) | 0.4074 | 0.3674 † |
| Relative validity | not exposed | 0.2160 |
| Points in negative-DBCV clusters | 11.85 % | 13.12 % |
| Mean membership probability | 0.8284 | 0.8105 |
| Cluster persistence (median / min / max) | 1.0 / 1.0 / 1.0 (degenerate) | 0.1086 / 0.0009 / 0.6069 |
| Fit / label time | 18.7 s / 127.0 s | 10.7 s / 120.1 s † |

**Backend note (important).** cuML and CPU HDBSCAN are two implementations of the same
algorithm and do not produce identical partitions: across the full 24-cell sweep grid they
differ on **every** cell, with cuML's cluster count ranging from 7.3 % below to 22.6 % above
the CPU count. The cuML fit is the deployed model because it is the one the sweep selected
this cell on and the one all reported clustering and signature metrics were computed from.
cuML does not populate `outlier_scores_` or `exemplars_` and returns a degenerate
`cluster_persistence_` of exactly 1.0 for every cluster, so persistence, GLOSH outlier scores
and exemplars can only be quoted from the CPU refit — and must be labelled as coming from a
181-cluster partition, not the deployed 175-cluster one.

DBCV **is** comparable across the two columns above, which it was not previously: both are now
scored with the `hdbscan` backend on independent 400-points-per-cluster stratified samples.

† The CPU column comes from the August-2026 `param_sweep_refit_cpu` run of this cell, not from
the `sweep_both_backends` run that produced the deployed cuML model. The two CPU fits use
identical parameters, but CPU `hdbscan` figures here have not been re-verified against the
later sweep. Read them as the reference implementation's behaviour on this cell, not as
numbers measured alongside the shipped model.

An earlier version of this sheet reported 170 clusters at ms = 1, eps = 0.05, 6.88 % noise.
Those values came from the `low_noise_hdbscan` run — a **different parameter cell** — and were
incorrect for this section regardless of backend. See
`umap_hdbscan_sweep/hdbscan/hdbscan_sweep_comparison.xlsx` for the full 24-cell backend
comparison.

### Saved Artefacts

| Artefact | cuML | CPU |
|---|---|---|
| `cohort_labels.npy`, `cohort_probabilities.npy` | ✓ | ✓ |
| `cluster_persistence.npy` | ✓ (degenerate) | ✓ |
| `outlier_scores.npy` | — | ✓ |
| `exemplars.npz` | — | ✓ |
| `hdbscan_model.pkl` | ✓ | ✓ |
| `fit_indices.npy` | ✓ | ✓ |
| SigProfiler `uv_only` assignment | ✓ | — |