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

## 3. HDBSCAN — Low-Noise Cohort Clustering

### Parameters

| Parameter | Value |
|---|---|
| min_cluster_size (mcs) | 2,500 |
| min_samples (ms) | 1 |
| cluster_selection_epsilon | 0.05 |
| Fit rows | 1,000,000 (random subsample of 2D UMAP coordinates) |
| Input | 2D UMAP coordinates |
| Clusters found | 170 |
| Noise fraction (cohort) | 6.88 % |
| Fit time | 86.4 s |
| Label time (full cohort, approximate_predict) | 46.0 s |
| GPU VRAM (cuML RMM pool) | 36 GB pre-allocated; 9.944 GB total device usage at fit time; HDBSCAN delta ≈ 0 for 1M-row fit |