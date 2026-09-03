# HDBSCAN Sweep — Results and Caveats

## Pipeline summary

**Parquet → VAE → UMAP → HDBSCAN → SigProfiler**

| Stage | Details |
|---|---|
| Data | 95-sample cohort, 5.08 B filtered rows (`st=MIXED, et=MIXED, FILT=1`) |
| VAE | 16-D latent, β=0.005, batch=1,048,576, early stopped epoch 30/38, val_loss=0.220917 |
| UMAP | Parametric encoder, fit on 25 M rows, nn=15, min_dist=0.1 |
| HDBSCAN | Fit on 1 M rows, **mcs=2500, ms=15**, EOM, cuML backend |
| SigProfiler | `uv_only` reference (lab-specific, not full COSMIC v3.5) |

## Final cohort result

- **175 clusters**, 7.3553% noise (−1 label)
- 157,501,580 deduplicated loci embedded; 145,916,931 mutations assigned to clusters
- Selected sweep cell: `fit1000000_mcs2500_ms15_eom` (cuML)

## What the sweep browser shows

Each cell in the grid is one HDBSCAN parameter combination (fit size × mcs × ms).
Three panels per cell:

| Panel | What it shows |
|---|---|
| **Cluster** | 2-D UMAP coloured by cluster label |
| **Substitution** | Dominant C>T / C>A / C>G / T>A etc. per cluster |
| **Cosine** | Cosine similarity heatmap between per-cluster SBS96 spectra |
| **SigProfiler** | Per-cluster `uv_only` signature activities |
| **Feature atlas** | UMAP coloured by individual input features (one panel per feature) |

## Why mcs=2500, ms=15 was selected

- Larger `min_cluster_size` suppresses fragmentation of noisy regions
- `min_samples=15` keeps the noise fraction reasonable (~7%) without over-pruning
- Cosine similarity between clusters is low (clusters are distinct spectra)
- ARI is stable across seeds at this parameter set

## Caveats

### 1. cuML HDBSCAN is not bit-reproducible
Two fits on identical input with the same seed and parameters give the same cluster
count but move the noise boundary by ~0.3 percentage points. The shipped
`hdbscan_model.pkl` is the definitive labelling — refitting will not reproduce it exactly.

### 2. cuML model requires a GPU with cuML to unpickle
`hdbscan_model.pkl` was saved with `cuml.cluster.hdbscan.HDBSCAN`. It cannot be
loaded on a CPU-only machine. Every other file in the deployment (coordinates, JSON
reports, SigProfiler outputs) is readable anywhere.

### 3. uv_only, not full COSMIC
SigProfiler was run against the lab's `uv_only_SBS_GRCh38.tsv` reference, not the
full COSMIC v3.5 database. Activities and decompositions are only comparable to other
runs using the same reference.

### 4. Row alignment is fixed to this dedup run
The five arrays (`vae_latent_16d.npy`, `umap_coords_2d.npy`, `cohort_labels.npy`,
`cohort_probabilities.npy`, `context.parquet`) are aligned row-for-row to the output
of the stage-0 dedup. Re-running stage-0 produces a different row order and
invalidates all five arrays — they must be regenerated together.

### 5. Parametric UMAP, not standard UMAP
The 2-D coordinates come from a neural encoder trained to approximate UMAP, not from
running UMAP directly. This means: (a) new rows project in one forward pass without
re-fitting, and (b) the embedding is deterministic for new data given the same
encoder weights, but will differ slightly from a fresh UMAP fit on the same points.

### 6. 5 M row GPU limit for HDBSCAN
The RTX PRO 5000 Blackwell (47 GB) runs out of memory above ~5 M fit rows for cuML
HDBSCAN. The 70 M row attempt OOMed after 26 h. The 1 M fit size was chosen to keep
wall time under 10 min while still capturing the global cluster structure.
