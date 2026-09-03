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

## Caveats

### 1. cuML HDBSCAN is not bit-reproducible
Two fits on identical input with the same seed and parameters give the same cluster
count but move the noise boundary by ~0.3 percentage points. The shipped
`hdbscan_model.pkl` is the definitive labelling — refitting will not reproduce it exactly.

### 2. Row alignment is fixed to this dedup run
The five arrays (`vae_latent_16d.npy`, `umap_coords_2d.npy`, `cohort_labels.npy`,
`cohort_probabilities.npy`, `context.parquet`) are aligned row-for-row to the output
of the stage-0 dedup. Re-running stage-0 produces a different row order and
invalidates all five arrays — they must be regenerated together.

### 3. Parametric UMAP, not standard UMAP
The 2-D coordinates come from a neural encoder trained to approximate UMAP, not from
running UMAP directly. This means: (a) new rows project in one forward pass without
re-fitting, and (b) the embedding is deterministic for new data given the same
encoder weights, but will differ slightly from a fresh UMAP fit on the same points.

### 4. GPU memory limit for large HDBSCAN fit sizes
cuML HDBSCAN on the RTX PRO 5000 Blackwell (47 GB) completed fits up to 25 M rows.
The 70 M row attempt OOMed after 26 h — MST construction at that scale exceeds the
card's memory. The sweep therefore covers fit sizes up to 25 M; the 1 M fit was
selected as the production cell to keep per-sample relabelling via `fast_predict`
fast and memory-bounded.

### 5. DBCV scores are missing for CPU-backend cells at large fit sizes
DBCV (Density-Based Cluster Validity) requires access to the full distance graph
from the fit, which the CPU sklearn backend returns but cuML does not expose in the
same form. Additionally, the CPU refit runs for the 25 M fit-size cells were not
completed — those cells show blank DBCV entries in the sweep comparison table. Only
cells where a CPU refit finished have valid DBCV scores. cuML cells at all fit sizes
report cluster count and noise fraction but not DBCV.
