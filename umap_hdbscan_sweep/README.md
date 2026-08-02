# UMAP × HDBSCAN sweep over the 95-file cohort

Sweeps UMAP and HDBSCAN hyperparameters over the deduplicated cohort, runs SigProfiler at
every grid cell, and can project all 5,078,201,907 reads through a chosen cell.

This is additive to `clust_regime_sweep/`, which sweeps *what data the VAE saw* at fixed
UMAP/HDBSCAN settings. Here the VAE checkpoint is fixed and the **clustering** parameters
are what vary. It does not modify the core package.

## The four stages

| Stage | Script | In | Out |
|---|---|---|---|
| 0 | `stage0_dedup.py` | 95 parquets, 5.08B reads | one row per `(CHROM,POS,REF,ALT)` |
| 1 | `stage1_embed.py` | dedup rows | `latent.npy` + `context.parquet` |
| 2 | `stage2_sweep.py` | latents | 480 cells, each with metrics + SigProfiler |
| 3 | `stage3_apply_full.py` | all 5.08B reads + one cell | read-weighted SigProfiler |

Each stage writes a manifest the next reads, so they run and resume independently.

```bash
STAGE=0 bash umap_hdbscan_sweep/run_sweep.sh                       # READ THE OUTPUT FIRST
STAGE=1 CHECKPOINT=<.../model.pt> bash umap_hdbscan_sweep/run_sweep.sh
STAGE=2 bash umap_hdbscan_sweep/run_sweep.sh
STAGE=3 CELL=nn30_md0.05_nc2/mcs500_ms25 bash umap_hdbscan_sweep/run_sweep.sh
```

Stage 2 resumes by default — cells with a `metrics.json` are skipped, so a killed sweep
restarts where it stopped. `DRY_RUN=1` prints the grid and fits nothing.

## Run stage 0 before planning anything else

Stage 0 prints the **unique-locus count**, and that number decides whether stage 2 can fit
on everything. It is not currently known: each file holds ~6M unique loci, but the union
across 95 samples depends on how much the samples overlap, so the answer is somewhere
between ~6M and ~570M. Everything below is sized off it.

`locus_reads` (how many reads collapsed into each surviving row) is computed in the same
window pass, so it costs nothing and is available for weighting later.

## What actually fits on the GPU

cuML's UMAP holds a kNN graph of roughly `N × n_neighbors × 8` bytes, and HDBSCAN's MST is
heavier still. Against a 48 GB card, the graph alone:

| N (loci) | nn=15 | nn=30 | nn=50 | nn=100 |
|---|---|---|---|---|
| 10M | 1.2 GB | 2.4 GB | 4 GB | 8 GB |
| 50M | 6 GB | 12 GB | 20 GB | 40 GB |
| 100M | 12 GB | 24 GB | 40 GB | **80 GB** |
| 200M | 24 GB | 48 GB | **80 GB** | **160 GB** |

`n_neighbors=100` is the binding constraint. If stage 0 reports ≳50M loci, expect the
high-`n_neighbors` cells to OOM.

**A cell that OOMs is recorded as a failure and the sweep continues** — it does not kill the
run. Check `cells_failed` in `sweep_summary.json`, then re-run just those with a cap:

```bash
STAGE=2 FIT_ROWS=20000000 UMAP_N_NEIGHBORS=100 bash umap_hdbscan_sweep/run_sweep.sh
```

With `FIT_ROWS` set, the rows beyond the cap are labelled by `transform` +
`approximate_predict`, so **every locus still reaches SigProfiler either way**. Only the
model fit is subsampled, never the population being clustered.

## Sharing the GPU with a running trainer

The caps `gpu_budget` installs are **per process**, so two capped processes sum. Holding
the total at 16 GB means the two caps must add up to it:

```
GPU_TOTAL_GB (16)  =  TRAINER_GPU_GB (10)  +  SWEEP_GPU_GB (6)
```

```bash
GPU_TOTAL_GB=16 TRAINER_GPU_GB=10 STAGE=2 bash umap_hdbscan_sweep/run_sweep.sh
```

**The trainer's half is only real if the trainer was started with it.** A cap is fixed at
launch and cannot be changed from outside, so a trainer started with the default 16 GB is
still *permitted* 16 GB and the total can reach 16 + `SWEEP_GPU_GB`. To actually enforce
the sum, relaunch the trainer with `GPU_TOTAL_GB=10`.

The two caps also behave differently, which matters when reading `nvidia-smi`:

| Allocator | Capped by | Behaviour |
|---|---|---|
| torch (trainer) | `set_per_process_memory_fraction` | a **ceiling** — only reached if training allocates that much |
| RMM (cuML sweep) | `rmm.reinitialize(maximum_pool_size=…)` | allocates **¼ of the pool up front**, so it appears immediately |

Before any fitting starts, `sweep_core.apply_gpu_budget` reads `torch.cuda.mem_get_info` —
which sees every process on the card — and refuses to start if the slice is not actually
free. Without that check an over-subscribed slice does not fail at startup (torch's cap is
a ceiling, not a reservation); it fails hours later as a mid-fit CUDA OOM.

Each stage passes its own RMM share, because one global value would be wrong for two of
the three:

| Stage | Uses | RMM share | 6 GB slice becomes |
|---|---|---|---|
| 1 embed | torch only | 0.0 | torch 6.0 GB |
| 2 sweep | cuML only | 0.9 | RMM 5.4 GB + torch 0.6 GB |
| 3 apply | torch **and** cuML | 0.5 | RMM 3.0 GB + torch 3.0 GB |

(0.9 is `resolve_rmm_share`'s ceiling — it deliberately refuses to starve torch entirely.
Stage 2 never loads torch, so the 0.6 GB is unused headroom.)

### What a 6 GB slice can actually fit

This is the real cost of running in parallel. The kNN graph alone is `N × n_neighbors × 8`
bytes, against RMM's 5.4 GB:

| n_neighbors | max fit rows (graph only) | realistic `FIT_ROWS` |
|---|---|---|
| 15 | ~45M | ~10M |
| 30 | ~22M | ~5M |
| 50 | ~13M | ~3M |
| 100 | ~6.7M | ~1.5M |

So **`FIT_ROWS=all` is not compatible with a 6 GB slice** unless stage 0 reports a small
locus count. Running alongside the trainer means fitting on a subsample and labelling the
rest by `transform` + `approximate_predict` — every locus still reaches SigProfiler, only
the model fit is subsampled:

```bash
GPU_TOTAL_GB=16 TRAINER_GPU_GB=10 FIT_ROWS=5000000 \
STAGE=2 bash umap_hdbscan_sweep/run_sweep.sh
```

Wait for the trainer to finish if you want `FIT_ROWS=all`.

## Cost

Stage 2's dominant cost is 32 UMAP fits (one per `n_neighbors × min_dist × n_components`);
the 15 HDBSCAN cuts within each are comparatively cheap. SigProfiler's `cosmic_fit` runs on
a 96 × n_clusters matrix and takes seconds to a couple of minutes, so 480 of them is
meaningful but not dominant.

Stage 3 is one full encode + transform + predict pass over 5.08B reads. Budget several
hours **per cell** — it is worth running for the cells that matter, not for all 480.

## Stage 2 vs stage 3: two different questions

Stage 2 clusters the **deduplicated** loci, so every locus counts once no matter its read
depth. Stage 3 pushes **every read** through, so high-depth loci contribute proportionally
more to the signature counts.

Stage 3 cannot be derived from stage 2's `locus_reads` column. Reads at one locus share an
SBS96 context but not a cluster — they differ in SNVQ, QUAL, MAPQ, DP and RAW_VAF, so the
VAE places them at different latent positions. That is why the full pass exists.

Stage 3 never writes per-read labels. SigProfiler consumes a 96 × n_clusters count matrix,
which is associative, so each batch scatter-adds into it and is discarded. Labelling 5.08B
reads would be ~20 GB nothing ever reads back. Peak memory is one batch.

## Measured findings

**UMAP `transform` is not batch-invariant, and cannot be made so.** umap-learn prunes each
transform's graph at `graph.data.max() / n_epochs` — a threshold computed from the batch —
and its numba negative-sampling loop consumes a shared `rng_state` in an order that depends
on how many points are in flight.

`UmapConfig.n_epochs` is pinned at 200 (umap-learn's own fit default above 10k rows, so the
fit is unchanged) which removes a third source of drift: with `n_epochs=None`, transform
picks 100 or 30 epochs *by input size*, so batch size changed the answer outright.

Measured on 3 separated 16-D blobs, 1500 held-out rows, batch 400 vs single-shot:

| | result |
|---|---|
| max coordinate drift | 2.73 units |
| HDBSCAN labels identical | **100.00%** |
| ARI between the two labelings | **1.0000** |

So the drift moves points *within* their cluster rather than across a boundary. **Do not
assume that carries to the real latent space**, where clusters are far less separated —
boundary reads can flip. `transform_batch_size` is recorded in both stage 2 and stage 3
output; treat it as part of the result and hold it fixed when comparing runs.

`approximate_predict` itself is exactly batch-invariant on fixed coordinates (verified) —
it is a pure kd-tree query with no optimization.

**SBS96 canonicalization verified against the pipeline.** `stage3_apply_full.sbs96_expr`
reimplements the reverse-complement-to-pyrimidine logic in `run_variant_cluster_pipeline`
(which cannot be imported without SigProfilerAssignment). Checked over all 1,296
REF × ALT × PREV × NEXT combinations including `N` and null: **0 mismatches**, exactly 96
distinct contexts over ACGT flanks.

**`hdbscan` rejects `memory=None`.** It calls `memory.cache(...)` unconditionally; its own
default is `Memory(None, verbose=0)`, a no-op cache object. `sweep_core.fit_hdbscan`
converts. The same latent bug exists in `clust_regime_sweep/clustering_core.py:hdbscan_grid`
— harmless today only because its one caller always passes a cache directory.

## Outputs

```
stage2_sweep/
  sweep_summary.json                  # every cell, best by DBCV, agreement vs best
  nn30_md0.05_nc2/
    umap_config.json                  # config, timing, backend, model path
    umap_model.joblib                 # reused by stage 3
    mcs500_ms25/
      metrics.json                    # internal panel + DBCV + SigProfiler paths
      analysis.parquet                # context + cluster_label + probability (+ umap_1/2)
      agreement_labels.npy            # labels on the shared subsample, for cross-cell ARI
      sigprofilerassignment_uv_only_grch38_v3.5/
```

Pairwise ARI across 480 cells would be ~115k comparisons, so every cell is instead compared
against the DBCV-best cell on one fixed 200k-row subsample — affordable, and the more
readable question.

## Gotchas

- `sed -i 's/\r$//' umap_hdbscan_sweep/run_sweep.sh` before running if it has been touched
  on Windows.
- `CHECKPOINT` must be a `model.pt` from the interleaved trainer. Its `feature_report` is
  what defines the column selection in stage 0, so stages 0/1/3 must all use the same one.
- `FIT_ROWS` subsamples only the model fit, never the clustered population.
- Stage 0 spills heavily. `TEMP_DIR` needs room for roughly the cohort's on-disk size.
