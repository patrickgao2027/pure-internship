# 2-D latent, no UMAP

Trains the cohort VAE with `latent_dim=2`, deletes UMAP from the pipeline, and hands the
raw latent straight to HDBSCAN → SigProfiler.

```
current:  VAE(16-D)  →  UMAP(2-D)  →  HDBSCAN  →  SigProfiler
this:     VAE( 2-D)  →               HDBSCAN  →  SigProfiler
```

Additive to `umap_hdbscan_sweep/` — it reuses that pipeline's stage 0 and stage 1
unchanged and replaces only stage 2. Nothing in the core package is modified. The one
shared-code change is additive: `sweep_core.HdbscanConfig` gained
`cluster_selection_method` and `cluster_selection_epsilon`, both defaulted so existing
grids and their directory names are unchanged.

## Running it

```bash
sed -i 's/\r$//' latent2d_test/run_latent2d.sh
```

```bash
STAGE=train bash latent2d_test/run_latent2d.sh
```

```bash
STAGE=1 bash latent2d_test/run_latent2d.sh
```

```bash
STAGE=2 bash latent2d_test/run_latent2d.sh
```

`STAGE=all` chains train → 0 → 1 → 2. `STAGE=2 DRY_RUN=1` prints the grid and fits
nothing. Stage 2 resumes: a cell with a `metrics.json` is skipped.

**Stage 0 is reused, not re-run.** Deduplication depends on the checkpoint only through
its `feature_report` — which columns to keep — and a 2-D-latent VAE built from the same
`ml_features.json` has an identical feature list. `run_latent2d.sh` therefore points
`DEDUP_DIR` at the 16-D sweep's `stage0_dedup/` when it exists. That saves the most
expensive stage *and* guarantees both runs cluster the same population, which is what
makes the ARI comparison below valid. If no baseline dedup exists it makes its own.

## What to change for a 2-D latent

### 1. `kl_weight` — the one that actually matters

Lowered from the baseline 0.05 to **0.01** by default, and it is the first thing to sweep.

β pulls the aggregate posterior toward a single N(0, I) blob. With UMAP downstream that is
nearly harmless: UMAP rebuilds structure from the kNN graph and rescales everything, so it
recovers neighbourhoods out of a compressed latent. With UMAP gone, **the latent's density
*is* what HDBSCAN reads**, and a well-regularised VAE latent is a smooth unimodal Gaussian
with no density gaps. HDBSCAN's honest answer to a Gaussian is *one cluster, everything
else noise*.

So this test can fail for a reason that has nothing to do with the science: β too high.
Sweep it before concluding anything.

```bash
for beta in 0.05 0.01 0.001; do
  KL_WEIGHT=$beta RUN_ID="beta$beta" STAGE=train bash latent2d_test/run_latent2d.sh
done
```

The cost of lowering β is a less regular latent (larger extent, no calibrated prior) and a
drift toward a plain autoencoder. That is an acceptable trade here — nothing downstream
samples from the prior.

### 2. Watch for a collapsed dimension

One posterior-collapsed dimension out of 16 is a capacity footnote. One out of 2 means the
"2-D" latent is a **line**, and HDBSCAN will produce confident-looking clusters that are
just cuts along it. Stage 2 prints per-dimension std and fills
`latent_stats.collapsed_dims` in the summary; a non-empty list invalidates the run rather
than weakening it. Fix by lowering β and retraining.

### 3. Early stopping's active-unit rule goes quiet

`early_stopping` stops when val ELBO stagnates **and** active units stop moving. With two
dimensions the AU count saturates at 2/2 within a couple of epochs and then never moves,
so the AU half of the criterion is vacuous and stopping is driven by val ELBO alone.
Nothing breaks — but `final AU count: 2` is not evidence of convergence. Read
`stop_reason` and the ELBO curve.

### 4. `cluster_selection_epsilon` is in latent units now

UMAP output has a conventional extent (roughly tens of units) that the existing grid was
tuned against. A VAE latent's extent depends on β and is not knowable in advance, so any
epsilon carried over from UMAP space is meaningless. Stage 2 prints the per-dimension
extent before it fits anything — read that number first, then set `SELECTION_EPSILONS`.

### 5. Add `cluster_selection_method=leaf` to the grid

Default here is `eom,leaf` rather than `eom` alone. A raw VAE latent tends toward a few
broad density regions rather than the many tight islands UMAP manufactures, and `eom`
answers that with a handful of enormous clusters. `leaf` cuts the condensed tree at its
finest level instead.

### 6. Consider standardising the two dimensions

UMAP output is roughly isotropic; a VAE latent is not — one dimension routinely carries
most of the variance. Euclidean HDBSCAN then effectively clusters on that dimension alone.
`variance_share` in the summary is the diagnostic; if it is lopsided (say 0.8 / 0.2), also
run `SCALE=standardize`. It is off by default because it is a real change to the geometry,
not a formatting step.

### 7. `min_samples` above 5 should fit now

The 16-D sweep OOMed for `min_samples` ∈ {25, 50} — the RMM pool peaked at 13.6 GB of 14.4
and never shrank. In 2-D the kNN/MST structures are far smaller, so the default grid keeps
5, 25 and 50. If it still OOMs, the fix is the RMM pool reset noted in
`AGENT_CONTEXT_2026-08-04.md` §3, not a smaller grid.

### 8. `FIT_ROWS` — left at 5,000,000 deliberately

The 2-D fit is much cheaper than the 16-D one, so `all` is realistic here in a way it was
not before. It is left at 5M for the first run because that is what the baseline used, and
changing it changes what is being compared. Re-run at `all` once the comparison is banked.

### 9. Things worth leaving alone the first time

`hidden_dims` (256,128), `hidden_dropout` (0.4), batch size, learning rate, epochs. Two are
plausible follow-ups but confound the comparison if changed now:

- `HIDDEN_DIMS=256,128,32` — 128 → 2 is a 64× funnel in one step; a gentler taper may
  train better.
- `HIDDEN_DROPOUT=0.2` — dropping 40 % of the layer feeding a 2-unit bottleneck is a much
  harsher constraint than feeding a 16-unit one.

Expect reconstruction MSE to rise substantially versus the 16-D model. That is the cost
being measured, not a bug.

## The confound, and the two controls that remove it

`2-D latent + no UMAP` vs `16-D latent + UMAP` differs in **two** things at once: the VAE's
capacity and the UMAP step. A difference in the results does not say which caused it.

Both controls are cheap once stage 1 exists:

**Control A — does removing UMAP hurt?** 16-D latent, HDBSCAN direct. Uses the baseline's
existing stage-1 embedding:

```bash
python latent2d_test/latent2d_cluster.py --embed-dir <baseline>/stage1_embed --output-root <run>/control_16d_noumap --allow-any-latent-dim --fit-rows 5000000
```

**Control B — does the 2-D bottleneck hurt?** 2-D latent, UMAP still on. Runs the existing
stage 2 against this test's stage-1 output:

```bash
python umap_hdbscan_sweep/stage2_sweep.py --embed-dir <run>/stage1_embed --output-root <run>/control_2d_umap --umap-n-neighbors 15 --umap-min-dist 0.0 --umap-n-components 2
```

That completes a 2×2 over {2-D, 16-D} × {UMAP, no UMAP}.

## Reading the result

`latent2d_summary.json` at the output root; one `metrics.json`, `analysis.parquet`,
`latent_clusters.png` and SigProfiler directory per cell.

- **`agreement_vs_reference.ari`** — the headline: agreement with the 16-D + UMAP baseline
  on a fixed 200k-locus subsample. Valid only when `--seed`, `--agreement-rows` and the
  stage-0 dedup population all match the run that produced the reference (which is why
  stage 0 is shared). It backfills onto already-complete cells, so you can add
  `AGREEMENT_REFERENCE=` to a finished run without re-clustering.
- **`selection_metric`** — cuML does not expose `relative_validity_`, so DBCV is `null` on
  the GPU path and best-cell selection falls back to silhouette. This is stated in the
  summary rather than silently producing no best cell (which is what happened to the 16-D
  sweep). `umap_hdbscan_sweep/rescore_dbcv.py` can fill DBCV in afterwards.
- **`latent_clusters.png`** — worth actually looking at. The clustering space is 2-D and
  therefore directly plottable, which the 16-D pipeline never was.
- **SigProfiler cosine** — the real endpoint. Read the **full distribution**, not the top
  of the file sorted by cosine: the baseline's mean mutation-weighted cosine is ≈ 0.27 with
  only ~11 % of clusters reaching 0.70 (see `AGENT_CONTEXT_2026-08-04.md` §2). More
  clusters or lower noise is not a better result if the cosine distribution is unchanged.

## Known gap

This is the **locus-level** experiment (the stage-2 equivalent): every deduplicated locus
counts once. The read-weighted pass over all 5.08B reads (`stage3_apply_full.py`) is *not*
wired up for the no-UMAP path — it expects a saved UMAP model. Each cell does persist its
fitted clusterer to `hdbscan_clusterer.joblib`, so adapting stage 3 means skipping its
transform step, not refitting.
