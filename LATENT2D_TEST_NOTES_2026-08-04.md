# 2-D latent / no-UMAP test — built, then reverted (2026-08-04)

Two commits were written, pushed, and then reverted at your request. **Nothing is lost** —
the code is still in git history and can be restored with one command each. This file is
the record so you can review when you are not tired and decide what, if anything, comes
back.

Working tree is byte-identical to `fb705d6` for every file that was touched (verified with
`git diff fb705d6 HEAD -- <paths>`, empty).

## The commits

| What | Original | Revert |
|---|---|---|
| Fix the broken `save_predictor_state` call | `c1e0c37` | `55115d3` |
| 2-D latent no-UMAP test + `sweep_core` extension | `5b2824b` | `6f7d6f5` |

Restore either by reverting its revert:

```bash
git revert 55115d3    # brings back the pipeline bug fix
```

```bash
git revert 6f7d6f5    # brings back the 2-D latent test
```

They are independent — the bug fix touches nothing the 2-D test needs.

---

## Change 1 — the `run_variant_cluster_pipeline.py` bug fix

**Recommendation: restore this one.** It is a small, self-contained correctness fix with
no design question attached.

`main()` called `save_predictor_state(..., umap_latent_columns=fit_coords)`, but that
function only accepts `output_dir, umap_model, clusterer, fit_df`. Running the script
standalone raised `TypeError` immediately *after* the UMAP/HDBSCAN fit — i.e. after the
expensive work was already done. Separately, `fit_umap_and_hdbscan` was annotated as
returning a 3-tuple but returned four values.

The fix drops the 4th return value rather than adding the parameter, because `fit_coords`
is *exactly* the `umap_1`/`umap_2` columns it was used to build, in the same row order,
and `save_predictor_state` already persists them from `fit_df`. Returning both invited the
two copies to drift. Three lines changed; code and annotation now agree.

Nobody had hit it because every current consumer (`stage2_sweep.py`,
`clustering_regime_sweep.py`) imports the module's helpers and never calls `main()`.

**Two things found while investigating, which stand regardless of the revert:**

1. A previous session already made this exact fix in the `vae-code-seeding-ff77ac`
   worktree. It never landed on main.
2. `VAE_Stability_Testing/docs/clustering_regime_plan.md` (in the
   `hpc-sample-testing-regime-de938d` worktree) claims *"✅ was fixed (2026-07-23) … the
   bad `save_predictor_state()` argument removed. It now byte-compiles."* That claim has
   been **false on main for six weeks**. That doc does not exist on main so nothing there
   needs editing, but treat its ✅ markers as unverified.

---

## Change 2 — the 2-D latent test

The question: can the VAE produce the 2-D clustering space itself, so UMAP is deleted from
the pipeline rather than tuned?

```
current:  VAE(16-D) -> UMAP(2-D) -> HDBSCAN -> SigProfiler
tested:   VAE( 2-D) ->              HDBSCAN -> SigProfiler
```

### What was built

- **`latent2d_test/latent2d_cluster.py`** (~660 lines) — a drop-in replacement for
  `umap_hdbscan_sweep/stage2_sweep.py` that reads the *same* stage-1 output
  (`latent.npy` + `context.parquet`) and clusters the raw latent with no UMAP. Grid over
  `min_cluster_size × min_samples × cluster_selection_method × cluster_selection_epsilon`;
  per-cell `analysis.parquet` → SigProfiler → `metrics.json`; resumable; a 2-D scatter per
  cell.
- **`latent2d_test/run_latent2d.sh`** — `STAGE=train | 0 | 1 | 2 | all`, tmux runner in the
  house style. Training needed **no new code**: `tmux_train_multi.sh` already takes
  `LATENT_DIM`.
- **`latent2d_test/README.md`** — the parameter analysis below, in full.
- **`umap_hdbscan_sweep/sweep_core.py`** — `HdbscanConfig` gained
  `cluster_selection_method` and `cluster_selection_epsilon`, both defaulted so existing
  grids and their directory names are byte-identical (stage 2 resumes by matching those
  names). Epsilon forwarded only when non-zero, so an older cuML still runs every existing
  grid. This was the only shared-code change.

### The design decision worth keeping

**Stage 0 is reused, not re-run.** Deduplication depends on the checkpoint only through
its `feature_report` — which columns to keep — and a 2-D-latent VAE built from the same
`ml_features.json` has an identical feature list. So `DEDUP_DIR` points at the 16-D sweep's
`stage0_dedup/`. That skips the most expensive stage *and* guarantees both runs cluster the
same population, which is the precondition for comparing them by ARI at all.

---

## The parameter analysis (the part worth re-reading)

This was the substantive answer to "is there anything I should change for a 2-D latent",
and it holds whether or not the code comes back.

### 1. `kl_weight` is the parameter that decides whether this test can work at all

β pulls the aggregate posterior toward a single N(0, I) blob. With UMAP downstream that is
nearly harmless — UMAP rebuilds structure from the kNN graph and rescales everything. With
UMAP gone, **the latent's density *is* what HDBSCAN reads**, and a well-regularised VAE
latent is a smooth unimodal Gaussian with no density gaps. HDBSCAN's honest answer to a
Gaussian is *one cluster, everything else noise*.

So the test can fail for a reason that has nothing to do with the biology: β too high. The
build defaulted it to **0.01** (down from 0.05) and recommended sweeping
{0.05, 0.01, 0.001} before drawing any conclusion. Cost of lowering it: a less regular
latent, larger extent, drift toward a plain autoencoder — acceptable, since nothing
downstream samples from the prior.

### 2. A collapsed dimension is fatal here, not a footnote

One dead dim out of 16 is a capacity note. **One out of 2 means the latent is a line**, and
HDBSCAN will produce confident-looking clusters that are just cuts along it. The script
measured per-dimension std of mu across rows and flagged it loudly. If you rebuild this,
keep that check — it is the difference between a null result and a wrong one.

### 3. `cluster_selection_epsilon` is in latent units, not UMAP units

UMAP output has a conventional extent (roughly tens of units) the existing grid was tuned
against. A VAE latent's extent depends on β and is not knowable in advance, so any epsilon
carried over from UMAP space is meaningless. The script printed per-dimension extent before
fitting anything.

### 4. Add `cluster_selection_method=leaf`

A raw VAE latent tends toward a few broad density regions rather than the many tight
islands UMAP manufactures, and `eom` answers that with a handful of enormous clusters.
`leaf` cuts the condensed tree at its finest level instead. (This is what the `sweep_core`
extension existed to enable.)

### 5. The two latent dimensions will not have equal variance

UMAP output is roughly isotropic; a VAE latent is not. Euclidean HDBSCAN then effectively
clusters on the dominant dimension alone. A `--scale standardize` option existed, off by
default because it is a real change to the geometry, with `variance_share` as the
diagnostic for when to reach for it.

### 6. Early stopping's active-unit rule goes quiet at 2 dimensions

`early_stopping` stops when val ELBO stagnates **and** active units stop moving. With two
dimensions the AU count saturates at 2/2 within a couple of epochs and never moves again,
so that half of the criterion is vacuous and stopping is driven by val ELBO alone. Nothing
breaks, but `final AU count: 2` is not evidence of convergence.

### 7. `min_samples` > 5 should stop OOMing

The 16-D sweep died for `min_samples` ∈ {25, 50} — the RMM pool peaked at 13.6 GB of 14.4
and never shrank. In 2-D the kNN/MST structures are far smaller, so the full grid should
fit. If it still OOMs, the fix is the RMM pool reset in `AGENT_CONTEXT_2026-08-04.md` §3,
not a smaller grid.

### 8. Leave alone the first time (they confound the comparison)

`hidden_dims` 256,128 — though 128 → 2 is a 64× funnel in one step and `256,128,32` may
train better. `hidden_dropout` 0.4 — dropping 40 % of the layer feeding a **2-unit**
bottleneck is far harsher than feeding a 16-unit one; 0.2 is the obvious follow-up.
`FIT_ROWS` was pinned at 5,000,000 purely to match the baseline, even though a 2-D fit is
cheap enough that `all` is realistic.

Expect reconstruction MSE to rise substantially vs the 16-D model. That is the cost being
measured, not a bug.

---

## The confound, and the controls that remove it

**This matters more than any single parameter.** `2-D latent + no UMAP` versus
`16-D latent + UMAP` differs in **two** things at once — the VAE's capacity and the UMAP
step. A difference in results does not say which caused it.

Both controls are cheap once a stage-1 embedding exists:

- **Control A — does removing UMAP hurt?** 16-D latent → HDBSCAN direct, on the baseline's
  existing embedding. (This is variant B from `clust_regime_sweep`, at cohort scale.)
- **Control B — does the 2-D bottleneck hurt?** 2-D latent → UMAP → HDBSCAN, i.e. run the
  existing `stage2_sweep.py` against the 2-D stage-1 output.

Together with the main cell that is a 2×2 over {2-D, 16-D} × {UMAP, no UMAP}.

## How the result should be read

- **ARI vs the 16-D + UMAP baseline** on a fixed 200k-locus subsample was the headline
  number. Valid only when seed, subsample size and the stage-0 dedup population all match —
  which is why stage 0 was shared rather than recomputed.
- **cuML does not expose `relative_validity_`**, so DBCV is `null` on the GPU path and
  "best by DBCV" silently selects nothing — this is what happened to the 16-D sweep, whose
  `best_cell` came back `None`. The build fell back to silhouette and recorded which
  criterion was used. `rescore_dbcv.py` can fill DBCV in afterwards.
- **SigProfiler cosine is the real endpoint**, and the full distribution must be read, not
  the top of a file sorted by cosine. Baseline: mean mutation-weighted cosine ≈ 0.27, only
  ~11 % of clusters reaching 0.70 (`AGENT_CONTEXT_2026-08-04.md` §2). More clusters or
  lower noise is **not** a better result if the cosine distribution is unchanged.

## What was verified, and what was not

**Verified** on synthetic 2-D data (conda base, CPU): 16 cells across all four grid axes;
fit-subsample + `approximate_predict`; resume; ARI backfill onto already-complete cells;
the collapsed-dimension warning; the latent-dim guard; `standardize`; plot rendering. The
`sweep_core` change was checked to leave every existing stage-2 directory name unchanged.
The bug fix was verified statically (return arity 3 = annotation arity 3 = unpack arity 3;
call kwargs match the signature exactly) and byte-compiles.

**Not verified:** anything requiring the HPC. SigProfiler is not installed on the Windows
box, so that block was never executed here — it is byte-identical to the one `stage2_sweep`
already runs on miletus. Nothing ran against a real featuremap parquet, at cohort scale, or
on a GPU. `run_variant_cluster_pipeline.main()` still has not been executed end-to-end;
the fix removes the `TypeError` but does not prove the rest of that function works.

## Also noticed, unrelated and left alone

`Early_Stopping_Tests/scripts/tmux_train_multi.sh` and `uv_vae/uv_vae/multi_streaming.py`
have **real uncommitted changes** (not CRLF churn) — a matched pair implementing the
decode-worker clamp under GPU decode, plus the runner line that reports it. Not mine, so
not committed. It would not have blocked the 2-D test (the runner leaves `DECODE_WORKERS`
at 1 and never sets `UV_VAE_GPU_DECODE`), but miletus will not have it until you commit it.
