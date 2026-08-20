# Final models + per-parquet pipeline — runbook

Produces, for the selected sweep cell `fit1000000_mcs2500_ms15_eom`:

1. a saved HDBSCAN model for **both** backends (cuML and CPU `hdbscan`),
2. the full 95-sample pipeline run against the **cuML** model,
3. per-sample UMAP coordinates + cluster labels in a form that joins back onto each source
   parquet.

Everything below runs on **miletus**. Run each long step under `tmux`.

---

## Why this run exists

The sweep that selected this cell predates `hdbscan_param_sweep.py` saving
`hdbscan_model.pkl`, so the selected clustering exists only as metrics and a cohort label
array — there is no model object to label new reads with. Separately, the previous 95-sample
run (`per_parquet_results/`, 170 clusters) was built on `low_noise_hdbscan`, a **different
parameter cell**: every read in it was labelled against a clustering that no reported metric
describes. Both problems are fixed by refitting the selected cell and re-running against it.

Backends are kept in separate directories that name themselves, because cuML and CPU HDBSCAN
do not produce the same partition — measured across all 24 sweep cells, cuML's cluster count
runs from 7.3 % below to 22.6 % above the CPU count (this cell: 175 vs 181). Cluster ids are
therefore **not comparable across backends**. cuML is the deployed model; the CPU fit exists
because cuML exposes no `outlier_scores_` or `exemplars_` and returns a degenerate
`cluster_persistence_` of exactly 1.0 for every cluster.

---

## Step 0 — sync and set paths

```bash
cd ~/pure-internship && git pull
```

Every script below already defaults to these paths. **Nothing needs exporting** — the table
is here so you can override one, not so you have to set them:

| Variable | Default |
|---|---|
| `COORDS` | `~/pure-internship/umap_hdbscan_sweep/hdbscan/results/hdbscan_scaling/coords.npy` |
| `CONTEXT` | `~/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed/context.parquet` |
| `PARQUET_GLOB` | `/data/lab/ppmseq_parquets/*.parquet` |
| `OUTPUT_ROOT` | `~/pure-internship/umap_hdbscan_sweep/hdbscan/results/final_models` |

Do **not** write `--coords "$COORDS"` unless you have exported `COORDS` in that shell: an
unset variable expands to an empty string rather than to nothing, so the flag arrives as
`--coords ''` and overrides the default with a filename that cannot exist.

---

## Step 1 — probe the cuML round trip *(≈2 min, do not skip)*

Everything downstream loads a **pickled** clusterer and drives `fast_predict` with it. The
CPU package supports that by design; cuML does not advertise it. Find out in two minutes,
not twenty hours into the 95-sample run.

```bash
python umap_hdbscan_sweep/probe_cuml_model.py --backend cuml
```

Reads `VERDICT: SAFE` on success. Anything else — `joblib.dump` raising, the model reloading
but `build_tables` failing, or query rows labelling differently after the round trip — means
cuML cannot be used as `--hdbscan-model`, and step 3 must run with `CLUSTER_BACKEND=cpu`
instead. Exit codes: `0` safe, `2` the round trip broke, `3` it survived but disagrees.

Run it for the CPU backend too if you want the baseline:

```bash
python umap_hdbscan_sweep/probe_cuml_model.py --backend cpu
```

---

## Step 2 — fit and save the final models, both backends

```bash
tmux new-session -d -s final_models 'bash umap_hdbscan_sweep/tmux_final_models.sh'
tmux attach -t final_models
```

Writes to `final_models/{cuml,cpu}/cells/fit1000000_mcs2500_ms15_eom/`:

| File | cuML | CPU |
|---|---|---|
| `metrics.json` (now records `cluster_backend`) | ✓ | ✓ |
| `cohort_labels.npy`, `cohort_probabilities.npy` | ✓ | ✓ |
| `hdbscan_model.pkl` | ✓ | ✓ |
| `fit_indices.npy` | ✓ | ✓ |
| `cluster_persistence.npy` | ✓ (all 1.0) | ✓ (real) |
| `outlier_scores.npy`, `exemplars.npz` | — | ✓ |
| SigProfiler `uv_only` assignment | ✓ | ✓ |

Checks worth making before moving on:

```bash
CELL=~/pure-internship/umap_hdbscan_sweep/hdbscan/results/final_models
python -c "
import json
for b in ('cuml','cpu'):
    m=json.load(open('$CELL/'+b+'/cells/fit1000000_mcs2500_ms15_eom/metrics.json'))
    print(b, 'backend=', m.get('cluster_backend'),
          'clusters=', m.get('cohort_n_clusters'),
          'noise=', round(m.get('cohort_noise_fraction',0)*100,2),
          'persist_med=', m['geometry'].get('persistence_median'))
"
```

Expect `cuml` → 175 clusters, 7.36 % noise, persistence 1.0; `cpu` → 181, 8.00 %, ~0.109.
A cuML row reporting anything other than 175 means the coords are not the ones the published
metrics came from — stop and check `coords.npy` before going further.

Useful variants:

```bash
DRY_RUN=1  bash umap_hdbscan_sweep/tmux_final_models.sh   # grid + projected cost only
BACKENDS=cuml bash umap_hdbscan_sweep/tmux_final_models.sh
FULL_GRID=1   bash umap_hdbscan_sweep/tmux_final_models.sh   # all 24 cells, both backends
```

`FULL_GRID=1` is what gives every sweep cell a saved model, not just the selected one. It is
much longer — 24 cells × 2 backends, with fits up to 5 M rows.

---

## Step 2b — verify the saved models are usable for inference

A cell directory that looks complete is not the same as one that works. The sweep holds the
clusterer in memory while it labels, so every number in `metrics.json` can be right while the
pickle beside it is unusable — and none of that surfaces until an inference run loads the
directory later.

```bash
python umap_hdbscan_sweep/verify_saved_models.py \
    --models-root ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/final_models \
    --json-out /tmp/verify_models.json
```

Three checks per cell, from disk only, nothing carried over from the fit:

| Check | What it proves |
|---|---|
| **present** | the files a consumer needs exist and the pickle loads |
| **consistent** | `clusterer.labels_` matches `cohort_labels.npy` at `fit_indices` — catches an index file paired with the wrong model |
| **reproduces** | held rows re-labelled from the reloaded model get the *same* labels the sweep recorded |

Only the third exercises the reloaded model end to end; it is the one that catches a pickle
that loads but does not work. Exit status is 0 only when every cell passes everything, so it
can gate step 3.

The probe rows are drawn with seed 7, deliberately not the sweep's 42, so the rows checked
are not the ones any fit selected. `--no-predict` runs checks 1–2 only — fast, but proves
nothing about labelling.

---

## Interlude — which `cohort_labels.npy` is which

Two arrays of identical shape `(157501580,)` and identical size are **not** the same
clustering. Measured:

| File | Clusters | Noise | Cell |
|---|---|---|---|
| `models/cohort_labels.npy` | 175 | 7.3553 % | selected cuML cell — **correct** |
| `models/coords/cohort_labels.npy` | 170 | 6.8835 % | `low_noise_mcs2500_ms1_eps0.05` — wrong cell |

ARI between them is **0.645** — a genuinely different partition, not a relabelling. Identical
file size proves only that both cover the same rows. Always check
`(labels.max()+1, (labels<0).mean())` before trusting a label array, because nothing about
the filename or size distinguishes them.

---

## Step 3 — 95-sample pipeline against the selected model

Per sample: DuckDB filter → VAE encode → parametric UMAP → HDBSCAN label → SBS96 →
SigProfiler `uv_only` → 4 plots. The previous run took ≈15 min/sample, so budget ~24 h.

```bash
tmux new-session -d -s per_parquet 'bash umap_hdbscan_sweep/tmux_per_parquet_inference.sh'
tmux attach -t per_parquet
```

Defaults to `CLUSTER_BACKEND=cuml` and writes to `per_parquet_inference_cuml/`. If step 1
returned anything other than SAFE:

```bash
CLUSTER_BACKEND=cpu tmux new-session -d -s per_parquet \
    'bash umap_hdbscan_sweep/tmux_per_parquet_inference.sh'
```

`SKIP_DONE=1` resumes without redoing finished samples. Confirm the header line reads the
intended model path and that the log's cohort cluster count matches step 2 before letting it
run overnight.

Per sample it writes `labels.npy`, `umap_coords.npy`, `row_assignments.parquet`,
`source_fingerprint.json`, the SigProfiler `uv_only` output, and four plots.

---

## Step 4 — collect the coordinates and labels

```bash
python umap_hdbscan_sweep/export_assignments.py \
    --results-dir ~/pure-internship/umap_hdbscan_sweep/per_parquet_inference_cuml
```

Writes `assignments_manifest.csv` / `.json`: per sample the source file, labelled row count,
clusters present, noise fraction, `file_row_number` range, and a re-check of the source
fingerprint. Exit status is non-zero if any source has drifted.

**Expect ≈5.08 billion rows in total, not 157.5 M.** These are two different populations and
confusing them misprices every storage estimate by ~30×:

| | Rows | What it is |
|---|---|---|
| `coords.npy` / `cohort_labels.npy` | 157,501,580 | the **cohort embedding** HDBSCAN was fit and labelled on |
| per-parquet assignments | ~5.08 B (~53 M/sample) | **every filtered row** of all 95 sources, re-encoded |

Per-sample noise will also not equal the cohort's 7.36 %: that figure describes the cohort
embedding, while each sample is labelled by prediction against the model and lands wherever
its own reads fall.

The joinable artefact is `row_assignments.parquet` in each sample dir:

| Column | Type | Meaning |
|---|---|---|
| `file_row_number` | int64 | positional row id in the source parquet — the join key |
| `umap_1`, `umap_2` | float32 | parametric UMAP coordinates |
| `cluster_label` | int32 | HDBSCAN cluster id, `-1` = noise |

To perform the join now:

```bash
python umap_hdbscan_sweep/export_assignments.py \
    --results-dir ~/pure-internship/umap_hdbscan_sweep/per_parquet_inference_cuml \
    --attach --out-dir /data/lab/ppmseq_enriched
```

Each output is every source column plus the three above. Rows the filter dropped keep NULL
coordinates and a NULL label so the file stays row-for-row comparable with its source; pass
`--filtered-only` for an inner join instead.

Or join it yourself — the recipe is the same one `--attach` runs:

```sql
SELECT src.*, asg.umap_1, asg.umap_2, asg.cluster_label
FROM read_parquet('<source>.parquet', file_row_number=true) AS src
LEFT JOIN read_parquet('<sample>/row_assignments.parquet') AS asg
       ON src.file_row_number = asg.file_row_number
```

**`file_row_number` is positional.** DuckDB derives it from physical layout rather than
reading it from the file, so it identifies a read only while the parquet is byte-identical to
the one that was labelled. A rewrite, re-sort or recompaction silently repoints every id at a
different read — the join still succeeds, it is just wrong. `export_assignments.py` verifies
the recorded fingerprint before attaching and refuses to write a drifted sample unless
`--allow-drift` is passed. Re-check before publishing anything derived from a join:

```bash
python umap_hdbscan_sweep/verify_source_fingerprint.py \
    --results-dir ~/pure-internship/umap_hdbscan_sweep/per_parquet_inference_cuml
```

---

## Step 5 — refresh the deployment

`UV_VAE_Deployment/models/hdbscan/` currently holds the **`low_noise_mcs2500_ms1_eps0.05`**
model (170 clusters), and `models/coords/cohort_labels.npy` holds that run's labels. Both are
the wrong cell. Replace from step 2's output:

```bash
DEST=~/pure-internship/UV_VAE_Deployment/models/hdbscan
SRC=~/pure-internship/umap_hdbscan_sweep/hdbscan/results/final_models/cuml/cells/fit1000000_mcs2500_ms15_eom
cp "$SRC"/{hdbscan_model.pkl,fit_indices.npy,cluster_persistence.npy,metrics.json} "$DEST"/
cp "$SRC"/cohort_labels.npy ~/pure-internship/UV_VAE_Deployment/models/coords/
```

Then confirm what landed, rather than trusting the copy:

```bash
python -c "
import numpy as np
x=np.load('$HOME/pure-internship/UV_VAE_Deployment/models/coords/cohort_labels.npy',mmap_mode='r')
print(int(x.max())+1,'clusters', round(float((np.asarray(x)<0).mean())*100,4),'% noise')
"
```

Must read `175 clusters 7.3553 % noise`. Anything else means the wrong array is deployed.

`umap_coords_2d.npy` does **not** change — it is the same rank-13 parametric encoder output
throughout, verified identical by md5 to the `coords.npy` every clustering run above reads.

---

## Reference — what each number should be

Selected cell `fit1000000_mcs2500_ms15_eom` (mcs 2500, ms 15, eom, eps 0.0, 1 M fit rows,
seed 42):

| | cuML (deployed) | CPU `hdbscan` |
|---|---|---|
| Clusters | 175 | 181 |
| Cohort noise | 7.36 % | 8.00 % |
| Fit / label | 18.8 s / 122.4 s | 10.7 s / 120.1 s |
| Relative validity | 0.3174 | 0.2160 |
| Persistence median | 1.0 (degenerate) | 0.1086 |
| Mean probability | 0.8284 | 0.8105 |

DBCV is not comparable across that table as previously recorded — the cuML pass scored with
the `kdbcv` backend (0.4313) and the CPU pass with the `hdbscan` backend (0.3674), on
independent 400-points-per-cluster stratified samples. `tmux_final_models.sh` pins
`DBCV_BACKEND=hdbscan` for both so the re-run's two numbers *are* comparable; expect the cuML
DBCV from it to differ from the 0.4313 in `pipeline_parameters.md` for that reason.

Full 24-cell backend comparison: `umap_hdbscan_sweep/hdbscan/hdbscan_sweep_comparison.xlsx`.
