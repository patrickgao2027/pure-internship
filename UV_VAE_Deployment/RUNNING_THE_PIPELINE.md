# Running the pipeline on new data

Applying the shipped checkpoints to parquet files that were not part of the cohort:
**parquet → DuckDB filter → VAE encode → parametric UMAP → HDBSCAN label → SBS96 →
SigProfiler `uv_only`**, then joining the coordinates and labels back onto the source rows.

Nothing here retrains anything. Every model is loaded from `models/` and applied, so results
stay comparable to the cohort run and to each other. What was used to *build* those models is
in [`pipeline_parameters.md`](pipeline_parameters.md); which file is what is in
[`DEPLOYMENT_MANIFEST.md`](DEPLOYMENT_MANIFEST.md).

---

## 0. Requirements

| | |
|---|---|
| GPU | CUDA device, ~44 GB budget by default. **cuML is mandatory** — `hdbscan_model.pkl` is a `cuml.cluster.hdbscan.HDBSCAN` and cannot be unpickled without it. |
| Environment | micromamba env `uv_vae` on miletus (`micromamba activate uv_vae`) |
| Input | one or more parquet files carrying the feature columns in `uv_vae/ml_features.json`, plus `CHROM`, `POS`, `REF`, `ALT`, `st`, `et`, `FILT` |

### Libraries

Everything the pipeline imports, and which stage needs it:

| Package | Needed for | Source |
|---|---|---|
| `torch` | VAE encoder, parametric UMAP encoder, GPU budget | `pyproject.toml` (CUDA-12.8 index on Linux) |
| `numpy` | everywhere | `pyproject.toml` |
| `polars` | frames handed to the VAE (`encode_frame` takes a `pl.DataFrame`) | `pyproject.toml` |
| `duckdb` | all parquet reading, filtering, and the join | `pyproject.toml` |
| `pyarrow` | Arrow batch streaming, parquet writing | `pyproject.toml` |
| `scikit-learn` | neighbour index fallback in `fast_predict` | `pyproject.toml` |
| `hdbscan` | CPU reference implementation, DBCV scoring | `pyproject.toml` (`>=0.8.42`) |
| `umap-learn` | UMAP utilities | `pyproject.toml` (`>=0.5.12`) |
| `joblib` | loading `hdbscan_model.pkl`, parallel workers | `pyproject.toml` (via scikit-learn) |
| `matplotlib`, `pillow` | the four per-sample plots | `pyproject.toml` |
| `sigprofilerassignment` | `uv_only` signature assignment | `pyproject.toml` (`>=1.1.3`) |
| `tqdm` | progress bars (set `TQDM_DISABLE=1` for batch runs) | `pyproject.toml` |
| **`cuml`** | **GPU HDBSCAN — required to unpickle the model** | **RAPIDS (conda), not pyproject** |
| **`cupy`** | GPU array ops in the UMAP apply path | **RAPIDS (conda), not pyproject** |
| **`rmm`** | GPU memory pool that `gpu_budget` configures | **RAPIDS (conda), not pyproject** |

**`uv sync` alone does not produce a working environment.** The three RAPIDS packages are
imported by the pipeline but are not in `pyproject.toml`, because they install from the
`rapidsai` conda channel rather than PyPI. On miletus they are already present in the
`uv_vae` micromamba env. Elsewhere, create the env first and add the Python deps into it:

```bash
micromamba create -n uv_vae -c rapidsai -c conda-forge -c nvidia \
    cuml cupy rmm python=3.12 cuda-version=12.8
micromamba activate uv_vae
cd uv_vae && uv sync --active
```

Check the GPU stack resolves before starting a long run — this is also the fastest way to
confirm a machine can load the shipped model at all:

```bash
python -c "
import cuml, cupy, rmm, torch
print('cuml', cuml.__version__, '| cupy', cupy.__version__, '| torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
import joblib; joblib.load('models/hdbscan/hdbscan_model.pkl'); print('model unpickles OK')
"
```

`sigprofilerassignment` pulls in TensorFlow transitively — the TF banner in the logs comes
from there, not from pipeline code. No pipeline file imports TensorFlow or Keras; the
parametric UMAP encoder is a plain torch `nn.Module` loaded from `.pt`.

Every path falls back to CPU **except** unpickling the HDBSCAN model. A CPU-only machine can
read every `.npy` and `.parquet` in this folder and run the VAE, but cannot load
`hdbscan_model.pkl`.

The row filter is fixed at `st = 'MIXED' AND et = 'MIXED' AND FILT = 1`. It is hard-coded in
`per_parquet_inference.py` rather than exposed as a flag, because the VAE's normalisation
statistics were computed over exactly that population — encoding a different one produces a
latent space the UMAP encoder was never fitted against.

Expect roughly **21 % of rows to survive the filter** (measured: 53.9 M of 255.3 M).

---

## 1. The five checkpoints

Everything below is driven by these. Paths assume the deployment folder is at `$DEPLOY`.

```bash
DEPLOY=~/pure-internship/UV_VAE_Deployment

VAE=$DEPLOY/models/vae/model.pt                                   # 16-D latent encoder
SPEC=$DEPLOY/uv_vae/ml_features.json                              # feature definitions
UMAP=$DEPLOY/models/umap/13_BEST_25M_nn15_md0.1_umap.pt           # parametric UMAP encoder
HDB=$DEPLOY/models/hdbscan/hdbscan_model.pkl                      # 175-cluster HDBSCAN
FIT=$DEPLOY/models/hdbscan/fit_indices.npy                        # which 1 M rows it was fit on
COORDS=$DEPLOY/models/coords/umap_coords_2d.npy                   # the 157.5 M cohort embedding
CONTEXT=$DEPLOY/models/coords/context.parquet                     # trinucleotide context
```

`fit_indices.npy` is not optional in practice. `fast_predict` indexes the fit set
**positionally** — it selects rows of `coords` by index to build the reference neighbourhood.
Passing the wrong index file, or letting it be re-derived from a mismatched
`(seed, fit_rows)`, produces labels computed against the wrong neighbours. The run will not
error; it will just be wrong.

---

## 2. Run the full pipeline

```bash
python umap_hdbscan_sweep/per_parquet_inference.py \
    --parquet-glob   '/path/to/new_samples/*.parquet' \
    --checkpoint     "$VAE" \
    --feature-spec   "$SPEC" \
    --umap-model     "$UMAP" \
    --coords         "$COORDS" \
    --context        "$CONTEXT" \
    --hdbscan-model  "$HDB" \
    --fit-indices    "$FIT" \
    --output-dir     /path/to/output \
    --mcs 2500 --ms 15 --epsilon 0.0 --fit-rows 1000000 --seed 42 \
    --genome-build GRCh38 --cosmic-version 3.5 \
    --n-workers 4 --sigprofiler-cpu 4 \
    --gpu-budget-gb 44 --predict-batch-rows 5000000 \
    --device auto
```

**Set `--epsilon 0.0` explicitly.** The script's default is `0.05`, left from an earlier
parameter cell. It is ignored when `--hdbscan-model` is given, but passing the deployed value
keeps the recorded configuration honest if anyone reads the log later.

`--mcs`, `--ms`, `--fit-rows` and `--seed` are likewise ignored when a model is supplied —
they only take effect if you omit `--hdbscan-model` and let the script fit its own. Don't:
a fresh fit produces a different partition whose cluster ids mean nothing next to the cohort's.

Long runs belong under tmux, and `tmux_per_parquet_inference.sh` wraps the above with the
deployment paths already filled in:

```bash
tmux new-session -d -s inference 'bash umap_hdbscan_sweep/tmux_per_parquet_inference.sh'
tmux attach -t inference
```

Override its defaults with environment variables — `PARQUET_GLOB`, `OUTPUT_DIR`, `MODEL_DIR`,
`CLUSTER_BACKEND`, `GPU_BUDGET_GB`. `SKIP_DONE=1` resumes without redoing finished samples.

**Before walking away**, check the header lines it prints: the model path, and that the
cohort cluster count reads **175**. A run pointed at the wrong cell is only obvious here.

### Cost

The 95-sample cohort took ~15 min/sample on one RTX PRO 5000 (48 GB) — about 24 h. Dominated
by VAE encode and SigProfiler, both of which scale with surviving rows rather than file size.

---

## 3. What each sample produces

```
<output-dir>/<sample>/
├── row_assignments.parquet     ← the joinable artefact
├── labels.npy                  ← same labels, array form
├── umap_coords.npy             ← same coordinates, array form
├── source_fingerprint.json     ← hash of the source parquet footer
├── sigprofiler_input/cluster_sbs96_matrix.tsv
├── sigprofilerassignment_uv_only_grch38_v3.5/
└── plots/  umap_cluster.png  umap_cosine.png  umap_sigprofiler.png  umap_substitution.png
```

`labels.npy` and `umap_coords.npy` are **exactly redundant** with columns of
`row_assignments.parquet` — verified identical, zero differences. They are ~60 % of the output
size. Delete them if space matters; nothing downstream reads them.

`row_assignments.parquet` holds one row per *surviving* read:

| Column | Type | Meaning |
|---|---|---|
| `file_row_number` | int64 | positional row id in the source parquet — the join key |
| `umap_1`, `umap_2` | float32 | parametric UMAP coordinates |
| `cluster_label` | int32 | HDBSCAN cluster id, `-1` = noise |

---

## 4. Attaching the results to the source parquets

### 4a. Check the sources first

```bash
python umap_hdbscan_sweep/export_assignments.py --results-dir /path/to/output
```

Writes `assignments_manifest.csv` / `.json`: per sample the source it keys against, labelled
row count, clusters present, noise fraction, `file_row_number` range, and a re-check of every
source fingerprint. **Exit status is non-zero if any source has drifted.** Do not join past
that — see §5.

### 4b. Join at query time (recommended)

Build a DuckDB database of views. The file is kilobytes; `SELECT *` returns every source
column plus the three new ones, with no data copied:

```bash
python umap_hdbscan_sweep/build_enriched_views.py \
    --manifest /path/to/output/assignments_manifest.json \
    --database /path/to/enriched.duckdb \
    --verify 3
```

```python
import duckdb
con = duckdb.connect("/path/to/enriched.duckdb", read_only=True)
con.execute("SELECT * FROM sample_index").df()          # what's in here
con.execute("SELECT * FROM <view> WHERE cluster_label = 106").df()
```

Full query guide: [`ACCESSING_ASSIGNMENTS.md`](ACCESSING_ASSIGNMENTS.md).

The join it encodes, should you want to run it yourself:

```sql
SELECT src.*, asg.umap_1, asg.umap_2, asg.cluster_label
FROM read_parquet('<source>.parquet', file_row_number=true) AS src
LEFT JOIN read_parquet('<output>/<sample>/row_assignments.parquet') AS asg
       ON src.file_row_number = asg.file_row_number
```

`LEFT`, so filtered-out rows stay in place with NULLs and the result remains row-for-row
aligned with the source.

### 4c. Materialise enriched files (only if you need standalone parquets)

```bash
python umap_hdbscan_sweep/export_assignments.py \
    --results-dir /path/to/output \
    --attach --out-dir /path/to/enriched \
    --threads 8 --memory-limit 32GB --temp-dir /data/lab/duckdb_tmp \
    --skip-existing
```

Writes `<sample>.enriched.parquet` = every source column plus `umap_1`, `umap_2`,
`cluster_label`. `--filtered-only` gives an INNER join instead.

**This duplicates the entire source dataset.** For the 95-sample cohort that is ~590 GB
against 569 GB of sources, and DuckDB spills to `--temp-dir`, so point that at a volume with
room. The view in §4b gives identical query results for kilobytes. Materialise only when the
files must travel somewhere the sources cannot.

After it runs, check each sample reports `status: ok` — the script counts non-null labels in
the output and fails a sample unless that equals the assignment row count. A LEFT join that
matched nothing writes a perfectly valid parquet in which all three new columns are NULL, and
only that count catches it.

---

## 5. NULL vs −1, and the one way this breaks

**In any joined result:**

| Value | Meaning |
|---|---|
| `NULL` | row failed `st`/`et`/`FILT` — never entered inference |
| `-1` | row was clustered and came out **noise** |

Conflating them folds ~79 % of the file into the noise cluster. `WHERE cluster_label IS NOT
NULL` selects the analysed population.

**`file_row_number` is positional.** DuckDB derives it from physical row order rather than
reading it from the file, so it identifies a read only while the source parquet is
byte-identical to the one that was labelled. A rewrite, re-sort, or recompaction repoints
every id at a different read — the join still succeeds, and every number it returns is wrong.

Guards, in order:

1. `per_parquet_inference.py` records `source_fingerprint.json` — a hash of the parquet
   footer, which is what any rewrite disturbs.
2. `export_assignments.py` re-checks every fingerprint when building the manifest, and
   `--attach` refuses a drifted source unless `--allow-drift`.
3. `build_enriched_views.py` refuses to build views over a drifted source.

Re-check before trusting a join made months earlier:

```bash
python umap_hdbscan_sweep/verify_source_fingerprint.py --results-dir /path/to/output
```

If a source legitimately changed, re-run inference for that sample. Do not reach for
`--allow-drift` to get past the refusal.

---

## 6. Verifying the models before a long run

```bash
python umap_hdbscan_sweep/verify_saved_models.py \
    --cell-dir <cell directory> --expect-complete --backend sklearn
```

Three checks from disk alone: the files a consumer needs exist and the pickle loads;
`clusterer.labels_` matches `cohort_labels.npy` at `fit_indices`; and held rows re-label to
the values the sweep recorded. Only the third exercises the reloaded model end to end.

The shipped model passes all three — 200,000/200,000 probe rows reproduced.

---

## 7. Encoding to the latent space only

If you want the 16-D VAE latent without the projection and clustering:

```python
import sys, polars as pl, duckdb
sys.path.insert(0, "<deploy>/uv_vae")
from uv_vae.inference import LatentInference

inf = LatentInference.from_checkpoint(
    "<deploy>/models/vae/model.pt",
    feature_spec_path="<deploy>/uv_vae/ml_features.json",
    device="cuda")                       # or "cpu" / None to autodetect

frame = duckdb.connect().execute("""
    SELECT * FROM read_parquet('new_sample.parquet')
    WHERE st = 'MIXED' AND et = 'MIXED' AND FILT = 1""").pl()

mu = inf.encode_frame(frame, batch_size=4096)     # (n, 16) float32
```

`encode_frame` takes a **polars** DataFrame and `batch_size` is required — DuckDB's `.pl()`
returns the right type directly. Apply the same row filter here that the pipeline uses; the
normalisation statistics were computed over that population.

The checkpoint carries the normalisation statistics in `preprocess_report.json`, so new data
is standardised exactly as the training data was. Encoding with different statistics yields a
latent space the UMAP encoder was never fitted against, and the resulting clusters would be
meaningless — this is why the row filter is not configurable.
