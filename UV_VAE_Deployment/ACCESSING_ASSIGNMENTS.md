# Accessing the UMAP coordinates and cluster labels

Every read in the 95-sample cohort carries two derived facts: where it landed in the
parametric UMAP projection, and which HDBSCAN cluster it belongs to. This document is how to
read those back **alongside the original columns of the source parquet** — same query, same
result set, as though the three columns had been written into the source files.

Nothing was written into the source files. The reason, and what that costs you, is in
[§5](#5-why-the-data-was-not-merged-into-the-source-parquets).

---

## 1. The one command

```bash
duckdb ~/pure-internship/umap_hdbscan_sweep/enriched.duckdb
```

```sql
SELECT * FROM csb0_1_ppm0058 LIMIT 5;
```

That returns every column of `csb0-1-ppm0058.featuremap.parquet` plus four more:

| Column | Type | Meaning |
|---|---|---|
| `file_row_number` | int64 | positional row id in the source parquet — the join key |
| `umap_1`, `umap_2` | float32 | parametric UMAP coordinates |
| `cluster_label` | int32 | HDBSCAN cluster id; `-1` is noise, `NULL` means the row did not pass the inference filter |

From Python, without the CLI:

```python
import duckdb
con = duckdb.connect("/home/patrick/pure-internship/umap_hdbscan_sweep/enriched.duckdb",
                     read_only=True)
df = con.execute("SELECT * FROM csb0_1_ppm0058 WHERE cluster_label = 42").df()
```

---

## 2. What is in the database

`sample_index` is the table of contents — start there rather than guessing view names:

```sql
SELECT view_name, sample, labelled_rows, noise_pct, source FROM sample_index;
```

| Object | What it is |
|---|---|
| 95 per-sample views | one per source parquet, named after the sample |
| `all_samples` | all 95 unioned, with a `sample` column in front |
| `sample_index` | view name ↔ original sample name ↔ source path ↔ row counts |

View names are the sample name with `.featuremap` dropped and `-` replaced by `_`, because
`csb0-1-ppm0058.featuremap` is not a legal SQL identifier — the dashes parse as subtraction
and the dot as a schema qualifier. `sample_index.sample` keeps the original spelling.

---

## 3. Worked queries

**A row, whole.** Source columns and derived columns are peers:

```sql
SELECT CHROM, POS, REF, ALT, st, FILT, umap_1, umap_2, cluster_label
FROM csb0_1_ppm0058 ORDER BY file_row_number LIMIT 4;
```

```
 CHROM |     POS | REF | ALT |    st | FILT | umap_1 | umap_2 | cluster_label
  chr1 | 1000000 |   A |   G | MIXED |    1 |  1.621 |  2.524 |            -1
  chr2 | 1000037 |   C |   T |  PLUS |    0 |   NULL |   NULL |          NULL
  chr3 | 1000074 |   G |   A |  PLUS |    0 |   NULL |   NULL |          NULL
  chr4 | 1000111 |   T |   C | MIXED |    1 |  1.544 |  2.572 |             3
```

Rows 2 and 3 are `st='PLUS'`, `FILT=0` — they never entered inference, so they carry NULLs
and stay in place. The view is row-for-row identical to its source; nothing is dropped.
`WHERE cluster_label IS NOT NULL` restricts to labelled reads.

**Aggregate over a source column, grouped by a derived one:**

```sql
SELECT cluster_label,
       count(*)            AS reads,
       round(avg(rq), 4)   AS mean_rq,
       round(avg(RL), 1)   AS mean_read_length
FROM csb0_1_ppm0058 WHERE cluster_label IS NOT NULL
GROUP BY 1 ORDER BY reads DESC LIMIT 10;
```

**A cluster's mutation spectrum:**

```sql
SELECT REF || '>' || ALT AS substitution, count(*) AS n
FROM csb0_1_ppm0058 WHERE cluster_label = 42
GROUP BY 1 ORDER BY n DESC;
```

**How one cluster distributes across the cohort:**

```sql
SELECT sample, count(*) AS reads,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM all_samples WHERE cluster_label = 42
GROUP BY 1 ORDER BY reads DESC;
```

**A locus to its cluster:**

```sql
SELECT sample, CHROM, POS, REF, ALT, cluster_label, umap_1, umap_2
FROM all_samples
WHERE CHROM = 'chr7' AND POS BETWEEN 55000000 AND 55100000
  AND cluster_label IS NOT NULL;
```

**Export a cluster for downstream work:**

```sql
COPY (SELECT * FROM all_samples WHERE cluster_label = 42)
  TO 'cluster_42.parquet' (FORMAT PARQUET);
```

---

## 4. Performance

DuckDB pushes predicates down into both parquets and skips row groups that cannot match, so
a selective query reads a fraction of the data.

- **Per-sample views are the fast path.** `SELECT … FROM csb0_1_ppm0058 WHERE cluster_label = 42`
  touches one source file (~53 M rows) and one assignment file.
- **`all_samples` spans 5.08 billion rows.** Filtered queries prune well; an unfiltered
  aggregate genuinely reads everything and will take a long time. Constrain by `sample`,
  `cluster_label`, or `CHROM` wherever you can.
- **Give it memory and threads** for cohort-wide work:

  ```sql
  SET memory_limit='32GB'; SET threads=8;
  SET temp_directory='/data/lab/ppmseq_parquets/duckdb_tmp';
  ```

  The temp directory matters: DuckDB spills to disk when a query exceeds the memory limit,
  and the default location is on the root partition, which has ~49 GB free. Point it at
  `/data/lab` (13 TB) before running anything cohort-wide.

---

## 5. Why the data was not merged into the source parquets

Parquet has no append-column operation. Every tool that appears to add a column in place
actually rewrites the entire file. Merging would therefore have meant writing ~590 GB of new
parquet — a second copy of 569 GB of source data, so that three columns could travel beside
it — and, if done destructively, replacing lab-shared files owned by another user with no way
back.

The view holds the sentence that describes the join instead of the rows the join produces.
The database is kilobytes. Query results are identical.

What a materialised merge would genuinely add is **independence from the source paths**: an
enriched parquet keeps working if the sources move or change, whereas these views do not. If
that independence is needed, `export_assignments.py --attach` writes the enriched files, and
`/data/lab` has 12 TB free. The trade is 590 GB and a set of files that silently go stale
when inference is re-run, against a database that stays correct or fails loudly.

---

## 6. The one thing that can silently break this

`file_row_number` is **positional**. DuckDB derives it from physical row order rather than
reading it from the file, so it identifies a read only while the source parquet is
byte-identical to the one that was labelled. A rewrite, re-sort, or recompaction repoints
every id at a different read: the join still succeeds and every number it returns is wrong.

Guards, in the order they run:

1. `per_parquet_inference.py` records a `source_fingerprint.json` per sample — a hash of the
   parquet footer, which is what any rewrite disturbs.
2. `export_assignments.py` re-checks all 95 fingerprints when it builds the manifest.
3. `build_enriched_views.py` **refuses to build** over a source whose fingerprint has drifted,
   rather than producing views that resolve cleanly and return wrong rows.
4. `--verify N` reads rows back from N views and fails one whose join matched nothing — the
   quiet failure a LEFT join otherwise hides behind a valid-looking, all-NULL result.

Re-check before trusting anything derived from these views months from now:

```bash
python umap_hdbscan_sweep/verify_source_fingerprint.py \
    --results-dir ~/pure-internship/umap_hdbscan_sweep/per_parquet_inference_cuml
```

If a source has legitimately changed, re-run inference for that sample; do not pass
`--allow-drift` to get past the refusal.

---

## 7. Rebuilding the database

It is disposable — kilobytes, reconstructed in seconds:

```bash
python umap_hdbscan_sweep/build_enriched_views.py \
    --manifest ~/pure-internship/umap_hdbscan_sweep/per_parquet_inference_cuml/assignments_manifest.json \
    --database ~/pure-internship/umap_hdbscan_sweep/enriched.duckdb \
    --verify 3
```

If the manifest is missing, regenerate it first (this also re-checks every fingerprint):

```bash
python umap_hdbscan_sweep/export_assignments.py \
    --results-dir ~/pure-internship/umap_hdbscan_sweep/per_parquet_inference_cuml
```

To build against copies on another machine, rewrite the recorded path prefixes:

```bash
python umap_hdbscan_sweep/build_enriched_views.py \
    --manifest .../assignments_manifest.json --database enriched.duckdb \
    --source-root /data/lab/ppmseq_parquets --new-source-root /mnt/data/parquets \
    --assignments-root /home/patrick/pure-internship/umap_hdbscan_sweep/per_parquet_inference_cuml \
    --new-assignments-root /mnt/data/per_parquet_inference_cuml
```

---

## 8. Provenance

The labels these views expose come from the selected cuML cell
`fit1000000_mcs2500_ms15_eom` — **175 clusters, 7.36 % cohort noise**. Full parameters for
every stage are in [`pipeline_parameters.md`](pipeline_parameters.md); the model files are
listed in [`DEPLOYMENT_MANIFEST.md`](DEPLOYMENT_MANIFEST.md).

Cluster ids are **not comparable across HDBSCAN backends**. The CPU `hdbscan` refit of this
same cell gives 181 clusters / 8.00 %, and its cluster 42 is not this cluster 42. Everything
reachable through these views is cuML.

Per-sample noise will not equal the cohort's 7.36 %: that figure describes the 157.5 M-row
cohort embedding HDBSCAN was fit on, while each sample is labelled by prediction against the
model and lands wherever its own reads fall. The manifest records the real per-sample figure.
