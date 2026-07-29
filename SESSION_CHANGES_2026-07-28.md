# Session change log — 2026-07-28

Work toward training the VAE on all 95 per-sample featuremap parquets. Everything
added is **additive**: no existing module was edited, so every prior sweep result
remains comparable.

---

## 1. Files added

| file | lines | what it is |
|---|---|---|
| `SAMPLING_STRATEGY.md` | 328 | Advisor-facing decision document: 7 sampling strategies (A–G) with verdicts, resource budget, open questions |
| `uv_vae/uv_vae/stats_cache.py` | 798 | Single-pass, cached, composable normalisation statistics |
| `uv_vae/uv_vae/splitting.py` | 286 | Content-hash train/val split predicates |
| `uv_vae/uv_vae/multi_parquet.py` | 572 | 95-way proportional interleaved dataset + epoch sharding |
| `uv_vae/uv_vae/multi_streaming.py` | 507 | The trainer that consumes it |
| `Early_Stopping_Tests/scripts/tmux_train_multi.sh` | 239 | Two-stage runner: statistics, then interleaved training |
| `uv_vae/tests/test_stats_cache.py` | 326 | 13 tests |
| `uv_vae/tests/test_multi_parquet.py` | 453 | 20 tests |

## 1b. Files edited

`Early_Stopping_Tests/Python Files/train_with_early_stopping.py` — the only
pre-existing file changed. Purely additive:

- `--parquet-path` and the new `--parquet-paths` (globs accepted) are now a
  required mutually exclusive group; passing `--parquet-path` behaves exactly as
  before.
- New group, ignored unless `--parquet-paths` is given: `--split-strategy`,
  `--val-fraction`, `--epoch-shards`, `--stats-cache-path`,
  `--duckdb-memory-limit`.
- `--parquet-paths` combined with `--streaming` / `--use-all` / `--sample-rows` is
  rejected with a message rather than silently ignoring the flag — argparse cannot
  express "mutually exclusive with a whole other group".
| `~/.claude/plans/give-me-a-road-mutable-bengio.md` | — | 5-phase roadmap (outside the repo) |

## 2. Files NOT modified

Verified with `git diff`:

- **`uv_vae/uv_vae/streaming.py` — byte-identical.** This was the design constraint.
  The training math, the epoch loop, and the diagnostics are reused by import, never
  copied.
- `training.py`, `early_stopping.py`, `data.py`, `preprocess.py`, `model.py`
- All CLIs and all tmux/SLURM runners

The three files `git status` shows as modified (`gpu_preflight.py`, `tmux_lib.sh`,
`gpu_budget.py`) were already dirty before this session.

---

## 3. What each module does

### `stats_cache.py`

`streaming.py` makes **three complete passes** over the data before the first
gradient step — `get_row_count` (line 667), `get_non_null_counts` (673), and
`compute_streaming_stats` (682) are the same scan with different aggregates. At
4.75B rows that is three passes over ~a terabyte.

`compute_streaming_stats` also has exactly one call site, is unconditional, and its
output lands in a **freshly timestamped run directory** every run
(`mkdir(exist_ok=False)`, line 658), so the numbers are persisted but never re-read.
Only checkpoints read them back, post-training.

This module folds the three queries into one and caches the answer, keyed on
(path + size + mtime + row filter + feature-spec hash + split predicate).

Statistics are stored **per file** and combined analytically, so adding a 96th
sample costs one scan rather than 96. Combination uses the Chan–Golub–LeVeque
pairwise update rather than accumulating `sum(x²)`:

```
delta = mu_B - mu_A
n     = n_A + n_B
mu    = mu_A + delta * n_B / n
M2    = M2_A + M2_B + delta² * n_A * n_B / n
```

Sources are cited in the module docstring (Chan/Golub/LeVeque 1983 *Am. Stat.*
37(3):242–247; the 1979 Stanford report; Welford 1962; Pébay SAND2008-6212).

Two details that matter more than the formula:

- **Each column carries its own `n`** — its non-null count, not the row count,
  because SQL aggregates skip NULLs. `ml_features.json` has columns that are 100%
  null on the production featuremap, so weighting by row count would misweight
  every column with nulls.
- **Files combine as a balanced binary tree**, not a left fold. Chan et al. §4 show
  the update is least accurate when partitions differ greatly in size, which is
  exactly what a fold produces. The 95 files have very unequal row counts.

The all-null column drop uses counts **summed across all files**, so a feature that
is null in one sample but populated in another is not dropped — deriving the model
shape from a single file would build the wrong network.

### `splitting.py`

Replaces `row_index % val_denominator == seed % val_denominator` with a hash of row
*contents*. Three reasons:

1. The positional split admits only `val_denominator` distinct splits — **10** at
   `train_fraction=0.9`, since the remainder is `seed % 10`. Seeds 42, 52 and 62
   give the byte-identical validation set. Every script in the repo pins `SEED=42`.
2. It depends on scan order, which is why `preserve_insertion_order` must be pinned
   and why train/val use two scans that must agree. Interleaving 95 files destroys
   any meaningful global scan position, so a positional split cannot survive it.
3. `train_fraction` quantises to `1/round(1/(1-f))`.

Default strategy is **`per_sample_row_hash`**: row-level, per-sample, proportional.
Each file holds out `val_fraction` of its own rows, so validation composition
mirrors the proportional training batches by construction. `RN` (read name) makes
`(CHROM, POS, REF, ALT, RN)` row-unique, so every row gets an independent draw.

`per_sample_site_hash` and `global_site_hash` are implemented but not default, so
switching is a config change and a re-run rather than a rewrite.

### `multi_parquet.py`

`InterleavedParquetDataset` keeps one reader per file and builds every batch from
all of them, drawing `k_i = w_i × batch_size` rows from file *i*. Every batch
contains every sample in true proportion; all readers exhaust together, so one pass
uses every row exactly once.

**Row-group shuffling is the core of it.** Each file has ~1,570 independently
addressable row groups of ~123k rows. Each reader shuffles its row-group order
every epoch and reads via `pyarrow.read_row_group`, so consecutive reads come from
random genomic locations. A sequential scan cannot achieve this at any affordable
buffer size — 95 readers × 500k rows is ~10 GB of host RAM.

**Decode is strictly sequential, buffers are concurrent.** `_encode_chunk_gpu`
decodes through cuDF against a 4 GB RMM pool; 95 concurrent decodes at the
streaming default chunk size would need ~5.7 GB and spill. The batch loop pulls
from one reader at a time, so peak decode memory is one row group.

`allocate_draws` uses largest-remainder allocation: plain truncation would floor
small samples to zero, so a sample at 0.2% of the pool would never appear.

**Epoch sharding** (`epoch_shards=E`): row groups are shuffled once per *cycle* of
E epochs, and epoch *e* takes the stride `order[e::E]`. The E shards are disjoint
and their union is every row group, so a full pass costs E epochs instead of one.
Seeding on the cycle rather than the epoch is what guarantees disjointness.
Applies to **training only** — a sharded validation set would make each epoch's val
loss a measurement over different rows, and early stopping compares those across
epochs.

### `multi_streaming.py`

The trainer. Assembly, not new math: `_run_training_epoch`,
`run_val_epoch_with_diagnostics` and `_build_model` are **imported** from
`streaming.py`, `EarlyStoppingMonitor` from `early_stopping.py`, and
`seed_everything`/`write_json` from `training.py`. Nothing is reimplemented, so a
95-file run is numerically comparable to a single-file streaming run by
construction rather than by inspection.

`TrainingConfig` was deliberately **not** modified. It carries a single
`parquet_path`, and `inference.py:122` reads that field as a fallback when no
parquet is passed explicitly — so the caller sets it to the first sample, keeping
that fallback resolvable, and the full list is recorded under `parquet_paths` in
every report.

Writes all six run artifacts in the existing format, plus a `sampling` block
recording per-file row counts, interleave weights, per-file batch draws, the split
strategy, validation composition, epoch shards, and the DuckDB version (the split
depends on `hash()`, which is not stable across DuckDB majors).

---

## 4. Measurements taken (on the real files in `parquet_files/`)

Previously assumed values were wrong; these are measured.

| quantity | measured |
|---|---|
| rows per file, total | 192,810,508 and 203,091,722 — **not ~50M** |
| rows per file, post-filter | 50,591,627 (**26.2%**) — this is what "~50M" referred to |
| implied total | ~4.75B post-filter across 95 samples (~19B raw) |
| file ordering | **perfectly sorted by CHROM, POS** — 0 out-of-order steps in 2M rows |
| same-site row spacing | **strictly adjacent** — row-index gap = 1 at median, p90, p99 *and max* |
| depth | 8.99 rows/site post-filter; 80% of rows in sites with ≥10 rows; max 4,336 |
| row groups | 1,570 and 1,653, averaging ~122,880 rows |
| schema | 69 columns; `RN` is a read name, `(CHROM,POS,REF,ALT,RN)` is unique |
| scan floor | 126M rows/s, 3 narrow columns with predicate, 6 threads |
| `RN` cost in the split key | 0.6 s baseline → 1.2 s site key → **2.8 s with RN** per 50.6M rows (~3.5 min per 95-file pass) |

**The finding that drove the design:** all structural correlation in the data is
local and is purely an artefact of read order. Breaking that ordering is the
primary job, which is what row-group shuffling does.

**Documented consequence of the row-level split:** at 8.99 rows/site, an
independent 10% row draw leaves **44.8% of sites straddling train/val, covering
75.3% of rows** (computed over all 5,624,433 sites). Those rows share `REF`, `ALT`,
`X_PREV1..3`, `X_NEXT1..3` — 8 of the 11 categorical features. Validation loss will
read slightly low and early stopping will fire slightly late. This is a recorded
trade; `training_report.json` will stamp which strategy produced each run.

---

## 5. Bugs found and fixed while building

### DuckDB `hash()` salting was structurally broken

`hash(k,'A') XOR hash(k,'B')` equals `hash('A') XOR hash('B')` **for every k** —
changing one argument XORs a constant offset, independent of the others. Verified
directly. Because the threshold tests the top few bits of the range, two salts
whose offset has a high bit set produce splits that are **mutually exclusive**.

Measured on the production file: two 10% draws over the same 50.6M rows with
**exactly zero** overlap, where independence predicts ~506k.

Each draw was still individually uniform, so no single file's split was ever wrong.
What broke was comparison *across* salts — varying `seed` to measure
validation-split sensitivity would have produced structured, not independent,
resamples.

**Fix:** fold the salt into a key column's *value* (`CAST(CHROM AS VARCHAR) || salt`)
so it changes the hash input rather than XOR-ing its output. Overlap is now 10.00%
for both a different sample id and a different seed. Regression test:
`test_salts_produce_independent_draws`.

### Row-group order was not reproducible across processes

The shuffle was seeded with Python's `hash()`, which is **randomised per process**
unless `PYTHONHASHSEED` is set before interpreter start. `training.seed_everything`
sets it with `os.environ.setdefault`, far too late to affect the running process.

Proof: `hash('s0.parquet')` returned three different values in three processes.

**Fix:** `stable_seed()` using CRC32. Verified byte-identical row-group order across
three separate processes. Regression test: `test_stable_seed_does_not_use_python_hash`.

### Split threshold was float-rounded

`int(0.1 * 2**64)` gives `1844674407370955264`, not the exact `1844674407370955161`
— the float product rounds. Irrelevant to the split fraction but platform-dependent.

**Fix:** `int(Fraction(str(val_fraction)) * 2**64)`, which parses the decimal
literal exactly. Regression test: `test_threshold_is_exact_not_float_rounded`.

### Convergence tracking was missing from the new trainer

An AST check comparing the artifacts each trainer writes found that
`multi_streaming.py` produced five of `streaming.py`'s six JSON files — it had no
`convergence_report.json` because it had no `ConvergenceTracker` at all.

The run-artifact contract treats that file as conditional ("when a test parquet is
given"), so this would not have broken any consumer. But latent-geometry stability
is this project's actual research question, and the 95-sample run is the one most
worth measuring, so omitting it would have been a real gap rather than an optional
extra. Now wired through with `--test-parquet-path` / `convergence_rows`, matching
`streaming.py`: per-epoch Procrustes / linear CKA / trustworthiness, recorded in
`history`, the tqdm postfix, `convergence_report.json`, and the checkpoint.

### The two runner stages would have used different splits

The runner computes the validation share twice: `awk '1-0.9'` for the statistics
stage, and `1.0 - args.train_fraction` inside the CLI for the training stage. In
float those are **not the same number** — `1.0 - 0.9` is `0.09999999999999998`,
giving threshold `...954792`, where a literal `0.1` gives `...955161`.

That is a genuinely different partition of the data. The statistics stage would
have measured and cached one split, the trainer would have used another, the
per-file cache would never have hit (the predicate is part of the cache key), and
every one of the 95 parquets would have been rescanned — with the reported
validation composition describing a split the trainer never used.

**Fix:** `SplitConfig.__post_init__` normalises `val_fraction` to 10 decimal
places, so any caller's arithmetic converges on the same threshold. The runner
also now derives `VAL_FRACTION` once and passes it to *both* stages. Regression
test: `test_val_fraction_derived_by_subtraction_gives_the_same_split`.

### Two smaller ones

- `.arrow()` returns a `RecordBatchReader` on current DuckDB, not a Table, so it
  cannot be length-checked. Now prefers `to_arrow_table()` with a
  `fetch_arrow_table()` fallback for older DuckDB.
- Empty row groups after filtering ended an epoch prematurely. A validation reader
  keeps ~10% of rows, so an empty result is routine; `_decode_next` now loops.

---

## 6. Test status

| suite | result |
|---|---|
| Chan combination vs numpy | 18 assertions, all pass |
| `stats_cache` against real parquet | 30 assertions, all pass |
| Split predicates against the 50.6M-row production file | 12 checks, all pass |
| `test_multi_parquet.py` | **18 pass, 1 deselected** |
| `multi_streaming.py` static checks | imports resolve, call-site kwargs match, artifact set matches `streaming.py` |

### What has NOT been executed

- `test_encoder_matches_the_streaming_encoder` — the anti-drift check pinning
  `RowEncoder` to `StreamingParquetDataset._encode_chunk`. Needs real torch.
- `test_stats_cache.py` has never run *as pytest*; the same assertions were verified
  through a DuckDB-backed stub, because pyarrow was missing when it was written.
- **`multi_streaming.py` has never been run at all.** It imports `torch.nn`, so it
  cannot even be imported on this machine. It was verified by AST analysis only:
  every imported name exists in its source module, every keyword argument at each
  call site matches the real signature, every `monitor.*` attribute is defined on
  `EarlyStoppingMonitor`, and the six artifact filenames match `streaming.py`. The
  `ConvergenceTracker.from_parquet`, `evaluate_epoch` and `_build_model` signatures
  were checked separately (they are imported inside functions, so the automated
  pass skipped them).

All local validation used a **torch stub**. Real pyarrow, polars and DuckDB were
exercised against the real 5.6 GB files; the numpy→tensor conversion and the entire
training loop are untested.

**First action on miletus:** `uv run pytest tests/` — the existing suite must stay
green (proving `streaming.py` is untouched), then the two new files.

---

## 7. How to run it

On miletus, after `sed -i 's/\r$//' Early_Stopping_Tests/scripts/tmux_train_multi.sh`:

```bash
PARQUET_GLOB='/cta/users/patrickgao765/parquet_files/*.featuremap.parquet' \
STATS_ONLY=1 bash Early_Stopping_Tests/scripts/tmux_train_multi.sh
```

`STATS_ONLY=1` runs stage 1 only: it builds the per-file statistics cache and
prints each sample's interleave weight and validation share. **Do this first.** It
is the cheapest way to confirm the 95 files parse, the row filter behaves as
expected, and no sample is starved — and the cache it writes is reused by every
later run, so it is not throwaway work.

Then drop `STATS_ONLY` to train. `EPOCH_SHARDS=20` makes one full pass over every
row cost 20 epochs instead of one.

## 8. Not done

- `stats_cache` is not wired into `streaming.py`; the single-file path still makes
  its three startup scans. Only `multi_streaming.py` benefits.
- **No end-to-end run on real data.** Nothing in the training path has executed.
- The advisor questions in `SAMPLING_STRATEGY.md` §9 remain open, chiefly whether
  validation should hold out loci or whole samples.
- `SAMPLING_STRATEGY.md` still describes this work as unimplemented — it was
  written before any of it was built.

## 9. Environment

`pyarrow`, `polars` and `pytest` were installed with `pip --target` into the session
scratchpad (~289 MB) purely to exercise the new modules locally. **The `uv`
environment and the project were never touched** — verified after removal: all four
of `pyarrow`, `polars`, `pytest` and `torch` are absent from the interpreter, as
they were before the session. The scratch installs have been **deleted** (291 MB →
129 KB; only small text files and synthetic test parquets remain).
