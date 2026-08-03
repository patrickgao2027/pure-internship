# What has and hasn't been checked (sweep GPU fix + DBCV rescore)

Last updated 2026-08-03, after the smoke test on miletus.
Plain-language companion to the code. Read this before trusting either change.

---

## 1. What went wrong, and why it took a while to see

The stage-2 sweep ran one cell successfully, then every cell after it died with
`maximum pool size exceeded` — the same three numbers every time (3.7 GB in use,
4.0 GB ceiling, 1.1 GB requested).

It turned out to be **two separate bugs that happened to line up**.

### Bug A — a file that changes GPU settings just by being imported

cuML (the GPU clustering library) takes its memory from a **pool**: one big block
reserved up front that everything else allocates out of.

1. The sweep starts and tries to set the pool's size. It fails (that's Bug B), logs
   something misleading, and carries on. cuML then makes its **own** pool, which grows
   as UMAP and the first HDBSCAN need room — up to about 3.7 GB.
2. The first cell finishes and calls SigProfiler. Doing that imports a helper file,
   `run_variant_cluster_pipeline.py`. **Merely importing that file resets the pool** —
   it was written for training, and caps the pool at 4 GB.
3. So now: 3.7 GB already in use, new ceiling 4.0 GB, next cell wants 1.1 GB. Dead. And
   it stays dead, because the ceiling never lifts.

### Bug B — the pool size was rounded, the *starting* size wasn't

A pool has a size and a *starting* size. The code rounded the size to a multiple of 256
as the library requires, then computed the starting size as "a quarter of that" and
never rounded it. A quarter of a multiple of 256 is only *sometimes* another multiple
of 256.

So whether a memory limit got applied **at all** depended on the arithmetic of the
particular numbers involved:

| who | asked for | starting size | result |
|---|---|---|---|
| the sweep | 14.4 GB | 3,865,470,528 | **rejected — no limit applied** |
| the SigProfiler helper | 4 GB | 1,073,741,824 | accepted |
| stage 1 | 1.5 GB | 402,653,184 | accepted |

That's the conspiracy. The limit the sweep *asked for* couldn't be applied; the limit
that *overwrote* it could. The only ceiling ever in force was the wrong one, and it
landed on a pool already 3.7 GB full.

This bug has been in the file since it was written. **Any run whose quarter-size wasn't
256-aligned has silently had no GPU memory limit — training included** (see section 6).

### What changed

| commit | change |
|---|---|
| `83234e4` | sweep sets `UV_VAE_DISABLE_CUML=1` before importing the helper — the actual fix for Bug A. The helper already supported this flag; the sweep just never used it |
| `83234e4` | the log now says *why* a pool wasn't set. Three different problems used to print the same misleading `(not installed)` |
| `83234e4` | a pool, once set, is never replaced by a smaller one — insurance so nothing else can repeat Bug A |
| `83234e4` | each cell's clusterer is released before the next starts, not after; two were briefly held at once |
| `deaaa35` | both pool sizes are now rounded to 256 — the fix for Bug B |
| `00afcb8` | `rescore_dbcv.py`, to recover DBCV scores (section 4) |

---

## 2. What has been verified on real hardware

One smoke test, run on miletus at commit `04ebbff` (so it covers Bug A's fix, **not**
Bug B's). It reproduced the exact failure sequence — cuML allocates, the SigProfiler
helper is imported, HDBSCAN allocates again — and passed:

- HDBSCAN still allocated after the helper import. Under the old code this is the step
  that died. **Bug A's fix works.**
- The new log line printed the real error, which is how Bug B was found at all.
- It confirmed the pool was never capped (`pool after import: None`), meaning the sweep
  had been running with no GPU limit whatsoever.

**That is the only real-hardware verification that exists.**

## 3. What has *not* been verified

- **The Bug B fix has not run on a GPU.** It was checked against a stand-in that imitates
  the library's 256-byte rule across 112 budget/share combinations, and against the exact
  failing case. But no real RMM has accepted it yet. Re-running the smoke test confirms it
  in about a minute.
- **`stage2_sweep.py` has still never been executed end-to-end.** The smoke test performed
  the same *steps* as `run_sigprofiler`, by hand — it did not call that function. The
  edited function itself remains unrun. A plain typo there would surface hours into a run.
- **Everything else was tested with stand-ins or invented data**, on a Windows laptop with
  no `cuml` and no `rmm`. Those tests confirm the logic is self-consistent and confirm
  nothing about how the real libraries behave. Bug B is the proof: a stand-in accepts
  whatever you hand it, so the local tests were *structurally incapable* of catching it.
  One minute on real hardware found what the whole local suite could not.
- **The project's own test suite passes (87 pass, 7 skip) but is weak evidence here** —
  all 7 skips are the GPU tests, and no test touches `gpu_budget` at all.
- **The full sweep has not been re-run**, so nothing above is confirmed at scale.

---

## 4. The DBCV rescore tool — design limits, not bugs

Background: cuML's HDBSCAN can't produce a DBCV score, so every GPU-fitted cell records
`dbcv: null`, and the sweep's "best cell" selection — which ranks by DBCV — would come
back empty across all 480 cells. `rescore_dbcv.py` recovers a score afterwards without
refitting anything.

These are permanent properties of the approach, and are recorded in its output rather
than hidden:

- **It is not exact, and can't be.** The scoring method compares every pair of clusters,
  so cost grows with the *square* of the number of points. 157.5M rows is unreachable by
  a wide margin. It scores a fixed 25,000-row sample that every cell shares, so cells stay
  comparable to each other. It's a **ranking score, not an absolute DBCV.**
- **Half the grid can't be scored.** Cells using 16-dimension UMAP never saved their
  coordinates (only 2-D cells did), so there's nothing to score them against. Not a bug —
  a decision baked into how the sweep writes its output. `--reload-umap-models` might
  rescue them, but **that code path has never been run.**
- **Heavily fragmented cells can't be scored either.** Cost also grows with the square of
  the *cluster count*: 200 clusters takes 5 seconds, 400 takes 17. A 5,709-cluster cell
  extrapolates to roughly an hour on its own, so cells above 500 clusters are skipped by
  default. They aren't cells you'd choose anyway.
- **Clusters with fewer than 4 sampled points are treated as noise**, because the scoring
  code divides by (points − 1) and crashes otherwise. How many clusters and points this
  discarded is recorded per cell — a score taken after throwing away most of a fragmented
  clustering is not comparable to one that kept everything.
- **It was tested on three tidy round blobs, far apart, at 50,000 rows.** Real data is a
  continuous smear with fuzzy edges and ~50% noise, at 157.5M rows. The test proves it can
  tell "correct" from "random". It says nothing about ranking two *plausible* clusterings,
  which is the actual job.
- **The "about 25 minutes for all cells" estimate is not measured.** It assumes reading the
  saved files only touches the 2 columns needed and skips the other ~40. That's how the
  library is meant to work, but it was never timed on a real file. If that's wrong, the
  step is roughly 20× slower.
- **`--top-n` has never been run either.**

---

## 5. What could still go wrong, most likely first

1. **A cell now runs out of memory that previously wouldn't have.** Note the direction of
   the change: stage 2 used to run with *no* GPU limit (that was Bug B) and now runs with a
   real 14.4 GB one. The largest demand actually observed was about 4.8 GB, so there's
   roughly 3× headroom — but anything wanting more than 14.4 GB will now fail where before
   it quietly took what it needed. If that happens, raise `GPU_TOTAL_GB`.
2. **Half the grid can't be DBCV-scored** (section 4). You'll be selecting among the 2-D
   cells unless `--reload-umap-models` works, and that's untested.
3. **A plain mistake in the edited sweep file.** Low probability, high annoyance — it costs
   hours before showing up, because that file has never been run.

---

## 6. Side effect worth knowing about past runs

Because Bug B has existed since `gpu_budget.py` was written, **the GPU memory ceiling
recorded in past `training_report.json` files was frequently never actually enforced.**

Training results are unaffected — those runs had the whole card to themselves, so an
unenforced ceiling changed nothing about the numbers. But if you cite the GPU budget from
a past run as a constraint the run operated under, that claim isn't reliable. What the
report recorded is what was *requested*, not necessarily what was *applied*.

Runs where the quarter-size happened to land on a 256-byte boundary (like stage 1's
1.5 GB) did have a real limit. There is no way to tell which was which from the old logs,
because the failure printed the same text as success-with-no-RMM-installed. From `83234e4`
onward the log distinguishes them.
