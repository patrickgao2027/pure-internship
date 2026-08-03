# What has and hasn't been checked (sweep GPU fix + DBCV rescore)

Written 2026-08-03, after the stage-2 sweep failed on every cell but the first.
Plain-language companion to the code. Read this before trusting either change.

---

## 1. What went wrong in the sweep

Stage 2 ran fine for one cell, then every cell after it died with
`maximum pool size exceeded`.

The cause, in order:

1. cuML (the GPU clustering library) gets its memory from a **pool** — one big block
   reserved up front, which everything else allocates out of.
2. The sweep starts, tries to set that pool's size, and fails. It logs
   `rmm 0 GB (not installed)` and carries on. cuML then makes its **own** pool instead,
   which quietly grows as UMAP and the first HDBSCAN need more room — up to about 3.7 GB.
3. The first cell finishes and calls SigProfiler. To do that it imports a helper file,
   `run_variant_cluster_pipeline.py`. **Just importing that file resets the pool** — it
   was written for training, and it caps the pool at 4 GB.
4. So now there's a pool that already has 3.7 GB in use and a new ceiling of 4 GB. The
   next cell asks for 1.1 GB, there's only 0.3 GB of headroom, and it dies. Every
   remaining cell dies the same way, forever, because the ceiling never lifts.

That's why the failures were byte-for-byte identical each time: same 3.7 / 4.0 / 1.1 GB.

### What was changed

- **The sweep now sets `UV_VAE_DISABLE_CUML=1` before importing that helper.** The
  helper already supports this flag — its own comments warn about exactly this hazard —
  the sweep just never used it. This is the actual fix.
- **A pool, once set, is never replaced by a smaller one.** Insurance, so no other file
  can do the same thing in future.
- **The log now says *why* the pool wasn't set.** Previously three different problems all
  printed the same misleading `(not installed)`, which is why this took hours to find.
- **Each cell's clusterer is released before the next one starts**, instead of after.
  Previously two were briefly held on the GPU at once.

---

## 2. What I actually checked, and what it proves

| What I ran | What it genuinely proves |
|---|---|
| Fake-memory test of the pool rules | The *bookkeeping* is right — the correct number wins when the pool is set twice |
| End-to-end test of the DBCV tool on made-up data | The tool runs, and the plumbing lines up correctly |
| The project's own test suite (87 pass, 7 skip) | I didn't break anything else in the package |
| Timing measurements of the DBCV scoring | Rough cost, on made-up data, on a Windows laptop |

---

## 3. What I could NOT check — the important part

**Nothing was tested on a real GPU.** My machine has no `cuml` and no `rmm`. Every test
replaced them with fake stand-ins that just record what they were asked to do. So the
tests confirm the *logic* is consistent, and confirm nothing at all about how the real
libraries behave.

This matters more than it sounds: **the bug was caused by a real library doing something
unexpected, and my tests are structurally incapable of catching that kind of bug.**

Specifically:

- **I never reproduced the original failure.** I diagnosed it by reading the code and the
  log. The reasoning is solid and every number in the log matches, but that is an argument,
  not proof.
- ~~**I still don't know why the pool failed to be set at the start.**~~ **RESOLVED by the
  smoke test on 2026-08-03 — see section 6.** It was a second, older bug.
- **The fixed sweep file has never been run.** Not once, not even on tiny fake data. The
  function I edited most — the SigProfiler one — has not executed. A typo there would only
  surface hours into the real run.
- **The 7 skipped tests in the project's suite are the GPU ones.** The suite passing is
  real but weak evidence: not one of its tests touches the code I changed.

For the DBCV rescoring tool specifically:

- **The test data was three tidy round blobs, far apart.** Real data is a continuous smear
  with fuzzy edges, very unequal cluster sizes, and about half the points marked as noise.
  My test proves the tool can tell "correct" from "random". It says nothing about whether
  it can sensibly rank two *plausible* clusterings against each other — which is the whole
  job.
- **Tested on 50,000 rows; the real run is 157,501,580.** Three thousand times bigger.
- **My "about 25 minutes for all cells" estimate is not measured.** It assumes reading the
  saved files only touches the 2 columns needed, ignoring the other ~40. That's how the
  library is supposed to work, but I never timed it on a real file. If I'm wrong, that step
  is roughly 20× slower.
- **The "1 hour for a 5,709-cluster cell" figure is an extrapolation**, stretched 14× past
  anything I measured, from only two reliable data points.
- **Two features have never run at all:** `--reload-umap-models` and `--top-n`. The first
  is a whole function that could fail on its first line.

---

## 4. What could still go wrong, most likely first

1. ~~**The pool still isn't capped.**~~ This is what happened, it was a real bug, and it is
   now fixed (section 6). **Note the direction of the change:** stage 2 used to run with no
   GPU limit at all, and now runs with a real 14.4 GB one. The largest demand actually seen
   was about 4.8 GB, so there is roughly 3x headroom — but a cell that wants more than
   14.4 GB will now fail where before it would have quietly taken what it needed. If that
   happens, raise `GPU_TOTAL_GB`.
2. **Half the grid can't be DBCV-scored.** The 16-dimension cells never saved coordinates,
   so there's nothing to score them against. Not a bug — a decision made when the sweep was
   written. `--reload-umap-models` might rescue them, but that code is untested.
3. **The heavily-fragmented cells can't be scored either**, because the scoring cost grows
   with the *square* of the cluster count. Cells with thousands of clusters are skipped by
   default. They're not cells you'd pick anyway.
4. **A plain mistake in the edited sweep file.** Low probability, high annoyance — it would
   cost you hours before showing up.

---

## 5. How you'll find out

Run the fast smoke test (in the commit message for the RMM fix). It takes about a minute and
reproduces the exact failure sequence — cuML allocates, the SigProfiler helper gets imported,
then HDBSCAN allocates again. Under the old code, that last step is what died.

**Do that before committing days of GPU time to the full sweep.**

---

## 6. Update, 2026-08-03: the smoke test found a second bug

It passed, and the new log line paid for itself immediately:

```
rmm.reinitialize failed (RuntimeError: Initial pool size required to be a
multiple of 256 bytes)
```

The pool has a size and a *starting* size. The code rounded the size to a multiple of 256
as the library demands, but computed the starting size as "a quarter of that" and never
rounded it — and a quarter of a multiple of 256 is only *sometimes* another multiple of 256.

So whether a memory limit got applied at all depended on the arithmetic of the particular
numbers involved:

| who | asked for | starting size | result |
|---|---|---|---|
| the sweep | 14.4 GB | 3,865,470,528 | **rejected — no limit applied** |
| the SigProfiler helper | 4 GB | 1,073,741,824 | accepted |
| stage 1 | 1.5 GB | 402,653,184 | accepted |

That is the whole conspiracy. The limit the sweep *asked for* was the one that couldn't be
applied; the limit that *overwrote* it was one that could. So the only limit ever in force
was the wrong one, and it landed on a pool already 3.7 GB full.

This bug has been in the file since it was written, so **any run whose quarter-size wasn't
256-aligned has silently had no GPU memory limit — training included.** Fixed in `deaaa35`.

Worth noting what this says about section 3: the local tests could not have found this.
They stubbed out the library, and a stub accepts whatever you hand it. Only the real
library enforces the rule that was being broken. The one-minute smoke test found in one
run what the entire local test suite was structurally unable to see.
