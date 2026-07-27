"""
Plot rq quality-filter sweep results vs the 52M full-data reference model.

Run locally:
    cd "C:\\Users\\Owner\\Documents\\PURE Files"
    python VAE_Stability_Testing/scripts/plot_rq_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE = Path(r"C:\Users\Owner\Documents\PURE Files\VAE_Stability_Testing")

# ── Load data ──────────────────────────────────────────────────────────────────
rq_results = json.loads((BASE / "rq_sweep" / "vs_52M_results.json").read_text())
rq_results.sort(key=lambda r: r["rq_threshold"])

# Random-subsample reference points at comparable row counts (density + extended sweep)
density  = json.loads((BASE / "seed_density_sweep"  / "vs_52M_results.json").read_text())
orig_ext = json.loads((BASE / "seed_sweep_extended" / "vs_52M_results.json").read_text())

def rand_at(rows: int) -> dict:
    """Mean Procrustes / CKA across all seeds at a given random-subsample row count."""
    combined = [r for r in density + orig_ext
                if r.get("rows") == rows or r.get("sample_rows") == rows]
    if not combined:
        return {}
    return {
        "proc_mean": np.mean([r["procrustes_disparity"] for r in combined]),
        "proc_std":  np.std( [r["procrustes_disparity"] for r in combined]),
        "cka_mean":  np.mean([r["linear_cka"]           for r in combined]),
        "cka_std":   np.std( [r["linear_cka"]           for r in combined]),
        "n": len(combined),
    }

RAND_ROWS = [2_500_000, 5_000_000, 10_000_000]

out_dir = BASE / "rq_sweep" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)

# Convenience arrays
RQ     = [r["rq_threshold"]       for r in rq_results]
NROWS  = [r["n_rows"]             for r in rq_results]
PROC   = [r["procrustes_disparity"] for r in rq_results]
CKA    = [r["linear_cka"]         for r in rq_results]
TRUST  = [r["trustworthiness"]    for r in rq_results]
COLLAP = [r["latent_collapse"]    for r in rq_results]
VLOSS  = [r["val_total_loss"]     for r in rq_results]
VKLLOSS= [r.get("val_kl_loss", float("nan")) for r in rq_results]

x = list(range(len(RQ)))
rq_labels = [f"rq<{v}" for v in RQ]

C_RQ   = "#c0392b"   # red — rq filtered
C_RAND = "#2a78d6"   # blue — random subsample reference

# ── Figure 1: Main result — Procrustes + CKA vs 52M ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "rq quality-filter sweep vs full 52M reference model\n"
    "Comparison: does filtering to higher-quality reads help match the full model?",
    fontsize=11,
)

for ax, vals, ylabel, better, rand_key_m, rand_key_s in [
    (axes[0], PROC, "Procrustes Disparity (vs 52M)", "lower", "proc_mean", "proc_std"),
    (axes[1], CKA,  "Linear CKA (vs 52M)",           "higher", "cka_mean",  "cka_std"),
]:
    # rq sweep line
    ax.plot(x, vals, "o-", color=C_RQ, linewidth=2.5, markersize=8,
            zorder=5, label="rq-filtered (this sweep)")
    for xi, (v, n) in enumerate(zip(vals, NROWS)):
        ax.annotate(f"{n/1e6:.1f}M rows", (xi, v),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7.5, color=C_RQ)

    # Random-subsample reference bands at 2.5M, 5M, 10M
    for rows in RAND_ROWS:
        stats = rand_at(rows)
        if not stats:
            continue
        m, s = stats[rand_key_m], stats[rand_key_s]
        n = stats["n"]
        ax.axhline(m, color=C_RAND, linewidth=1, linestyle="--", alpha=0.6)
        ax.axhspan(m - s, m + s, color=C_RAND, alpha=0.08)
        ax.annotate(f"rand {rows/1e6:.0f}M (n={n})", xy=(len(x) - 0.5, m),
                    fontsize=7, color=C_RAND, va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(rq_labels, fontsize=10)
    ax.set_xlabel("rq threshold (lower = stricter quality filter)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}\n({better} = more similar to 52M)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

from matplotlib.patches import Patch
legend_handles = [
    plt.Line2D([0], [0], color=C_RQ,   linewidth=2, marker="o", label="rq-filtered models"),
    plt.Line2D([0], [0], color=C_RAND, linewidth=1, linestyle="--",
               label="random-subsample mean ± std (2.5M/5M/10M)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=[0, 0.06, 1, 1])
p = out_dir / "procrustes_cka.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved: {p}")
plt.close()

# ── Figure 2: Supporting metrics ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("rq sweep — supporting metrics", fontsize=12)

panels = [
    (axes[0, 0], VLOSS,   "Val Total Loss",       "lower = better", "#2c3e50"),
    (axes[0, 1], VKLLOSS, "Val KL Loss",          "lower = better", "#8e44ad"),
    (axes[1, 0], TRUST,   "Trustworthiness",      "higher = better","#27ae60"),
    (axes[1, 1], COLLAP,  "Latent Collapse Score","higher = more active dims", "#e67e22"),
]

for ax, vals, title, direction, color in panels:
    ax.plot(x, vals, "o-", color=color, linewidth=2, markersize=7)
    for xi, (v, n) in enumerate(zip(vals, NROWS)):
        ax.annotate(f"{n/1e6:.1f}M", (xi, v),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(rq_labels, fontsize=9)
    ax.set_title(f"{title}\n({direction})", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_xlabel("rq threshold")

plt.tight_layout()
p = out_dir / "supporting_metrics.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved: {p}")
plt.close()

# ── Figure 3: Head-to-head — rq-filtered vs random subsample at similar Nrows ─
# Closest row-count pairs:
#   rq<0.025 (4.4M)  vs random 5M
#   rq<0.075 (9.9M)  vs random 10M
#   rq<0.15  (19.2M) — no random equivalent; show alone

pairs = [
    (0.025, 4_435_615,  5_000_000),
    (0.075, 9_935_706,  10_000_000),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(
    "Head-to-head: rq-filtered vs random subsample at similar row counts\n"
    "Same test set encoded through both; compared to 52M reference",
    fontsize=11,
)

for ax, (metric, ylabel, better) in zip(
    [axes[0], axes[1]],
    [("procrustes_disparity", "Procrustes Disparity", "lower"),
     ("linear_cka",           "Linear CKA",           "higher")],
):
    bar_x, bar_h_rq, bar_h_rand, bar_err_rand, bar_labels = [], [], [], [], []
    pos = 0
    for rq_thresh, rq_n, rand_n in pairs:
        rq_row = next(r for r in rq_results if r["rq_threshold"] == rq_thresh)
        rand   = rand_at(rand_n)

        bar_x.append(pos)
        bar_h_rq.append(rq_row[metric])

        bar_x.append(pos + 0.8)
        rand_m_key = "proc_mean" if metric == "procrustes_disparity" else "cka_mean"
        rand_s_key = "proc_std"  if metric == "procrustes_disparity" else "cka_std"
        bar_h_rand.append(rand[rand_m_key])
        bar_err_rand.append(rand[rand_s_key])

        bar_labels.append(f"rq<{rq_thresh}\n({rq_n/1e6:.1f}M)")
        bar_labels.append(f"rand {rand_n/1e6:.0f}M\n({rand['n']} seeds)")
        pos += 2.2

    colors = []
    for i in range(len(bar_x)):
        colors.append(C_RQ if i % 2 == 0 else C_RAND)

    bars = ax.bar(bar_x, bar_h_rq + bar_h_rand, width=0.7, color=colors, alpha=0.75)

    # error bars only on random
    rand_pos  = [bar_x[i] for i in range(1, len(bar_x), 2)]
    rand_vals = bar_h_rand
    rand_errs = bar_err_rand
    ax.errorbar(rand_pos, rand_vals, yerr=rand_errs, fmt="none",
                ecolor="black", capsize=4, linewidth=1.5)

    ax.set_xticks(bar_x)
    ax.set_xticklabels(bar_labels, fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}  ({better} = better match to 52M)",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

legend_handles = [
    Patch(color=C_RQ,   alpha=0.8, label="rq-filtered"),
    Patch(color=C_RAND, alpha=0.8, label="random subsample (mean ± std)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=[0, 0.06, 1, 1])
p = out_dir / "headtohead.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved: {p}")
plt.close()

# ── Figure 4: Summary table ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 3.5))
ax.axis("off")

headers = ["rq<", "N rows", "Procrustes\n(vs 52M)", "CKA\n(vs 52M)",
           "Trustworthiness", "Latent Collapse", "Val Loss", "Val KL Loss",
           "vs rand subsample\n(same ~Nrows)"]

def fmt_compare(rq_val, rand_val, better):
    if rand_val is None:
        return "no ref"
    delta = rq_val - rand_val
    if better == "lower":
        verdict = "WORSE" if delta > 0.05 else ("similar" if abs(delta) <= 0.05 else "BETTER")
    else:
        verdict = "WORSE" if delta < -0.05 else ("similar" if abs(delta) <= 0.05 else "BETTER")
    return f"{verdict}\n(Δ={delta:+.3f})"

rand_ref = {
    0.025: rand_at(5_000_000),
    0.05:  rand_at(5_000_000),
    0.075: rand_at(10_000_000),
    0.1:   rand_at(10_000_000),
    0.15:  None,
}

rows_tbl = []
for r in rq_results:
    rr = rand_ref.get(r["rq_threshold"])
    if rr:
        cmp = fmt_compare(r["procrustes_disparity"], rr["proc_mean"], "lower")
    else:
        cmp = "no ref"
    rows_tbl.append([
        str(r["rq_threshold"]),
        f"{r['n_rows']:,}",
        f"{r['procrustes_disparity']:.4f}",
        f"{r['linear_cka']:.4f}",
        f"{r['trustworthiness']:.4f}",
        f"{r['latent_collapse']:.4f}",
        f"{r['val_total_loss']:.4f}",
        f"{r.get('val_kl_loss', float('nan')):.2f}",
        cmp,
    ])

tbl = ax.table(cellText=rows_tbl, colLabels=headers, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 2.2)
for (i, j), cell in tbl.get_celld().items():
    if i == 0:
        cell.set_facecolor("#1c3f6e")
        cell.set_text_props(color="white", fontweight="bold")
    elif i % 2 == 0:
        cell.set_facecolor("#fef5f5")
    # red highlight if Procrustes column (j=2) is very high
    if i > 0 and j == 2:
        try:
            v = float(rows_tbl[i-1][2])
            if v > 0.5:
                cell.set_facecolor("#ffe0e0")
        except:
            pass

ax.set_title(
    "rq sweep summary — all metrics vs 52M full-data reference",
    fontsize=10, fontweight="bold", pad=16,
)
plt.tight_layout()
p = out_dir / "summary_table.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved: {p}")
plt.close()

# ── Terminal summary ──────────────────────────────────────────────────────────
print(f"\n{'='*78}")
print(f"  rq sweep -- Procrustes & CKA vs full 52M reference")
print(f"{'='*78}")
print(f"{'rq<':>6}  {'N rows':>12}  {'Procrustes':>11}  {'CKA':>8}  "
      f"{'Trust':>7}  {'Collapse':>9}  {'Val Loss':>9}")
print("-"*78)
for r in rq_results:
    print(f"{r['rq_threshold']:>6.3f}  {r['n_rows']:>12,}  "
          f"{r['procrustes_disparity']:>11.4f}  {r['linear_cka']:>8.4f}  "
          f"{r['trustworthiness']:>7.4f}  {r['latent_collapse']:>9.4f}  "
          f"{r['val_total_loss']:>9.4f}")
print("-"*78)

print(f"\nRandom-subsample reference (mean Procrustes across all seeds):")
for rows in RAND_ROWS:
    s = rand_at(rows)
    print(f"  rand {rows/1e6:.0f}M ({s['n']} seeds): proc={s['proc_mean']:.4f}±{s['proc_std']:.4f}  "
          f"cka={s['cka_mean']:.4f}±{s['cka_std']:.4f}")
