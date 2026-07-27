"""
Plot seed density sweep (20 data seeds × 3 row counts) vs 52M reference.

Both sweeps fix --seed 42 (VAE weight init / split / batch order) and vary
only --data-seed (DuckDB row sampling), so all variability shown is from
which rows were sampled, not from stochastic training.

Outputs to seed_density_sweep/plots/:
  procrustes_vs_52M.png  — main result: Procrustes + CKA vs full model
  cv_stability.png       — CV% confirming/refuting the 5M threshold
  summary_table.png      — numeric table for 25 combined seeds

Run locally:
    cd "C:\\Users\\Owner\\Documents\\PURE Files"
    python VAE_Stability_Testing/scripts/plot_seed_density.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = Path(r"C:\Users\Owner\Documents\PURE Files\VAE_Stability_Testing")

# ── Load data ─────────────────────────────────────────────────────────────────

density  = json.loads((BASE / "seed_density_sweep"   / "vs_52M_results.json").read_text())
orig_ext = json.loads((BASE / "seed_sweep_extended"  / "vs_52M_results.json").read_text())

def by_rows(data: list[dict], rows: int) -> list[dict]:
    return [r for r in data if r.get("rows") == rows or r.get("sample_rows") == rows]

def vals(records: list[dict], key: str) -> list[float]:
    return [r[key] for r in records if r.get(key) is not None]

def cv(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = np.mean(xs)
    return float(np.std(xs) / m * 100) if abs(m) > 1e-12 else float("nan")

# Row counts in the density sweep
DENSITY_ROWS   = [2_500_000, 5_000_000, 10_000_000]
DENSITY_LABELS = ["2.5M",    "5M",       "10M"]

# Full row range from the original 5-seed sweep
ALL_ROWS   = [750_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000, 13_000_000, 26_000_000]
ALL_LABELS = ["750K",  "1M",      "2.5M",    "5M",       "10M",      "13M",       "26M"]

# Colours
C_ORIG    = "#e34948"   # red   — original 5 seeds
C_DENSITY = "#2a78d6"   # blue  — 20 new seeds
C_MED     = "#1c1c1e"   # near-black median tick

out_dir = BASE / "seed_density_sweep" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)

# ── Figure 1: Procrustes + CKA vs 52M ────────────────────────────────────────
# Left panel: Procrustes disparity (lower = more similar to 52M full model)
# Right panel: Linear CKA (higher = more similar)
#
# For both metrics:
#   • Line: median of original 5 seeds across all 7 row counts (context curve)
#   • Strip: individual seed values at 2.5M / 5M / 10M
#     - Red diamonds = original 5 seeds  (same 5 used in line)
#     - Blue circles  = 20 fresh seeds

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Latent geometry vs full 52M model — data-sampling seed variability\n"
    "Training seed fixed at 42 for all runs; only DuckDB row sample varies",
    fontsize=12,
)

for ax, key, title, better in [
    (axes[0], "procrustes_disparity", "Procrustes Disparity", "lower"),
    (axes[1], "linear_cka",           "Linear CKA",           "higher"),
]:
    # ── Background curve: original 5-seed median across all row counts ──
    orig_medians, orig_q1, orig_q3 = [], [], []
    for r in ALL_ROWS:
        v = vals(by_rows(orig_ext, r), key)
        if v:
            orig_medians.append(np.median(v))
            orig_q1.append(np.percentile(v, 25))
            orig_q3.append(np.percentile(v, 75))
        else:
            orig_medians.append(float("nan"))
            orig_q1.append(float("nan"))
            orig_q3.append(float("nan"))

    x_all = list(range(len(ALL_ROWS)))
    ax.plot(x_all, orig_medians, "-", color=C_ORIG, linewidth=2,
            zorder=4, label="5-seed median (750K–26M)")
    ax.fill_between(x_all, orig_q1, orig_q3, color=C_ORIG, alpha=0.12,
                    zorder=3, label="5-seed IQR")

    # ── Strip plots at the 3 density-sweep row counts ──
    for i, (r, rl) in enumerate(zip(DENSITY_ROWS, DENSITY_LABELS)):
        x_pos = ALL_ROWS.index(r)   # align with the background curve

        d_v = vals(by_rows(density,  r), key)
        o_v = vals(by_rows(orig_ext, r), key)

        # jitter horizontally
        jd = rng.uniform(-0.22, 0.22, len(d_v))
        jo = rng.uniform(-0.22, 0.22, len(o_v))

        ax.scatter([x_pos + j for j in jd], d_v,
                   color=C_DENSITY, alpha=0.55, s=22, zorder=6)
        ax.scatter([x_pos + j for j in jo], o_v,
                   color=C_ORIG, alpha=0.85, s=40, zorder=7, marker="D")

        # combined median tick
        combined = d_v + o_v
        if combined:
            med = np.median(combined)
            ax.hlines(med, x_pos - 0.35, x_pos + 0.35,
                      colors=C_MED, linewidths=2, zorder=8)

    ax.set_xticks(x_all)
    ax.set_xticklabels(ALL_LABELS, fontsize=9)
    ax.set_xlabel("Training rows")
    ax.set_ylabel(title)
    ax.set_title(f"{title}  ({better} = more similar to 52M)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    # Annotation arrows at the 3 overlap points
    for r, rl in zip(DENSITY_ROWS, DENSITY_LABELS):
        xp = ALL_ROWS.index(r)
        ax.axvline(xp, color="grey", linewidth=0.6, linestyle=":", zorder=1)

legend_handles = [
    mpatches.Patch(color=C_ORIG,    alpha=0.8,  label="Original 5 seeds (7,13,42,67,99)"),
    plt.Line2D([0], [0], color=C_ORIG, linewidth=2, label="5-seed median curve"),
    mpatches.Patch(color=C_ORIG,    alpha=0.15, label="5-seed IQR band"),
    plt.Line2D([0], [0], marker="o", color=C_DENSITY, linestyle="none",
               markersize=6, alpha=0.7, label="20 new seeds (density sweep)"),
    plt.Line2D([0], [0], color=C_MED, linewidth=2, label="Combined median"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=8.5, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=[0, 0.06, 1, 1])
p = out_dir / "procrustes_vs_52M.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved: {p}")
plt.close()


# ── Figure 2: CV% stability curve ────────────────────────────────────────────
# Shows whether more seeds push CV% above the 0.5 % "stable" threshold at 5M.
# CV% = std/mean × 100, computed over all seeds at that row count.

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "CV% across data seeds — does the 5M stability threshold hold with 25 seeds?\n"
    "(CV% = std/mean × 100 of Procrustes / CKA across data seeds; training seed = 42 fixed)",
    fontsize=11,
)

THRESHOLD_COLOR = "#888"

for ax, key, title in [
    (axes[0], "procrustes_disparity", "Procrustes Disparity CV%"),
    (axes[1], "linear_cka",           "Linear CKA CV%"),
]:
    # Original 5 seeds — all 7 row counts
    cv_orig = []
    for r in ALL_ROWS:
        v = vals(by_rows(orig_ext, r), key)
        cv_orig.append(cv(v))

    # 20 new seeds only — 3 row counts
    cv_new = {}
    for r in DENSITY_ROWS:
        v = vals(by_rows(density, r), key)
        cv_new[r] = cv(v)

    # Combined 25 seeds — 3 row counts
    cv_combined = {}
    for r in DENSITY_ROWS:
        d = vals(by_rows(density,  r), key)
        o = vals(by_rows(orig_ext, r), key)
        cv_combined[r] = cv(d + o)

    x_all = list(range(len(ALL_ROWS)))

    ax.plot(x_all, cv_orig, "o-", color=C_ORIG, linewidth=2,
            markersize=6, label="5 seeds (orig sweep)", zorder=4)

    # 20 new seeds at 3 points
    x_new  = [ALL_ROWS.index(r) for r in DENSITY_ROWS]
    y_new  = [cv_new[r]      for r in DENSITY_ROWS]
    y_comb = [cv_combined[r] for r in DENSITY_ROWS]

    ax.plot(x_new, y_new, "s--", color=C_DENSITY, linewidth=1.8,
            markersize=7, label="20 new seeds only", zorder=5)
    ax.plot(x_new, y_comb, "^-", color="#1a7a30", linewidth=2,
            markersize=8, label="25 seeds combined", zorder=6)

    # 0.5% stability threshold
    ax.axhline(0.5, color=THRESHOLD_COLOR, linewidth=1, linestyle=":")
    ax.text(len(ALL_ROWS) - 0.05, 0.52, "0.5% stability threshold",
            fontsize=8, color=THRESHOLD_COLOR, ha="right", va="bottom")

    # shade the 3 overlap columns
    for r in DENSITY_ROWS:
        xp = ALL_ROWS.index(r)
        ax.axvspan(xp - 0.4, xp + 0.4, color=C_DENSITY, alpha=0.06, zorder=1)

    ax.set_xticks(x_all)
    ax.set_xticklabels(ALL_LABELS, fontsize=9)
    ax.set_xlabel("Training rows")
    ax.set_ylabel("CV%")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_ylim(bottom=0)

plt.tight_layout()
p = out_dir / "cv_stability.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved: {p}")
plt.close()


# ── Figure 3: Summary table ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 3.5))
ax.axis("off")

headers = ["Rows", "Seeds (orig+new)", "Proc mean±std", "Proc CV%",
           "CKA mean±std", "CKA CV%", "Val Loss mean±std"]
rows_tbl = []

for r, rl in zip(DENSITY_ROWS, DENSITY_LABELS):
    d = by_rows(density,  r)
    o = by_rows(orig_ext, r)
    combined = d + o
    n_orig = len(o)
    n_new  = len(d)

    def stat(key):
        v = vals(combined, key)
        if not v:
            return "n/a", "n/a"
        return f"{np.mean(v):.4f}±{np.std(v):.4f}", f"{cv(v):.2f}%"

    proc_ms, proc_cv = stat("procrustes_disparity")
    cka_ms,  cka_cv  = stat("linear_cka")
    loss_ms, _       = stat("val_total_loss")

    rows_tbl.append([rl, f"{n_orig}+{n_new}={n_orig+n_new}",
                     proc_ms, proc_cv, cka_ms, cka_cv, loss_ms])

tbl = ax.table(cellText=rows_tbl, colLabels=headers, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.2)
for (i, j), cell in tbl.get_celld().items():
    if i == 0:
        cell.set_facecolor("#1c3f6e")
        cell.set_text_props(color="white", fontweight="bold")
    elif i % 2 == 0:
        cell.set_facecolor("#eef3ff")

ax.set_title(
    "Combined 25-seed summary vs 52M reference  "
    "(training seed=42 fixed; only DuckDB data-sampling seed varies)",
    fontsize=10, fontweight="bold", pad=16,
)
plt.tight_layout()
p = out_dir / "summary_table.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved: {p}")
plt.close()


# ── Terminal summary ──────────────────────────────────────────────────────────
print("\n" + "═" * 72)
print("  VERIFICATION: both sweeps use --seed 42 (VAE init) fixed;")
print("  only --data-seed varies (DuckDB row sampling).")
print("═" * 72)
print(f"\n{'Rows':>6}  {'N orig':>7}  {'N new':>6}  "
      f"{'Proc CV% (orig)':>17}  {'Proc CV% (new)':>15}  {'Proc CV% (combined)':>20}")
for r, rl in zip(DENSITY_ROWS, DENSITY_LABELS):
    d = by_rows(density,  r)
    o = by_rows(orig_ext, r)
    pv_o = vals(o, "procrustes_disparity")
    pv_d = vals(d, "procrustes_disparity")
    pv_c = pv_o + pv_d
    print(f"{rl:>6}  {len(o):>7}  {len(d):>6}  "
          f"{cv(pv_o):>17.3f}%  {cv(pv_d):>15.3f}%  {cv(pv_c):>20.3f}%")

print()
