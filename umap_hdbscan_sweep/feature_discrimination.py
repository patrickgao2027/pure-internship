#!/usr/bin/env python
"""Why does a UMAP panel look flat? Separate the five reasons.

A colour panel produced by ``cohort_cluster_report.py`` can look undifferentiated for
reasons that have nothing to do with each other, and the fix (or the correct conclusion) is
different for each:

1. DEAD          -- the column is all-null or single-valued, so the VAE never received it.
                    ``ml_features.json`` lists it, but preprocessing drops all-null numerics.
                    The panel is plotting a constant; matplotlib's +/-0.1 colourbar is its
                    zero-variance fallback, not a real range.
2. DEGENERATE    -- the column has a real dtype but one value dominates (mode_share high).
                    There is almost no information for any method to use.
3. NON-DISCRIMINATIVE -- the column varies honestly, the VAE consumed it, but the variance is
                    WITHIN clusters rather than between them. This is a genuine scientific
                    result about the latent space, not a defect.
4. OUTLIER-DRIVEN -- eta2 looks high, but a handful of tiny extreme-valued clusters supply
                    nearly all of it. The feature isolates those few groups; it does not
                    structure the space. Reads as "flat" because the groups are too small to
                    see. Treat as undiscriminating for general purposes.
5. SCALE-COMPRESSED -- the column discriminates broadly, but a long tail eats the colour
                    palette: matplotlib maps min->max, so if p99 is 13 and max is 236, 99% of
                    points land in the bottom 5% of the colormap and look identical.

Cases 1-4 are properties of the data. Case 5 is a plotting artefact and is the only one a
different colour scale would fix -- which is why it is reported separately.

NOTE ON COVERAGE: only the 29 numeric ``color_columns`` are plotted at all. The categorical
context features are never rendered, so the five most cluster-defining columns in this data
(X_PREV1, trinuc192, X_NEXT1, ALT, REF -- NMI 0.57-0.76) have no panel, while 13 DEAD columns
each get one. Absence of an image is not evidence about a feature.

The discrimination statistic is eta-squared for numerics (fraction of total variance
explained by cluster membership) and normalised mutual information for categoricals, both
against ``cluster_label``. Noise rows (label -1) are excluded: they are not a cluster, and
including them inflates every score by adding a large heterogeneous group.

Usage
-----
    python umap_hdbscan_sweep/feature_discrimination.py \
        --analysis <cell>/analysis.parquet

    # compare the same feature across cells
    python umap_hdbscan_sweep/feature_discrimination.py \
        --analysis a/analysis.parquet b/analysis.parquet --csv out.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# A column whose modal value covers at least this share carries little usable signal even
# when its dtype suggests otherwise.
DEGENERATE_MODE_SHARE = 0.90
# Below this eta2/NMI the feature does not separate clusters in any practically useful way.
LOW_DISCRIMINATION = 0.05
# If 99% of points occupy less than this share of the min-max range, the default colour scale
# is hiding whatever variation exists.
PALETTE_COMPRESSION = 0.25
# When the top five clusters supply at least this share of the between-group sum of squares,
# eta2 is reporting a few outlier groups rather than general cluster structure.
OUTLIER_DRIVEN_CONCENTRATION = 0.50

# Not features -- coordinates, identifiers, and the label itself.
EXCLUDE = {"umap_1", "umap_2", "cluster_label", "cluster_probability", "POS"}


def eta_squared(values: pd.Series, labels: pd.Series) -> tuple[float, float]:
    """Fraction of a numeric column's total variance explained by cluster membership.

    Returns ``(eta2, concentration)`` where concentration is the share of the between-group
    sum of squares contributed by the five highest-contributing clusters.

    The second number is not optional bookkeeping -- eta2 alone is actively misleading on this
    data. Each cluster contributes ``n_i * (mean_i - grand_mean)**2``, so a handful of tiny
    groups with extreme means can dominate a score that then reads as "this feature separates
    clusters". Measured case: SCED at 25M fit rows scores eta2=0.721, but 25 clusters of 15-54
    rows (means 27-134 against a grand mean near 0.5) supply nearly all of it, while the other
    847 clusters are indistinguishable. Reported alone, that 0.721 said the opposite of the
    truth. A high eta2 with high concentration means "isolates a few outlier clusters", not
    "separates clusters generally".
    """
    s = values.astype("float64")
    grand = s.mean()
    grouped = s.groupby(labels).agg(["mean", "count"])
    contributions = grouped["count"] * (grouped["mean"] - grand) ** 2
    between = contributions.sum()
    total = ((s - grand) ** 2).sum()
    if total <= 0:
        return 0.0, 0.0
    top5 = contributions.nlargest(5).sum()
    concentration = float(top5 / between) if between > 0 else 0.0
    return float(between / total), concentration


def normalised_mutual_info(a: pd.Series, b: pd.Series) -> float:
    """MI(a; b) / min(H(a), H(b)) -- 1.0 when one variable determines the other.

    Normalising by the smaller entropy rather than by sqrt(H(a)H(b)) keeps the score
    interpretable when the two variables have very different cardinality, which is the case
    here: 4 bases against several hundred clusters.
    """
    table = pd.crosstab(a, b).to_numpy(dtype="float64")
    n = table.sum()
    if n == 0:
        return 0.0
    joint = table / n
    p_a = joint.sum(axis=1, keepdims=True)
    p_b = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (p_a @ p_b)[nz])).sum())
    h_a = float(-(p_a[p_a > 0] * np.log(p_a[p_a > 0])).sum())
    h_b = float(-(p_b[p_b > 0] * np.log(p_b[p_b > 0])).sum())
    floor = min(h_a, h_b)
    return mi / floor if floor > 0 else 0.0


def classify(row: dict) -> str:
    """Which reason explains this feature's panel.

    Discrimination is tested before mode share, because the two are independent: a column can
    be 96% one value and still carry cluster information in its tail. But a raw eta2 is not
    enough to establish that, which is why OUTLIER-DRIVEN is checked first among the
    discriminating cases -- see ``eta_squared`` for the SCED case that forced it.
    """
    if row["non_null_pct"] < 5.0 or row["n_unique"] <= 1:
        return "DEAD"
    if row["score"] >= LOW_DISCRIMINATION:
        if row["concentration"] >= OUTLIER_DRIVEN_CONCENTRATION:
            # Five clusters supply nearly the whole score: the feature flags a few outlier
            # groups rather than structuring the space. Verified against SCED/SCST at 25M.
            return "OUTLIER-DRIVEN"
        # It does separate clusters generally. The only remaining question is whether the
        # default colour scale lets you see it.
        compressed = (
            row["palette_used"] is not None and row["palette_used"] < PALETTE_COMPRESSION
        )
        return "SCALE-COMPRESSED" if compressed else "ok"
    if row["mode_share"] >= DEGENERATE_MODE_SHARE:
        return "DEGENERATE"
    return "NON-DISCRIMINATIVE"


def profile(frame: pd.DataFrame) -> pd.DataFrame:
    assigned = frame[frame["cluster_label"] >= 0]
    if assigned.empty:
        raise SystemExit("no assigned rows -- every row is noise (label -1)")
    labels = assigned["cluster_label"]

    records = []
    for column in frame.columns:
        if column in EXCLUDE:
            continue
        series = assigned[column]
        non_null = float(series.notna().mean() * 100)
        n_unique = int(series.nunique(dropna=True))
        counts = series.value_counts(normalize=True, dropna=True)
        mode_share = float(counts.iloc[0]) if len(counts) else 1.0

        numeric = pd.api.types.is_numeric_dtype(series)
        # A dead column has no variance for eta2 to explain and no entropy for MI to use;
        # both would return 0 anyway, but computing them on 2M rows is wasted work.
        if n_unique <= 1:
            score, concentration, palette_used = 0.0, 0.0, None
        elif numeric:
            score, concentration = eta_squared(series, labels)
            s = series.astype("float64")
            lo, hi, p99 = s.min(), s.max(), s.quantile(0.99)
            palette_used = float((p99 - lo) / (hi - lo)) if hi > lo else None
        else:
            # NMI has no analogue of the outlier-group inflation eta2 suffers from: it is
            # built from probabilities, so a tiny group contributes in proportion to its mass.
            score = normalised_mutual_info(series, labels)
            concentration, palette_used = 0.0, None

        record = {
            "feature": column,
            "kind": "numeric" if numeric else "categorical",
            "statistic": "eta2" if numeric else "nmi",
            "score": score,
            "concentration": concentration,
            "non_null_pct": non_null,
            "n_unique": n_unique,
            "mode_share": mode_share,
            "palette_used": palette_used,
        }
        record["verdict"] = classify(record)
        records.append(record)

    return pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)


def render(table: pd.DataFrame, source: str) -> None:
    print(f"\n=== {source} ===")
    header = (
        f"{'feature':>16} {'stat':>5} {'score':>7} {'top5':>6} {'nonnull':>8} "
        f"{'uniq':>6} {'mode':>7} {'palette':>8}  verdict"
    )
    print(header)
    print("-" * len(header))
    for _, r in table.iterrows():
        # None survives the DataFrame round-trip as NaN, so test for it rather than identity.
        palette = "-" if pd.isna(r["palette_used"]) else f"{r['palette_used'] * 100:.0f}%"
        conc = "-" if r["statistic"] == "nmi" else f"{r['concentration'] * 100:.0f}%"
        print(
            f"{r['feature']:>16} {r['statistic']:>5} {r['score']:>7.3f} "
            f"{conc:>6} {r['non_null_pct']:>7.1f}% {r['n_unique']:>6} "
            f"{r['mode_share'] * 100:>6.1f}% {palette:>8}  {r['verdict']}"
        )

    print()
    for verdict, note in (
        ("DEAD", "all-null or constant -- dropped before the VAE ever saw it"),
        ("DEGENERATE", f"one value covers >={DEGENERATE_MODE_SHARE:.0%} of rows, and it does not separate clusters"),
        ("NON-DISCRIMINATIVE", "varies, but within clusters rather than between them"),
        ("SCALE-COMPRESSED", "DOES separate clusters -- only the default colour scale hides it; replot with percentile limits"),
        ("OUTLIER-DRIVEN", "high eta2 comes from a few tiny extreme clusters, NOT general structure -- treat as undiscriminating"),
    ):
        hits = table.loc[table["verdict"] == verdict, "feature"].tolist()
        if hits:
            print(f"  {verdict:>19}: {', '.join(hits)}")
            print(f"  {'':>19}  {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--analysis",
        nargs="+",
        required=True,
        type=Path,
        help="one or more analysis.parquet written by cohort_cluster_report.py",
    )
    parser.add_argument("--csv", type=Path, help="write the combined table here")
    args = parser.parse_args()

    combined = []
    for path in args.analysis:
        frame = pd.read_parquet(path)
        table = profile(frame)
        render(table, path.parent.name or str(path))
        table.insert(0, "cell", path.parent.name)
        combined.append(table)

    if args.csv:
        pd.concat(combined, ignore_index=True).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
