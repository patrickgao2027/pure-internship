#!/usr/bin/env python
"""Internal validity metrics for a density-based clustering: DBCV, probabilities,
persistence, connectivity.

These measure the geometry of the partition in the space it was fitted in. **None of them
knows anything about mutational biology**, which is both why they are cheap and why they must
never be read alone: the 2026-08-04 UMAP cells separate cleanly by every density criterion
here and are still 87 % single-trinucleotide-context, i.e. they score well precisely because
UMAP manufactured the separation the metric then rewards. Read these beside
``spectrum_metrics.top_channel_share``, never instead of it.

Nothing here is ranked or combined. The module returns numbers.

What is and is not computed
---------------------------
* **DBCV** -- delegated to ``rescore_dbcv.score_cell`` (already in this repo), which scores a
  shared subsample. Two limits it inherits and this module does not hide: HDBSCAN's own
  ``relative_validity_`` is a cheaper MST approximation and *not* true DBCV, and the sampled
  score drops clusters with too few sampled points.
* **Probabilities** -- mean and median are reported but the threshold fractions are the
  honest summary. ``probabilities_`` is lambda_p / lambda_max *within each cluster*, so it is
  renormalised per cluster: a partition of many small tight clusters posts a high mean without
  being better resolved.
* **Persistence** -- ``cluster_persistence_``, HDBSCAN's own stability measure. Partly
  circular under ``cluster_selection_method='eom'`` because that is the quantity EOM
  maximises; genuinely informative under ``leaf``.
* **Connectivity** -- fraction of each point's k nearest neighbours carrying its label.
  **Monotone in cluster count**: merge everything into one cluster and it is 1.0 by
  construction. It is reported next to ``n_clusters`` for that reason and is meaningless
  without it.
* **CDbw** -- deliberately absent. No maintained, validated Python implementation exists, it
  needs multiple representatives per cluster (O(n^2)-ish), and it would land as a third
  subsample-based density score next to DBCV without adding an independent question.
* **Dunn / generalised Dunn** -- deliberately absent, see the note below.

References
----------
* HDBSCAN cluster stability, and the excess-of-mass selection EOM maximises:
  Campello, Moulavi & Sander (2013), "Density-Based Clustering Based on Hierarchical Density
  Estimates", PAKDD, LNCS 7819, 160-172. Extended: Campello, Moulavi, Zimek & Sander (2015),
  ACM TKDD 10(1):5.
* ``cluster_persistence_`` / ``relative_validity_``: McInnes, Healy & Astels (2017),
  "hdbscan: Hierarchical density based clustering", JOSS 2(11):205. The library documents
  ``relative_validity_`` as "a fast approximation of the DBCV score" computed from the
  mutual-reachability MST, and warns it "might not be an objective measure of the goodness of
  clustering. It may only be used to compare results across different choices of
  hyper-parameters".
* DBCV: Moulavi, Jaskowiak, Campello, Zimek & Sander (2014), "Density-Based Clustering
  Validation", SDM, 839-847, doi:10.1137/1.9781611973440.96.
* Connectivity: Handl & Knowles (2005), "Computational cluster validation in post-genomic
  data analysis", Bioinformatics 21(15):3201-3212.
* CDbw: Halkidi & Vazirgiannis (2008), Pattern Recognition Letters 29(6):773-786.

On Dunn
-------
There is no canonical method called the "robust Dunn index". What exists is the family of
**generalised Dunn indices** in Bezdek & Pal (1998), "Some new indexes of cluster validity",
IEEE Trans. SMC-B 28(3):301-315, which identifies two deficiencies making the original Dunn
index (Dunn 1974, J. Cybernetics 4(1):95-104) "overly sensitive to noisy clusters" and
proposes generalisations "not as brittle to outliers".

Those generalisations are not adopted here, for a reason stated in that same paper: its
finding that minimum interset distance is the least reliable basis for a validity index holds
"when the clusters are expected to form volumetric clouds". HDBSCAN clusters are arbitrarily
shaped by construction, which is precisely the case DBCV was introduced to handle -- Moulavi
et al. open by noting that indices designed for globular clusters "may fail" on density-based
ones. A Dunn variant would also compress 1,000+ clusters into one scalar driven by the single
closest pair out of ~660,000, which cannot say WHICH clusters are poor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Connectivity and DBCV are O(n log n) and O(n^2)-ish respectively; both are computed on a
# subsample. Fixed seeds keep the sample identical across cells so scores are comparable.
DEFAULT_CONNECTIVITY_ROWS = 200_000
DEFAULT_DBCV_ROWS = 25_000
PROBABILITY_THRESHOLDS = (0.5, 0.8)


def probability_stats(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Membership strength over the ASSIGNED points only.

    Noise carries probability 0 by definition, so including it would make this a restatement
    of the noise fraction with extra steps.
    """
    assigned = labels >= 0
    if not assigned.any():
        return {"prob_n_assigned": 0}

    values = np.asarray(probabilities, dtype=np.float64)[assigned]
    stats = {
        "prob_n_assigned": int(assigned.sum()),
        "prob_mean": float(values.mean()),
        "prob_median": float(np.median(values)),
        "prob_p10": float(np.percentile(values, 10)),
        "prob_p90": float(np.percentile(values, 90)),
    }
    for threshold in PROBABILITY_THRESHOLDS:
        key = str(threshold).replace(".", "")
        stats[f"prob_frac_above_{key}"] = float((values > threshold).mean())
    return stats


def persistence_stats(clusterer) -> dict:
    """HDBSCAN's per-cluster stability, if the backend exposes it.

    Stability is the excess of mass of a cluster in the condensed hierarchy -- the integral
    of its membership over the density scales it survives (Campello, Moulavi & Sander 2013).
    ``cluster_persistence_`` normalises it to 0-1: 1.0 is "persists over all distance
    scales", 0.0 is "perfectly ephemeral".

    EOM selection chooses the set of clusters MAXIMISING total stability, so under
    ``cluster_selection_method='eom'`` the sum is the objective the fit already optimised and
    is not independent evidence about the fit. Under ``leaf`` it is.
    """
    values = getattr(clusterer, "cluster_persistence_", None)
    if values is None:
        return {}
    values = np.asarray(_to_host(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "persistence_mean": float(values.mean()),
        "persistence_median": float(np.median(values)),
        "persistence_min": float(values.min()),
        "persistence_max": float(values.max()),
        # Total stability mass. EOM maximises the sum, so under eom this is close to the
        # objective the fit already optimised and is not independent evidence.
        "persistence_sum": float(values.sum()),
    }


def connectivity(space: np.ndarray, labels: np.ndarray, n_neighbours: int = 10,
                 max_rows: int = DEFAULT_CONNECTIVITY_ROWS, seed: int = 0) -> dict:
    """Fraction of each point's k nearest neighbours that share its label.

    Neighbours are searched in the FULL space but scored only for sampled query points, so
    the answer is about the real neighbourhood structure rather than the subsample's.

    Rises monotonically as clusters merge -- a single cluster scores 1.0 -- so it says
    'the labels are locally coherent', not 'the clustering is good'. Always read with
    n_clusters.
    """
    from sklearn.neighbors import NearestNeighbors

    assigned = np.nonzero(labels >= 0)[0]
    if assigned.size < n_neighbours + 1:
        return {"connectivity_n": 0}

    reference = np.asarray(space[assigned], dtype=np.float32)
    reference_labels = np.asarray(labels)[assigned]

    rng = np.random.default_rng(seed)
    if assigned.size > max_rows:
        query_positions = np.sort(rng.choice(assigned.size, size=max_rows, replace=False))
    else:
        query_positions = np.arange(assigned.size)

    index = NearestNeighbors(n_neighbors=n_neighbours + 1, algorithm="auto").fit(reference)
    _, neighbours = index.kneighbors(reference[query_positions])
    # Column 0 is the query point itself; scoring it would add a constant 1/(k+1).
    neighbour_labels = reference_labels[neighbours[:, 1:]]
    agreement = (neighbour_labels == reference_labels[query_positions][:, None]).mean(axis=1)

    return {
        "connectivity_n": int(query_positions.size),
        "connectivity_k": int(n_neighbours),
        "connectivity_mean": float(agreement.mean()),
        "connectivity_median": float(np.median(agreement)),
        # Points whose entire neighbourhood agrees -- the interior of a cluster.
        "connectivity_frac_pure": float((agreement == 1.0).mean()),
        # Points that agree with fewer than half; a boundary or a mis-assignment.
        "connectivity_frac_below_half": float((agreement < 0.5).mean()),
    }


def dbcv(space: np.ndarray, labels: np.ndarray, max_rows: int = DEFAULT_DBCV_ROWS,
         max_clusters: int = 500, seed: int = 0) -> dict:
    """Density-based validity over a fixed subsample, via rescore_dbcv.score_cell.

    Delegated rather than reimplemented so this and the standalone rescore path cannot drift.
    Refuses above ``max_clusters`` because validity_index divides by per-cluster internal
    distances and raises outright when clusters have too few sampled points -- measured at
    2,000 clusters over 12,500 sampled rows.
    """
    import rescore_dbcv

    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
    if n_clusters > max_clusters:
        return {"dbcv": None,
                "dbcv_note": f"{n_clusters} clusters exceeds max_clusters={max_clusters}"}

    total = labels.shape[0]
    rng = np.random.default_rng(seed)
    sample = (np.sort(rng.choice(total, size=max_rows, replace=False))
              if total > max_rows else np.arange(total))
    try:
        result = rescore_dbcv.score_cell(np.asarray(space, dtype=np.float64),
                                         np.asarray(labels), sample)
    except Exception as exc:  # noqa: BLE001 - a DBCV failure must not cost the cell
        return {"dbcv": None, "dbcv_note": f"{type(exc).__name__}: {exc}"}
    # score_cell explains a null score under "note"; this wrapper uses "dbcv_note" for the
    # cases it declines itself. Two keys for one concept means a caller checking the wrong
    # one reports "no reason given" for a refusal that did give a reason -- so normalise.
    if "note" in result:
        result["dbcv_note"] = result.pop("note")
    return result


def _to_host(array):
    for attribute in ("to_numpy", "get"):
        method = getattr(array, attribute, None)
        if callable(method):
            return np.asarray(method())
    return np.asarray(array)


def summarise(space: np.ndarray, labels: np.ndarray, probabilities: np.ndarray | None = None,
              clusterer=None, *, connectivity_rows: int = DEFAULT_CONNECTIVITY_ROWS,
              dbcv_rows: int = DEFAULT_DBCV_ROWS, dbcv_max_clusters: int = 500,
              seed: int = 0) -> dict:
    """Every geometry metric for one clustering. No ranking, no combination."""
    record: dict = {
        "n_clusters": int(labels.max()) + 1 if (labels >= 0).any() else 0,
        "noise_fraction": float((labels < 0).mean()),
    }
    if probabilities is not None:
        record.update(probability_stats(probabilities, labels))
    if clusterer is not None:
        record.update(persistence_stats(clusterer))
    record.update(connectivity(space, labels, max_rows=connectivity_rows, seed=seed))
    record.update(dbcv(space, labels, max_rows=dbcv_rows,
                       max_clusters=dbcv_max_clusters, seed=seed))
    return record
