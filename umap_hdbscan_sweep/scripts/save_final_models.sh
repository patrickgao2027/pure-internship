#!/bin/bash
# Collect the 8 encoders chosen for the HDBSCAN phase into one folder.
#
# The sweep wrote its encoders under runC/encoders and runD/encoders with names keyed
# on fit rows, graph config, mode and seed. Those names are precise but unreadable at a
# glance, and the two runs split the fit sizes, so working from them directly in a later
# session means re-deriving which file is which every time. This copies the eight into
# umap_tests/final_models/ under names that carry the configuration, and writes a README
# holding the metrics each one was selected on.
#
# Copies rather than symlinks: the sweep directories are working output and may be
# cleaned. Eight encoders is ~3 MB.
#
# All eight are seed42 (replicate 0). Cluster counts move by roughly +-25 at 25M on the
# seed alone, so anything concluded downstream should be checked against replicate 1 or 2
# before it is treated as final -- those encoders sit in the same source directories under
# seed43 and seed44.
#
#     bash umap_hdbscan_sweep/save_final_models.sh
#
# Windows edit note:  sed -i 's/\r$//' umap_hdbscan_sweep/save_final_models.sh

set -euo pipefail

if [ -z "${SWEEP_DIR:-}" ] && [ -d "$HOME/pure-internship/umap_hdbscan_sweep" ]; then
    SWEEP_DIR="$HOME/pure-internship/umap_hdbscan_sweep"
fi
SWEEP_DIR="${SWEEP_DIR:-$HOME/umap_hdbscan_sweep}"
SRC="$SWEEP_DIR/umap_tests/parametric_sweep"
DST="${DST:-$SWEEP_DIR/umap_tests/final_models}"
SEED="${SEED:-seed42}"

# rank|run|source stem|destination name
MODELS=(
"13|runD|25000000__nn15_md0.1_nc2__umap|13_BEST_25M_nn15_md0.1_umap"
"14|runD|25000000__nn15_md0.25_nc2__umap|14_25M_nn15_md0.25_umap"
"11|runD|25000000__nn15_md0.25_nc2__hybrid|11_25M_nn15_md0.25_hybrid"
"07|runC|5000000__nn15_md0.25_nc2__umap|07_5M_nn15_md0.25_umap"
"02|runC|5000000__nn15_md0.25_nc2__hybrid|02_5M_nn15_md0.25_hybrid"
"05|runC|5000000__nn30_md0.25_nc2__umap|05_5M_nn30_md0.25_umap"
"01|runC|5000000__nn30_md0.25_nc2__hybrid|01_5M_nn30_md0.25_hybrid"
"09|runC|2000000__nn15_md0.25_nc2__umap|09_2M_nn15_md0.25_umap"
)

# Check every source before copying anything, so a missing encoder fails the whole run
# rather than leaving a folder that looks complete and is not.
missing=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r rank run stem dest <<< "$entry"
    if [ ! -f "$SRC/$run/encoders/${stem}__${SEED}.pt" ]; then
        echo "MISSING: $SRC/$run/encoders/${stem}__${SEED}.pt" >&2
        missing=$((missing + 1))
    fi
done
if [ "$missing" -gt 0 ]; then
    echo "$missing of ${#MODELS[@]} encoders not found -- nothing copied." >&2
    exit 1
fi

mkdir -p "$DST"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r rank run stem dest <<< "$entry"
    cp -p "$SRC/$run/encoders/${stem}__${SEED}.pt" "$DST/${dest}.pt"
    printf '  %-38s <- %s/%s\n' "${dest}.pt" "$run" "$stem"
done

cat > "$DST/README.md" <<'EOF'
# Final models for the HDBSCAN phase

Eight parametric UMAP encoders, all `seed42` (replicate 0), copied from the sweep's
`runC/encoders` and `runD/encoders`. Load with `torch.load(path)` -> `{"state_dict", "mode"}`
into `parametric_umap.ParametricEncoder(latent_dim, output_dim=2, hidden=(256,256,128))`,
then `ParametricUmap(encoder, device, mode).transform(latent)`.

Latent: `uv_vae/runs/train_multi_20260802T192756Z/stage1_embed/latent.npy` (157,501,580 x 16).
Embedding the full cohort through one encoder takes ~22 s.

## The set

| # | fit rows | nn | min_dist | mode | clusters | rnx ratio | cluster % | NMI | rep ARI | rep knn | role |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 13 | 25M | 15 | 0.1  | umap   | 505 | 0.9748 | 98.2  | 0.7918 | 0.261 | 0.185 | **chosen model** |
| 14 | 25M | 15 | 0.25 | umap   | 361 | 1.0536 | 109.2 | 0.7255 | 0.238 | 0.191 | min_dist control vs 13 |
| 11 | 25M | 15 | 0.25 | hybrid | 325 | 0.9966 | 98.3  | 0.7389 | 0.370 | 0.180 | mode control vs 14 |
| 07 | 5M  | 15 | 0.25 | umap   | 189 | 0.9735 | 109.2 | 0.7154 | 0.467 | 0.121 | fit-size control vs 14 |
| 02 | 5M  | 15 | 0.25 | hybrid | 169 | 0.9639 | 97.7  | 0.7378 | 0.596 | 0.143 | stability contrast |
| 05 | 5M  | 30 | 0.25 | umap   | 167 | 0.9564 | 88.7  | 0.7144 | 0.527 | 0.120 | n_neighbors control vs 07 |
| 01 | 5M  | 30 | 0.25 | hybrid | 160 | 0.9487 | 85.3  | 0.7337 | 0.679 | 0.155 | highest rep ARI in sweep |
| 09 | 2M  | 15 | 0.25 | umap   | 139 | 0.9697 | 79.6  | 0.6893 | 0.438 | 0.111 | smallest fit size |

Three matched umap/hybrid pairs at identical graph settings: 14/11, 07/02, 05/01.

## What the columns mean

- **clusters** -- HDBSCAN on 500k probe rows at `min_cluster_size=250, min_samples=25`.
- **rnx ratio** -- encoder RNX AUC / non-parametric UMAP RNX AUC. How much of a full UMAP
  fit's neighbourhood quality the encoder retains. Above 1.0 means it beat the reference.
- **cluster %** -- encoder cluster count / reference cluster count. 100 = same granularity;
  below is merging, above is splitting.
- **NMI / rep ARI / rep knn** -- agreement with the reference clustering; cross-encoder
  agreement on cluster labels; cross-encoder agreement on 15-NN neighbourhoods.

## Carry these caveats forward

1. **Cluster counts are seed-sensitive.** 508-533 at 25M and 142-184 at 5M with *nothing*
   changing but the training seed. Treat any single count as approximate.
2. **rep ARI and rep knn disagree systematically.** rep knn ranks by fit size (25M best);
   rep ARI ranks the opposite way. rep knn measures the embedding, rep ARI measures the
   embedding plus a clustering step with its own instability. For the clustering phase,
   rep ARI is the relevant one -- 0.261 for model 13 is the baseline to improve on.
3. **`min_cluster_size` is the main lever**, currently 250. The 139-505 spread in cluster
   count across this set exists so you can test whether a fine model with a high
   `min_cluster_size` lands where a coarse model with a low one does.
4. **The acceptance test is signature stability, not label stability.** Cluster two
   independently trained encoders and compare aggregated SigProfiler profiles rather than
   point assignments -- boundaries can shuffle while the aggregate profile holds.
5. **Global-structure (Spearman) scores do not separate these models.** Seed-to-seed
   variation exceeded the between-model differences. Do not rank on it.

Full metrics: `umap_tests/parametric_sweep_results.xlsx`, sheets `Shortlist` and `Seed tests`.
EOF

echo
echo "wrote $DST"
ls -la "$DST"
