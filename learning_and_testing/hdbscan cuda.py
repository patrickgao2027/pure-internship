import time
import numpy as np
from sklearn.datasets import make_blobs
import hdbscan  # Standalone package using Borůvka's algorithm
import cuml

# 1. Generate a dataset large enough to measure the speed gap
print("Generating dataset (100,000 samples, 10 features)...")
X, _ = make_blobs(n_samples=100_000, n_features=10, centers=5, random_state=42)
X = X.astype(np.float32) # Ensure fast float32 arrays for optimal GPU execution

# ========================================================
# BENCHMARK 1: Standalone HDBSCAN (CPU - Borůvka Algorithm)
# ========================================================
print("\n--- Running Standalone hdbscan (CPU w/ Borůvka) ---")
start = time.time()
# 'boruvka_kdtree' leverages the accelerated tree layout across all CPU cores
hdb_model = hdbscan.HDBSCAN(min_cluster_size=15, algorithm='boruvka_kdtree', core_dist_n_jobs=-1) 
hdb_model.fit(X)
hdb_time = time.time() - start
print(f"Standalone Borůvka Time: {hdb_time:.4f} seconds")

# ========================================================
# BENCHMARK 2: RAPIDS cuML (GPU)
# ========================================================
print("\n--- Running RAPIDS cuML (GPU) ---")
# Pin data directly to device memory to bypass CPU-to-GPU transfer overhead
X_gpu = cuml.Utils.input_to_dev_array(X) 

start = time.time()
cuml_model = cuml.cluster.HDBSCAN(min_cluster_size=15)
cuml_model.fit(X_gpu)
cuml_time = time.time() - start
print(f"RAPIDS cuML Time: {cuml_time:.4f} seconds")

# ========================================================
# Final Performance Summary
# ========================================================
print("\n================ SUMMARY ================")
print(f"RAPIDS cuML (GPU) is {hdb_time / cuml_time:.1f}x faster than Standalone Borůvka.")