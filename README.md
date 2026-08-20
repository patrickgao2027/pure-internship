My learning / code for the Pure Summer Program at Sabanci University

7/6 
Met team and went through UV_VAE /  Single Read SNV to understand project and optimization

7/7
Got HPC access met with Ogulan to discuss goals, read more of Single Read SNV

7/8
Researched cuML, Adjusted rand index, Procrustes Distance

7/9
Finally got HPC to work and went through first run through of uv_vae with 5000 samples

7/10
Switched to VAE encoding problem -> figure out how to subsample

7/12 
Plan for 7/13 
I will test VAE with procrustes distance based upon subsamples of the data input
I have two options for drawing a test dataset based upon concerns of rare UV signatures occuring

1.  Randomly sample with a different seed and the same filter as current UV_VAE
    The current UV_VAE has only the filter passed in the pipeline and a random sample
    It takes some quality readings and then a random sample based upon the seed.
    I would sample a new seed and then I would use that as my test dataset
    It would have the same random rate of rare Uv signatures
2. Stratify sample based upon COSMIC sigatures of before and after for UV
    I would have claude code write the SQL filter to get the rows with before and after that have similar signatures to what is in COSMIC
    Stratify sample from there -> get at least N different samples that have the specific nucleotide contexts
    Build the rest of the sample from rows without those specific contexts
    Rest of sample would include rare uv signatures that don't follow common contexts through randomness.  (I don't see a way to select for this)
 
Compute: Sweep through VAE encoding for different subsamples and max data input.  

7/13 
Created test dataset of 100,000 rows selected from parquet with 192 million rows
"/cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet"
Same filter as default in pipeline "WHERE st = 'MIXED' AND et = 'MIXED' AND FILT = 1"
SEED = 99

Claude Code wrote VAE stability test script for subsamples of input N rows and gives different stability metrics
Went through 1M sweep
7/14
Went through 10M sweep, seed testing, and full data sweep\
7/15
Interpret all data sweep results + plan next tests \
Planned Tests: Test more seed swaps across 750K to 26M, Epoch sweeps from 750K to 26M, Subsample sweep for other parquet dataset\
Started seed checks -> going through code base verifying seeds and how they are affected by changing input seeds

set deterministic seeds at the beginning 
def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

row sampling
with connect_duckdb(threads=1) as sample_conn:
    sampled_frame = sample_frame(
        conn=sample_conn,
        parquet_path=config.parquet_path,
        feature_names=feature_names,
        sample_rows=sample_rows,
        seed=config.seed,
        where=config.row_filter,
    )

train/val split
rng = np.random.default_rng(seed)
indices = rng.permutation(row_count)
train_size = max(1, min(row_count - 1, int(row_count * train_fraction)))
train_indices = indices[:train_size]
val_indices = indices[train_size:]
if val_indices.size == 0:
    val_indices = train_indices[:1]

train_numeric = numeric_matrix[train_indices]
train_numeric_mask = numeric_mask_matrix[train_indices]
val_numeric = numeric_matrix[val_indices]
val_numeric_mask = numeric_mask_matrix[val_indices]
\
Most seeds are correct and set.  Changed code to only change seed for data sampling.  

7/16
Saw results from epochs sweep, 5 seed testing of 750K to 26M, and other non-UV damaged subsample full data sweep. \
Results were that input data sizes of 2.5M to 5M has the best performance relative to data input size.  Compared to the 26M input the results were similar or better depending on the seed with less variability.  Based upon the 5 seeds, Procrustes distance, Linear CKA, and Trustworthiness metrics had the smallest variability and best metrics. Data input sizes above get about the same performance but need more time to train. The best epochs for a data input of 2.5M to 5M would be 10 epochs. The results are about the same for the other non-UV damaged sample training. However the ~5M subsample result is a lot worse than expected. This could be due to a bad seeding.  \

Next Directions:  
Go through 10-20 more seeds of 2.5M 5M 10M to see if the results even out  

From chat with Ogulan -> IDK  

I believe that if we can get a subset of the full data that is truly representative of full data.  Training a VAE on the subset will have similar performance and latent encoding with an approximated VAE model compared to the full dataset.  

Test Dataset Selection:  
Randomly sample 1 billion rows from the full merged dataset.  Inference doesn't take that long so having a lot of rows to test on is reasonable.  

Training Dataset Size:
52M * 95 ~ 5 Billion rows
I think we should start at 1 billion rows to sample from and go up / down depending on computing requirements

Idea 1 for sampling a dataset to train on:  
Select a billion rows from the 
From the full 95 samples -> select 0.095 from each dataset to get an even stratified sample
From within the 0.095 of each sample -> go through the nucleotides and get an even representation within each nucleotide substitution context  
Or just keep the same distribution as seen within each sample -> purely random as before 

Idea 2 for sampling a dataset to train on:  
Train HDBSCAN on full dataset (on GPU)
Pass all data through BIRCH + K Means, the inversely sample according to distance  
Sample enough of the rare ones -> points outside / further away from clusters  
Sample enough of the common ones -> points close to the center of clusters  

Idea 3 for sampling a dataset to train on:
SNVQ Filtering -> choose a set filter (prob 50 or above)
Anything below filter is not used and anything above is passed into training dataset -> gets a lot less data

7/20  
Ideas for training full dataset  
1. Dropout with high rate 0.4 to 0.5 and add input dropout p = 0.1 smaller DONE
2. Early stopping -> early stop for 1 or 2 epochs if loss does not get better DONE
3. Higher validation set so less data gets passed through -> 0.2 DONE
4. Change to float16 on GPU   DONE
5. Add cuML and cuDF libraries for zero code change acceleration DONE
6. potentially add KL weight -> DON'T ADD, I do not see the reasoning behind this and it is not used within ML benchmarking studies
7. Look at active units 
8. Procrustes distance again check if anything changes between epochs DONE
9. KL shift -> DON'T ADD, I do not see the reasoning behind this and it is not used within ML benchmarking studies
10. Drop unimportant columns after training run and retrain -> check difference with procrustes, etc
11. Tailor batch size
12. Add something about latent dimension reduction participation ratio / covariance matrix 
13. Get characteristics of the clusters

Files to transfer (5 files):

uv_vae/uv_vae/streaming.py — cuDF + convergence integration  
uv_vae/uv_vae/convergence.py — new convergence tracker module  
Early_Stopping_Tests/Python Files/train_with_early_stopping.py — --test-parquet-path + --convergence-rows args  
Early_Stopping_Tests/scripts/run_train_only.sh — TEST_PARQUET env var  
uv_vae/scripts/run_full_pipeline.sh — parameterized values + TEST_PARQUET  

7/21 
Batch Size Tuning:  
32,768 is the current batch size 
Larger Batch Sizes can speed up training and potentially reduce the number of epochs required but might lead to overfitting if not monitored properly.  Alleviate the overfitting problem by more aggressive dropout percentage and early stopping.  Larger batch size requires larger learning rate that should be scaled according to some rules.

Also add warm up for changed learning rate

https://www.geeksforgeeks.org/machine-learning/how-to-choose-batch-size-and-number-of-epochs-when-fitting-a-model/  
https://www.geeksforgeeks.org/deep-learning/how-should-the-learning-rate-change-as-the-batch-size-changes/  
https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e  

7/22 + 7/24
To Do: 
1. Batch size limits for GPU NVME SSD GBs cap thing  
3. Get characteristics of the clusters -> what else needs to be computed apart from what is seen in sigprofiler? 
5.  Auto encoder instead of variational auto encoder 
    a. https://www.ibm.com/think/topics/autoencoder 
    b. https://medium.com/data-science/difference-between-autoencoder-ae-and-variational-autoencoder-vae-ed7be1c038f2   
    c. https://arxiv.org/html/2604.22099v1#S4 alternatives to VAE
6. Do full pipeline runs of what we already have -> check for rq filtering, full, and some subsamples...


Done:
1. Tailor specific batchsize, learning rate, and learning rate warmup for GPU training  
    a. For batchsize, learning rate, and learning rate warmup have slurm scripts to test on GPU
    b. only feasible for full dataset on GPU, with speed ups for batchsize, cuDF, cuML etc  
2. Scripts for dropout and early stopping depending on val loss / active units
4. Think of KL weight as trainable parameter -> give sweeps to see performance.  
    a. Messes with disentanglement / entanglement of VAE encoding
    b. Disentanglement is when one latent dimension captures one source of variation in the data, entanglement is when its a mesh together
    c. Needs full pipeline runs which is computationally expensive
    d. https://openreview.net/pdf?id=Sy2fzU9gl  
    e. beta parameter is added to loss function in KL divergence, allows to choose between reconstruction and KL 
1. Formulate why Active Units is so useful for detecting when latent space is learned
    a. https://arxiv.org/pdf/1509.00519
    b. measures activity of a latent dimension U through the statistic cov(E---)
## VAE Theory  
- True posterior p(z|x) is intractable → approximate with q_φ(z|x) = N(μ(x), σ²(x))
- Encoder outputs μ and log σ² per latent dimension j — this is the variational assumption
## ELBO  
- Can't minimize KL(q‖p) directly → derive ELBO as tractable surrogate
- L = E[log p(x|z)] - D_KL(q_φ(z|x) ‖ p(z))
- Maximizing ELBO = better reconstruction + keeping q close to prior
## KL Closed Form  
- Choose prior p(z) = N(0,I) → off-diagonal covariance = 0 → dimensions independent
- Closed form per dimension: KL_j = ½(μ² + σ² - log σ² - 1)
- In code: -0.5 * sum(1 + log_var - mu² - exp(log_var))
## Posterior Collapse & Active Units  
- KL term pushes μ_j(x) → 0 for all x, collapsing the x conditioning
- Active unit: Var_x[μ_j(x)] > ε → dimension responds to input
- Dead unit: Var_x[μ_j(x)] ≈ 0 → KL won, dimension ignores x
- Compute: all_mus.var(dim=0) → shape (latent_dim,), threshold at ε
## Mitigations  
- KL annealing: delay KL penalty early in training
- Free bits: clamp per-dim KL to floor λ so optimizer can't zero it out
7. KL Collapse
    a. how to track https://arxiv.org/pdf/1911.02469
    b. https://arxiv.org/abs/1804.03599 mean of KL term
    c. https://arxiv.org/pdf/1911.02469 as a distribution percentage below threshold
    d. https://www.emergentmind.com/topics/posterior-collapse

Rejected:
1. Add something about latent dimension reduction through participation ratio / covariance matrix -> could the covariance matrix reveal something about dimensions not being correlated / could be reduced  DONT DO
2. Rank importance by KL divergence between each feature and the prior, larger KL divergence means more importance https://arxiv.org/abs/1804.03599  
    A. NEW Direction -> NO need to use KL divergence method as method of feature selection
3. Feature selection methods DONT NEED
    a. https://academic.oup.com/bib/article/27/1/bbag006/8441040 review of specific unsupervised feature selection methods
    b. PCA, ICA
    c. Honestly after more thought, it doesn't seem as useful as intended.  
    d. could be useful if we don't want autoencoder structure 


# Week of 7/27 to 8/31
7/27 
I got GPU access and was able to start doing tests on the VAE model to finalize a model that can be used for inference.  At the beginning of the week, I had to figure out a way to fit the data into the GPU memory without running out and allocate a specific amount such that I don't take up the entire GPU memory in the cluster.  I allocated myself 16/48 gigabytes to do my training and had Claude generate me code to load the data into the GPU in batches.  Also created my micromamba environment in the miletus HPC.    
7/28  
More edits to GPU budgeting and looked at strategies to load data into the GPU / sampling from the 95 parquet files to ensure representative batches.  
7/29  
Settled on a sampling strategy.  To make sure that each sample was represented proportionally to the amount each parquet is represented in the full data, I had the weights calculated for each parquet to get the amount that is represented in each batch size pushed into the VAE for training.  Since the data was structured such that groups of ALT, REF, POS, and CHROM were all stacked on top of each other, I had to shuffle the row groups such that the structuring was destroyed.  During one forward phase, the row groups would be shuffled for each parquet and each group would be visited during one batch.  Within the batch N rows are sampled and passed into the batch.  In the next batch this gets repeated, and after one forward pass the groups are shuffle again so that during the next forward pass a different set of row groups are seen first.  The validation set was created by using a hash function (SplitMix64) to generate a hash key for each row in the data, a specific cutoff was set such that 10% of the data would be unseen by the VAE model during training and would be held out as the validation set.  The hash function results in rows that are uniformly distributed for a 10% cutoff is very easy.  Early stopping was settled on a week ago.  I would be looking at val_loss, KL divergence, and active units.  Early stopping would be set to whatever I felt was good, batchsize, learning rate warm up, dropout would all be tuned in the tests.  I did not want to do a sweep, just changing parameters to whatever would work best depending on the results of the initial test.  
7/30 
Tests initially had bad results for VAE.  Most of the time, the data was not loading fast enough from the CPU into the GPU ( the problem was converting the data into tensors that are loaded into the GPU for training).  After the data was loaded into the GPU, the calculations were fast but then the GPU was sitting idle waiting for the CPU to continue loading more data.  A potential fix I thought for this was to simply load the data raw into the GPU from CPU and then let the GPU do the convert.  This made the data decoding process slower.  Also for CPU process to decode tried to do make CPU work on decoding the next batch while GPU is doing it's work.    
7/31
Did tests on VAE  
8/1  
Claude generated UMAP/HDBSCAN sweeps.  
8/2
Check claude generated code. Drop SNVQ from dedup process.  Made it so that decode (create tensors + val split creation) uses multiple cores instead of 1 core.  Pre buffer parquet file reader with PyArrow, just puts more of parquet in memory at once and allows for faster loading.  
8/3  
Went to bank.  

## Full Cohort VAE Training Runs (95 files, 5.08B eligible rows, latent_dim=16, hidden_dims=[256,128], seed=42)

| Run | Started | Batch | LR | KL Wt | In DO | Hid DO | Pat | Decode | Epochs | Best | AU | Collapsed | Val Total | Val KL | Val Num | Val Cat | Train Total | **Wall Clock** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260730T071636Z | 07-30 07:16 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | failed, no artifacts |
| 20260730T084507Z | 07-30 08:45 | 131,072 | 0.001 | 0.05 | 0.1 | 0.4 | 8 | CPU | 20 | 19 | 3/16 | 81.25% | 1.4943 | 5.0760 | 0.4566 | 0.7839 | 1.5768 | **4h 46m** |
| 20260730T182607Z | 07-30 18:26 | 131,072 | 0.001 | 0.05 | 0.1 | 0.4 | 8 | **GPU** | 20 | 19 | 3/16 | 81.25% | 1.4943 | 5.0760 | 0.4566 | 0.7839 | 1.5768 | **10h 52m** |
| 20260731T065905Z | 07-31 06:59 | — | — | — | — | — | — | CPU | 30+ | — | — | — | — | — | — | — | — | killed mid-run |
| 20260731T144544Z | 07-31 14:45 | 1,048,576 | 0.001 | 0.005 | 0.1 | 0.1 | 8 | CPU ×1 | 38 | 30 | 16/16 | 0% | 0.2225 | 28.9258 | 0.0756 | 0.0023 | 0.3126 | **10h 30m** |
| 20260801T054132Z | 08-01 05:41 | 262,144 | 0.001 | 0.005 | 0.1 | 0.1 | 4 | CPU | 8 | 4 | 16/16 | 0% | 0.2471 | 29.9646 | 0.0933 | 0.0040 | 0.3309 | **1h 45m** |
| A_patience | 08-01 20:52 | 262,144 | 0.001 | 0.005 | 0.1 | 0.1 | 10 | CPU | 22 | 12 | 16/16 | 0% | 0.2379 | 28.7418 | 0.0912 | 0.0029 | 0.3049 | not logged |
| B_lowlr | 08-02 02:00 | 262,144 | **0.0005** | 0.005 | 0.1 | 0.1 | 10 | CPU | 25 | 24 | 16/16 | 0% | 0.2377 | 28.3460 | 0.0914 | 0.0046 | 0.2968 | not logged |
| 20260802T180314Z | 08-02 18:03 | 32,768 | 0.001 | 0.05 | 0.1 | 0.4 | 8 | CPU | 2 | 2 | 4/16 | 81.25% | 1.5778 | 4.5893 | 0.5000 | 0.8483 | 1.5969 | not logged (2 epochs) |
| 20260802T192756Z | 08-02 19:27 | 1,048,576 | 0.001 | 0.005 | 0.1 | 0.1 | 8 | CPU ×8 | 38 | 30 | 16/16 | 0% | 0.2225 | 28.9258 | 0.0756 | 0.0023 | 0.3126 | **4h 25m** |

### Two controlled comparisons

**GPU decode cost 2.3×** — 084507Z vs 182607Z: identical config, identical results, CPU 4h 46m → GPU 10h 52m.

**Parallel decode workers gained 2.4×** — 144544Z vs 192756Z: identical config, byte-identical results, 1 worker 10h 30m → 8 workers 4h 25m.

Wall times come from the tmux runner logs (`===== END: train =====`), which only exist for 5 of the 8 completed runs. A_patience, B_lowlr and 180314Z have empty `logs/` dirs.

8/4 and 8/5  
It was a bad idea to just do sweeps through all the potential parameters for UMAP and HDBSCAN at once.  Instead I will settle on a UMAP model first, and then finalize the HDBSCAN.  From 480 trained models that would have come from the sweeps, only 5/480 were able to be completedd.  I analyzed the results of the 5 and saw the UMAP embeddings.  From the UMAP representation, there is better separation of the clusters than before and the cosine similarity had some clusters in the upper 90s.  Silhoutte score was also at least 0.5 for the 5 sweeps of the HDBSCAN parameters.  The problem is that the sweep would have taken 10 days due to the biggest time consumers being transforming the points after UMAP is trained and transforming the points after HDBSCAN is trained.  So the bottleneck right now is transforming the points after the UMAP model is trained.  A solution I am looking at right now is to use parametric UMAP which uses a trained neural network model that has loss function based upon the loss used in UMAP training as a transformer for the points.  

Parametric UMAP sources:
https://pmc.ncbi.nlm.nih.gov/articles/PMC8516496/
https://umap-learn.readthedocs.io/en/latest/parametric_umap.html    
https://arxiv.org/abs/2009.12981
https://github.com/timsainb/ParametricUMAP_paper/
original UMAP  
https://arxiv.org/pdf/1802.03426
Metrics to use to compare different UMAP embeddings:
Jaccard Score, Trustworthiness (extent to which local structure is maintained), Davies-Boudin Score (average similarity of cluster to most similar cluster, similarity calculated as ratio of within cluster distance to between cluster distance), Calinski Harabasz Score (), Area Under the Curve (AUC) of RNX 
Sources: 
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html  
https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html  
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
https://www.sciencedirect.com/science/article/pii/S0925231215003641

8/6 Testing Plan
Checked parametric UMAP code yesterday and learned how the UMAP algorithm really works.
Today I will settle on an input size for the UMAP, see if the data input eventually converges to a stable point
    1. Test data inputs 1M, 2.5M, 5M, 10M, 25M, 50M and see how long it takes to train
    2. Check with metrics trustworthiness, continuity, spearman pearson correlation, procrustes

8/6 – 8/10: 2D VAE Latent Experiment (no UMAP)
Instead of settling UMAP first, tested whether the VAE's 2D latent space alone (without UMAP projection) can support direct HDBSCAN clustering.  Model: same 95-sample cohort checkpoint (run_20260806T132009Z, latent_dim=2, hidden=[256,128,64,16], β=0.01, 157,501,580 loci embedded).

Latent space stats: dim0 std=0.703 (var share 35.4%), dim1 std=0.950 (var share 64.6%).  Both dims alive (collapsed_dims=[]), but anisotropic and showing a cross-shaped filament structure — vertical arm at latent1≈0 suggesting conditional posterior collapse in dim0 for a large subpopulation.

## 2D Latent HDBSCAN Parameter Grid

Swept min_samples (5, 25) × cluster_selection_method (eom, leaf), min_cluster_size fixed at 1000, metric=euclidean.  HDBSCAN fit on 5M rows (GPU memory limit), remaining 152.5M assigned via approximate_predict.  Backend: cuML GPU.

### Results — noise-contaminated (sklearn includes noise label −1 as a cluster)

| Cell | Clusters | Noise % | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|---|---|---|---|---|---|
| mcs1000_ms5 (eom) | 346 | 52.7% | −0.297 | 2.385 | 389,765 |
| mcs1000_ms25 (eom) | 293 | 61.8% | −0.417 | 2.118 | 416,935 |
| mcs1000_ms5_leaf | 1,050 | 86.0% | −0.773 | 3.926 | 27,116 |
| mcs1000_ms25_leaf | 604 | 84.5% | −0.738 | 2.382 | 85,570 |

### Results — noise excluded (recomputed from analysis.parquet, non-noise rows only)

| Cell | Non-noise rows | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|---|---|---|---|---|
| mcs1000_ms5 (eom) | 74,483,676 | **+0.434** | 0.903 | 19,401,181 |
| mcs1000_ms25 (eom) | 60,174,009 | **+0.426** | 0.664 | 16,727,322 |
| mcs1000_ms5_leaf | 22,006,811 | **+0.426** | 2.254 | 7,585,673 |
| mcs1000_ms25_leaf | 24,397,092 | **+0.516** | 0.824 | 4,954,254 |

All reproduction checks passed (delta=0.0): the negative silhouette values were entirely noise contamination artifacts.  The real clusters are well-separated (silhouette 0.43–0.52).  The problem is coverage: 53–86% of loci unassigned, driven by continuous density gradients in the 2D latent with no sharp gaps for HDBSCAN to exploit.

**Best cell: mcs1000_ms5 (eom)** — lowest noise fraction (52.7%), most clusters (346), best coverage.  mcs1000_ms25 has the tightest clusters (DB=0.664) but assigns fewer loci.  leaf selection makes noise substantially worse (84–86%) with no improvement in cluster quality.

### SigProfiler Assignment (UV-only SBS, GRCh38, v3.5)

Mean cosine similarity ~0.30 across all 4 cells — nearly identical regardless of parameter choice.  Parameters move assignment rate (how many loci get a cluster), not assignment quality (how well clusters match UV signatures).  mcs1000_ms5 yielded 14.9M loci in clusters with cosine ≥ 0.7.

**Conclusion:** The 2D VAE latent space forms real, well-separated clusters but leaves ~half the data as noise due to the continuous-gradient geometry of the space.  Adding UMAP before HDBSCAN (the standard pipeline) remains the expected path forward — UMAP explicitly optimises for density gaps, which is exactly what HDBSCAN needs.

8/12 
HDBSCAN 
25M is theoretical limit, 50M OOMs
approximate_predict time is also a constraint, 
https://docs.rapids.ai/api/cuml/nightly/api/generated/cuml.neighbors.nearestneighbors/
https://arxiv.org/pdf/1103.2635
change KNN algorithms to RBC which is faster
HDSBCAN backend uses KNN to find closest cluster

8/18 columns included
Expected: 69  In atlas: 69  PNGs: 69
Numeric: 51  Categorical: 18
Missing: none
Extra:   none
---

## Mutational Signature Assignment — SigProfiler

| Parameter | Value |
|---|---|
| Tool | SigProfilerAssignment |
| Reference genome | GRCh38 |
| COSMIC version | 3.5 |
| Signature database | `uv_only` (lab-specific reference; not full COSMIC) |
| Input | Per-cluster SBS96 mutation spectrum (96 trinucleotide channels) |
| SBS96 construction | Vectorised lookup table over (PREV, REF, ALT, NEXT) base codes |

---

## Engineering Methods for Scale

### Proportional Interleaved Multi-Parquet Loading

Training across 95 parquet files simultaneously. Each file's contribution to every batch is proportional to its post-filter row count (largest-remainder allocation), ensuring no sample is over- or under-represented. Row groups within each file are read in shuffled order each epoch to break genomic position clustering (~88 % of adjacent rows share the same locus in the sorted parquet files).

### Content-Hash Train/Val Split

Row assignment to train or val is determined by a deterministic hash of the genomic site (chromosome + position), not position within the file. This keeps all reads at the same locus on the same split side, preventing data leakage across the train/val boundary when reading interleaved from 95 files.

### Epoch Sharding

The 5B-row dataset is split into 20 shards per epoch. Each shard processes a different shuffled subset of row groups, giving the model variation across training without loading the full dataset into memory at once.

### Streaming Inference with Batch-Bounded Memory

Per-sample inference (VAE encode → UMAP → HDBSCAN label) is fully streaming: rows are read from each parquet in 5M-row batches via DuckDB Arrow record-batch reader. Peak memory scales with batch size (~1.2 GB GPU), not sample size (largest sample: 226M rows). Between samples, glibc memory arenas are returned to the OS via `malloc_trim(0)` to prevent RSS accumulation.

### Parametric UMAP for Scalable Projection

Training a parametric UMAP encoder once on 25M rows and applying it via forward pass to new batches reduced the per-sample projection step from hours (standard UMAP refit) to seconds per batch. This was the key step that made full-cohort per-sample inference practical.

### GPU-Accelerated HDBSCAN with RBC Approximate Predict

The cohort HDBSCAN model was reused across all 95 samples via `approximate_predict` with a Random Ball Cover (RBC) index built on the GPU (cuML). RBC precomputes a ball-cover index over the fit points, reducing nearest-neighbour search for new points from O(n·fit_size) to sub-linear. This reduced per-sample labelling from minutes to seconds.

### Vectorised SBS96 Construction

Trinucleotide context (SBS96 channel) assignment was originally computed row-by-row via Python string operations, stalling on 200M-row samples. Replaced with a 625-entry lookup table (LUT) over integer base codes, enabling vectorised numpy indexing over the full batch with no Python-level iteration.

### AMP Mixed Precision Training

Automatic Mixed Precision (AMP) was enabled for the VAE training loop. Forward passes and loss computation run in float16; gradients are accumulated and applied in float32 (GradScaler). This approximately halved GPU memory use per batch, allowing the 1M-row batch size to fit within a 16 GB torch budget on the 48 GB card.

### Row Identity via file_row_number + Source Fingerprint

UMAP coordinates and HDBSCAN labels are joined back to source parquet rows using DuckDB's physical `file_row_number` index, which is stable while the source file is byte-identical. Each output directory records a `source_fingerprint.json` (file size, mtime, SHA-256 of last 1 MB / Parquet footer) to detect silent file drift before any join is trusted.

---

## Hardware

| Resource | Spec |
|---|---|
| GPU | NVIDIA RTX PRO 5000 Blackwell |
| VRAM | 48 GB (47.27 GB usable) |
| CUDA | 13.0 (sm_120) |
| PyTorch | 2.10.0 |
| GPU ML libraries | cuML / cuDF (RAPIDS) for HDBSCAN and UMAP GPU acceleration |
| CPU decode workers | 8 parallel parquet decoders during training |
| Node | miletus.sabanciuniv.edu (no SLURM; tmux session management) |

---

## GPU-Specific Training

Training the full 95-sample cohort requires a GPU with at least 16 GB of VRAM dedicated to PyTorch.
The runner handles environment activation, GPU budgeting, and tmux session management automatically.

### Quick start (miletus)

```bash
# 1. Verify all 95 files parse and print per-sample interleave weights (cheap, reuse cache)
STATS_ONLY=1 bash Early_Stopping_Tests/scripts/tmux_train_multi.sh

# 2. Run training with the final configuration used for the cohort model
EPOCH_CEILING=40 EPOCH_SHARDS=20 PATIENCE=8 BATCH_SIZE=1048576 \
DECODE_WORKERS=8 KL_WEIGHT=0.005 INPUT_DROPOUT=0.1 HIDDEN_DROPOUT=0.1 \
bash Early_Stopping_Tests/scripts/tmux_train_multi.sh
```

### Key environment variables

| Variable | Default | Effect |
|---|---|---|
| `PARQUET_GLOB` | auto-detected | Glob for input parquet files; miletus defaults to `/data/lab/ppmseq_parquets/*.featuremap.parquet` |
| `BATCH_SIZE` | 32768 | Rows per forward pass; final cohort run used **1,048,576** |
| `DECODE_WORKERS` | 1 | Parallel CPU workers decoding parquet batches; set to **8** for the 2.4× speed gain |
| `EPOCH_SHARDS` | 20 | Splits one full pass over 5B rows into N mini-epochs; early stopping checks between shards |
| `EPOCH_CEILING` | 40 | Hard cap on epochs; cohort run stopped at epoch 30 of 38 run |
| `PATIENCE` | 8 | Epochs without val ELBO improvement (and no active-unit change) before stopping |
| `KL_WEIGHT` | 0.05 | β in β-VAE loss; final run used **0.005** to avoid posterior collapse |
| `UV_VAE_GPU_MEM_GB` | auto | Torch VRAM budget in GB; auto-detected, cohort run capped at 16 GB |
| `UV_VAE_ENABLE_CUML` | 0 | Set to 1 to use cuML/cuDF GPU acceleration for HDBSCAN and UMAP (requires RAPIDS) |
| `STATS_ONLY` | 0 | Set to 1 to run only the statistics pre-flight and stop before training |
| `SEED` | 42 | Controls model init, row shuffling, and train/val split |

### GPU decode (not recommended)

Setting `UV_VAE_GPU_DECODE=1` routes parquet decoding through cuDF on the GPU.
In practice this is **slower** than CPU decode (GPU was idle waiting for CPU in CPU mode, flipping to GPU decode
made the CPU process wait for the GPU decode instead). The 8-worker CPU decode path (`DECODE_WORKERS=8`) is 2.4× 
faster than the 1-worker baseline and was used for the final cohort run.

### AMP and VRAM budgeting

AMP (Automatic Mixed Precision) is always on: forward pass and loss run in float16, gradients in float32.
A GPU preflight check runs before training and rejects configs that exceed the VRAM budget.
To skip it (e.g. if you have verified the config manually): `SKIP_PREFLIGHT=1`.

```bash
# Run preflight only without training
python uv_vae/scripts/gpu_preflight.py \
    --batch-size 1048576 \
    --budget-gb 16 \
    --feature-spec-path uv_vae/ml_features.json
```

---

## Inference-Only Passes

Use this when you have a trained VAE checkpoint and want to encode new parquet data without retraining.
All three model files (VAE, UMAP encoder, HDBSCAN model) must be from the same pipeline run to be consistent.

### Full per-sample inference (VAE → UMAP → HDBSCAN → SigProfiler)

```bash
python umap_hdbscan_sweep/per_parquet_inference.py \
    --parquet-glob '/data/lab/ppmseq_parquets/*.parquet' \
    --checkpoint  <path-to>/run_20260802T192814Z/model.pt \
    --umap-model  <path-to>/13_BEST_25M_nn15_md0.1_umap.pt \
    --feature-spec uv_vae/ml_features.json \
    --coords      <path-to>/umap_coords_2d.npy \
    --context     <path-to>/context.parquet \
    --output-dir  results/per_parquet_inference
```

Each sample gets its own subdirectory under `--output-dir` containing cluster labels, SigProfiler assignments, 
and four plots (UMAP coloured by cluster, substitution, SigProfiler cosine, and per-sample coverage).

### VAE encode only (latent µ vectors)

Use `LatentInference.from_checkpoint` directly when you only need the 16-D latent vectors:

```python
from uv_vae.inference import LatentInference

inf = LatentInference.from_checkpoint(
    checkpoint_path="<path-to>/model.pt",
    feature_spec_path="uv_vae/ml_features.json",
    device="cuda",          # or "cpu"
)

# Encode one parquet file (streams in batches, returns concatenated mu)
mu = inf.encode_parquet(
    parquet_path="<path-to>/sample.parquet",
    row_filter="st = 'MIXED' AND et = 'MIXED' AND FILT = 1",
    batch_size=5_000_000,
)
# mu.shape == (n_rows, 16), dtype float32
```

### Checkpoint path layout

A run directory produced by any of the three trainers always contains:

```
run_YYYYMMDDTHHMMSSZ/
├── model.pt                 # VAE weights (load with LatentInference.from_checkpoint)
├── feature_report.json      # which features were active and their stats
├── preprocess_report.json   # standardisation means/stds per numeric feature
├── training_report.json     # loss curves, early-stopping diagnostics
└── summary.json             # single-line summary of the run
```

Point `--checkpoint` or `LatentInference.from_checkpoint` at `model.pt`; the loader reads the
sibling `feature_report.json` and `preprocess_report.json` automatically to reconstruct the exact
preprocessing that was applied during training.
