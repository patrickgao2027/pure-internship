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




