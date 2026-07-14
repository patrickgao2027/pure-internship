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
