# Stability Metrics Ranked by Usefulness Based Upon Personal Opinion

## 1. Procrustes Distance 
Implementation Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html

Method Sources: \
https://arxiv.org/pdf/2408.01379 \
https://www.emergentmind.com/topics/procrustes-shape-distance \
https://arxiv.org/pdf/1802.03426 \
https://arxiv.org/pdf/2503.09101 

A similarity test for two different datasets after a set of geometric transformations.  Imagine you have two scatter plots of points, procrustes distance gives the difference between the two scatter plots after the best possible shifts, rotations, and transformations to get similar scatter plots.  0 is a perfect match and scores that are smaller indicate more similarity between sets of points.  This is used in machine learning stability metrics to test how different embeddings / latent spaces compare to each other.  

## 2. Trustworthiness Score
Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html 

Method Sources / Papers where idea is proposed: \
https://dl.acm.org/doi/abs/10.1016/j.neunet.2006.05.014 \
https://link.springer.com/chapter/10.1007/3-540-44668-0_68 \ 

Shows how much the local structure is kept after shifting to a latent space /  new embedding on a scale of 0 to 1.  Points in the original space have N neighbors considered.  New neighbors that are not a part of the original N neighbors in the original space are penalized.  This metric shows how much the local structure is retained.  

## 3. Linear Centered Kernel Alignment (CKA)
Paper Source: https://arxiv.org/abs/1905.00414

Summary Source: https://www.emergentmind.com/topics/centered-kernel-alignment-cka-similarity

Paper tries to come up with a new method to compare neural networks that is less sensitive to reversible linear transformations.  The traditional method of canonical correlation analysis tests how linear transformations can make neural networks similar and is sensitive to all linear transformations. Linear CKA improves upon the former by being blind to rotations, reflections, and scaling that only change the coordinate system of the neural network.  It takes into account linear transformations that change the actual geometry of the space (matrix transforms that aren't orthogonal X*X^T=I).  Metric is on a scale of 0 to 1 with values closer to 1 meaning more similarity.  


## Val Loss (Standard VAE ELBO Loss)
Constructed from reconstruction loss and KL divergence.  This is the metric that is aimed to lower in forward and backward passes of neural networks.  

## Latent collapse score (mean std across dims)
Paper Source: https://arxiv.org/pdf/1911.02469

Don’t Blame the ELBO! A Linear VAE Perspective on Posterior Collapse

Paper shows that "Posterior Collapse" sometimes occurs in VAE models where latent dimensions no longer encode information because the VAE no longer uses data inputs.  
From ChatGPT: Mean standard deviation across dims are used to show latent collapse because with values near 0 there is little to no information being encoded in the latent space. Values near zero indicate collapse.  

I honestly have no idea what this is