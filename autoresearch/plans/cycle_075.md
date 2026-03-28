# Task
Item 35: K-means on per-token FFN Jacobians as proxy for "linear regions" in GPT-2. Cumulative reconstruction error as clusters added in descending size order. Test for Pareto structure and correlation with compression difficulty.

# Hypothesis
- **Confirmed if**: top 20% covers >=70% output variance in majority of blocks AND Gini correlates with (1-compression difficulty) at Spearman >= 0.35.
- **Refuted if**: cluster size distribution uniform (Gini < 0.2).
- **Inconclusive if**: Pareto holds in toy but LLM distribution is flat.

# Pre-flight Results
- P1 (data): shakespeare.txt probe corpus, GPT-2 pretrained. Both available from prior cycles.
- P2 (deps): `_gelu_grad_autograd`, `_standard_weight` in `experiments/ffn_jacobian_svd_compression.py`. sklearn KMeans available.
- P3 (ground truth): `cycle_048_local_complexity_gpt2.json` has damage_8x/16x for 12 blocks.
- P4 (compute): 12 blocks x ~500 tokens x 768x768 Jacobian each. Per-token SVD profile extraction ~5min. K-means on 768D profiles is fast.

# Steps
Script: `experiments/region_volume_pareto_gpt2.py`

1. **Load GPT-2 + probe corpus.** 3 seeds [42, 123, 7]. Per seed: tokenize shakespeare.txt, take 4 non-overlapping 128-token windows (512 tokens). Hook `h[i].mlp.c_fc` output to capture pre-GELU activations z.

2. **Per-token Jacobian singular value profiles.** For each block, token: compute J(x) = W_out @ diag(GELU'(z_t)) @ W_in (768x768). Extract singular values via `torch.linalg.svdvals` (768D vector). Normalize to unit L2. This is the "region fingerprint" — captures the local linear map's spectral shape without 589K dimensions.

3. **K-means clustering.** Per block, per seed: K-means on normalized SV profiles (N_tokens x 768D) with K=50 clusters (matching ~50 expected effective behaviors from toy). Assign each token to its cluster.

4. **Volume distribution + Pareto analysis.** Per block: cluster volumes = fraction of tokens per cluster. Sort descending. Compute Gini coefficient. Cumulative reconstruction error in SVD-profile space: total_SS = sum_t ||p_t - p_global_mean||^2. For each cluster c, within_SS_c = sum_{t in c} ||p_t - p_c_mean||^2. Add clusters in descending volume order; cumulative coverage = 1 - (remaining within_SS / total_SS). Extract top-20% coverage. Note: this measures how much of the profile variance the largest clusters explain, which is the SVD-profile analog of "output variance."

5. **Null baseline.** Per block, per seed: assign tokens uniformly to K=50 clusters at random. Same Gini + Pareto analysis. This gives expected Gini and coverage under no structure.

6. **Correlation with compression difficulty.** Load damage_8x from cycle 048. Per block: compute Gini. Spearman(Gini, 1 - damage_8x) across 12 blocks. Also Spearman with damage_16x.

7. **Save.** JSON to `autoresearch/results/cycle_075_pareto_gpt2.json`. Plot (2x2): (a) Gini per block vs null, (b) cumulative variance curve for 3 representative blocks, (c) Lorenz curves, (d) Gini vs damage scatter. Save PNG.

Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/region_volume_pareto_gpt2.py`

# Validation
- K-means converges for all blocks (inertia finite).
- Gini > null baseline Gini for majority of blocks (else no structure detected).
- Cross-seed std of Gini < 50% of mean Gini per block.
- Spearman p-value reported alongside rho.

# Artifacts
- `experiments/region_volume_pareto_gpt2.py`
- `autoresearch/results/cycle_075_pareto_gpt2.json`
- `autoresearch/results/cycle_075_pareto_gpt2.png`

# Fallbacks
- If SVD profiles cluster trivially (all nearly identical): use top-50 singular values only (reduce to 50D).
- If K=50 too many (empty clusters): sweep K in [10, 20, 50, 100], report best silhouette K.
- If Jacobian computation exceeds memory: process tokens in mini-batches of 64.
