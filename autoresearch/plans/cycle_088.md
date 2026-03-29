# Task
Item 102: Input-space token clustering + local Jacobian — GPT-2 FFN. For blocks [7, 5, 4, 1], cluster FFN inputs x (768D) with k-means, compute local Jacobian J_c = W_out @ diag(GELU'(W_in @ x_c + b_in)) @ W_in at each cluster centroid, reconstruct y_approx = J_c @ (x - x_c) + FFN(x_c). Sweep K in {4, 8, 16, 32, 64}. Find K* per block for <5% reconstruction error. Measure effective rank of each J_c.

# Hypothesis
- **Confirmed if**: K* <= 64 achieves < 5% error for >= 3/4 blocks.
- **Refuted if**: K* > 128 still > 10% error for majority (input space also too continuous).
- **Inconclusive if**: works for some blocks but not others.

# Pre-flight Results
- P1: Not duplicate. Cycle 085 (item 97) clustered GELU' patterns in *output* space (3072D) and found K* > 256 for all 12 blocks. This clusters in *input* space (768D) and uses local Jacobian at the centroid, not average GELU' vectors. Different space, different linearization.
- P2: No dependencies. Item says "Depends on: none".
- P3: Concrete — single script, 4 blocks, k-means sweep with 5 k-values, deterministic Jacobian computation per centroid, clear metric (relative reconstruction error).
- P4: No program.md changes needed.
- P5 (null baseline): For iid Gaussian inputs in 768D, k-means centroids are roughly uniformly distributed on a shell. GELU'(W_in @ x_c + b_in) varies smoothly with x_c, so the Jacobian changes slowly across nearby centroids. First-order Taylor remainder scales as ||x - x_c||^2 × curvature. With K clusters, mean ||x - x_c|| ~ sigma / K^{1/768} ≈ sigma (curse of dimensionality — k-means barely helps in raw 768D). But real LM activations live on a low-dimensional manifold (cycle 086: steep eigenspectrum), so effective dimension is much lower and K clusters can partition the manifold meaningfully. Report k=1 (global linearization) error as null baseline — must exceed 5% for experiment to be informative.
- P6 (methods): Jacobian J_c = W_out @ diag(GELU'(W_in @ x_c + b_in)) @ W_in is 768×768, computed analytically from weights (no torch.func.jacrev needed — just evaluate GELU' at the centroid and multiply). This is consistent affine linearization: GELU' evaluated at the actual centroid input z_c = W_in @ x_c + b_in, not averaged over cluster members. GPT-2 uses tanh-approx GELU. Effective rank of J_c via SVD at 95%/99% energy thresholds.
- P7 (power): 4096 mixed-corpus tokens: 3072 train / 1024 holdout. At K=64, 3072/64 = 48 tokens per cluster — adequate for k-means. At K=4, 3072/4 = 768 — large clusters, high reconstruction error expected. Holdout split prevents overfitting cluster assignments.

# Steps
Script: `experiments/input_space_jacobian_clustering_gpt2.py`. Adapt infrastructure from `experiments/gelu_pattern_diversity.py` (corpus setup) and `experiments/ffn_jacobian_rank_census.py` (weight extraction, GELU').

1. **Setup.** Load GPT-2 (float32, CPU). Build 4096-token mixed-corpus probe: 1024 tokens each from shakespeare, wikitext2, OWT, C4 (8 sequences × 128 each per source). Split: first 3072 = train, last 1024 = holdout. Blocks = [7, 5, 4, 1].

2. **Collect FFN inputs and outputs.** For each block, hook `h[b].ln_2` output (FFN input after LayerNorm) to get x (n, 768). Hook `h[b].mlp` output for ground-truth y (n, 768). Forward in batches of 4. Flatten sequence dim. Extract W_in = c_fc.weight.T (3072, 768), b_in = c_fc.bias (3072), W_out = c_proj.weight.T (768, 3072), b_out = c_proj.bias (768).

3. **K-means sweep.** For each block, for K in {4, 8, 16, 32, 64}: sklearn MiniBatchKMeans on x_train (3072, 768). Record centroids x_c (K, 768). Assign holdout tokens to nearest centroid. Report cluster sizes, flag degenerate clusters (<5 members).

4. **Local Jacobian computation.** For each centroid x_c: z_c = W_in @ x_c + b_in (3072,). d_c = gelu_grad(z_c) (3072,). J_c = W_out @ diag(d_c) @ W_in (768, 768). Also compute f(x_c) = W_out @ GELU(z_c) + b_out (768,) as the expansion point.

5. **Reconstruction.** For each holdout token x in cluster c: y_approx = J_c @ (x - x_c) + f(x_c). Error = ||y_approx - y_true|| / ||y_true||. Report mean and p95 error per K per block.

6. **Find K*.** Per block: K* = min K with mean relative error < 5%. If K=64 still > 5%, report K* = >64. Also report K=1 (global linearization at dataset mean) as null baseline.

7. **Effective rank of J_c.** For each centroid at K=16 (or K*), SVD of J_c. Report rank at 95% and 99% energy thresholds. Are local Jacobians low-rank? If so, item 103 (low-rank factorization) becomes more promising.

8. **Save results.** JSON: `autoresearch/results/cycle_088_input_jacobian_clustering.json`. Plot: `autoresearch/results/cycle_088_input_jacobian_clustering.png` — (a) reconstruction error vs K curves for 4 blocks, (b) K* bar chart, (c) effective rank histogram across centroids.

9. **Bookkeeping.** Update agenda item 102 with outcome, cycle tag, artifact paths. Commit: `[autoresearch] cycle 088: input-space token clustering + local Jacobian — GPT-2`.

# Validation
- GELU'(0) = 0.5, GELU'(3) ≈ 1.0, GELU'(-3) ≈ -0.012. Assert before proceeding.
- K=1 error must be > 5% (else FFN is already globally linear and experiment is trivial).
- Reconstruction at K=64 must be strictly better than K=4 (monotonic improvement with K).
- J_c @ zero_vector = 0 (linearity check: reconstructing centroid itself should give f(x_c) exactly).
- Spot-check: for one centroid, compare J_c @ (x - x_c) + f(x_c) against torch.func.jacrev result to verify analytic Jacobian matches autograd.

# Artifacts
- `experiments/input_space_jacobian_clustering_gpt2.py`
- `autoresearch/results/cycle_088_input_jacobian_clustering.json`
- `autoresearch/results/cycle_088_input_jacobian_clustering.png`
- Agenda update: item 102 outcome + cycle tag.
- Commit: `[autoresearch] cycle 088: input-space token clustering + local Jacobian — GPT-2`

# Fallbacks
- MiniBatchKMeans slow at K=64 × 4 blocks → already using MiniBatchKMeans, should be fast.
- OOM on 4096 tokens × 4 blocks → process blocks sequentially, free tensors after each.
- K=64 still > 5% → extend sweep to K=128. If K=128 still > 10%, verdict is REFUTED.
- Any data source missing → fall back to 3 sources × 1365 tokens (trim to 4096).
