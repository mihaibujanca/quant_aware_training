# Task
Item 95: Data-conditioned Jacobian clustering for merge candidates — GPT-2 FFN. Block 7 only. Compute per-token GELU'(z) diagonal patterns (3072-dim) on 1024 mixed-corpus tokens. K-means with k in {8,16,32,64}. Reconstruct block output using centroid Jacobians. Find K* for <5% output norm error. Check centroid Jacobian low-rank structure.

# Hypothesis
- Confirmed if: K* ≤ 32 centroids achieve <5% output norm reconstruction error on held-out tokens.
- Refuted if: K* > 128 needed for <5% error.
- Inconclusive if: K* in 32–128 range.

# Pre-flight Results
- P1: Not duplicate. Item 23 (OBSOLETE) used full Jacobian k-means; item 95 clusters GELU' diagonal patterns. Item 80 (cycle 068) computed rank census but no clustering. Item 94 (cycle 083) did near-linearity census — different question.
- P2: Depends on item 97 (GELU activation pattern diversity). Item 97 is PENDING (P0, not yet run). However, item 95 is self-contained: it computes its own k-means clustering and K* search. Item 97 would provide prior knowledge of cluster structure, but is not a hard blocker. Proceeding without it. Item 80 (cycle 068) confirmed rank_99 = 434–510 per block (genuinely low-rank), which motivates checking whether clusters reduce rank further.
- P3: Concrete — single script, sweep over k, clear error threshold, SVD analysis.
- P4: No program.md changes needed.

# Steps
Script: `experiments/jacobian_clustering_gpt2.py`

1. **Setup.** Load GPT-2, mixed-corpus probe: 256 tokens each from Shakespeare, WikiText-2, OWT, C4 = 1024 train. Holdout: 512 tokens (128 each). seq_len=128. Block 7 only. Extract W_in (3072×768), W_out (768×3072), biases. Use `gelu_grad()` from `ffn_jacobian_rank_census.py` — verify it uses `approximate='tanh'` for GPT-2 compatibility (GPT-2 uses tanh-approximate GELU).

2. **Collect GELU' patterns and pre-activations.** For each token x (768-dim, FFN input from residual stream after LN): z = W_in @ x + b_fc, d = GELU'(z) ∈ [0,1]^3072. Store D matrix (1024×3072) for train, (512×3072) for holdout. Also store Z matrices (pre-activations z, same shapes) — needed for per-cluster centroid z_c in the affine linearization. Store FFN outputs y = W_out @ GELU(z) + b_proj for ground truth.

3. **K-means sweep.** For k in {8, 16, 32, 64, 128}: sklearn KMeans on D_train (1024×3072), n_init=10, max_iter=300. Record cluster sizes, inertia. Assign holdout tokens to nearest centroid. Also compute per-cluster centroid pre-activation z_c = mean of z vectors in each train cluster.

4. **Centroid affine reconstruction.** GELU is not homogeneous, so the correct linearization around centroid pre-activation z_c is: GELU(z) ≈ GELU(z_c) + diag(d_c) @ (z - z_c). For each cluster c with centroid derivative d_c and centroid pre-activation z_c: J_c = W_out @ diag(d_c) @ W_in (768×768). For each holdout token x assigned to c with z = W_in @ x + b_fc: y_hat = W_out @ [GELU(z_c) + d_c * (z - z_c)] + b_proj = J_c @ x + W_out @ [GELU(z_c) - d_c * z_c + d_c * b_fc] + b_proj. The per-centroid constant offset W_out @ [GELU(z_c) - d_c * z_c + d_c * b_fc] + b_proj is precomputed once per cluster. Error: ||y - y_hat|| / ||y||. Report mean/p95 relative error per k.

5. **Find K*.** K* = min k such that mean relative reconstruction error < 5% on holdout. If k=128 still > 5%, try k=256 (fallback).

6. **SVD of centroid Jacobians.** For K* centroids, compute SVD of each J_c. Report rank_95, rank_99. Compare to unclustered per-token rank (434–510 from cycle 068). If centroid Jacobians have lower rank, clusters expose additional low-rank structure.

7. **Save results.** JSON: `autoresearch/results/cycle_084_jacobian_clustering.json` — per-k: mean_error, p95_error, cluster_sizes, inertia, centroid_ranks. Plot: `autoresearch/results/cycle_084_jacobian_clustering.png` — (a) reconstruction error vs k, (b) centroid SVD spectra for K*.

8. **Bookkeeping.** Move item 95 from NEXT to PAST EXPERIMENTS in `autoresearch/agenda.md` with outcome, commit hash, artifacts. Commit: `[autoresearch] cycle 084: Jacobian clustering for merge candidates — GPT-2`.

# Validation
- GELU' values must be in [0, 1]. If any < -0.01 or > 1.01, implementation bug.
- Ground truth FFN outputs must match `model.transformer.h[7].mlp(x)` within float32 tolerance (~1e-5). Verify on 10 tokens before proceeding.
- Cluster sizes should not be degenerate (no cluster with <1% of tokens at k=8). If so, increase n_init.
- k=1 error provides null baseline (single global Jacobian). Must be > 5% for the experiment to be informative.

# Artifacts
- `experiments/jacobian_clustering_gpt2.py`
- `autoresearch/results/cycle_084_jacobian_clustering.json`
- `autoresearch/results/cycle_084_jacobian_clustering.png`

# Fallbacks
- K* > 128 (refuted) -> still report SVD of centroid Jacobians at k=64; this informs whether clusters have any low-rank benefit even if too many are needed.
- KMeans slow at 3072-dim -> use MiniBatchKMeans with batch_size=256.
- Mixed-corpus data loading fails for any source -> fall back to Shakespeare-only 1024 tokens (weaker coverage but still valid).
