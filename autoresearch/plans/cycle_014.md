# Task
RTG spectral coarsening on GPT-2 FFN blocks using k-NN Jacobian graph.

# Hypothesis
From agenda item 25:
- **Confirmed if**: spectral coarsening reduces reconstruction error by ≥ 15% relative to k-means at matched K for a majority of blocks.
- **Refuted if**: spectral coarsening performs at parity with or worse than k-means on raw Jacobians.
- **Inconclusive if**: the benefit appears only at small K (K ≤ 4) and vanishes as K grows.

# Pre-flight Results
- P1: Not a duplicate. Past experiments include k-means on toy (cycle 012) and RTG spectral on toy (cycle 013). No GPT-2 spectral experiment exists.
- P2: Depends on items 9 (confirmed, cycle 004) and 24 (inconclusive, cycle 013 — not refuted). No blockers.
- P3: Concrete: compute Jacobians, build graph, cluster, measure reconstruction error. One cycle.
- P4: No prompt/program modifications needed.

# Scope Reduction
Full 768×768 Jacobians for 256 tokens × 12 blocks exceeds 30-min cap. Reduce to:
- 4 blocks: 0 (hard), 2 (easy), 5 (easy), 11 (hard) — spanning cycle 005 damage range.
- 128 tokens per block.
- K ∈ {2, 4, 8, 16} (matching brief).

# Steps
1. **Write** `experiments/rtg_spectral_gpt2.py`:
   - Load gpt2 + Shakespeare probe (reuse `lm_geometric_compression` helpers).
   - For each of 4 selected blocks, collect 128-token preactivations.
   - Compute per-token Jacobian J(x) = W_out @ diag(σ'(z(x))) @ W_in via factored form (768×768).
   - Flatten Jacobians to vectors (589,824-D). Build k-NN graph (k=10) using pairwise L2 on flattened Jacobians (torch.cdist).
   - Build graph Laplacian L = D − A. Compute K smallest eigenvectors (scipy sparse eigsh). Cluster via KMeans on eigenvector embedding.
   - Also run k-means directly on flattened Jacobians (baseline).
   - For each cluster: fit per-cluster affine A_c = mean J(x) over cluster members. Reconstruction error = mean_x ||J(x) − A_{c(x)}||_F / ||J(x)||_F.
   - Save per-block, per-K, per-method results to JSON.

2. **Run**: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/rtg_spectral_gpt2.py`

3. **Evaluate**: spectral vs k-means error at each (block, K). Majority of blocks with ≥15% relative improvement = confirmed.

4. **Move item 25** to PAST, commit, write manifest.

# Validation
- Reconstruction error should decrease monotonically with K for both methods.
- k-NN graph should be connected (check n_components) — if fragmented, report as cycle 013 repeat.
- At K=128 (=N), error should be ~0 for both methods (sanity check, don't run — just verify logic).

# Fallbacks
- If 128 tokens × 768×768 OOMs: use random projection to 128-D before distance computation.
- If scipy eigsh fails on dense Laplacian: use numpy.linalg.eigh on dense matrix (128×128 is small).
- If >30 min: drop to 2 blocks (0 and 5).

# Artifacts
- `experiments/rtg_spectral_gpt2.py`
- `autoresearch/results/cycle_014_rtg_spectral_gpt2.json`
- `autoresearch/results/cycle_014_summary.md`
