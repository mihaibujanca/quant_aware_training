# Item
277 (P0): Block 11 c_proj output rank decomposition.
For blocks {1, 5, 11}: collect mlp_out activation on calib, SVD covariance, eff_rank, top-K energy. SVD of c_proj.weight, top-20 directions. Cosine of top SVs with lm_head rows.

# Context
- Cycle 173: block 11 dominates single-block PPL (+4.02%).
- Cycle 264: block 11 is 11/12 in κ disruption.
- Cycle 263: block 11 has ~10× higher local linearity defect than mid blocks.
- Hypothesis: block 11 PPL load is from output-channel concentration OR alignment with lm_head.

# Dependencies
None.

# Key Files
- experiments/block5_geometric_cycle252.py — analysis harness.
- HF gpt2 weights.

# Constraints
- Calib: 200 wikitext-2 val × 128, seed 42.
- For block ∈ {1, 5, 11}:
  - Forward pass with hook on mlp.c_proj OUTPUT (post c_proj, the residual contribution).
  - mlp_out shape (25600, 768).
  - cov = mlp_out.T @ mlp_out / N
  - SVD of cov → eigenvalues; eff_rank = exp(entropy(normalized eigenvalues))
  - Top-K energy fractions: k ∈ {1, 5, 20, 50, 100, 200}
  - Top-20 right SVs of mlp_out (or eigenvectors of cov)
- For each block: SVD of c_proj.weight (3072, 768) → top-20 right SVs (in R^768)
- Cosines between top-20 SVs of mlp_out and rows of lm_head.weight (50257, 768).
- For each top-5 SV of mlp_out, find max cos with any lm_head row → median across the 5.
- Confirmed if eff_rank ≤ 50 OR median top-5 cosine with top-200 lm_head rows ≥ 0.5.
- Refuted if eff_rank ≥ 200 AND median cosine ≤ 0.2.
- Inconclusive otherwise.
- Trainable params: 0.

# Handoff
Plan agent: design via reusing cycle 252's hook pattern. New script `experiments/block11_cproj_decomp_cycle267.py`. Output JSON + summary.
