# Item
310 (P1): What are the 2 directions? — semantic decomposition of cycle 280's frozen A factor.

# Context
- Cycle 280: block 1 rank-2 retrain. A = U[:, :2] @ diag(S[:2]) frozen (3072, 2). B (2, 768) trainable.
- Cycle 299: SVD init is 83% of the optimal basin; Adam fine-tunes small rotation.
- Open: what do A's 2 columns represent at the FFN-intermediate side and B's 2 rows at the residual-stream side?

# Dependencies
- Cycle 280/299 trained models.

# Key Files
- experiments/block1_lowrank_cycle280.py
- experiments/cycle280_decomp_cycle299.py
- aleph/pullback_geometry.py

# Constraints
- Reuse cycle 299's trained B or retrain.
- For each of 2 directions i ∈ {0, 1}:
  - Input (A[:, i]): top-20 channels by |A[c, i]|. For those: σ_c (cycle 261 pattern), ||W_fc[:, c]||, pre-activation stats.
  - Output (B[i, :]): cosine with wte rows → top-20 boost/supp tokens. Cosine with block 2 Q/K/V (cycle 218 infra). Vocab projection.
  - Compare retrained B to SVD init Vh[:2].
- Verdict per item 310.

# Handoff
Plan agent: new script `experiments/cycle280_A_semantic_cycle300.py`. Reuse cycle 299 infrastructure for loading.
