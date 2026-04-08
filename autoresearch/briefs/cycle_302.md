# Item
312 (P2): Unified geometric curvature residual framework.
Implement a single script subsuming cycles 217/218/235/258/311 as parameterized panels.

# Context
- Cycle 217: per-token Δ SVD (rank_95).
- Cycle 218: centred Δ̃ alignment with c_fc/Q/K/V reading subspaces.
- Cycle 235: T1 latent feature extraction (top-k direction vocab decoding).
- Cycle 258: pullback κ under ReLU+b_leak.
- Cycle 301: (GELU, ReLU) curvature lives IN cycle 280's rank-2 kept subspace.
- Open: unify these as a single parameterized framework.

# Dependencies
- All referenced cycle scripts.

# Key Files
- experiments/delta_per_token_cycle217.py
- experiments/delta_centred_alignment_cycle218.py
- experiments/bump_t1_features_cycle235.py
- experiments/pullback_geom_variants_cycle258.py
- experiments/curvature_residual_A_proj_cycle301.py

# Constraints
- Build experiments/curvature_residual_geometric.py with 6 panel computations (217, 218, 235, 258, 311, plus metadata).
- Takes (model, σ, σ_lin, blocks, calib, A_factor_optional).
- Validation: run with (GELU, ReLU, GPT-2, all blocks), verify reproduction within margin.
- Confirmed if all 3 reproductions within published margins.
- Refuted if any >2× off.

# Handoff
Plan agent: design the unified framework. May need to simplify (e.g., skip cycle 258 panel if κ is expensive to compute). Prioritize rank + alignment + A-projection panels.
