# Item
316 (P0): Block 8 deep-dive — why is E_kept lowest at block 8?

# Context
- Cycle 302: E_kept_top_2 per block, minimum at block 8 = 0.47 (all others 0.54-0.98).
- Cycle 278: block 8 eff_rank = 291.41 (high-rank diffuser).
- Cycle 304: k_95 at block 8 = 511.
- Open: what's special about block 8?

# Dependencies
- Cycle 302/304 framework.

# Key Files
- experiments/curvature_residual_geometric.py.
- experiments/all_blocks_eff_rank_cycle278.py.

# Constraints
- Analysis only, no training.
- At block 8: compute Δy (GELU-ReLU curvature residual).
- Measure: (a) full SV spectrum of Δy, (b) per-token ||Δy||, (c) singular value concentration (top-1 energy fraction), (d) cosine between Δy's top right singular vectors and W_proj's top right singular vectors.
- Compare with neighbors block 7, block 9.
- Trainable params: 0.

# Handoff
Plan agent: design via extending cycle 302 framework. New script `experiments/block8_deep_dive_cycle306.py`.
