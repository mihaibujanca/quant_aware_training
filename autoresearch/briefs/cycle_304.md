# Item
314 (P0): E_kept_top_k curve per block — what k captures 95% of curvature residual energy?

# Context
- Cycle 302: E_kept_top_2 per block {0: 0.59, 1: 0.75, 2: 0.92, 3: 0.86, 4: 0.84, 5: 0.81, 6: 0.72, 7: 0.66, 8: 0.47, 9: 0.54, 10: 0.84, 11: 0.98}.
- Cycle 301: same but only measured 4 blocks.
- Open: what k gives E_kept = 0.95 per block?

# Dependencies
- Cycle 302 unified framework.

# Key Files
- experiments/curvature_residual_geometric.py — has panel_A_projection.
- experiments/unified_curvature_cycle302.py — driver.

# Constraints
- For each block l ∈ {0..11}, compute E_kept_top_k for k ∈ {2, 5, 10, 20, 50, 100}.
- The "A factor" is the top-k right singular vectors of c_proj_l.weight — compute dynamically from SVD of c_proj.weight.
- Find k_95 = smallest k where E_kept_top_k ≥ 0.95.
- Plot per-block curve.
- Trainable params: 0.

# Handoff
Plan agent: adapt cycle 302 framework. New script `experiments/e_kept_curve_cycle304.py`.
