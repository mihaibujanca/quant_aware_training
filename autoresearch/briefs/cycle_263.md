# Item
273 (P2): Local linearity defect ε_l(x, h) = ‖f(x+h) − f(x) − J(x)h‖_G / ‖h‖²_G.
Per token, per block, per direction. 64 tokens × 32 directions × 4 ε scales × 12 blocks × 2 activations.

# Context
- Extends cycle 260's ΔJ analysis with a SECOND-order curvature measurement.
- Predictions: ReLU defect ≈ 0 unless crossing hyperplane; GELU defect quadratic in ‖h‖.

# Dependencies
- Cycle 271 / cycle 260 J construction (we have ΔJ; we need J itself per token).
- Cycle 181 / cycle 258 patch (ReLU+b_leak).

# Key Files
- experiments/jacobian_delta_cycle260.py — has the J = c_proj.weight.T @ diag(σ'(z)) @ c_fc.weight.T pattern.
- experiments/pullback_geom_variants_cycle258.py — ReLU+b_leak patch.
- aleph/pullback_geometry.py — pullback_norm.

# Constraints
- 64 tokens from calib (200 wikitext-2 val × 128, take first 64).
- For each block l ∈ {0..11}, for σ ∈ {GELU, ReLU+b_leak}:
  - For each token x:
    - Compute J_l(x) = c_proj.weight.T @ diag(σ'(z(x))) @ c_fc.weight.T  (768, 768)
    - Compute f(x) = mlp output at this block (use HF block hooks or manual: ln_2 → c_fc → activation → c_proj)
    - Sample 32 isotropic Gaussian directions h ∼ N(0, I_768), normalize to unit length.
    - For each ε ∈ {0.01, 0.03, 0.1, 0.3} × ‖x‖ (where x is the residual stream input):
      - Scale h to magnitude ε
      - Compute f(x + h) and J(x) @ h
      - residual = f(x + h) - f(x) - J(x) @ h
      - ε_l(x, h) = pullback_norm(residual, U) / pullback_norm(h, U)^2
- Aggregate per block: median, p90 over (token, direction). Verify quadratic ‖h‖ scaling.
- Confirmed if median ε_G/ε_R ≥ 5 at small ‖h‖ AND p99 ε_R/ε_G ≥ 1 in ≥3 of blocks 0-4.
- Refuted if both functions have indistinguishable defect profiles (median and p99 ratios ∈ [0.5, 2] at all blocks).
- Inconclusive otherwise.
- Trainable params: 0.

# Handoff
Plan agent: design carefully — this is computationally heavy (64 × 32 × 4 × 12 × 2 = 196,608 forward passes if naive). Vectorize: for a fixed token, compute J once, then batch all 32×4=128 directions in one forward pass.

Note: f(x) here means the BLOCK's MLP output (not the full model). x is the residual stream input AFTER LN2.

New script `experiments/local_linearity_defect_cycle263.py`. Output JSON + summary.
