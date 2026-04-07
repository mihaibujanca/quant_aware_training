# Item
270 (P0): Per-token trajectory divergence under depth.
For 200×128 wikitext-2 val tokens, single forward pass with GELU vs ReLU+b_leak (cycle 181 patch). At each layer l ∈ {0..12}, compute d_l(x) = ‖x_G^l − x_R^l‖_G per token. Report median/p10/p90 vs l, fit growth law, and median per-token cos(d_l, d_{l+1}).

# Context
- Cycle 189 measured cos(e_l, e_{l-1}) = 0.72 AGGREGATED over tokens. This is the per-token version.
- Cycle 258 (just done): pullback κ comparison shows ReLU+b_leak kinks κ in layers 2-6 by 14-48%.
- aleph/pullback_geometry.py provides pullback_norm.

# Dependencies
None directly. Cycle 258's b_leak precomputation can be reused.

# Key Files
- aleph/pullback_geometry.py — collect_residual_trajectory, pullback_norm
- experiments/pullback_geom_variants_cycle258.py — has the ReLU+b_leak patching code

# Constraints
- Same calib slice (200 wikitext-2 val × 128 tokens, seed 42).
- For each variant {GELU, ReLU+b_leak}, single forward pass capturing residual stream at all 13 trajectory points.
- Compute d_l(x) = pullback_norm(x_G^l − x_R^l, U) per token at each layer.
- Report:
  (i) median/p10/p90 of d_l vs l (13 layer points)
  (ii) fit log-linear, linear, saturating curves to median d_l; report best-fit type and growth constant
  (iii) per-layer median cos(d_l, d_{l+1}) over the (B*T) token axis (per-token analogue of cycle 189's aggregate cosine)
- Confirmed if log-fit growth ≥ 0.20/layer in l ∈ [0, 6] AND median per-token cos(d_l, d_{l+1}) ≥ 0.6 in ≥4 of those layers.
- Refuted if growth sub-linear OR median per-token cos ≤ 0.2 in ≥3 early layers.
- Inconclusive between.

# Handoff
Plan agent: design via reusing cycle 258's variant setup. New script `experiments/per_token_divergence_cycle259.py`. Output JSON + summary + plot.
