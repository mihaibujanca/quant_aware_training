# Item
19 — Jacobian variance atlas: which FFN blocks are near-linear on probe data?

# Context
- Existing `_mean_jacobian_spectrum` computes J(x) = W_out @ diag(σ'(z)) @ W_in per token
- Need: linearity index L = 1 − mean_x(||J(x) − J̄||_F / ||J̄||_F) per block
- Can compute J(x) − J̄ efficiently: δ = d(x) − d̄, then J_diff = (W_out * δ) @ W_in
- Cycle 005 has per-layer damage for correlation: 16x ppl values for 12 blocks
- Note: agenda says "24 GPT-2 blocks" but gpt2 only has 12 FFN blocks

# Dependencies
- Item 9 (cycle 004): DONE

# Key Files
- `aleph/lm_geometric_compression.py` — probe_ffn_geometry, _mean_jacobian_spectrum, FfnLayerGeometry
- Cycle 005 data: layer damage rankings

# Constraints
- 768×768 Jacobian per token. Use ≤256 tokens/block for memory.
- Proxy-only (gpt2+Shakespeare)

# Handoff
Write `experiments/jacobian_variance_atlas.py`. Compute L per block, correlate with cycle 005 damage.
