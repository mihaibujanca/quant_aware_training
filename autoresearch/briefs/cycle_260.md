# Item
271 (P1): Jacobian delta ΔJ_l(x) = J_GELU − J_ReLU per token, per block.
For 256 calib tokens × 12 blocks: J_l(x) = W_down @ diag(σ'(z(x))) @ W_up for σ ∈ {GELU, ReLU}. Save ΔJ. Per token: SVD, rank_95. Per block: stack along token, mode-1 SVD, alignment with cycle 218 c_fc reading basis of next block.

# Context
- Cycle 217: rank_95(Δ̃) ≤ 64 in 9/12 blocks (zeroth-order).
- Cycle 218: cycle 218 c_fc reading basis. top-5 cos sum 2.97 vs random 1.27.
- Cycle 195: J_GELU rank_95 ~472/768.
- This is the OPERATOR-LEVEL difference: ΔJ = W_down @ diag(σ'_GELU − σ'_ReLU) @ W_up. Note: σ'_ReLU is 0 or 1 (Heaviside), σ'_GELU is smooth. The diagonal difference is concentrated where z is small (around the GELU "knee").

# Dependencies
None.

# Key Files
- experiments/pullback_geom_variants_cycle258.py — for variant patches if needed.
- aleph/pullback_geometry.py — for U.
- experiments/delta_centred_alignment_cycle218.py — for c_fc reading basis pattern.

# Constraints
- Sample 256 tokens from calib (200 wikitext-2 val × 128 SEQ_LEN, seed 42).
- For each block l ∈ {0..11}:
  - For each token x in 256:
    - Capture pre-GELU activation z(x) via hook on c_fc input → preact = ln_out @ c_fc.weight + c_fc.bias (shape (3072,))
    - σ'_GELU(z) = derivative of GELU at z. For tanh-approx: GELU(z) = 0.5*z*(1 + tanh(sqrt(2/π)*(z + 0.044715*z^3))). Derivative is messy. Use torch.autograd or direct formula.
    - σ'_ReLU(z) = (z > 0).float()
    - delta_diag = σ'_GELU(z) - σ'_ReLU(z)  shape (3072,)
    - J_l(x) is implicitly W_down.T @ diag(delta_diag) @ W_up = sum over c of delta_diag[c] * W_down[:,c] * W_up[c,:]^T (outer product). Each token gives a (768, 768) ΔJ.
    - More efficiently: ΔJ = (W_down * delta_diag.unsqueeze(0)) @ W_up.T, where W_down is (768, 3072) and W_up is (3072, 768). Wait — HF Conv1D layout: c_fc.weight is (768, 3072), c_proj.weight is (3072, 768). So W_up = c_fc.weight (768, 3072) and W_down = c_proj.weight (3072, 768).
    - Δactivation' = σ'_GELU(z) - σ'_ReLU(z) shape (3072,). For each token, ΔJ = c_proj.weight.T @ diag(Δactivation') @ c_fc.weight.T. That's (768, 3072) @ (3072, 3072 diag) @ (3072, 768) = (768, 768).
    - Vectorized: For each token, diag operation = c_proj.weight.T * Δactivation'.unsqueeze(0). Shape (768, 3072). Then @ c_fc.weight.T (3072, 768) = (768, 768).
- Per token: SVD ΔJ → rank_95 (smallest k where cumulative SV² ≥ 0.95). Top-3 left SVs.
- Per block aggregate: stack ΔJ across 256 tokens → tensor (256, 768, 768). Mode-1 SVD: reshape to (256, 768*768) → SVD → top right SVs are the "stable directions across tokens". Or: average ΔJ over tokens, then SVD that. Simpler.
- Compute alignment of stable subspace with c_fc reading basis of block l+1 (top-20 left SVs of c_fc(l+1)). Top-5 principal cosine sum. Compare to cycle 218's random baseline 1.27.
- Confirmed if median rank_95 ≤ 100 in ≥6 blocks AND mode-1 alignment top-5 cos sum ≥ 2.0 in ≥6 blocks.
- Refuted if median rank_95 ≥ 400 in ≥6 blocks.
- Inconclusive otherwise.
- Trainable params: 0.

# Handoff
Plan agent: design carefully — the per-token SVD over (768, 768) matrices for 256 tokens × 12 blocks could be slow. Consider whether to do SVD via torch.linalg.svdvals (only singular values, no U/V) for the rank_95 computation, then full SVD only for top-3 left SVs.

New script `experiments/jacobian_delta_cycle260.py`. Output JSON + summary.
