# Item
311 (P2): Curvature residual decomposition relative to cycle 280's rank-2 column space.
Does (GELU, ReLU) curvature Δy live in the kept or discarded subspace of cycle 280's A?

# Context
- Cycle 280: block 1 rank-2 A = U[:, :2] @ diag(S[:2]) frozen.
- Cycle 258: ReLU swap shifts κ +47.7% at layer 4.
- Cycle 217: Δ̃ rank_95 ≤ 64 in 9/12 blocks.
- Open: what fraction of Δy energy at block 1 lives in col(A)?

# Dependencies
- Cycle 280 saved A factor.

# Key Files
- experiments/cycle280_A_semantic_cycle300.py — has A factor extraction.
- experiments/pullback_geom_variants_cycle258.py — has Δy infrastructure.

# Constraints
- Calib: 200 wikitext-2 val × 128 tokens, seed 42.
- For block 1: compute z(x) = c_fc input, δ_σ(z) = GELU(z) - ReLU(z), Δy = δ_σ @ W_proj.T.
- Load cycle 280 frozen A (U[:, :2] @ diag(S[:2]) of c_proj_1.weight SVD). Note: A operates in R^(3072,2), mapping hidden-3072 → 2-dim bottleneck. But Δy is in R^768 residual-stream space.
- WAIT: the agenda says "col(A) is the kept 2D output subspace" but A ∈ R^(3072, 2) maps intermediate→2D. The kept OUTPUT subspace in R^768 is span(B) where B is the trained (2, 768) factor. Reread agenda carefully.
- Per agenda: "the kept 2D output subspace" — so it's the 2D subspace in R^768 that the rank-2 operator writes into. That's span of top-2 LEFT singular vectors of W (when W = c_proj), not A.
- Actually A_rank2 = U[:, :2] @ Σ[:2] @ V[:, :2]^T. The output subspace is col(A_rank2) = col(U[:, :2]) in R^3072. Wait c_proj.weight is (3072, 768): U is (3072, 3072), Vh is (768, 768). So c_proj.weight maps 3072 → 768; its column space is the output in R^3072. Hmm.
- Let me rethink. c_proj in HF Conv1D layout: output = input @ weight + bias. weight is (in=3072, out=768). Output is in R^768.
- The rank-r approximation is weight_r = U_r @ diag(S_r) @ Vh_r where U_r is (3072, r), Vh_r is (r, 768). Vh_r^T = (768, r) is the "output" subspace in R^768.
- So col(A) in the agenda's framing corresponds to the top-r right singular vectors of c_proj.weight = the top-r rows of Vh = span(Vh_r^T).
- This IS in R^768 (the residual stream output space). Δy(x) lives in R^768. Good.
- Project Δy onto Vh_r^T (2-dim subspace). E_kept = ||P @ Δy||_F^2 / ||Δy||_F^2.
- Also compute for blocks 0, 2, 11.
- Random null: 100 random 2D subspaces.
- Verdict: CONFIRMED if E_kept ≤ 0.05 at block 1. REFUTED if ≥ 0.30.

# Handoff
Plan agent: design carefully per the SVD conventions. New script `experiments/curvature_residual_A_proj_cycle301.py`.
