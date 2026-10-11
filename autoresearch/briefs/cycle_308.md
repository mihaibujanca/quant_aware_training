# Item
318 (P2): Synthetic rank-2 unigram projector for block 1 c_proj.
Hand-designed replacement using unigram log-frequencies + stopword indicator projected through W_E.

# Context
- Cycle 300 CONFIRMED_unigram: both rank-2 directions are unigram axes.
- Cycle 280: block 1 rank-2 retrain → 49.03.
- Cycle 301: E_kept_top_2 at block 1 = 0.75 (curvature residual lives in the kept subspace).
- Hypothesis: a hand-designed rank-2 c_proj from unigram statistics can reach PPL ≤ 60 with zero training.

# Dependencies
- Cycle 280/300 trained A factor.

# Key Files
- experiments/cycle280_A_semantic_cycle300.py — has A factor extraction.
- HF gpt2 for W_E (wte.weight).

# Constraints
- Compute unigram histogram p(t) on calib (200 wikitext-2 seqs × 128 = 25,600 tokens).
- log_p ∈ R^50257.
- d₁ = wte.weight.T @ log_p ∈ R^768, normalized.
- d₂ = wte.weight.T @ stopword_indicator, Gram-Schmidt against d₁.
- A_synth = [d₁ | d₂] ∈ R^(768, 2) — wait, that's rows not columns. In c_proj conventions, c_proj.weight is (3072, 768). The rank-r factorization is (3072, r) @ (r, 768).
- For the synthetic projector, we need A = (3072, 2) and B = (2, 768). The brief suggests using unigram directions in R^768 as the OUTPUT of the rank-2 operator. So the 768-dim directions d₁, d₂ should be the ROWS of B (the right factor) — they define the OUTPUT subspace.
- A factor: can be a fixed (3072, 2) matrix — perhaps random or from the real cycle 280 A. The agenda says "construct A = [d₁ | d₂] ∈ R^(768×2)" but that's for the B factor in R^768.
- Simpler reading: construct a rank-2 weight W_synth = (3072, 768) with only 2 effective output directions d₁, d₂. E.g. W_synth = A_random @ [d₁; d₂] where A_random is (3072, 2).
- Or: keep A from cycle 280 SVD, replace B with [d₁; d₂].
- Grid search 25 point scale combinations.

# Handoff
Plan agent: design the synthetic projector carefully. New script `experiments/synthetic_unigram_projector_cycle308.py`.
