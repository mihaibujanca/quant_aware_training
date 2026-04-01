# Cycle 174 — Brief

**Item:** 185. Geometry of token-dependent negative signal — GPT-2 FFN [INTERPRETABILITY]

**Goal:** Characterize the geometric structure and prediction-alignment of the token-dependent negative signal s_neg(x) = y_neg(x) - μ_neg across all 12 GPT-2 blocks.

**Prior results (cycle 173):**
- Negative regime is fully absorbable as bias in all 12 blocks (C3 ≈ C0 within +1.4% PPL)
- So even if s_neg is prediction-aligned, its practical impact is negligible
- This cycle characterizes the *geometry*, not the practical impact

**Prior results (cycle 109):**
- Negative regime is 87-98% bias energy (b_leak dominates s_neg)
- s_neg has only 1.5-13% of negative-regime L2 energy

**Method:**
- For each block (0-11) on 2048 OWT tokens:
  (a) Stack s_neg into S_neg ∈ R^{2048×768}, SVD → effective rank at 95% energy
  (b) Same for S_pos → compare ranks
  (c) Compute J(x) = d(CE_loss)/d(y_block) per token via backward hooks
  (d) Per-token cosine alignment cos(s_neg(x), J(x)) and cos(s_pos(x), J(x))
  (f) Report mean|cos| and ranks

**Confirmed if:** |cos(s_neg, J)| > 0.2 in ≥8/12 blocks AND rank(S_neg) < rank(S_pos)/4.
**Refuted if:** |cos(s_neg, J)| < 0.05 across blocks.

**Budget:** ~15 min. Forward + backward passes per block.
