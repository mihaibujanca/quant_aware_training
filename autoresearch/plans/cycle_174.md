# Cycle 174 Plan: Geometry of token-dependent negative signal — GPT-2 FFN

**Item:** 185 | **Script:** `experiments/neg_signal_geometry_cycle174.py`

## Criteria (verbatim from agenda)
- **Confirmed if:** |cos(s_neg, J)| > 0.2 in ≥8/12 blocks AND rank(S_neg) < rank(S_pos)/4.
- **Refuted if:** |cos(s_neg, J)| < 0.05 — orthogonal to predictions (noise).
- **Inconclusive if:** between.

## Pre-flights
**P1 -- Prior:** Cycle 173 confirmed negative regime absorbable as bias. Cycle 109 showed s_neg is 1.5-13% of negative L2. This cycle is geometry-only.
**P2 -- Cost:** 12 blocks × (1 forward + 1 backward) = 24 passes. ~15 min on MPS.
**P3 -- Abort:** Float PPL outside [20, 30]. If gradient hook fails on >2 blocks, abort.
**P4 -- Data:** 2048 OWT tokens, seq_len=128 (16 sequences).

## Config
- Model: HF `gpt2` (12 blocks, d_model=768, d_ff=3072)
- Data: 2048 tokens from OpenWebText (16 seqs × 128 tokens)
- Device: MPS (fallback CPU)

## Steps

### Phase 1: Setup and baseline (1 pass)
1. Load GPT-2, eval float PPL on 200 seqs. Abort if outside [20, 30].
2. Load 16 sequences of 128 tokens from OWT.

### Phase 2: Per-block signal decomposition (12 passes)
For each block l in 0..11:
3. Forward pass, hook post-GELU activations h_gelu ∈ R^{2048×3072}.
4. Compute y_block = h_gelu @ W_out.T + b_out (the FFN output contribution).
5. Mask: neg_mask = (h_gelu < 0), pos_mask = (h_gelu >= 0).
6. y_neg(x) = (h_gelu * neg_mask) @ W_out.T — negative-channel contribution per token.
7. y_pos(x) = (h_gelu * pos_mask) @ W_out.T — positive-channel contribution per token.
8. μ_neg = y_neg.mean(dim=0), s_neg = y_neg - μ_neg.
9. μ_pos = y_pos.mean(dim=0), s_pos = y_pos - μ_pos.
10. SVD of S_neg, S_pos → effective rank at 95% cumulative energy.

### Phase 3: Gradient computation (12 backward passes)
For each block l in 0..11:
11. Forward pass with CE loss against next-token targets.
12. Register backward hook on block l output to capture d(loss)/d(y_block) per token.
13. Backward pass → J(x) ∈ R^{2048×768} is the per-token gradient.

### Phase 4: Alignment analysis
14. Per block: compute cos(s_neg(x), J(x)) per token → mean|cos|.
15. Per block: compute cos(s_pos(x), J(x)) per token → mean|cos|.
16. Report table: block, rank_neg, rank_pos, mean|cos_neg_J|, mean|cos_pos_J|.

### Phase 5: Verdict
17. Count blocks where mean|cos(s_neg,J)| > 0.2 AND rank_neg < rank_pos/4.
18. Apply criteria. Save JSON + summary.

## Artifacts
- `experiments/neg_signal_geometry_cycle174.py`
- `autoresearch/results/cycle_174_neg_signal_geometry.json`
