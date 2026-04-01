# Cycle 175 Brief — Item 186

## What
Per-token prediction-space impact of replacing GELU with ReLU in GPT-2 FFN blocks.
Not hidden-state error — actual KL divergence on next-token distributions.

## Why now
- Cycle 173: negative regime absorbable as constant bias (all 12 blocks, max +1.4% PPL).
- Cycle 174: token-dependent s_neg is random noise (cos with J ≈ 0.03).
- Missing piece: direct prediction-space measurement. KL(P_float || P_relu) per token
  tells us whether the negative regime matters for *what the model predicts*, not just
  hidden-state geometry.

## What we measure
For each block b (0-11), independently replace GELU→ReLU:
1. P_float = softmax(logits_float), P_relu = softmax(logits_modified)
2. Per-token KL(P_float || P_relu) — 2048 tokens from OWT
3. Top-50 tokens by KL per block — report text and context
4. Fraction where P_relu(correct) > P_float(correct) (neg regime hurts prediction)
5. Correlate mean KL with ||y_neg|| and ||s_neg|| from cycle 174 methodology

## Success criteria
- Confirmed: mean KL > 0.1 nats in ≥8/12 blocks AND top-50 tokens show semantic pattern
- Refuted: mean KL < 0.01 nats — negligible per-block prediction impact
- Inconclusive: moderate KL or no semantic pattern

## Expected outcome
Given cycles 173-174, expect REFUTED: mean KL < 0.01. The negative regime is noise.
