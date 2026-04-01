# Cycle 181 — Brief

**Item:** 191 — Full-model ReLU + per-block bias vector swap — GPT-2
**Type:** empirical
**Depends on:** none (builds on cycle 173 single-block result)

## Context

Cycle 173 confirmed that replacing GELU with ReLU in any single GPT-2 FFN block, then adding a per-block constant bias μ_neg_l to the block output, recovers within +2% PPL (12/12 blocks). That test modified ONE block at a time while all others stayed float GELU.

Cycle 166 tested a related full-model ReLU+bias swap and saw +87% PPL cascade. However, cycle 166 added the bias in 3072-dim intermediate space. Item 191 computes μ_neg_l = E[W_down @ (GELU(z) ⊙ 1[z≤0])] — the mean negative-regime output contribution projected through W_down — and adds it as a 768-dim constant to the block output. Mathematically equivalent (linearity of expectation), but the experimental question is whether simultaneous 12-block swap cascades or stays within +10%.

## What this cycle measures

Precompute μ_neg_l for each of 12 blocks on 2048 OWT tokens. Replace all 12 GELU activations with ReLU. Add μ_neg_l as constant to each block's FFN output. Measure full-model PPL.

Total added params: 12 × 768 = 9,216.

## Criteria

- **Confirmed:** full-model ReLU+bias PPL within +10% of float baseline
- **Refuted:** PPL > +50% of float baseline
- **Inconclusive:** +10% to +50%
