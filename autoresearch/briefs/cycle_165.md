# Cycle 165 — Item 175 — Weight-data alignment decomposition

## Question
Why is z_mean ≈ -1 in GPT-2 FFN hidden layers? Given z = W_up @ x + b, b_up contributes only -0.08, so the negativity arises from W_up @ x. Is this alignment negative-learned or geometric accident?

## Analysis Plan
(a) E[W_up @ x] decomposition over 4096 tokens
(b) Per-channel cosine(W_up[j], E[x]) — row-mean alignment
(c) PCA on x, project W_up rows onto top-k PCs
(d) Random-init W_up z_mean baseline (test learned vs accidental)
(e) Per-block comparison across 12 layers

## Context
- Cycle 164: positive channels essential (zeroing → +3M% PPL)
- Cycle 163: error uniformly distributed, no Pareto blocks
- Working hypothesis: negative bias is functionally critical (cycle 176); sign patterns structured enough for piecewise lstsq (cycle 177)
- If W_up negatively biased toward data: confirms network learned to gate computation

## Hypothesis
**Confirmed if**: trained z_mean shift > 0.5 vs random init (learned anti-alignment)
**Refuted if**: random init z_mean ≈ -1 (geometric property of high-D projection)
**Inconclusive**: 0.1-0.5 shift

## Estimate
~5 min compute; 5 analyses on 4096-token window
