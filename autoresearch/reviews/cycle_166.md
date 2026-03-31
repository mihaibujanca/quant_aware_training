# Cycle 166 Review — Constant Negative Bias (Item 176)

Item 176. Reviewer: design-review agent.

## R1: Premise validity — OK

Cycle 165 confirmed negativity is learned (W_up-driven). Cycle 164 showed positive
channels are essential. Testing whether the negative regime is functionally just a
bias is the natural next step. The b_leak construction directly targets the 87-98%
token-invariant fraction identified in the brief.

## R2: Measurement soundness — OK

PPL is the right metric. Baseline sanity check is in place. Per-block isolation
correctly tests each block independently before the full-model cascade. The b_leak
is computed on float model data (not on modified model), which is correct — it
captures what the negative regime actually produces.

## R3: Capacity and statistical adequacy — OK

No training involved — pure ablation. 200 sequences × 1024 tokens = ~200K eval
tokens, sufficient for stable PPL estimates. b_leak computed on 4096 tokens is
adequate for a mean estimate of 3072-dim vectors.

## R4: Method-architecture alignment — OK

GPT-2 uses NewGELU at `model.transformer.h[i].mlp.act`. The wrapper replaces the
activation module, same pattern as cycle 164 (tested and working). b_leak is added
post-activation (same dimension as GELU output = 3072), which is correct since
W_down expects the full 3072-dim vector including the negative contribution.

One note: b_leak must be added AFTER ReLU, not before. The plan's ReLU(z) + b_leak
is correct — it replaces the full gelu(z) output with relu(z) + constant. This
preserves the positive regime and substitutes the negative regime with its mean.

## R5: Known pitfall scan — OK

- MP6 (verdict misclassification): the metric (PPL) directly tests the hypothesis.
  A PPL change is informative either way.
- MP7 (post-LayerNorm): measurements are on pre-activation z and post-activation
  output, not post-LayerNorm. Not applicable.
- No MP1-MP5, MP8 issues.

## Summary

No blocking or minor issues found. The plan is clean and well-scoped.

**Verdict: ACCEPT.**
