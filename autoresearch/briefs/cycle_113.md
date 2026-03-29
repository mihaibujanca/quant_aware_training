# Cycle 113 Brief — Attention-Mediated Cascade: Per-Head Amplification Census — GPT-2

**Item:** 124 | **Type:** analysis | **Depends on:** none

## Context
Cycle 104 (item 115) showed attention dominates cascade amplification (>=70% in 8/11
transitions, c~0.41). But it treated attention as a monolithic block. Item 124 asks:
which individual heads amplify FFN compression error? If a small subset dominates,
they become surgical targets for cascade suppression (item 126 downstream).

## Question
Is attention-mediated cascade amplification concentrated in a few heads per layer, or
uniformly distributed across all 12 heads?

## Approach
For each of the 12 FFN blocks: compress at k=1 (6x) with topk+lstsq+500 Adam, inject
the resulting residual-stream error into the NEXT block's attention layer, and measure
per-head output perturbation ||Attn_h(x+delta) - Attn_h(x)||_2 / ||delta||_2. Aggregate
over 2048 OWT tokens. Report heatmap (12x12), top-5 heads, Spearman vs ablation PPL.

## Risk
Attention heads may amplify error unevenly across token positions. Mean-over-tokens
aggregation could hide position-dependent concentration.
