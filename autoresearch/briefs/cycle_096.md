# Cycle 096 Brief — Cascade Attribution: FFN vs Attention (GPT-2)

**Item:** 111 | **Type:** analysis | **Depends on:** none

## Context
Cycles 089-095 established a c~0.41/block cascade rate that is ratio-independent (093),
allocation-independent (094), and ordering-independent (089). Attention SVD (095) shows
the same inter-block propagation bottleneck (rank-16 ~ rank-11 in stacking). The cascade
ceiling is structural, but we don't know which sub-block (FFN or attention) amplifies it.

## Question
When k=4 FFN-compressed blocks are stacked, is the cascade amplification primarily
mediated by the attention sub-block (which reads corrupted residuals) or the FFN sub-block
(which propagates its own approximation error)?

## Approach
Instrument residual stream at each sub-block boundary. Measure per-block error delta
after attention vs after FFN. Ablation: freeze one sub-block's error contribution,
measure cascade reduction. Attribute the c~0.41 amplification factor to FFN vs attention.

## Risk
Attribution may vary by block position (early vs late), yielding INCONCLUSIVE.
Block-0 attention is known to be catastrophically sensitive (095) -- may dominate.

## Assets
- `experiments/compositional_accumulation_curve_gpt2.py` (FFN stacking infra)
- `experiments/attention_compression_v2_gpt2.py` (attention SVD infra)
- `data/openwebtext_sample.txt`
