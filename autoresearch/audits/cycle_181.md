# Cycle 181 — Audit

**Item:** 191 — Full-model ReLU + per-block bias vector swap — GPT-2
**Verdict:** REFUTED (PPL +87.9%, threshold was +50%)

## Result summary

| Metric | Value |
|--------|-------|
| Baseline PPL | 22.700 |
| Full-model ReLU+bias PPL | 42.643 |
| Delta | +87.9% |
| Single-block max delta | +5.9% (block 4) |
| Single-block mean delta | +2.4% |

## Key findings

1. **Single-block swaps confirm cycle 173:** All 12 blocks individually within +6% PPL (mean +2.4%). The constant bias μ_neg_l absorbs the GELU negative regime per-block.

2. **Full-model cascade is severe:** Simultaneous 12-block swap gives +87.9% PPL, matching cycle 166's +87% exactly. The biases precomputed on float model activations become stale as upstream ReLU swaps change the residual stream distribution.

3. **Cascade is smooth and monotonic:** Incremental accumulation shows no cliff — blocks 0-2 at +9.3%, blocks 0-4 at +30%, blocks 0-6 at +56%, all 12 at +88%. Each additional block contributes roughly +5-10% PPL degradation.

4. **Block 11 μ_neg norm anomaly:** ||μ_neg_11|| = 120.3, compared to 15-53 for blocks 0-10. The last block's negative-regime output contribution is 4x larger, amplifying accumulated upstream errors.

## Methodology check

- Baseline PPL (22.7) within expected range, consistent with prior cycles.
- μ_neg_l computed correctly as E[y_neg] @ W_down (GPT-2 Conv1D shape: (3072, 768)).
- 2048 tokens for bias estimation as specified in item text.
- Both full-model and incremental evaluations on same 200 OWT sequences.

## Next steps

Item 192 (iterative bias recomputation) is now eligible. The incremental curve strongly suggests distribution shift is the mechanism — recomputing each block's bias on the partially-swapped model should help. Whether it helps enough (≥10 PPL points) is the question.
