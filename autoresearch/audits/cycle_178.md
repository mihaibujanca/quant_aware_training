# Cycle 178 — Audit

## G1: Was the baseline verified?
YES. Mean baseline PPL = 52.3 across 3 seeds (range 48.8-57.7), within expected [25, 65] range. SEQ_LEN=128 (shorter than cycle 166/173's 1024) causes higher baseline variance and different absolute PPL.

## G2: Was the experiment faithful to the plan?
MOSTLY. The review (R3) correctly identified that absorbing all 12 blocks was unfair and narrowed to 4 blocks. The plan was followed with this fix. One divergence: the plan said "200 OWT eval sequences" but the library's `build_probe_corpus` was used with `eval_val_sequences=200`, which is consistent.

## G3: Are the measurements trustworthy?
YES. 3 seeds, consistent direction across all three:
- Seed 42: A=57.9, B=79.3 (B worse by +37%)
- Seed 137: A=44.9, B=66.4 (B worse by +48%)
- Seed 7: A=47.5, B=65.8 (B worse by +38%)
No seed shows B even close to A. The effect is large and stable.

## G4: Is the verdict correct given the data?
YES. Both refutation criteria met:
1. Absorption cascade: +12.3% > 5% threshold
2. Positive-only worse: 70.5 > 50.1 (much worse than standard topk)

## G5: Were there surprises or confounds?
YES — important finding: the item's premise that "~9-13% of channels are active" was about per-token activation rates, not about the fraction of channels that are ever positive. In practice, 93.8-99.3% of channels are sometimes positive (z>0 in >=1% of tokens). This means the "positive-only" condition doesn't actually reduce the channel set — it still selects 512 from ~3000 candidates, same as standard topk.

The real degradation comes from GELU->ReLU substitution: ReLU(z) vs GELU(z) differs most near z=0, which cycle 175 showed is where prediction-relevant differences concentrate. The lstsq targets (original GELU outputs) can't be well-matched by ReLU hidden activations.

## G6: Does this inform next steps?
YES. The negative-regime absorption direction is now firmly closed:
- Per-block absorption is cheap (+1-6% per block), but stacking cascades badly
- The GELU positive-regime shape near z=0 carries real information that ReLU discards
- Future work should preserve GELU, not replace it. The topk+lstsq recipe with GELU activation remains the strongest baseline.
