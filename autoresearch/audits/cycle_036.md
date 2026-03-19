# Audit: Cycle 036 — Item 53

**Heterogeneous-ratio multi-block accumulation — GPT-2 FFN**

## Verdict check

- k=4 PPL = 131.85, rel = +9.59%.
- Confirmed if: k=4 <= 8% AND >4x avg. Result is +9.59% > 8% -> not confirmed.
- Refuted if: k=4 > 12%. Result is +9.59% < 12% -> not refuted.
- Inconclusive if: 8-12%. Result is 9.59% -> INCONCLUSIVE is correct.

**Phase 1 result: PASS** — verdict matches criteria.

## Quality checks

| Check | Status | Note |
|-------|--------|------|
| G1: Code diff | PASS | Identical to cycle 033 (`multiblock_accumulation_6block.py`) except `HIDDEN_FRACTION = 1/6` (was 0.25), updated thresholds (8%/12% vs 10%/20%), and overlay of cycle 033 reference curve. No structural changes to compression or evaluation logic. |
| G2: Statistical power | FLAG | Single seed (42), 50 eval sequences. Same limitation as cycle 033. Marginal result (+9.59% sitting in the 8-12% grey zone) makes this especially problematic — a different seed could push it either way. |
| G3: Interpretation | PASS | k=4 at +9.59% (6x) vs +8.84% (4x) = only +0.75pp for 50% more per-block compression. The result is genuinely informative: degradation is dominated by block count, not per-block ratio. k=3 curves nearly identical (6.40% vs 6.38%). |
| G4: Consistency | PASS | k=1 at -0.16% (sub-baseline) is consistent with cycle 035's compression-as-regularization finding for single-block 6x. Per-block ratios are uniform 5.995x across all blocks as expected. |
| G5: Method failure | N/A | Not a refutation. |

## Flags

1. **Single-seed fragility on a marginal result.** The +9.59% sits squarely in the inconclusive band. Multi-seed confirmation (3+ seeds) would resolve whether this tips to confirmed or remains borderline. Low priority since the qualitative finding (block count dominates per-block ratio) is robust regardless.

## Summary

INCONCLUSIVE — correctly classified. The experiment is well-executed and the comparison against cycle 033 is the right design. The key takeaway (block count >> per-block ratio for degradation) is solid even without resolving the verdict.
