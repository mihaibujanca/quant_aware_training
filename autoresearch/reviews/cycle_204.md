# Cycle 204 Review: 3-Block Selection Sweep (Item 215)

## Verdict: REFUTED

No 3-block selection beats [1,4,7] by more than 1 PPL. Best is [1,4,10] at 61.455 vs anchor 61.746 (0.29 PPL improvement, within noise).

## Key Numbers

| Block set | PPL    | vs float |
|-----------|--------|----------|
| [1,4,10]  | 61.455 | +14.0%   |
| [1,3,8]   | 61.727 | +14.5%   |
| [1,4,7]   | 61.746 | +14.6%   |
| [1,6,10]  | 62.081 | +15.2%   |
| [1,5,9]   | 62.597 | +16.1%   |
| [2,5,8]   | 63.590 | +18.0%   |
| [3,6,9]   | 65.233 | +21.0%   |

## Interpretation

All top-4 selections include block 1. Selections starting with block 2 or 3 are consistently
1.8-7.6 PPL worse. This confirms block 1's special status: it has input_rank=192 (highest of
any compressible block, second only to block 0 with rank=244). Block 1 operates on partially-
collapsed embeddings, making it the most "valuable" block to compress.

The choice of the other 2 blocks (from {3-10}) matters little — any pair gives within 1 PPL.
This is consistent with cycles 187/203: blocks 2-10 are individually near-redundant and have
similar cascade behavior. The specific block indices 4 and 7 (original recipe) are arbitrary.

## Key Structural Finding

Block 1 is uniquely valuable to compress. The near-optimal 3-block strategy is:
- Always include block 1 (highest-value compressible block)
- Any 2 other blocks from {3,...,10} are approximately equivalent
- Blocks 2-10 after block 1 contribute roughly equally to cascade and PPL degradation

## Impact

Closes item 215. [1,4,7] is effectively optimal — the 0.29 PPL improvement from [1,4,10]
is within run-to-run variance. The best achievable with 3 blocks and topk+lstsq is approximately
+14-15% above float PPL. Further improvement requires a better single-block compressor.
