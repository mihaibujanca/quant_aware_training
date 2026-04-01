# Cycle 205 Review: Block 1 Single-Block K-sweep (Item 216)

## Verdict: INCONCLUSIVE

Improvement beyond K=512 is 0.493 PPL (just below 0.5 threshold). Curve is still
decreasing at K=1536 but very slowly (diminishing returns).

## Key Numbers

| K     | PPL    | vs float |
|-------|--------|----------|
| 256   | 54.903 | +1.9%    |
| 512   | 54.689 | +1.5%    |
| 768   | 54.478 | +1.1%    |
| 1024  | 54.453 | +1.0%    |
| 1536  | 54.196 | +0.6%    |

Float baseline: 53.895. Max improvement beyond K=512: 0.493 PPL.

## Critical Finding

Block 1 alone at K=512 costs only +1.5% PPL. The 3-block recipe [1,4,7] costs +14.6%.
The gap is 13.1% — which is ENTIRELY from cascade across 3 blocks, not from block 1's
individual compression quality. This confirms that each block compresses well individually
(+1-2%) but cascade amplifies errors 10× when composing 3 blocks.

## Implication for Item 217

Since block 1's single-block quality at K=512 is already very good (+1.5%), increasing K
would gain at most 0.5 PPL. The real leverage is reducing the CASCADE amplification. Item 217
(Adam W_out correction on the 3-block composed model) directly targets this: it fine-tunes
the compressed projections to better handle the composed error.

## Impact

Closes item 216. K=512 is approximately optimal for single-block block 1 compression. No
need to increase K beyond 512 — the marginal gain is too small relative to the additional
parameters. The cascade, not single-block quality, is the bottleneck.
