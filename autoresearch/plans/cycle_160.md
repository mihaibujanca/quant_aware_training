# Cycle 160 — Plan

## Task
170. **Block-adaptive compression ratio via per-block sensitivity — GPT-2 FFN**

## Hypothesis
Non-uniform channel allocation across blocks, guided by per-block sensitivity sweeps, can beat uniform k=4 stacked PPL by ≥2 points at equal total parameter count.

## Pre-flight Results

### P1: Duplicate check — FAIL

Item 170 is a duplicate of two already-refuted items:

- **Item 110 (cycle 094) — REFUTED**: "Non-uniform compression ratio allocation — GPT-2 FFN". Same 4-block set [7,5,4,1], same topk+lstsq pipeline, same equal-budget constraint. Result: uniform 6x PPL=69.40, mild nonuniform PPL=69.56, aggressive nonuniform PPL=70.25. Best non-uniform delta = +0.24% (negligible). "Polish absorbs single-block damage differences; allocation is irrelevant after output-projection is refit." 3 seeds.

- **Item 65 (cycle 050) — REFUTED**: "Oracle-guided adaptive compression ratio — GPT-2 FFN". Oracle allocation was *worse* than uniform (+12.33% vs +8.45%). Variable dims produce incompatible residual structure.

Item 170 proposes the same mechanism (sweep per-block sensitivity → allocate budget non-uniformly) on the same model/block set. The brief says "12 blocks" but standard set is 4 blocks [7,5,4,1] (brief line 23 confirms). The k∈{2,3,4,5,6} sweep range is the only nominal difference from cycle 094, which already tested mild and aggressive non-uniform allocations.

The fundamental reason uniform wins — polish absorbs single-block differences — has not changed. No new method (e.g., per-block correction, different polish strategy) is proposed that would invalidate the prior refutation.

## PLAN INVALID: P1 failed — Item 170 duplicates refuted items 110 (cycle 094) and 65 (cycle 050). Polish absorbs per-block sensitivity differences, making non-uniform allocation irrelevant. The sweep range (k∈{2,3,4,5,6}) does not address the mechanism that caused prior refutations.

## Recommendation
Mark item 170 as SKIP/DUPLICATE in the agenda, citing cycles 050 and 094. Advance to the next agenda item.
