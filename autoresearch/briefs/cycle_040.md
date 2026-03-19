# Cycle 040 Brief — Item 54

## Task
**Compression ceiling scan: 8x and 10x single-block — GPT-2 FFN**

Cycles 033-035 established that 6x single-block compression is tolerable (all 6 easiest blocks within 5% PPL, most sub-baseline due to lstsq regularization). Item 54 extends to 8x and 10x on blocks 7, 5, 4 to find the actual per-block compression wall.

## Hypothesis
At least 1 of 3 blocks tolerates 8x with mean relative PPL <=10%.

## Success Criteria
- **Confirmed**: >=1 block tolerates 8x at mean <=10% relative PPL.
- **Refuted**: all 3 blocks exceed 10% at 8x (6x is the ceiling).
- **Inconclusive**: block-dependent results with no clear pattern.

## Dependency
- Item 52: DONE (cycle 035, multi-seed 6x confirmed real sub-baseline effect)

## Key Context
- Cycle 035 6x results (mean relative PPL): block 7 +0.51%, block 5 -0.20%, block 4 -0.13%
- Sub-baseline PPL at 6x is confirmed real (lstsq regularization effect)
- Pipeline: topk channel selection + lstsq refit, 3 seeds, 200 eval sequences
