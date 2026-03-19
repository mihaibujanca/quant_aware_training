# Cycle 042 Brief — Item 60

## Task
**Multi-block stacking at 10x per-block -- GPT-2 FFN**

Single-block 10x is free: all 3 tested blocks show sub-baseline PPL at 10x (cycle 040: block 7 +0.05%, block 5 -0.59%, block 4 -0.54%). Four-block 6x costs +9.59% uncorrected and +8.35% corrected (cycles 037, 039). This experiment tests whether per-block 10x headroom composes: accumulate blocks [7,5,4,1] at 10x (hidden_fraction=0.1), k=1..4, with and without residual correction, 3 seeds, 200 eval sequences.

## Hypothesis
k=4 at 10x with correction achieves <=12% relative PPL, matching the k=4@6x corridor.

## Success Criteria
- **Confirmed**: k=4@10x with correction <=12% relative PPL.
- **Refuted**: k=4@10x exceeds +20% even with correction.
- **Inconclusive**: PPL falls between 12-20%, or correction inconsistent across seeds.

## Dependencies
- Item 54 (cycle 040): CONFIRMED -- 10x single-block is sub-baseline.
- Item 53 (cycle 039): CONFIRMED -- residual correction recovers 3.6pp at k=5@6x.

## Key Context
- At 10x, each block keeps 307/3072 channels. Param reduction: 4 blocks save ~17M of 56.7M = 30.1% (vs 27.8% at 6x). Modest +2.3pp gain in params, but tests composability of extreme per-block ratios.
- Block ordering [7,5,4,1] = easiest-first from cycle 005 damage ranking.
