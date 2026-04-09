# Cycle 369 Brief — Item 360

## Task
Test topk+lstsq accumulation order sensitivity on GPT-2 FFN blocks.
Three orderings on blocks [b1, b4, b7, b11]:
- (a) Easiest-first: lowest single-block swap damage
- (b) Block-index order [1,4,7,11]
- (c) Hardest-first: highest single-block swap damage

Measure PPL after each cumulative stage (1→2→3→4 blocks).

## Success Criterion
>5% PPL gap between best and worst ordering at 4-block stage.

## Dependency
Item 355 (required, should be in PAST — not verified in agenda due to item renumbering).

## Motivation
Cycle 007 on toy models showed easiest-first beats hardest-first by 27%.
Block importance ranking from cycle 367 (item 376):
ascending: [1,7,2,6,9,10,8,3,5,4,11,0]
So: b1=least critical (+1.60 PPL), b11=2nd most critical (+9.12 PPL)

## Context
- Working recipe: topk k=512 + lstsq + MLP adapter (h=64-128)
- Best baseline: 3-block joint=47.43 PPL, 6-block=65.54 (composition barrier)
- Per-block Frobenius: b1=1385, b11=6293 (4.5x spread)
- Block errors are orthogonal (cosine mean=0.016)

## Experimental Design
Run on [b1, b4, b7, b11] (subset spanning low→mid→high importance).
For each ordering, compress sequentially, measure cumulative PPL at stages 1,2,3,4.
Compare final (4-block) PPL across orderings; check for >5% gap.

## Expected Outcome
Order sensitivity may reveal whether composition barrier is due to:
1. Early-block capacity exhaustion (easiest-first fails)
2. Late-block error amplification (hardest-first recovers)
3. Path independence (all orderings similar → barrier is structural)
