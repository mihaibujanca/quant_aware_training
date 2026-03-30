# Cycle 114 Brief — Linearity-Adaptive Compression Ratio Allocation — GPT-2 FFN

**Item:** 125 | **Type:** empirical | **Depends on:** none

## Context
Cycle 107 confirmed a block linearity gradient: middle blocks (4, 7) are nearly linear
(direct regression matches topk PPL within noise), while early (1) and late (11) are
nonlinear (topk dominates). Cycle 094 tested damage-based allocation on different blocks
[7,5,4,1] and found REFUTED — allocation irrelevant after polish (+0.24%).

## Question
Can linearity-adaptive allocation beat uniform 6x? Compress linear blocks harder (8x),
give spare budget to nonlinear blocks (4x), at matched total channel count.

## Approach
k=4 stacking on blocks [1,4,7,11]. Three conditions: (a) uniform 6x, (b) middle-heavy
(4,7 at higher compression), (c) edge-heavy (1,11 at higher compression). topk+lstsq+500
Adam, 3 seeds, 200 eval seqs OWT.

## Risk
Cycle 094 showed allocation has negligible effect after gradient polish (polish absorbs
the differences). This may repeat. Also: stated 8x/4x ratios don't budget-match —
must adjust k values to match total channel count exactly.
