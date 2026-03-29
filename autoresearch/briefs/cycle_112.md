# Cycle 112 Brief — GELU Negative-Leak Aware Compression (Corrected) — GPT-2 FFN

**Item:** 123 | **Type:** empirical | **Depends on:** none

## Context
Cycle 105 (item 116) was plan-rejected: Condition B's lstsq target was `y` instead of
`y - b_leak`, so adding b_leak post-hoc overshoots. Item 123 fixes this one-line bug.
Cycle 109 confirmed the negative leak is 87-99% constant bias (b_energy 0.87-0.99 across
all 12 blocks), meaning it can be predicted from per-channel means on probe data alone.

## Question
Does separating the constant negative leak from the approximation target let topk+lstsq
focus on the informative positive-regime signal, improving compressed PPL?

## Approach
Three conditions at 6x on blocks [1,4,7,11], 3 seeds, 500 Adam:
- **A** (baseline): standard topk+lstsq, k=512 from all 3072 channels
- **B** (leak-aware): topk from positive-only channels, lstsq target = `y - b_leak`, add b_leak back
- **C** (positive-only, no b_leak): isolates filtering effect from bias correction

Probe/fit data split prevents b_leak overfitting.

## Risk
Most blocks have neg_frac ~85% (cycle 111), so positive-dominated channels may be few.
If N_pos < 512 in a block, Condition B can't match k=512; must exclude or reduce k.
