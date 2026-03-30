# Cycle 115 Brief — Compositional Stacking with Per-Block Cascade Correction — GPT-2 FFN

**Item:** 118 | **Type:** empirical | **Depends on:** none

## Context
k=4 stacking at 6x yields ~+5% PPL (cycle 098 baseline ~52 PPL). The cascade rate
c~0.41 is attention-driven (cycle 104), uniform across heads (cycle 113). Error spans
85% of dims (cycle 099) — not low-rank as a subspace. But this experiment tests learned
rank-r *corrections* (I + U@V), not projections. The correction only needs to capture
the error *direction*, not the full activation subspace.

## Question
Can tiny learned per-block corrections (I + U_l @ V_l, rank r in {4,8,16}) absorb the
cascade error that compounds through k=4 stacking, at negligible param cost (49K at r=8)?

## Approach
Compress k=4 blocks [7,5,4,1] at 6x with topk+lstsq+Adam (standard recipe). Freeze
everything, insert rank-r corrections after each compressed block. Train corrections
jointly under next-token CE. Compare r={4,8,16} vs uncorrected baseline.

## Risk
Cycle 099 showed error spans 85% dims — rank-16 may be too low to capture meaningful
correction directions. But CE training may find the task-relevant subset (cf. Exp 1:
energy recovery != task recovery). The corrections see full model loss, not just MSE.
