# Cycle 284 Design Review — Item 294

**Status: APPROVE**

## Plan check
Plan combines cycle 280 (block 1 rank-2 SVD c_proj) and cycle 270 (block 11 rank-22 SVD c_proj) into a single homogeneous low-rank weight-SVD hybrid. Both LowRankCProj modules use the same factorisation convention (frozen A=Uᵣ·Σᵣ, trainable B=Vhᵣ + bias). Joint Adam over the 19,968-param trainable set, all other model parameters frozen.

## Hazards (none blocking)
- Different from cycle 275 (which mixed topk+lstsq W_out at block 1 with weight-SVD at block 11). Here both blocks use the SAME compression operator family — the heterogeneous-mix hypothesis from cycle 275 should not apply.
- Init PPL on the joint hybrid is unmeasured (will be larger than either single-block init since both errors compose). This is recorded but not used in verdict.
- 19,968 params is in the same order as cycle 270 (17,664) and cycle 282 (13,824) — directly comparable on the param axis.

## Verdict thresholds
Per item 294 verbatim: CONFIRMED iff best_mean ≤ 55.0; REFUTED iff best_mean ≥ 58.0; INCONCLUSIVE in (55.0, 58.0).

APPROVE — proceed to Execute.
