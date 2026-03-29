# Cycle 099 Brief — Inter-Block Residual-Stream Projection Correction (GPT-2 FFN)

**Item:** 112 | **Type:** implementation + empirical | **Depends on:** none

## Context
FFN cascade rate c~0.41/block is allocation/order-independent (cycles 093, 094). Per-block
Adam polish reduces single-block error but does NOT reduce c. At k=4 blocks [7,5,4,1] at 6x,
polished PPL is ~+5%. At k=8 (all blocks compressed), PPL explodes to +914% (cycle 089).
The cascade is the binding constraint: each block's FFN error gets amplified by downstream blocks.

## Question
Does the residual-stream error after each compressed block live in a low-rank subspace? If so,
projecting it out (no training, just SVD on calibration data) could break the cascade.

## Approach
Collect float vs compressed residual states at each block boundary on probe data. SVD the error
matrix. Build correction P_i = I - U_r @ U_r^T. Insert after each compressed FFN. Sweep r in
{4, 8, 16, 32}. Measure PPL and refit cascade rate c.

## Risk
If error is diffuse (not low-rank), projection removes useful signal along with error.
LayerNorm after each block may scatter error across all dimensions (MP7-adjacent concern).

## Assets
- `experiments/compositional_accumulation_curve_gpt2.py` (stacking infra, cascade measurement)
- `experiments/cascade_polish_gpt2.py` (per-block error measurement)
