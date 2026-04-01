# Cycle 190: Positive-Regime Shape Isolation via GELU-neg + ReLU-pos Hybrid

## Objective
Isolate E_shape (dominant error source in cycles 181/182) from E_replacement by testing a hybrid activation: GELU(z) for z < 0, ReLU(z) for z ≥ 0. Applied to all 12 FFN blocks. This preserves the negative regime (no bias correction needed) while measuring the cost of positive-regime shape change alone.

## Motivation
- Cycle 188 showed shape change accounts for ~14.3 PPL of the 18.9-PPL gap (C3 constant replacement: only ~4.6 PPL)
- Cycles 181/182 full swap: PPL ~42.6 (+88%), but confounds shape change, negative-regime loss, and bias interaction
- Hybrid design decouples: negative regime unchanged (GELU), positive regime shape changed (ReLU), no bias compensation
- Direct test of whether positive-regime shape dominates composition error

## Implementation
Hybrid activation: `torch.where(z < 0, F.gelu(z), F.relu(z))`
Applied to all 12 blocks with no bias adjustment.
Evaluation: SEQ_LEN=1024, N_SEQS=200, baseline PPL ~22.7

## Verdict Criteria
- **Confirmed** (E_shape dominant): hybrid PPL > 35 (+54%)
- **Refuted** (negative regime matters more): hybrid PPL < 28 (+23%)
- **Inconclusive**: 28–35 PPL range

## Expected Outcome
If hybrid PPL ≈ 27–28, validates that positive-regime shape change drives ~14 PPL of error. If < 28, suggests negative-regime loss or replacement effect larger than shape cost.
