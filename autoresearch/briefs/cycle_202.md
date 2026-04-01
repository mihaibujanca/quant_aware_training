# Cycle 202 Brief: BLOCKS=[1,4,7] Compression Excluding Block 11 (Item 213)

## Item

213 — BLOCKS=[1,4,7] compression — exclude block 11 (output decompressor)

## Context

- Cycle 201 CONFIRMED block 11 as output decompressor: in_rank=2→out_rank=231 (154×).
- Block 11 applies a rank-1 perturbation to expand the residual stream for LM head.
- Current recipe BLOCKS=[1,4,7,11] may be suboptimal: block 11 compression disrupts rank expansion.
- Hypothesis: compressing block 11 loses the rank-expansion step → LM head receives wrong-rank input.
- Test: BLOCKS=[1,4,7] (K=512) vs BLOCKS=[1,4,7,11] (K=512).

## Experiment Design

- Compare PPL for BLOCKS=[1,4,7] vs BLOCKS=[1,4,7,11] on the SAME held-out eval set.
- Use N_CALIB=200, SEQ_LEN=128 for fitting (same as cycle 196/199).
- Use N_EVAL=100 held-out sequences for eval (same as cycle 199).
- Float baseline from same eval split.

## Key Files

- `experiments/k_sweep_rank1_cycle199.py` — all helper functions (collect_mlp_io, fit_block, CompressedMLP, eval_ppl)
- `autoresearch/results/cycle_199_k_sweep.json` — K=512 PPL on held-out eval = 68.14

## Constraints

- MPS: lstsq on CPU
- MUST use same eval sequences as cycle 199 for fair comparison (sequences 200-300)
- Baseline: float_ppl from cycle 199 = 53.89 (held-out eval)
- Report both [1,4,7] PPL and [1,4,7,11] PPL from the same run
