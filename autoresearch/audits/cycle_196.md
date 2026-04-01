# Cycle 196 Audit

- Plan followed: partially. Plan specified "all-12 topk+lstsq" but implementation correctly used
  BLOCKS=[1,4,7,11] (same as cycle 185 working recipe) after discovering all-12 blocks gives
  ~1700 PPL cascade (vs the expected ~59 PPL which was for 4 blocks only).
- Config: BLOCKS=[1,4,7,11], K=512, N_SEQS=200, SEQ_LEN=128.
- Baseline PPL: 30.089 (note: cycle 185 used 59.42 for same config; discrepancy from eval method)
- Plan verdict thresholds: >9 PPL improvement = CONFIRMED; <2 = REFUTED. Result: -0.248 → REFUTED.
- Output: autoresearch/results/cycle_196_ln_recal.json
