# Cycle 197 Audit

- Plan deviation: tested sequential progressive fitting rather than true "joint student"
  (single compressed block approximating 2 blocks). Sequential fitting is simpler
  and directly tests the distribution mismatch hypothesis.
- Config: BLOCKS=[1,4,7,11], K=512, N_SEQS=200×128=25600 tokens. Same as cycle 196.
- Verdict thresholds: >2 PPL improvement = CONFIRMED; <0 = REFUTED; 0-2 = INCONCLUSIVE.
- Result: 0.135 PPL improvement → INCONCLUSIVE.
- Output: autoresearch/results/cycle_197_joint_compression.json
