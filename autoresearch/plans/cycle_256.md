# Cycle 256 Plan — Item 266
## Item (verbatim)
266 (P0): 6-block K=256 — fill the missing K=256 6-block data point. BLOCKS=[1,2,4,5,7,8] K=256 joint Adam W_out+b_out, 4000 steps, 3 seeds.
## Hypothesis
K=256 6-block ~63 (midpoint of K=512 59.515 and K=128 67.562); beats cycle 247 (64.453) and 249 (65.275) hetero mixes at matched ~1.18M params.
## Baselines
c221 K=512 59.515, c231 K=128 67.562, c247 hetero 64.453, c249 hetero 65.275, float 53.895.
## Setup
Copy `experiments/adam_6block_earlystop_cycle221.py` → `experiments/six_block_k256_cycle256.py`. Change `K = 512` → `K = 256`. Keep BLOCKS=[1,2,4,5,7,8], N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024], best-state restore. Update OUTPUT_PATH/SUMMARY_PATH/cycle/item labels and verdict thresholds. Trainable ≈ 1.18M.
## Verdict
CONFIRMED ≤62.0; REFUTED ≥65.0; else INCONCLUSIVE.
## Execute
`MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/six_block_k256_cycle256.py 2>&1 | tee autoresearch/logs/cycle_256_run.log`
## Bookkeeping
Write JSON (per-seed+aggregate+baselines), summary md; agenda delete item 266 from NEXT/P0, prepend Cycle 256 PAST entry; TSV row; commit; manifest; bookkeeping commit; exec_result.
## Risks
Run cap ~10 min; reuse cycle 221 recipe verbatim aside from K=256 and labels.
