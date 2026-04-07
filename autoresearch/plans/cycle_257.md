# Cycle 257 Plan — Item 267
## Item (verbatim)
267 (P1): 5-block K=64 — extend the K-vs-PPL curve at 5 blocks below K=128. BLOCKS=[1,2,4,7,8] K=64, 4000 steps, 3 seeds.
## Hypothesis
5-block tolerates K=64 with manageable degradation → best_mean ≤ 64.0 (5 PPL above K=128 plateau, well below K=1 territory).
## Baselines
c245 K=512 5-block 53.826, c251 K=128 5-block 59.119, c230 K=64 3-block 50.781, c240 K=1 6-block 96.886, float 53.895.
## Setup
Copy `experiments/five_block_k128_cycle251.py` → `experiments/five_block_k64_cycle257.py`. Change `K = 128` → `K = 64`. Keep BLOCKS=[1,2,4,7,8], N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024], best-state restore, disjoint calib/eval split. Update OUTPUT_PATH/SUMMARY_PATH/cycle/item labels, verdict thresholds, and baseline constants. Trainable ≈ 246,720.
## Verdict
CONFIRMED ≤64.0; REFUTED ≥75.0; else INCONCLUSIVE.
## Execute
`MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/five_block_k64_cycle257.py 2>&1 | tee autoresearch/logs/cycle_257_run.log`
## Bookkeeping
Write JSON (per-seed + aggregate + baselines) and summary md; agenda delete item 267 from NEXT/P1, prepend `### Cycle 257` PAST entry; TSV row; commit; manifest; bookkeeping commit; exec_result.
## Risks
Run cap ~10 min; reuse cycle 251 recipe verbatim aside from K=64 and label updates.
