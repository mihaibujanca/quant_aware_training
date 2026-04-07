# Cycle 287 Plan — Item 297

## Dependencies
None (cycle 271 template; cycle 286 baseline on disk).

## Item (verbatim Hypothesis)
Cycle 271 [1] K=512 extended budget — 8000 steps — can Adam push the cascade floor below 43.82? [1] K=512 1-block has the FEWEST trainable params of any cascade tested at K=512 (394K) → most Adam slack per param → most likely to benefit from longer training. Hypothesis: 8000 steps drops cycle 271 [1] K=512 below 43.0 — establishes a new dominant PPL Pareto anchor.

## Approach
Mirror `experiments/one_block_sweep_cycle271.py` verbatim with **SUBSETS=[[1]]**, K=512, **N_STEPS=8000**, **EVAL_EVERY=200** (40 eval points). Joint Adam over W_out+b_out, 3 seeds {42,137,2024}, best-state restore. Same scorer, same lstsq init.

## Files
- `experiments/one_block_1_k512_8k_cycle287.py` (new); `autoresearch/results/cycle_287_one_block_1_k512_8k.json`, `_summary.md`, `_manifest.txt`; `autoresearch/logs/cycle_287_run.log`.

## Config
SUBSETS=[[1]]; K=512; LR=1e-4; N_STEPS=8000; EVAL_EVERY=200; SEEDS=[42,137,2024]; N_CALIB=200; N_EVAL=100 disjoint; SEQ_LEN=128; MPS. Trainable=393,984. Wall ~14 min.

## Verification
Float within 0.5 of 53.895; trainable==393,984 asserted; restore `|restored-best|<1e-3`; init PPL deterministic at 54.689 (matches cycle 271 [1]); step-4000 eval within 0.05 of cycle 271 per-seed (43.826/43.846/43.789).

## Verdict logic (item 297 verbatim)
- CONFIRMED iff best_mean ≤ 43.0; REFUTED iff best_mean ≥ 43.7; INCONCLUSIVE otherwise.

## Baselines to record
float 53.895; cycle 271 [1] K=512 4k = 43.821 (393,984 params); cycle 286 [2] K=512 4k = 43.563 (393,984 params, current PPL Pareto winner).
