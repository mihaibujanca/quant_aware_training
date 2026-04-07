# Cycle 265 Plan — Item 275: K=128 4-block sweep

## Hypothesis (verbatim)
K=128 at 4 blocks closes most of the K=512 gap → min over 3 subsets best_mean ≤ 53.0 (within 3.4 PPL of cycle 255 K=512 49.609, beats cycle 251 K=128 5-block 59.119 by ≥6 PPL).

## Verdict thresholds
- CONFIRMED: min over 3 subsets best_mean ≤ 53.0
- REFUTED: all 3 subsets best_mean ≥ 56.0
- INCONCLUSIVE: min ∈ (53, 56)

## Setup
Trivial K swap on cycle 255 template. SUBSETS=[[1,2,4,7], [1,4,7,8], [1,2,7,8]], K=128, 3 seeds {42,137,2024}, 9 fits total. Hyperparams identical to cycle 255: N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, disjoint splits, best-state restore. Trainable params/config ≈ 394,752 (4 × 98,688).

## Steps
1. Create `experiments/four_block_k128_sweep_cycle265.py` (clone cycle 255, K=512→128, update output paths, add cycle 251/223 baselines).
2. Run with `MPLCONFIGDIR=.mplconfig .venv/bin/python ...`, log to `autoresearch/logs/cycle_265_run.log`. Wall cap ~27 min; drop seed 2024 if exceeded after 6 fits.
3. Save JSON to `autoresearch/results/cycle_265_four_block_k128_sweep.json` with per-subset/per-seed + aggregates + cycle 255 / 251 / 223 baselines + float 53.895.
4. Write `autoresearch/results/cycle_265_summary.md`.

## Baselines
cycle 255 K=512 4-block [1,2,4,7]=49.609 / [1,4,7,8]=50.712 / [1,2,7,8]=50.208; cycle 251 K=128 5-block 59.119; cycle 223 K=128 3-block [1,4,7]=48.989; float 53.895.

## Bookkeeping
Update agenda.md: delete item 275 from NEXT/P0; add `### Cycle 265` entry to top of PAST EXPERIMENTS. Append TSV row. Stage cycle files only. Commit with verdict. Manifest + bookkeeping commit. Audit.

## Expected outcome
linear K=128 extrapolation (cycle 231 → 251 gradient ≈ 8.4 PPL/block) predicts best_mean ≈ 50.7 — likely CONFIRMED.
