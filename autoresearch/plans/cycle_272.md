# Cycle 272 Plan — Item 282: K=256 4-block sweep

## Hypothesis (verbatim)
K-curve concave-up holds at 4 blocks → min best_mean ≤ 52.0 (between cycle 265 K=128 53.65 and cycle 255 K=512 49.61, expected ~51.6 from linear midpoint).

## Verdict thresholds
- CONFIRMED: min over 3 subsets best_mean ≤ 52.0
- REFUTED: all 3 subsets best_mean ≥ 54.0
- INCONCLUSIVE: min ∈ (52, 54)

## Setup
Trivial K=128→K=256 swap on cycle 265 template. SUBSETS=[[1,2,4,7],[1,4,7,8],[1,2,7,8]], K=256, 3 seeds {42,137,2024}, 9 fits. N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, disjoint splits, best-state restore. Trainable params/config = 789,504 (4 × (256×768 + 768)).

## Steps
1. Create `experiments/four_block_k256_sweep_cycle272.py` (clone cycle 265, K=128→256, update output paths, baselines, 52/54 thresholds).
2. Run with `MPLCONFIGDIR=.mplconfig .venv/bin/python ...`, log to `autoresearch/logs/cycle_272_run.log`. Wall cap ~27 min; drop seed 2024 if exceeded after 6 fits.
3. Save JSON to `autoresearch/results/cycle_272_four_block_k256_sweep.json` with per-subset/per-seed + cycle 255 K=512 4-block (49.609) + cycle 265 K=128 4-block (53.647) + cycle 223 K=256 3-block (47.509) + float (53.895).
4. Write `autoresearch/results/cycle_272_summary.md`.

## Baselines
cycle 255 K=512 4-block [1,2,4,7]=49.609; cycle 265 K=128 4-block [1,2,4,7]=53.647; cycle 223 K=256 3-block=47.509; cycle 256 K=256 6-block=64.181; float=53.895.

## Bookkeeping
Update agenda.md: delete item 282 from NEXT/P0; add `### Cycle 272` entry to top of PAST EXPERIMENTS. Append TSV row. Stage cycle files only. Commit with verdict. Manifest + bookkeeping commit. Audit.

## Expected outcome
Linear midpoint K=256 ≈ 51.6 PPL (between K=128 53.65 and K=512 49.61); concave-up curve predicts slightly above midpoint. Same subset ordering [1,2,4,7] best as cycle 255/265.
