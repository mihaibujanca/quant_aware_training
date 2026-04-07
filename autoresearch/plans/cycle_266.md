# Cycle 266 Plan — Item 276: 2-block K=512 sweep

## Hypothesis (verbatim)
count-driven cascade extrapolates → min over 3 subsets best_mean ≤ 45.0 (≥4.6 PPL below cycle 255 4-block, ≥8.9 PPL below float).

## Verdict thresholds
- CONFIRMED: min over 3 subsets best_mean ≤ 45.0
- REFUTED: all 3 subsets best_mean ≥ 47.0
- INCONCLUSIVE: min ∈ (45, 47)

## Setup
Trivial SUBSETS swap on cycle 255 template. SUBSETS=[[1,4], [1,7], [4,7]], K=512, 3 seeds {42,137,2024}, 9 fits total. Hyperparams identical to cycle 255: N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, disjoint splits, best-state restore. Trainable params/config ≈ 787,968 (2 × 393,984 = 2 × (512×768 + 768)).

## Steps
1. Create `experiments/two_block_sweep_cycle266.py` (clone cycle 255, swap SUBSETS, update output paths, baselines + 45/47 thresholds).
2. Run with `MPLCONFIGDIR=.mplconfig .venv/bin/python ...`, log to `autoresearch/logs/cycle_266_run.log`. Wall cap ~27 min; drop seed 2024 if exceeded after 6 fits.
3. Save JSON to `autoresearch/results/cycle_266_two_block_sweep.json` with per-subset/per-seed + aggregates + cycle 210 K=512 3-block 46.556 + cycle 255 K=512 4-block min 49.609 + float 53.895.
4. Write `autoresearch/results/cycle_266_summary.md`.

## Baselines
cycle 210 K=512 3-block [1,4,7]=46.556; cycle 255 K=512 4-block [1,2,4,7]=49.609; cycle 245 K=512 5-block 53.826; cycle 221 K=512 6-block 59.515; float 53.895.

## Bookkeeping
Update agenda.md: delete item 276 from NEXT/P0; add `### Cycle 266` entry to top of PAST EXPERIMENTS. Append TSV row. Stage cycle files only. Commit with verdict. Manifest + bookkeeping commit. Audit.

## Expected outcome
Count-driven extrapolation (49.6 → 46.6 → predicted ~43-44) gives best_mean ≈ 43-44; subset ordering uncertain (item 276 says [1,4] favoured if early-block easier hypothesis holds).
