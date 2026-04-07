# Cycle 271 Plan — Item 281: 1-block K=512 sweep

## Hypothesis (verbatim)
cascade extrapolates concavely → min over 3 subsets best_mean ≤ 43.0 (≥0.85 PPL below cycle 266 [1,4]=43.85, ≥10.9 PPL below float).

## Verdict thresholds
- CONFIRMED: min over 3 subsets best_mean ≤ 43.0
- REFUTED: all 3 subsets best_mean ≥ 46.0
- INCONCLUSIVE: min ∈ (43, 46)

## Setup
Trivial SUBSETS swap on cycle 266 template. SUBSETS=[[1], [4], [7]], K=512, 3 seeds {42,137,2024}, 9 fits total. Hyperparams identical to cycle 266: N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, disjoint splits, best-state restore. Trainable params/config = 393,984 (1 × (512×768 + 768)).

## Steps
1. Create `experiments/one_block_sweep_cycle271.py` (clone cycle 266, swap SUBSETS=[[1],[4],[7]], update output paths, baselines + 43/46 thresholds).
2. Run with `MPLCONFIGDIR=.mplconfig .venv/bin/python ...`, log to `autoresearch/logs/cycle_271_run.log`. Wall cap ~27 min; drop seed 2024 if exceeded after 6 fits.
3. Save JSON to `autoresearch/results/cycle_271_one_block_sweep.json` with per-subset/per-seed + aggregates + cycle 266 [1,4] 43.85 + cycle 210 [1,4,7] 46.56 + float 53.895.
4. Write `autoresearch/results/cycle_271_summary.md`.

## Baselines
cycle 266 K=512 2-block [1,4]=43.852; cycle 210 K=512 3-block [1,4,7]=46.556; cycle 255 K=512 4-block min 49.609; float 53.895.

## Bookkeeping
Update agenda.md: delete item 281 from NEXT/P0; add `### Cycle 271` entry to top of PAST EXPERIMENTS. Append TSV row. Stage cycle files only. Commit with verdict. Manifest + bookkeeping commit. Audit.

## Expected outcome
Concave cascade savings shrinking (5.69→4.22→3.05→2.70) suggests 2→1 saves ~2.0-2.5 PPL → 1-block min ≈ 41.5-42.0; uncertain whether [1] or [7] wins (cycle 173 ranking [7]≈[1]>[4]).
