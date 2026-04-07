# Cycle 291 Plan — Item 301

## Dependencies
Cycle 280 (block 1 r=2 retrain @ 49.0324) and cycle 289 (block 10 r=5 retrain @ 51.4659). Template: `experiments/rank_hybrid_1_11_cycle284.py`.

## Item (verbatim Hypothesis)
301 (P0): 2-block rank-hybrid [1] r=2 + [10] r=5 c_proj weight-SVD — cheap non-destructive composition. Hypothesis: joint Adam over rank-2 (block 1) + rank-5 (block 10) c_proj patches reaches ≤49.5 PPL at ~6,912 trainable params (2,304 + 4,608) — adds block 10 essentially for free on top of cycle 280's anchor and creates a new Pareto tile between cycle 280 (2.3K, 49.03) and cycle 282 (13.8K, 50.04).

## Approach
Adapt `experiments/rank_hybrid_1_11_cycle284.py` verbatim. Diff: TARGET_BLOCKS=(1, 10), RANKS=(2, 5); new OUTPUT_PATH/SUMMARY_PATH; new baseline constants and verdict thresholds; new expected_trainable=6,912. No structural change to LowRankCProj, the SVD loop, or the Adam retrain loop.

## Files
- `experiments/rank_hybrid_1_10_cycle291.py` (new)
- `autoresearch/results/cycle_291_rank_hybrid_1_10.json`, `_summary.md`, `_manifest.txt`, `_exec_result.txt`
- `autoresearch/logs/cycle_291_run.log`

## Config
TARGET_BLOCKS=(1, 10); RANKS=(2, 5); LR=1e-4; N_STEPS=4000; EVAL_EVERY=100; SEEDS=[42, 137, 2024]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; MPS; GPT-2 small; wikitext-2 val. Trainable per fit = (2·768+768) + (5·768+768) = 2,304 + 4,608 = 6,912. Wall ~8 min.

## Verification
- Float PPL within 0.5 of 53.895.
- Trainable param count == 6,912 asserted; everything else `requires_grad_(False)`.
- Per-seed best-state restore: assert `|restored - best| < 1e-3`.
- Cross-check: cycle 280 init (b1 r=2) was 57.658 and cycle 289 init (b10 r=5) was 59.918. Hybrid init combines BOTH replacements simultaneously, so it will exceed both individual inits.

## Verdict logic (item 301 verbatim)
- CONFIRMED iff best_mean ≤ 49.5 (≤0.47 PPL above cycle 280, non-destructive composition; new Pareto tile at ~6.9K params).
- REFUTED iff best_mean ≥ 51.5 (composition damages cycle 280's anchor by ≥2.5 PPL).
- INCONCLUSIVE otherwise.

## Baselines to record
float 53.895; cycle 280 b1 r=2 = 49.0324 (2,304); cycle 289 b10 r=5 = 51.4659 (4,608); cycle 282 b10 r=17 = 50.0445 (13,824); cycle 284 hybrid b1r2+b11r22; cycle 275 heterogeneous hybrid.
