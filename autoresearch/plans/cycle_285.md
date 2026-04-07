# Cycle 285 Plan — Item 295

## Dependencies
None (cycle 278 eff_rank, cycle 280 template already on disk).

## Item (verbatim Hypothesis)
rank-1 or rank-2 weight-SVD of block 2 c_proj + Adam retrain reaches ≤49 PPL at 1,536-2,304 trainable params — possibly beating cycle 280's 49.03 anchor on the ultra-low-param Pareto axis.

## Approach
Adapt `experiments/block1_lowrank_cycle280.py` verbatim with `TARGET_BLOCK=2`. Stage A: no-retrain forward swap of block 2 `c_proj.weight` with rank-r SVD approximations for r ∈ {1, 2, 5, 10}. Stage B: retrain at `RETRAIN_RANK=2`, joint Adam over `(B, bias) = (2,768)+(768) = 2,304` params, 3 seeds {42,137,2024}, best-state restore. Optional second retrain at r=1 only triggered if no-retrain r=1 lands within striking distance (≤55) — gated by an in-script flag, ~8 min extra.

## Files
- `experiments/block2_lowrank_cycle285.py` (new)
- `autoresearch/results/cycle_285_block2_lowrank.json`, `_summary.md`, `_manifest.txt`
- `autoresearch/logs/cycle_285_run.log`

## Config
TARGET_BLOCK=2; RANKS_NORETRAIN=[1,2,5,10]; RETRAIN_RANK=2; LR=1e-4; N_STEPS=4000; EVAL_EVERY=100; SEEDS=[42,137,2024]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; MPS; GPT-2 small; wikitext-2 val. Trainable per fit = 2*768 + 768 = 2,304. Wall ~8 min, +8 min iff r=1 gate fires.

## Verification
- Float PPL within 0.5 of 53.895.
- SVD shapes: U(3072,768), S(768,), Vh(768,768); print top-5 SVs and energies at r∈{1,2,5,10}.
- Restore original `c_proj.weight` between sweep stages; assert restore PPL ≈ float (<1e-3).
- Per-seed init PPL identical (deterministic SVD init).
- Trainable param count == 2,304 asserted; everything else `requires_grad_(False)`.
- Per-seed best-state restore: assert `|restored - best| < 1e-3`.

## Verdict logic (item 295 verbatim)
- CONFIRMED iff (r=2 retrain best_mean ≤ 50.0) OR (r=1 retrain best_mean ≤ 51.0).
- REFUTED iff r=2 retrain best_mean ≥ 53.0.
- INCONCLUSIVE otherwise.

## Baselines to record
float 53.895; cycle 280 b1 r=2 = 49.0324 (2,304 params); cycle 270 b11 r=22 = 54.4687 (17,664); cycle 282 b10 r=17 = 50.0445 (13,824); cycle 283 [2] K=16 = 49.46 (13,056); cycle 278 b2 c_proj OUTPUT eff_rank = 1.0503, top-1 energy = 0.9961.
