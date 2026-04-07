# Cycle 288 Plan — Item 298

## Dependencies
None (cycle 278 eff_rank census, cycle 282 template already on disk).

## Item (verbatim Hypothesis)
rank-24 retrain reaches ≤55 PPL at ~19.2K trainable params (24×768 + 768) — interpolates between cycle 282 (13.8K, 50.04) and cycle 270 (17.7K, 54.47). The geometric question is whether the early-vs-late distinction matters at intermediate eff_rank, OR whether eff_rank alone determines retrained PPL regardless of block depth.

## Approach
Adapt `experiments/block10_lowrank_cycle282.py` verbatim with `TARGET_BLOCK=3`. Stage A: no-retrain forward swap of block 3 `c_proj.weight` with rank-r SVD approximations for r ∈ {5, 10, 24}. Stage B: retrain at `RETRAIN_RANK=24`, joint Adam over `(B, bias) = (24,768)+(768) = 19,200` trainable params, 3 seeds {42,137,2024}, best-state restore.

## Files
- `experiments/block3_lowrank_cycle288.py` (new)
- `autoresearch/results/cycle_288_block3_lowrank.json`, `_summary.md`, `_manifest.txt`
- `autoresearch/logs/cycle_288_run.log`

## Config
TARGET_BLOCK=3; RANKS_NORETRAIN=[5,10,24]; RETRAIN_RANK=24; LR=1e-4; N_STEPS=4000; EVAL_EVERY=100; SEEDS=[42,137,2024]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; MPS; GPT-2 small; wikitext-2 val. Trainable per fit = 24*768 + 768 = 19,200. Wall ~7-9 min.

## Verification
- Float PPL within 0.5 of 53.895.
- SVD shapes: U(3072,768), S(768,), Vh(768,768); print top-5 SVs and energies at r∈{5,10,24}.
- Restore original `c_proj.weight` between sweep stages; assert restore PPL ≈ float (<1e-3).
- Per-seed init PPL identical (deterministic SVD init) and equal to noretrain r=24 PPL.
- Trainable param count == 19,200 asserted; everything else `requires_grad_(False)`.
- Per-seed best-state restore: assert `|restored - best| < 1e-3`.

## Verdict logic (item 298 verbatim)
- CONFIRMED iff rank-24 retrained best_mean ≤ 55.0.
- REFUTED iff rank-24 retrained best_mean ≥ 57.0.
- INCONCLUSIVE otherwise.

## Baselines to record
float 53.895; cycle 280 b1 r=2 = 49.0324 (2,304 params); cycle 282 b10 r=17 = 50.0445 (13,824 params); cycle 270 b11 r=22 = 54.4687 (17,664 params); cycle 285 b2 r=2 = 49.3290 (2,304 params); cycle 278 b3 c_proj OUTPUT eff_rank = 24.29, top-1 energy = 0.5887.
