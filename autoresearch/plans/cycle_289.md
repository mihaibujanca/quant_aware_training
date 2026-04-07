# Cycle 289 Plan — Item 299

## Dependencies
None (cycle 282 template + cycle 278 eff_rank census already on disk).

## Item (verbatim)
299 (P2): Block 10 c_proj rank-5 weight-SVD — can a sub-eff_rank patch still beat float? Cycle 282 confirmed block 10 rank-17 retrains to 50.04 PPL at 13.8K trainable params, below float by 3.85 PPL (eff_rank=16.70). Cycle 282 also tested r=5 no-retrain (59.92) and r=50 no-retrain (58.27); none beat retrained r=17. Untested: does rank-5 retrain (r << eff_rank=16.70) still beat float, mirroring cycle 280 b1 r=2? Block 10 r=5 tests "rank << eff_rank, with retrain" — does Adam absorb the missing 12 directions, or fail to converge below float? Params: 5×768 + 768 = 4,608 — between cycle 280 (2.3K) and cycle 282 (13.8K).

## Approach
Adapt `experiments/block10_lowrank_cycle282.py` verbatim with `RANKS_NORETRAIN=[5]`, `RETRAIN_RANK=5`, thresholds from item 299. Stage A: no-retrain rank-5 swap (re-verify cycle 282's 59.92). Stage B: retrain r=5, joint Adam over (B, bias) = (5,768)+(768) = 4,608 trainable params, 3 seeds {42,137,2024}, best-state restore.

## Files
- `experiments/block10_rank5_cycle289.py` (new)
- `autoresearch/results/cycle_289_block10_rank5.json`, `_summary.md`, `_manifest.txt`, `_exec_result.txt`
- `autoresearch/logs/cycle_289_run.log`

## Config
TARGET_BLOCK=10; RANKS_NORETRAIN=[5]; RETRAIN_RANK=5; LR=1e-4; N_STEPS=4000; EVAL_EVERY=100; SEEDS=[42,137,2024]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; MPS; GPT-2 small; wikitext-2 val. Trainable per fit = 5*768 + 768 = 4,608. Wall ~7 min.

## Verification
- Float PPL within 0.5 of 53.895.
- SVD shapes: U(3072,768), S(768,), Vh(768,768); print top-5 SVs + energy at r=5.
- Restore original c_proj.weight between sweep stages; assert restore PPL ≈ float (<1e-3).
- Per-seed init PPL identical (deterministic SVD init) and equal to noretrain r=5 PPL.
- Trainable param count == 4,608 asserted; everything else `requires_grad_(False)`.
- Per-seed best-state restore: assert `|restored - best| < 1e-3`.
- Cross-check noretrain r=5 PPL against cycle 282 (59.92 reference) within 0.01.

## Verdict logic (item 299 verbatim)
- CONFIRMED iff rank-5 retrained best_mean ≤ 53.0 (sub-eff_rank still beats float).
- REFUTED iff rank-5 retrained best_mean ≥ 56.0 (rank << eff_rank breaks block 10).
- INCONCLUSIVE otherwise.

## Baselines to record
float 53.895; cycle 280 b1 r=2 = 49.0324 (2,304 params); cycle 282 b10 r=17 = 50.0445 (13,824 params); cycle 282 b10 r=5 no-retrain = 59.92; cycle 278 b10 c_proj OUTPUT eff_rank = 16.70, top-1 energy = 0.5889.
