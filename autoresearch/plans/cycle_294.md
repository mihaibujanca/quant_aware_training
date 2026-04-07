# Cycle 294 Plan — Item 304

## Dependencies
None. Trivial budget extension of `experiments/one_block_k_sweep_cycle276.py` (cycle 276 4k baseline already on disk).

## Item (verbatim Hypothesis)
304 (P2): 1-block [1] K sweep extended budget — 8000 steps for K∈{32, 64, 128, 256}. Cycle 276 swept [1] K∈{32, 64, 128, 256} at N_STEPS=4000 and CONFIRMED K=256→45.385, K=128→46.932, K=64→48.045, K=32→49.649. All 12 fits were budget-bounded (best_step at or near 4000 cap). Cycle 287 extended [1] K=512 from 4k to 8k and bought −1.04 PPL. Hypothesis: 8k extension shifts each K-tier down by 0.5–1.5 PPL, potentially turning some Pareto-dominated points into new winners (especially K=32 at the lowest-param tier where Adam slack is largest).

## Approach
Adapt `experiments/one_block_k_sweep_cycle276.py` verbatim. Only changes: N_STEPS=4000→8000, EVAL_EVERY=100→200, output paths, baseline constants, verdict thresholds (item 304 verbatim), wall-time cap raised to 60 min. SUBSETS=[[1]] unchanged. K_VALUES=[32,64,128,256] unchanged. Joint Adam (W_out, b_out) per K, 3 seeds {42,137,2024}, best-state restore.

## Files
- `experiments/one_block_1_k_sweep_8k_cycle294.py` (new)
- `autoresearch/results/cycle_294_one_block_1_k_sweep_8k.json`, `_summary.md`, `_manifest.txt`, `_exec_result.txt`
- `autoresearch/logs/cycle_294_run.log`

## Config
SUBSETS=[[1]]; K_VALUES=[32,64,128,256]; LR=1e-4; N_STEPS=8000; EVAL_EVERY=200; SEEDS=[42,137,2024]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; MPS; GPT-2 small; wikitext-2 val. Trainable per fit = K×768+768 = {25344, 49920, 99072, 197376}. Wall ~28 min (4 K × 3 seeds × 2× cycle 276 per-K).

## Verification
- Float PPL within 0.5 of 53.895.
- Per-(K,seed) init PPL deterministic, must match cycle 276 within 1e-4.
- Step-4000 evals must match cycle 276 best_ppl per (K,seed) cell within 1e-4 (no init/LR drift).
- Trainable param count asserted == K×768+768 per fit; everything else `requires_grad_(False)`.
- Per-seed best-state restore: assert `|restored − best| < 1e-3`.

## Verdict logic (item 304 verbatim)
- CONFIRMED iff at least 2 of 4 K-tiers shift by ≥0.5 PPL (4k best_mean − 8k best_mean ≥ 0.5).
- REFUTED iff all 4 K-tiers shift by ≤0.1 PPL.
- INCONCLUSIVE otherwise.

## Baselines to record
float 53.895; cycle 276 [1] K∈{32,64,128,256} 4k best_means {49.6485, 48.0453, 46.9316, 45.3854}; cycle 287 [1] K=512 8k = 42.7853 (393,984 params); cycle 271 [1] K=512 4k = 43.8205.
