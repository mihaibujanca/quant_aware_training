# Cycle 290 Plan — Item 300

## Dependencies
None (cycle 287 template + cycle 286 [2] K=512 4k baseline already on disk).

## Item (verbatim)
300 (P0): 1-block [2] K=512 extended budget — 8000 steps — does block 2 cross [1] at matched 8k budget? Cycle 286 measured [2] K=512 best_mean=43.5627 ± 0.0258 at N_STEPS=4000 (393,984 trainable), all 3 seeds best at step 4000 — budget-bounded, still descending. Cycle 286 was the best 1-block K=512 result at 4k by 0.258 PPL over cycle 271 [1] K=512=43.821, but cycle 287 then extended [1] K=512 to 8000 steps and dropped it to 42.7853 ± 0.0125 (−1.04 PPL pure budget gain), pushing [1] back into the lead by 0.78 PPL. The [2]-vs-[1] comparison at matched 8k budget is therefore unsettled — cycle 286 is dominated by cycle 287 only because cycle 286 was capped at 4k. Cycle 283's per-K [2]-beats-[1] trend at K∈{16,32,64} suggests block 2 should still win at K=512 IF given the same Adam budget. Hypothesis: at 8k steps, [2] K=512 drops below 42.0 (matches or beats cycle 287 [1] K=512 by ≥0.78 PPL), establishing block 2 as the dominant 1-block K=512 target and the new PPL Pareto winner at 394K params.

## Approach
Adapt `experiments/one_block_1_k512_8k_cycle287.py` verbatim — change SUBSETS=[[1]] to SUBSETS=[[2]], output paths, baseline constants, verdict thresholds. No other changes. Joint Adam over (W_out, b_out)=393,984 params per fit, 3 seeds {42,137,2024}, best-state restore.

## Files
- `experiments/block2_k512_8k_cycle290.py` (new)
- `autoresearch/results/cycle_290_block2_k512_8k.json`, `_summary.md`, `_manifest.txt`, `_exec_result.txt`
- `autoresearch/logs/cycle_290_run.log`

## Config
SUBSETS=[[2]]; K=512; LR=1e-4; N_STEPS=8000; EVAL_EVERY=200; SEEDS=[42,137,2024]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; MPS; GPT-2 small; wikitext-2 val. Trainable per fit = 512×768 + 768 = 393,984. Wall ~13 min (matches cycle 287).

## Verification
- Float PPL within 0.5 of 53.895.
- Per-seed init PPL 55.46118 (deterministic, must match cycle 286 within 1e-4).
- Step-4000 evals must match cycle 286 per seed within 1e-4 (seed 42=43.53027, seed 137=43.59336, seed 2024=43.56444) — confirms only the budget extension differs.
- Trainable param count == 393,984 asserted; everything else `requires_grad_(False)`.
- Per-seed best-state restore: assert `|restored - best| < 1e-3`.

## Verdict logic (item 300 verbatim)
- CONFIRMED iff best_mean ≤ 42.5 (block 2 beats cycle 287 [1] K=512 8k=42.785 by ≥0.29 PPL — new dominant PPL Pareto winner at 394K params).
- REFUTED iff best_mean ≥ 43.0 (block 2 buys < 0.56 PPL from extension, [1]-vs-[2] gap closes but [1] still wins at matched 8k).
- INCONCLUSIVE otherwise (dead zone (42.5, 43.0) — both still useful but no clear winner).

## Baselines to record
float 53.895; cycle 286 [2] K=512 4k = 43.56269 (393,984 params); cycle 287 [1] K=512 8k = 42.7853 (393,984 params, current PPL Pareto winner); cycle 271 [1] K=512 4k = 43.82045.
