# Cycle 268 Audit

**Status: PASS**

## Item
278 (P1): K=64 4-block sweep — extend cycle 257's K=64 to 4 blocks.

## Verdict
CONFIRMED — min over 3 subsets of best_mean across 3 seeds = 55.9533 ([1,4,7,8]) ≤ 56.0 by 0.047 PPL margin.

## Verification

### Plan adherence
- Plan: `autoresearch/plans/cycle_268.md` — trivial K=128→K=64 swap on cycle 265 template. SUBSETS=[[1,2,4,7],[1,4,7,8],[1,2,7,8]], K=64, 3 seeds {42,137,2024}, hyperparams identical to cycle 265. Hypothesis verbatim from item 278. PASS
- Script: `experiments/four_block_k64_sweep_cycle268.py` — diffed against cycle 265: K constant flipped 128→64, OUTPUT_PATH/SUMMARY_PATH renamed, baseline constants swapped (CYCLE_257_K64_5BLOCK / CYCLE_265_K128_4BLOCK_MIN / CYCLE_255_K512_4BLOCK_MIN / CYCLE_230_K64_3BLOCK / FLOAT_BASELINE_REF), verdict thresholds updated (≤56.0 / ≥65.0), `cycle` and `item` fields updated to 268/278, baseline labels in delta/print/JSON/markdown blocks updated. Sweep loop, CompressedMLP, fit_block_lstsq, eval_ppl, snapshot/restore, hyperparams (LR/N_STEPS/EVAL_EVERY/N_CALIB/N_EVAL/SEQ_LEN/SEEDS/WALL_TIME_CAP_SEC) verbatim from cycle 265. PASS
- Hyperparameters: K=64, N_CALIB=200, N_EVAL=100 (disjoint), SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore, MPS, GPT-2 small, wikitext-2 val, SEEDS={42,137,2024}. Match item 278 brief and constraints. PASS
- Trainable params per config: 199,680 = 4 × (49,920 = 64×768 + 768). Brief estimate was 197,376 (= 4 × 49,344) — discrepancy is the bias term (768) which the brief computed as `49,344 = 64×768 + 192`? Actually 64×768 = 49,152 and +bias 768 = 49,920. The brief's 49,344 figure is off by 576, but trainable count is dominated by the 64×768 weight and the bookkeeping is correct in the script. Both sit at "≈ 200K". PASS (script-internal value reported in JSON is correct).

### Run completion
- Script executed with `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/four_block_k64_sweep_cycle268.py > autoresearch/logs/cycle_268_run.log 2>&1`. PASS
- Wall time: 20.9 min (cycle 257's K=64 5-block was ~13 min in 5 fits; 9 fits at 4 blocks scales linearly). Under 30 min cap, no seed drop. PASS
- All 9 fits completed. PASS
- Per-fit `assert abs(final_ppl_restored - best_ppl) < 1e-3` passed (in script; would have raised AssertionError otherwise — log shows clean exit and JSON saved). PASS
- Float baseline computed once: 53.895 (matches FLOAT_BASELINE_REF). PASS

### Numerical consistency
- Per-subset best_mean from JSON aggregates matches per-seed values:
  - [1,2,4,7]: mean(56.1251, 56.0600, 56.0174) = 56.0675; std (population) = 0.0443. PASS
  - [1,4,7,8]: mean(56.0171, 56.0199, 55.8228) = 55.9533; std = 0.0923. PASS
  - [1,2,7,8]: mean(57.3503, 57.3720, 57.1075) = 57.2766; std = 0.1199. PASS
- min_best_mean = 55.9533 = aggregates['1_4_7_8']['best_mean']. PASS
- max_best_mean = 57.2766 = aggregates['1_2_7_8']['best_mean']. PASS
- Init PPLs deterministic per subset (same model, same calib, no randomness in lstsq fit): all 3 seeds for [1,2,4,7] = 78.258, all 3 seeds for [1,4,7,8] = 78.873, all 3 seeds for [1,2,7,8] = 79.628. Confirms lstsq fit is seed-independent (only Adam initialisation differs across seeds via shuffle order). PASS
- best_step values cluster at 3800-4000 across seeds (best_step_mean ≈ 3900-3967) — Adam still descending at the budget cap, late-stop pattern matches cycles 255/265. PASS
- Per-seed best_ppl < init_ppl by ~22 PPL across all 9 fits — Adam recovery is uniform. PASS
- All best_ppl values within ±0.2 PPL of their subset mean — low cross-seed variance, no outlier. PASS

### Verdict logic
- Per item 278 (verbatim from agenda):
  - CONFIRMED if best_mean ≤ 56.0
  - REFUTED if best_mean ≥ 65.0
  - INCONCLUSIVE if best_mean ∈ (56, 65)
- Script-implemented logic operates on `min over subsets best_mean`: 55.9533 ≤ 56.0 → CONFIRMED. PASS
- REFUTED clause: `all(bm >= 65.0 for bm in best_means)` → max 57.28 < 65.0 → does NOT trigger (correct skip). PASS
- → CONFIRMED. PASS

### Cross-cycle comparison
- Delta vs cycle 257 K=64 5-block (62.181): -6.228 PPL — 5→4 block transition at K=64 saves the most of any K (cf. K=128 = -5.47, K=512 = -4.22). Marginal block-drop benefit GROWS as K shrinks. PASS
- Delta vs cycle 265 K=128 4-block min (53.647): +2.306 PPL — halving K from 128 to 64 at 4 blocks costs 2.31 PPL. PASS
- Delta vs cycle 255 K=512 4-block min (49.609): +6.344 PPL — total K=512→K=64 cost at 4 blocks is 6.34 PPL across two K-halvings (4.04 + 2.31 = 6.35, matches to rounding). PASS
- Delta vs cycle 230 K=64 3-block (50.781): +5.172 PPL — adding a 4th block at K=64 COSTS 5.17 PPL, mirroring cycle 265's finding that 3-block→4-block at K=128 was a +4.66 PPL cost. The 3-block recipe is the K=64 floor for fewer-block-counts. PASS
- Delta vs float (53.895): +2.058 PPL — does NOT beat float. The brief's "sub-300K-param recipe that beats float" prediction is REFUTED on the float-tie axis. PASS

### Subset ordering observation
- K=64 ordering: [1,4,7,8] (55.95) < [1,2,4,7] (56.07) < [1,2,7,8] (57.28).
- K=128 ordering (cycle 265): [1,2,4,7] (53.65) < [1,4,7,8] (53.73) < [1,2,7,8] (54.49).
- K=512 ordering (cycle 255): [1,2,4,7] (49.61) < [1,4,7,8] (50.71) < [1,2,7,8] (50.21). (Note: at K=512 the [1,2,7,8] vs [1,4,7,8] order is also slightly different from K=128.)
- The [1,2,4,7] vs [1,4,7,8] flip at K=64 is small (0.11 PPL ≈ 1.2 σ on the [1,4,7,8] std). Within noise but consistent across all 3 seeds: every [1,4,7,8] seed (56.017, 56.020, 55.823) is better than every [1,2,4,7] seed (56.125, 56.060, 56.017) except for the matched-pair [1,4,7,8]/seed42=56.017 vs [1,2,4,7]/seed2024=56.017. PASS
- [1,2,7,8] consistently last across all three K. block 4 is second most load-bearing after block 1. PASS

### Pareto frontier impact
- New low-param anchor: cycle 268 K=64 4-block [1,4,7,8] (200K, 55.95).
- DOMINATES cycle 251 K=128 5-block (495K, 59.12) — fewer params AND better PPL. PASS
- Does not dominate cycle 223 K=128 3-block (297K, 48.99) — 33% fewer params but 6.97 PPL worse — cycle 223 still Pareto. PASS
- The (200K, 55.95) point opens the LOW-param wing of the frontier. PASS

### Bookkeeping
- `autoresearch/plans/cycle_268.md` written (24 lines including the required hypothesis verbatim). PASS
- `autoresearch/briefs/cycle_268.md` already existed (read-only context). PASS
- `experiments/four_block_k64_sweep_cycle268.py` created fresh, K-swap of cycle 265 template. PASS
- `autoresearch/logs/cycle_268_run.log` created (full stdout including tqdm progress bars during model load). PASS
- `autoresearch/results/cycle_268_four_block_k64_sweep.json` contains cycle/item/verdict/config/float_ppl/subset_results (per-seed history)/aggregates/min/max/all baselines/all deltas/wall_time/fits_completed. PASS
- `autoresearch/results/cycle_268_summary.md` written. PASS
- `autoresearch/results/cycle_268_manifest.txt` written with verdict, commit hash, all per-subset numerical values, artifact list. PASS
- `autoresearch/results/cycle_268_exec_result.txt` written. PASS
- `autoresearch/agenda.md`: item 278 removed from P1 (P1 now has placeholder note); cycle 268 entry added at top of PAST EXPERIMENTS above cycle 267, with item 278 marked CONFIRMED, full reading + numerics + Pareto + open follow-ups. PASS
- TSV row appended to `autoresearch/logs/experiments.tsv` (line 266); hash field updated from `HASH` placeholder to `5664a91` after main commit. PASS
- Two commits: `5664a91` (cycle files: experiment script + plan + brief + log + JSON + summary + agenda) + `1afc8c4` (bookkeeping: TSV + manifest). PASS
- No pre-existing dirty files staged: verified via `git status --short` after first `git add` — only cycle 268 + agenda + TSV staged; pre-existing M files (program.md, results/cycle_170/205/255_manifest.txt, results/cycle_266_exec_result.txt, surveys/literature_map.md, logs/flagged.jsonl) untouched. PASS

### Pre-commit hook interaction
- First commit attempt failed because `fix end of files` hook added trailing newlines to `cycle_268_four_block_k64_sweep.json` and `cycle_268_summary.md`. Rule-compliant resolution: re-staged the modified files and re-ran `git commit` (NEW commit, not --amend), per CLAUDE.md git safety protocol. The second attempt passed all hooks. PASS

## Issues found
None substantive. One cosmetic note: brief's trainable param estimate (197,376) is 1.2% off from the actual (199,680) — the brief used 49,344 per block instead of 49,920 (= 64×768 + 768). Script-reported value is correct. Numerical consistency, plan adherence, verdict logic, and bookkeeping all check out.

## Status
PASS
