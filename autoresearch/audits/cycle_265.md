# Cycle 265 Audit

**Status: PASS**

## Item
275 (P0): K=128 4-block sweep — extend the K=128 curve, find the Pareto winner.

## Verdict
INCONCLUSIVE — min over 3 subsets best_mean = 53.647 sits in the (53.0, 56.0) inconclusive band.

## Verification

### Plan adherence
- Plan: clone cycle 255 template with K=512→K=128, SUBSETS=[[1,2,4,7], [1,4,7,8], [1,2,7,8]], 3 seeds, 4000 steps, 9 fits. ✔
- Script: `experiments/four_block_k128_sweep_cycle265.py` is verbatim cycle 255 template with K=128, updated output paths, updated baseline constants (cycle 251 K=128 5-block 59.119, cycle 223 K=128 3-block 48.989), updated verdict thresholds (53.0/56.0 from agenda item 275). ✔
- Hyperparameters: K=128, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024]. All match cycle 255 except K. ✔
- Disjoint splits: calib=all_ids[:200], eval=all_ids[200:300]. ✔
- Best-state restore + assertion check: confirmed in run log (`Restored best (step ...): X.XXX (recorded best=X.XXX)` for all 9 fits, max diff < 1e-3). ✔

### Run completion
- 9/9 fits completed in 21.0 min wall (under 27 min budget, no seed drop). ✔
- Init PPL deterministic per subset: [1,2,4,7]→74.216, [1,4,7,8]→76.107, [1,2,7,8]→75.862. ✔
- All 9 fits logged to `autoresearch/logs/cycle_265_run.log`. ✔

### Numerical consistency
Per-seed best_ppl recovered from JSON match the manifest:
- [1,2,4,7]: [53.6948, 53.6926, 53.5533] → mean 53.6469 ± 0.0656
- [1,4,7,8]: [53.7708, 53.8272, 53.5976] → mean 53.7319 ± 0.0976
- [1,2,7,8]: [54.5944, 54.5747, 54.2941] → mean 54.4877 ± 0.1366
- min over subsets = 53.6469 ([1,2,4,7]). ✔
- max over subsets = 54.4877 ([1,2,7,8]). ✔
- Trainable params per config = 396,288 = 4 × (128×768 + 768) = 4 × 98,688. ✔

### Verdict logic
- Per agenda item 275: CONFIRMED if min ≤ 53.0; REFUTED if all ≥ 56.0; else INCONCLUSIVE.
- min = 53.647 → not ≤ 53.0 → CONFIRMED FAIL by 0.647 PPL. ✔
- max = 54.488 → not all ≥ 56.0 → REFUTED FAIL. ✔
- → INCONCLUSIVE. ✔

### Baseline cross-checks
- Cycle 255 K=512 4-block min [1,2,4,7] = 49.609 — matches `cycle_255_four_block_sweep.json`. ✔
- Cycle 251 K=128 5-block = 59.119 — matches `cycle_251_five_block_k128.json` aggregate.best_mean = 59.1192. ✔
- Cycle 223 K=128 3-block [1,4,7] = 48.989 — matches `cycle_223_lower_k_adam.json` K=128 entry aggregate.best_mean = 48.9889. ✔
- Float baseline 53.895 — measured during cycle 265 run, matches cycle 255/251/223 reference. ✔
- Delta vs c255 min: 53.647 − 49.609 = +4.038 ✔
- Delta vs c251: 53.647 − 59.119 = −5.472 ✔
- Delta vs c223: 53.647 − 48.989 = +4.658 ✔
- Delta vs float: 53.647 − 53.895 = −0.248 ✔

### Bookkeeping
- agenda.md: item 275 deleted from NEXT/P0 (P0 now starts at 276). ✔
- agenda.md: cycle 265 entry added to top of PAST EXPERIMENTS, with verdict INCONCLUSIVE, full geometric / Pareto reading, and artifacts list. ✔
- TSV row appended to `autoresearch/logs/experiments.tsv` with hash 02a5936. ✔
- Manifest written: `autoresearch/results/cycle_265_manifest.txt` with all per-subset/per-seed values, baselines, file SHA256s, commit hash. ✔
- Two commits: `02a5936` (cycle files) + `510a7db` (bookkeeping). ✔
- exec_result written: `autoresearch/results/cycle_265_exec_result.txt`. ✔

### Pareto frontier sanity
- (297K params, 48.99 PPL) cycle 223 K=128 3-block [1,4,7] — Pareto winner low params.
- (1.58M params, 49.61 PPL) cycle 255 K=512 4-block [1,2,4,7] — Pareto winner low PPL.
- (396K params, 53.65 PPL) cycle 265 K=128 4-block [1,2,4,7] — DOMINATED by cycle 223 (more params, worse PPL). ✔
- The naive linear K=128 extrapolation 6b→5b→4b (67.56 → 59.12 → predicted 50.7) failed by 2.95 PPL — the K=128 cascade is sub-linear at low block counts. ✔

### Subset ordering generalises across K
- Cycle 255 K=512: [1,2,4,7]=49.61 < [1,2,7,8]=50.21 < [1,4,7,8]=50.71
- Cycle 265 K=128: [1,2,4,7]=53.65 < [1,4,7,8]=53.73 < [1,2,7,8]=54.49
- The [1,2,4,7] minimum is preserved across K but the [1,2,7,8] vs [1,4,7,8] order swaps — at K=128 [1,4,7,8] is better than [1,2,7,8] (opposite of K=512). Recorded; minor.

## Issues found
None. Verdict logic, numerical consistency, and bookkeeping all check out.

## Status
PASS
