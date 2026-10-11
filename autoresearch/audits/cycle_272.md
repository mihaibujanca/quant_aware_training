# Cycle 272 Audit

**Status: PASS**

## Item
282 (P0): K=256 4-block sweep — fill the missing K-curve data point.

## Verdict
CONFIRMED — min over 3 subsets best_mean = 51.9224 ≤ 52.0 (passes by 0.078 PPL). REFUTED clause fails (max = 52.6762 < 54.0). Verdict per item 282 verbatim thresholds.

## Verification

### Plan adherence
- Plan: trivial K=128→K=256 swap on cycle 265 template, SUBSETS retained, 3 seeds, 4000 steps, 9 fits. PASS
- Script `experiments/four_block_k256_sweep_cycle272.py` is verbatim cycle 265 template with K=128→256 swap, updated output paths, updated baselines (cycle 255 K=512 4-block 49.609, cycle 265 K=128 4-block 53.647, cycle 223 K=256 3-block 47.509, cycle 256 K=256 6-block 64.181, float 53.895), updated verdict thresholds (52.0/54.0 from agenda item 282). PASS
- Hyperparameters: K=256, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024]. All match cycle 265. PASS
- Disjoint splits: calib=all_ids[:200], eval=all_ids[200:300]. PASS
- Best-state restore + assertion check: confirmed in run log (`Restored best (step ...): X.XXX (recorded best=X.XXX)` for all 9 fits, max diff < 1e-3). PASS

### Run completion
- 9/9 fits completed in 21.3 min wall (under 27 min budget, no seed drop). PASS
- Init PPL deterministic per subset across seeds: [1,2,4,7]→70.869 (3/3 identical to 3 dp), [1,4,7,8]→72.818 (3/3 identical), [1,2,7,8]→72.450 (3/3 identical, 6 dp matches within float precision). PASS
- All 9 fits logged to `autoresearch/logs/cycle_272_run.log`. PASS
- Float baseline measured during run = 53.895, matches reference. PASS

### Numerical consistency
Per-seed best_ppl recomputed from JSON match the manifest:
- [1,2,4,7]: [51.9505, 51.9072, 51.9095] → mean 51.9224 ± 0.0199 (PASS — std unbiased=False as in cycle 265 template)
- [1,4,7,8]: [52.5492, 52.4601, 52.4056] → mean 52.4716 ± 0.0592 (PASS)
- [1,2,7,8]: [52.7813, 52.6685, 52.5789] → mean 52.6762 ± 0.0828 (PASS)
- min over subsets = 51.9224 ([1,2,4,7]). PASS
- max over subsets = 52.6762 ([1,2,7,8]). PASS
- Trainable params per config = 789,504 = 4 × (256×768 + 768) — matches expected formula. PASS

### Verdict logic
- Per agenda item 282: CONFIRMED if min ≤ 52.0; REFUTED if all ≥ 54.0; else INCONCLUSIVE.
- min = 51.9224 → ≤ 52.0 → CONFIRMED PASS by 0.078 PPL. PASS
- max = 52.6762 → < 54.0 → REFUTED clause FAIL. PASS
- → CONFIRMED. PASS

### Baseline cross-checks
- Cycle 255 K=512 4-block min [1,2,4,7] = 49.609 — matches script CYCLE_255_K512_4BLOCK_MIN. PASS
- Cycle 265 K=128 4-block min [1,2,4,7] = 53.647 — matches script CYCLE_265_K128_4BLOCK_MIN. PASS
- Cycle 223 K=256 3-block [1,4,7] = 47.509 — matches script CYCLE_223_K256_3BLOCK. PASS
- Cycle 256 K=256 6-block = 64.181 — matches script CYCLE_256_K256_6BLOCK. PASS
- Float baseline 53.895 — measured during cycle 272 run, matches reference. PASS
- Delta vs c255 K=512 4b: 51.9224 − 49.609 = +2.313. PASS
- Delta vs c265 K=128 4b: 51.9224 − 53.647 = −1.725. PASS
- Delta vs c223 K=256 3b: 51.9224 − 47.509 = +4.413. PASS
- Delta vs float: 51.9224 − 53.895 = −1.972. PASS

### K-curve at 4 blocks
- K=64 → 55.953 (cycle 268)
- K=128 → 53.647 (cycle 265)
- **K=256 → 51.922 (cycle 272 — this run)**
- K=512 → 49.609 (cycle 255)
- Per-K-halving cost: 512→256 = 2.313, 256→128 = 1.725, 128→64 = 2.306. The curve is roughly linear in log K with a flat midstep at 256→128. NOT strictly concave-up (brief hypothesis was qualitatively right on the threshold but the per-step structure is non-monotone). The 4-block K-curve is now fully mapped. PASS

### Subset ordering
- [1,2,4,7] (51.92) < [1,4,7,8] (52.47) < [1,2,7,8] (52.68). PASS
- Same canonical ordering as cycle 255 (K=512) and cycle 265 (K=128). The cycle 268 (K=64) flip to [1,4,7,8] < [1,2,4,7] does not generalise to K=256 — the canonical [1,2,4,7] ordering returns. PASS
- [1,2,4,7] vs [1,4,7,8] gap = 0.55 PPL; [1,4,7,8] vs [1,2,7,8] gap = 0.20 PPL — both above seed std bands.

### Pareto frontier
- Cycle 272 K=256 4-block (789K, 51.92) is **NOT** a Pareto winner.
- Dominated by cycle 271 K=512 1-block [1] (394K, 43.82): cycle 271 is at HALF the params for 8.10 PPL better — strict dominance on both axes. PASS
- vs cycle 266 K=512 2-block [1,4] (788K, 43.85): essentially same param count, cycle 266 is 8.07 PPL better — strictly dominated. PASS
- vs cycle 255 K=512 4-block (1.58M, 49.61): cycle 272 has fewer params but worse PPL → both non-dominated relative to each other.
- vs cycle 223 K=128 3-block (297K, 48.99): cycle 223 has fewer params AND better PPL → cycle 272 dominated by cycle 223 too. PASS
- The brief's prediction "(~790K, ≤51.5) Pareto data point that may DOMINATE cycle 255 4-block K=512" is REFUTED on the dominance claim — cycle 272 is dominated by both cycle 271 and cycle 223. The K-curve fill remains the headline finding. PASS

### Convergence sanity
- best_step distribution: subset [1,2,4,7] mean 3633 ± 170, [1,4,7,8] mean 3233 ± 368, [1,2,7,8] mean 3100 ± 294. Best_step well below 4000 cap on most fits — Adam has converged or is near-converged at K=256 (faster than K=128 which had best_step ≈3700-3900). PASS
- final@4000 vs best_mean: subset [1,2,4,7] final 52.07 vs best 51.92 (overshoots by 0.15); [1,4,7,8] final 52.78 vs best 52.47 (overshoots by 0.31); [1,2,7,8] final 53.05 vs best 52.68 (overshoots by 0.37). All three subsets show mild post-best overshoot — consistent with Adam past its minimum at the budget cap. Best-state restore is doing useful work. PASS

### Bookkeeping
- agenda.md: item 282 deleted from NEXT/P0 (P0 now empty between header and P1). PASS
- agenda.md: cycle 272 entry added to top of PAST EXPERIMENTS, with verdict CONFIRMED, full K-curve reading, Pareto frontier update, and artifacts list. PASS
- TSV row appended to `autoresearch/logs/experiments.tsv` with hash a61fdf6. PASS
- Manifest written: `autoresearch/results/cycle_272_manifest.txt` with all per-subset/per-seed values, baselines, file SHA256s, commit hash a61fdf6. PASS
- Two commits: `a61fdf6` (cycle files) + `861ed18` (bookkeeping: manifest + TSV hash + exec_result). PASS
- exec_result written: `autoresearch/results/cycle_272_exec_result.txt`. PASS
- Pre-commit end-of-file fixer modified JSON and summary on first commit; manifest hashes were updated in the bookkeeping commit to match the post-fix file contents (a3777d3.../1578ee6...). PASS

### Block 1 dominance (cross-cycle pattern)
- Cycle 272 [1,2,4,7] is the best 4-block subset at K=256. Block 1 included. PASS
- Same block-1 inclusion pattern as cycles 255 (K=512), 265 (K=128), 268 (K=64): every winning 4-block subset across all four K values includes block 1. PASS

## Issues found
None. Verdict logic, numerical consistency, K-curve fit, Pareto analysis, and bookkeeping all check out. CONFIRMED is the correct verdict per the agenda thresholds.

## Status
PASS
