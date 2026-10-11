# Cycle 266 Audit

**Status: PASS**

## Item
276 (P0): 2-block K=512 sweep — how far below float does the count-driven cascade go?

## Verdict
CONFIRMED — min over 3 subsets best_mean = 43.852 ≤ 45.0 (PASS by 1.15 PPL).

## Verification

### Plan adherence
- Plan: clone cycle 255 template with K=512 retained, SUBSETS swapped to [[1,4], [1,7], [4,7]], 3 seeds, 4000 steps, 9 fits. PASS
- Script: `experiments/two_block_sweep_cycle266.py` is verbatim cycle 255 template with SUBSETS swap, updated output paths, updated baselines (cycle 210 K=512 3-block 46.556, cycle 255 K=512 4-block min 49.609, cycle 245/221 reference), updated verdict thresholds (45.0/47.0 from agenda item 276). PASS
- Hyperparameters: K=512, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024]. All match cycle 255. PASS
- Disjoint splits: calib=all_ids[:200], eval=all_ids[200:300]. PASS
- Best-state restore + assertion check: confirmed in run log (`Restored best (step ...): X.XXX (recorded best=X.XXX)` for all 9 fits, max diff < 1e-3). PASS

### Run completion
- 9/9 fits completed in 22.2 min wall (under 27 min budget, no seed drop). PASS
- Init PPL deterministic per subset across seeds: [1,4]→57.057 (3/3 identical to 4 dp), [1,7]→58.018 (3/3 identical), [4,7]→59.777 (3/3 identical). PASS
- All 9 fits logged to `autoresearch/logs/cycle_266_run.log`. PASS
- Float baseline measured during run = 53.895, matches reference. PASS

### Numerical consistency
Per-seed best_ppl recovered from JSON match the manifest:
- [1,4]: [43.8658, 43.8771, 43.8145] → mean 43.8525 ± 0.0273 (PASS — std computed with unbiased=False as in cycle 255 template)
- [1,7]: [43.9460, 43.8333, 43.9352] → mean 43.9048 ± 0.0508 (PASS)
- [4,7]: [45.7520, 45.7673, 45.7917] → mean 45.7703 ± 0.0164 (PASS)
- min over subsets = 43.8525 ([1,4]). PASS
- max over subsets = 45.7703 ([4,7]). PASS
- Trainable params per config = 787,968 = 2 × (512×768 + 768) = 2 × 393,984 — matches expected formula. PASS

### Verdict logic
- Per agenda item 276: CONFIRMED if min ≤ 45.0; REFUTED if all ≥ 47.0; else INCONCLUSIVE.
- min = 43.852 → ≤ 45.0 → CONFIRMED PASS by 1.15 PPL. PASS
- max = 45.770 → not all ≥ 47.0 → REFUTED FAIL (correctly skipped). PASS
- → CONFIRMED. PASS

### Baseline cross-checks
- Cycle 210 K=512 3-block [1,4,7] = 46.556 — matches cycle 255 script's `CYCLE_210_3B_BASELINE`. PASS
- Cycle 255 K=512 4-block min [1,2,4,7] = 49.609 — matches `cycle_255_four_block_sweep.json` aggregate min. PASS
- Cycle 245 K=512 5-block = 53.826 — matches reference. PASS
- Cycle 221 K=512 6-block = 59.515 — matches reference. PASS
- Float baseline 53.895 — measured during cycle 266 run, matches cycle 255/265/210 reference. PASS
- Delta vs c210: 43.852 − 46.556 = −2.704. PASS
- Delta vs c255: 43.852 − 49.609 = −5.757. PASS
- Delta vs float: 43.852 − 53.895 = −10.042. PASS

### Cascade extrapolation sanity
- 6b 59.515 → 5b 53.826: −5.689 PPL
- 5b 53.826 → 4b 49.609: −4.217 PPL
- 4b 49.609 → 3b 46.556: −3.053 PPL
- 3b 46.556 → 2b 43.852: −2.704 PPL
- Marginal savings monotonically shrinking: TRUE (5.69 > 4.22 > 3.05 > 2.70). Cascade is concave. PASS
- Brief predicted "~43-44 PPL ≥10 PPL below float"; actual 43.85 with −10.04 PPL margin — prediction quantitatively correct on both axes. PASS

### Subset ordering
- [1,4] (43.85) < [1,7] (43.90) < [4,7] (45.77).
- Item 276 hypothesised "if cycle 255 ranking generalises, [1,4] should win" — TRUE.
- [1,4] vs [1,7] gap = 0.05 PPL, within 1σ (max σ across subsets = 0.051 for [1,7]) — statistically tied.
- [4,7] is 1.92 PPL worse than [1,4] and 1.87 worse than [1,7] — block 1 absence is the discriminating factor. PASS

### Pareto frontier
- (788K params, 43.85 PPL) cycle 266 K=512 2-block [1,4] — NEW lowest-PPL Pareto winner.
- (1.18M params, 46.56 PPL) cycle 210 K=512 3-block [1,4,7] — DOMINATED by cycle 266 (more params, worse PPL). PASS
- (1.58M params, 49.61 PPL) cycle 255 K=512 4-block [1,2,4,7] — DOMINATED by cycle 266 (more params, worse PPL). PASS
- (297K params, 48.99 PPL) cycle 223 K=128 3-block [1,4,7] — still Pareto winner on the low-param axis (cycle 266 has more params but lower PPL, so neither dominates the other). PASS

### Bookkeeping
- agenda.md: item 276 deleted from NEXT/P0 (P0 now starts at 277). PASS
- agenda.md: cycle 266 entry added to top of PAST EXPERIMENTS, with verdict CONFIRMED, full cascade / Pareto reading, and artifacts list. PASS
- TSV row appended to `autoresearch/logs/experiments.tsv` with hash c1ee079. PASS
- Manifest written: `autoresearch/results/cycle_266_manifest.txt` with all per-subset/per-seed values, baselines, file SHA256s, commit hash. PASS
- Two commits: `c1ee079` (cycle files) + `ef83384` (bookkeeping). PASS
- exec_result written: `autoresearch/results/cycle_266_exec_result.txt`. PASS

### Cross-K block 1 dominance
- Cycle 255 K=512 4-block: best subset includes block 1.
- Cycle 265 K=128 4-block: best subset includes block 1.
- Cycle 210 K=512 3-block: [1,4,7] (block 1).
- Cycle 223 K=128 3-block: [1,4,7] (block 1).
- Cycle 266 K=512 2-block: [1,4] and [1,7] both contain block 1; [4,7] (no block 1) is 1.9 PPL worse.
- Block 1 IS in every winning subset across both K levels and all explored block counts. PASS — confirmed pattern.

## Issues found
None. Verdict logic, numerical consistency, and bookkeeping all check out.

## Status
PASS
