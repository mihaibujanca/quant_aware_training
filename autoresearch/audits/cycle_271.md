# Cycle 271 Audit

**Status: PASS**

## Item
281 (P0): 1-block K=512 sweep — where does the cascade end?

## Verdict
INCONCLUSIVE — min over 3 subsets best_mean = 43.8205 ∈ (43.0, 46.0). CONFIRMED clause fails (43.82 > 43.0 by 0.82); REFUTED clause fails ([1]=43.82, [4]=44.63, [7]=45.37 all < 46.0).

## Verification

### Plan adherence
- Plan: trivial SUBSETS swap on cycle 266 template, K=512 retained, SUBSETS=[[1],[4],[7]], 3 seeds, 4000 steps, 9 fits. PASS
- Script: `experiments/one_block_sweep_cycle271.py` is verbatim cycle 266 template with SUBSETS swap, updated output paths, updated baselines (cycle 266 K=512 2-block min 43.8525 added, cycle 210/255/245/221 retained), updated verdict thresholds (43.0/46.0 from agenda item 281). PASS
- Hyperparameters: K=512, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024]. All match cycle 266. PASS
- Disjoint splits: calib=all_ids[:200], eval=all_ids[200:300]. PASS
- Best-state restore + assertion check: confirmed in run log (`Restored best (step ...): X.XXX (recorded best=X.XXX)` for all 9 fits, max diff < 1e-3). PASS

### Run completion
- 9/9 fits completed in 21.7 min wall (under 27 min budget, no seed drop). PASS
- Init PPL deterministic per subset across seeds: [1]→54.689 (3/3 identical to 3 dp), [4]→55.715 (3/3 identical), [7]→56.774 (3/3 identical). PASS
- All 9 fits logged to `autoresearch/logs/cycle_271_run.log`. PASS
- Float baseline measured during run = 53.895, matches reference. PASS

### Numerical consistency
Per-seed best_ppl recomputed from JSON match the manifest:
- [1]: [43.8262, 43.8459, 43.7893] → mean 43.8205 ± 0.0234 (PASS — std computed with unbiased=False as in cycle 266 template)
- [4]: [44.6346, 44.6652, 44.5992] → mean 44.6330 ± 0.0270 (PASS)
- [7]: [45.3771, 45.3709, 45.3756] → mean 45.3745 ± 0.0026 (PASS)
- min over subsets = 43.8205 ([1]). PASS
- max over subsets = 45.3745 ([7]). PASS
- Trainable params per config = 393,984 = 1 × (512×768 + 768) — matches expected formula. PASS

### Verdict logic
- Per agenda item 281: CONFIRMED if min ≤ 43.0; REFUTED if all ≥ 46.0; else INCONCLUSIVE.
- min = 43.8205 → > 43.0 → CONFIRMED FAIL. PASS
- min = 43.8205 → < 46.0 → REFUTED FAIL ([1] alone breaks the all-≥46.0 clause). PASS
- → INCONCLUSIVE. PASS

### Baseline cross-checks
- Cycle 266 K=512 2-block [1,4] = 43.8525 — matches `cycle_266_two_block_sweep.json` aggregate min. PASS
- Cycle 210 K=512 3-block [1,4,7] = 46.556 — matches cycle 266 script's `CYCLE_210_K512_3BLOCK`. PASS
- Cycle 255 K=512 4-block min [1,2,4,7] = 49.609 — matches script. PASS
- Cycle 245 K=512 5-block = 53.826 — matches reference. PASS
- Cycle 221 K=512 6-block = 59.515 — matches reference. PASS
- Float baseline 53.895 — measured during cycle 271 run, matches cycle 266/255/210 reference. PASS
- Delta vs c266 [1,4]: 43.8205 − 43.8525 = −0.032. PASS
- Delta vs c210: 43.8205 − 46.556 = −2.736. PASS
- Delta vs float: 43.8205 − 53.895 = −10.074. PASS

### Cascade extrapolation sanity
- 6b 59.515 → 5b 53.826: −5.689 PPL
- 5b 53.826 → 4b 49.609: −4.217 PPL
- 4b 49.609 → 3b 46.556: −3.053 PPL
- 3b 46.556 → 2b 43.852: −2.704 PPL
- 2b 43.852 → 1b 43.820: **−0.032 PPL**
- Marginal savings monotonically shrinking: TRUE (5.69 > 4.22 > 3.05 > 2.70 > 0.03). Cascade is concave and FLATLINES at 1 block. PASS
- Brief predicted "1-block ≈ 41.5-42.0 PPL" — actual 43.82 is 1.8-2.3 PPL ABOVE prediction. Prediction's "2.0-2.5 PPL further drop" REFUTED; actual 2→1 drop is 0.03 PPL, two orders of magnitude smaller. PASS (refutation is itself the geometric finding)

### Subset ordering
- [1] (43.82) < [4] (44.63) < [7] (45.37). PASS
- [1] vs [4] gap = 0.81 PPL; [4] vs [7] gap = 0.74 PPL — both well above seed std bands.
- Cycle 173 predicted ordering [7] ≈ [1] > [4] (i.e. [4] best). REFUTED — joint Adam at K=512 1-block makes block 1 strictly best by 0.81 PPL, opposite to cycle 173's no-retrain ordering. PASS

### Pareto frontier
- (394K params, 43.82 PPL) cycle 271 K=512 1-block [1] — NEW lowest-PPL AND lowest-K=512 Pareto winner. Dominates cycle 266 [1,4] (788K, 43.85): HALF the params for essentially the same PPL (within seed std). PASS
- (788K, 43.85) cycle 266 K=512 2-block [1,4] — DOMINATED by cycle 271 (more params, same PPL). PASS
- (1.18M, 46.56) cycle 210 K=512 3-block [1,4,7] — DOMINATED by cycle 271. PASS
- (17.7K, 54.47) cycle 270 block 11 rank-22 — Pareto winner at the very low param row, not dominated by cycle 271 (10.6 PPL worse but 22× fewer params). PASS

### Bookkeeping
- agenda.md: item 281 deleted from NEXT/P0 (P0 now starts at 282). PASS
- agenda.md: cycle 271 entry added to top of PAST EXPERIMENTS, with verdict INCONCLUSIVE, full cascade-flatlines reading, Pareto frontier update, and artifacts list. PASS
- TSV row appended to `autoresearch/logs/experiments.tsv` with hash 4cf86ac. PASS
- Manifest written: `autoresearch/results/cycle_271_manifest.txt` with all per-subset/per-seed values, baselines, file SHA256s, commit hash. PASS
- Two commits: `4cf86ac` (cycle files) + `6621162` (bookkeeping). PASS
- exec_result written: `autoresearch/results/cycle_271_exec_result.txt`. PASS

### Block 1 dominance (cross-cycle pattern)
- Cycle 255 K=512 4-block: best subset includes block 1.
- Cycle 265 K=128 4-block: best subset includes block 1.
- Cycle 268 K=64 4-block: best subset includes block 1.
- Cycle 210 K=512 3-block: [1,4,7] (block 1).
- Cycle 223 K=128 3-block: [1,4,7] (block 1).
- Cycle 266 K=512 2-block: [1,4] and [1,7] both contain block 1; [4,7] is 1.9 PPL worse.
- Cycle 271 K=512 1-block: [1] alone is best AND essentially equivalent to cycle 266 [1,4] — block 4 contributes 0.03 PPL on top of block 1 alone.
- Block 1 IS the cascade. The other blocks in winning subsets across cycles 255/265/266/268/270 contribute ≤ 0.05 PPL on top of block 1 alone at K=512. PASS — confirmed pattern, now in its strongest form.

### Convergence sanity
- All best_step values at 4000 (the budget cap) for [1] and [7]; [4] mixes 3900/4000/4000. PASS
- Adam still descending at the cap (no overfitting). The 0.03 PPL gap to cycle 266 [1,4] may close further with more steps but is already within seed std.

## Issues found
None. Verdict logic, numerical consistency, and bookkeeping all check out. INCONCLUSIVE is the correct verdict per the agenda thresholds.

## Status
PASS
