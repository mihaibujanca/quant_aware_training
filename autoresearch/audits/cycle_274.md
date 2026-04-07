# Cycle 274 Audit

**Status: PASS**

## Item
284 (P2): K=64 5-block extended budget (8000 steps) — does cycle 257 plateau? BLOCKS=[1,2,4,7,8] K=64, N_STEPS=8000, EVAL_EVERY=200, 3 seeds.

## Verdict
INCONCLUSIVE — best_mean = 61.37696 ± 0.04625 falls in the (60.7, 61.5) gap. CONFIRMED clause (best_mean ∈ [61.5, 62.5]) FAILS by 0.123 PPL on the low side. REFUTED clause (best_mean ≤ 60.7) FAILS by 0.677 PPL. Per item 284 verbatim thresholds.

## Verification

### Plan adherence
- Plan: clone `experiments/five_block_k64_cycle257.py` → `experiments/five_block_k64_8k_cycle274.py`. BLOCKS=[1,2,4,7,8], K=64, joint Adam W_out+b_out, LR=1e-4, **N_STEPS=8000**, **EVAL_EVERY=200**, SEEDS=[42,137,2024], best-state restore, disjoint N_CALIB=200/N_EVAL=100, SEQ_LEN=128. PASS
- Script sets exactly these constants. Diff vs cycle 257: N_STEPS 4000→8000, EVAL_EVERY 100→200, OUTPUT_PATH/SUMMARY_PATH renamed, baselines updated, `final_ppl_at_step_4000` → `final_ppl_at_step_8000` (consistent across script + JSON), verdict logic updated to item 284 thresholds. PASS
- All 3 seeds completed; no seed drop. PASS

### Numerical consistency
- Per-seed best from JSON: [61.4234, 61.3139, 61.3936] → mean (61.4234+61.3139+61.3936)/3 = 184.1309/3 = 61.37697. JSON aggregate.best_mean = 61.37696. Match within rounding. PASS
- Per-seed std: vals = [61.4234, 61.3139, 61.3936], var = ((61.4234-61.37697)^2 + (61.3139-61.37697)^2 + (61.3936-61.37697)^2)/3 = (0.04643^2 + (-0.06307)^2 + 0.01663^2)/3 = (0.002156 + 0.003978 + 0.000277)/3 = 0.006411/3 = 0.002137. std = sqrt(0.002137) = 0.04623. JSON aggregate.best_std = 0.04625. Match within rounding (population std, unbiased=False). PASS
- Per-seed best_step: [6200, 6200, 7400] → mean = 19800/3 = 6600. PASS
- Per-seed best_step std: ((6200-6600)^2 + (6200-6600)^2 + (7400-6600)^2)/3 = (160000+160000+640000)/3 = 320000. sqrt = 565.685. Match. PASS
- Per-seed init: 95.62036... all 3 identical (lstsq init is seed-independent). Matches cycle 257 init 95.620 to 3 dp. PASS
- All 3 best_step in [6200, 7400], well into the second 4k of training. None at the 8000 cap. PASS

### Verdict logic
- Per item 284 verbatim: CONFIRMED iff best_mean ∈ [61.5, 62.5]; REFUTED iff best_mean ≤ 60.7; else INCONCLUSIVE.
- best_mean = 61.37696. 61.37696 < 61.5 → NOT in CONFIRMED band → CONFIRMED clause FAIL by (61.5 − 61.37696) = 0.123 PPL. PASS
- 61.37696 > 60.7 → NOT ≤ 60.7 → REFUTED clause FAIL by (61.37696 − 60.7) = 0.677 PPL. PASS
- → INCONCLUSIVE. PASS
- The 0.123 PPL miss on the low side is unambiguous: cycle 274 best_mean is below the plateau band, indicating real descent (not a 4k plateau), but not enough descent to clear the REFUTED bar.

### Best-state restore
- Per-seed asserts in run log: "Restored best (step 6200): 61.423 (recorded best=61.423)" for all 3 seeds. The script asserts |restored - recorded| < 1e-3 — passes for all 3 fits. PASS

### Trainable count
- Script-internal `n_trainable = sum(p.numel() for p in trainable_params)` reports 249,600 = 5 × (64×768 + 768) = 5 × (49152 + 768) = 5 × 49920 = 249,600. PASS
- Matches plan/review expected value. (Cycle 257 manifest reported 246,720, which appears to have been a typo; the actual cycle 257 trainable count by the same formula should also be 249,600. Brief value 249,600 was correct.)

### Baseline cross-checks
- cycle 257 K=64 5-block 4k = 62.181 — matches script CYCLE_257_K64_4K_BASELINE. PASS
- cycle 254 K=128 5-block 8k = 59.018 — matches script CYCLE_254_K128_8K_BASELINE. PASS
- cycle 251 K=128 5-block 4k = 59.119 — matches script. PASS
- cycle 245 K=512 5-block = 53.826 — matches script. PASS
- cycle 269 K=1 6-block 16k = 91.6126 — matches script. PASS
- Float baseline 53.895 — measured during cycle 274 run, matches reference. PASS
- Δ best_mean vs cycle 257 4k: 61.37696 − 62.181 = **−0.804**. PASS
- Δ vs cycle 254 K=128 8k: 61.37696 − 59.018 = **+2.359**. PASS
- Δ vs cycle 245 K=512 5b: 61.37696 − 53.826 = **+7.551**. PASS
- Δ vs float: 61.37696 − 53.895 = **+7.482**. PASS

### Plateau-vs-descent comparison
- Cycle 254 K=128 5b 8k extension: best_mean drop = 59.018 − 59.119 = −0.101 PPL (4k → 8k). Cycle 254 trajectory: best at step 4400-5200, then climb back to ~60.5 at step 8000 (+1.5 PPL plateau-then-climb pattern).
- Cycle 274 K=64 5b 8k extension: best_mean drop = 61.377 − 62.181 = −0.804 PPL (4k → 8k). Cycle 274 trajectory: best at step 6200-7400 (none at 8000), final@8000 mean 61.550 (+0.173 above best, mild overshoot — NOT the +1.5 PPL plateau-then-climb of cycle 254).
- Cycle 274 8k descent (0.804 PPL) is **8× the cycle 254 8k descent (0.101 PPL)** at the same hyperparam template, only K differs. Cycle 274 best_step shifts later (6600 ± 566 vs cycle 254's ~4800). The K=64 surface has more uncrowded W_out parameter space than K=128, and Adam continues meaningful descent through ~step 7000 before mild overshoot. This is a real qualitative difference between K=64 and K=128 at 5 blocks.
- Cycle 274 is also distinct from cycle 269 (K=1 6-block 16k) which gained 5.27 PPL over 4× the budget — cycle 274 gained 0.804 PPL over 2× the budget, much smaller in absolute and relative terms. The K=64 5-block surface is in a slow-descent regime intermediate between K=128 5b plateau and K=1 6b extended-descent. PASS

### Pareto check
- Cycle 274 K=64 5-block 8k at (249,600 params, 61.377 PPL) is dominated by:
  - cycle 271 K=512 1-block [1] (393,984 params, 43.821 PPL): 1.6× the params for 17.6 PPL better — NOT a Pareto-domination on params (cycle 274 has fewer params), but cycle 271 dominates on PPL by 17.6. Both are on the Pareto frontier in the sense that neither strictly dominates the other on both axes.
  - Actually checking strict dominance: cycle 271 has 393,984 > 249,600 (cycle 274) → cycle 274 has FEWER params; cycle 271 has 43.821 < 61.377 → cycle 271 has BETTER PPL. Neither strictly dominates the other on both axes. Cycle 274 sits on the low-param-high-PPL row but is dominated by:
  - cycle 268 K=64 4-block [1,4,7,8] (199,680 params, 55.953 PPL): 80% the params for 5.42 PPL better — strict dominance on BOTH axes (fewer params AND better PPL). Cycle 274 is strictly Pareto-dominated by cycle 268. PASS
- Cycle 274 does NOT enter the Pareto frontier. PASS

### Overshoot check
- Final@8000 mean = 61.550 PPL, +0.173 above best_mean 61.377 — mild overshoot, all 3 seeds individually show final@8000 ≥ best (61.6516 vs 61.4234 for seed 42, +0.228; 61.3733 vs 61.3139 for seed 137, +0.059; 61.6263 vs 61.3936 for seed 2024, +0.233). The overshoot is mild and matches a "still-descending-but-fluctuating" pattern, NOT a clean plateau. PASS

### Bookkeeping
- agenda.md: item 284 deleted from NEXT/P2 (P2 section now empty between header and "### Closed"). PASS
- agenda.md: "### Cycle 274" block inserted at top of PAST EXPERIMENTS (above "### Cycle 273") with verdict INCONCLUSIVE, full plateau-vs-descent reading, Pareto check, and artifacts list. PASS
- TSV row appended to `autoresearch/logs/experiments.tsv` with cycle=274 item=284 outcome=inconclusive commit=bcb8803. (Initial append had a malformed first attempt that left a corrupted row; the row was rebuilt cleanly with 8 columns; no other rows affected.) PASS
- Manifest written: `autoresearch/results/cycle_274_manifest.txt` with all per-seed values, baselines, file SHA256s, commit=bcb8803. PASS
- exec_result written: `autoresearch/results/cycle_274_exec_result.txt` with commit=bcb8803. PASS
- Primary commit `bcb8803` with message `[autoresearch] cycle 274: K=64 5-block 8000 steps — INCONCLUSIVE`. PASS

## Issues found
None. Verdict logic, numerical consistency, plateau-vs-descent analysis, Pareto check, and bookkeeping all check out. INCONCLUSIVE is the correct verdict per the agenda thresholds — best_mean is 0.123 PPL below the CONFIRMED low band (61.5) and 0.677 PPL above the REFUTED threshold (60.7). The 8k extension at K=64 5-block bought meaningful descent (0.804 PPL) but not enough to clear the ≥1.5 PPL bar.

## Note on TSV append
The first TSV append attempt used a heredoc + printf invocation that zsh truncated at a `%_` literal sequence, leaving a partial duplicated row. The row was rebuilt programmatically with 8 columns and the duplicated cells removed. No other rows or files were affected. The final cycle 274 row in experiments.tsv is correct: cycle=274, item=284, type=empirical, outcome=inconclusive, key_metric (≈900 chars), description (≈2620 chars), commit=bcb8803, paths.

## Status
PASS
