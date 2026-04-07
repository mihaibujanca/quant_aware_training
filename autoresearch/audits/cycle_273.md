# Cycle 273 Audit

**Status: PASS**

## Item
283 (P1): Heterogeneous K guided by cycle 267 eff_rank — BLOCKS=[1,4,7,11], hetero K=[512,512,512,22] vs uniform K=22 control.

## Verdict
REFUTED — hetero best_mean = 52.6366 ≥ 47.0 (REFUTED clause triggered by 5.637 PPL). The CONFIRMED clause (hetero ≤ 44.5 AND control ≥ 60.0) fails on the hetero side by 8.14 PPL; the control gate (60.686 ≥ 60.0) is satisfied but irrelevant. Verdict per item 283 verbatim thresholds.

## Verification

### Plan adherence
- Plan: clone cycle 247 → `experiments/hetero_k_eff_rank_cycle273.py`, BLOCKS=[1,4,7,11], CONFIGS=[{name:hetero, k_per_block:{1:512,4:512,7:512,11:22}}, {name:uniform_k22, k_per_block:{1:22,4:22,7:22,11:22}}], 3 seeds × 2 configs = 6 fits, joint Adam W_out+b_out LR=1e-4 4000 steps EVAL_EVERY=100 best-state restore, disjoint calib/eval 200/100, SEQ_LEN=128. PASS
- Script sets BLOCKS=[1,4,7,11], CONFIGS match plan exactly. `run_seed` signature is `(seed, k_per_block, cfg_name)` so trainable count is recomputed per config with an assertion. PASS
- Hyperparameters: K_PER_BLOCK per config (see above), N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024]. All match plan and cycle 247 template. PASS
- Disjoint splits: calib_ids = all_ids[:200], eval_ids = all_ids[200:300]. PASS
- Best-state snapshot/restore + assert |restored - recorded| < 1e-3: confirmed in run log for all 6 fits (max diff well below 1e-3 — restored values match recorded best exactly to 3 dp across all fits). PASS

### Trainable counts
- hetero: 3 × (512×768 + 768) + 1 × (22×768 + 768) = 3 × 393,984 + 17,664 = 1,181,952 + 17,664 = **1,199,616**. Script assertion passes. PASS
- uniform_k22: 4 × (22×768 + 768) = 4 × 17,664 = **70,656**. Script assertion passes. PASS
- NOTE: plan text quoted "1,200,768" and "1,198,080" — both are wrong by small amounts (1,200,768 = 3×(512×768+768)+(22×768+768) with an arithmetic slip; 1,198,080 is also off). The correct value per the stated formula is 1,199,616 and the script computes this programmatically via `sum(k_per_block[b]*768 + 768 for b in BLOCKS)`. The numeric discrepancy is in the plan's narrative, not in the script — experiment integrity is intact. PASS

### Run completion
- 6/6 fits completed in ~25 min wall (no wall-cap mechanism needed; 6 fits is below the 30 min threshold and no seed drop logic was implemented for this cycle since the fit count is small). PASS
- Init PPL deterministic per config across seeds: hetero→72.1984 (3/3 identical to 3 dp), uniform_k22→82.7869 (3/3 identical to 3 dp). This confirms the lstsq init is seed-independent (only Adam shuffle differs). PASS
- All 6 fits logged to `autoresearch/logs/cycle_273_run.log`. PASS
- Float baseline measured during run = 53.895 (matches reference exactly). PASS

### Numerical consistency
Per-seed best_ppl from JSON:
- hetero: [52.6645, 52.6912, 52.5539] → mean 52.63655 ± 0.0595 (std unbiased=False). PASS
- uniform_k22: [60.7519, 60.6697, 60.6371] → mean 60.68622 ± 0.0483. PASS
- hetero best_step: [2600, 2800, 3000] → mean 2800 ± 163. PASS
- uniform_k22 best_step: [4000, 4000, 4000] → all at budget cap, still descending at termination. PASS

### Per-block lstsq residual MSE (seed 42)
- hetero: block 1 K=512 → 0.08432, block 4 K=512 → 0.16163, block 7 K=512 → 0.41485, **block 11 K=22 → 5.20024**. Block 11 K=22 residual is 12.5× block 7 K=512 and 60× block 1 K=512 — dominates the cascade init error budget. PASS
- uniform_k22: block 1 K=22 → 0.16103, block 4 K=22 → 0.26408, block 7 K=22 → 0.69959, block 11 K=22 → 5.20024. Block 11 K=22 residual is identical to hetero (expected — block 11 lstsq fit is independent of what happens at blocks 1/4/7 during the per-block collect-and-fit phase). PASS
- The block 11 K=22 residual is the dominant bottleneck in both configs — the ~5.2 MSE is an operator-level limit not an Adam-recoverable gap. Confirms the weight-SVD vs activation-SVD mismatch from cycle 270. PASS

### Verdict logic
- Per agenda item 283: CONFIRMED iff hetero ≤ 44.5 AND uniform_k22 ≥ 60.0; REFUTED iff hetero ≥ 47.0; else INCONCLUSIVE.
- hetero = 52.6366 → NOT ≤ 44.5 → CONFIRMED clause FAIL by 8.14 PPL. PASS
- hetero = 52.6366 → ≥ 47.0 → REFUTED clause TRIGGERED by 5.637 PPL. PASS
- uniform_k22 = 60.6862 → ≥ 60.0 → control gate MET (0.686 PPL above threshold) but irrelevant since CONFIRMED clause fails on hetero side. PASS
- → REFUTED. PASS

### Baseline cross-checks
- cycle 266 K=512 2-block [1,4] = 43.852 — matches script CYCLE_266_K512_2BLOCK. PASS
- cycle 270 block 11 rank-22 retrain = 54.469 — matches script CYCLE_270_BLOCK11_R22. PASS
- cycle 247 hetero 6-block = 64.45 — matches script CYCLE_247_HETERO_6BLOCK. PASS
- cycle 255 K=512 4-block min = 49.609 — matches script CYCLE_255_K512_4BLOCK. PASS
- cycle 268 K=64 4-block min = 55.953 — matches script CYCLE_268_K64_4BLOCK. PASS
- Float baseline 53.895 — measured during cycle 273 run, matches reference. PASS
- Δ hetero vs c266 [1,4]: 52.6366 − 43.852 = **+8.785**. PASS
- Δ hetero vs c255 4-block: 52.6366 − 49.609 = **+3.028**. PASS
- Δ hetero vs c270 block 11 r=22: 52.6366 − 54.469 = **−1.832**. PASS
- Δ hetero vs c272 K=256 4-block: 52.6366 − 51.922 = **+0.714**. PASS (cycle 272 strictly dominates cycle 273 hetero)
- Δ hetero vs float: 52.6366 − 53.895 = **−1.258**. PASS
- Δ control vs float: 60.6862 − 53.895 = **+6.791**. PASS

### Pareto frontier
- Cycle 273 hetero (1.20M, 52.64) is NOT a Pareto winner.
- Dominated by cycle 271 K=512 1-block [1] (394K, 43.82): third the params, 8.82 PPL better — strict dominance on both axes. PASS
- Dominated by cycle 272 K=256 4-block (789K, 51.92): 66% the params, 0.71 PPL better — strict dominance on both axes. PASS
- Dominated by cycle 266 K=512 2-block [1,4] (788K, 43.85): 66% the params, 8.78 PPL better — strict dominance. PASS
- Cycle 273 uniform_k22 (70K, 60.69) is also NOT a Pareto winner:
  - Dominated by cycle 270 block 11 r=22 (17.7K, 54.47): 25% the params, 6.22 PPL better. PASS
- Neither config enters the frontier. The brief's "matches cycle 266 [1,4] at 43.85" prediction is REFUTED by 8.78 PPL — the rank-22 block 11 budget is nowhere near "free" at the cascade level.

### Convergence sanity
- hetero best_step = 2600/2800/3000 (all well below the 4000 cap) — Adam converges around step 2800 then overshoots: final@4000 = 53.059/53.092/52.835 (overshoot of +0.39/+0.40/+0.28 PPL relative to best). The best-state restore is doing useful work on hetero. PASS
- uniform_k22 best_step = 4000 for all seeds — still descending at the budget cap. final@4000 = 60.752/60.670/60.637 exactly matches best_ppl for each seed (monotone descent through the full budget, no overshoot). Extending uniform_k22 to 8000 steps would likely yield further improvement but extrapolating the late-budget slope the descent is slow (~0.5 PPL over the last 1000 steps) and would not close the ~5 PPL gap to the hetero floor. PASS

### Bookkeeping
- agenda.md: item 283 removed from NEXT/P1 (P1 now empty between header and "### P2"). PASS
- agenda.md: "### Cycle 273" block inserted at top of PAST EXPERIMENTS (above "### Cycle 272") with verdict REFUTED, full geometric/operator-vs-activation reading, Pareto frontier update, and artifacts list. PASS
- TSV row appended to `autoresearch/logs/experiments.tsv` with placeholder `PENDING` commit hash — will be updated in bookkeeping commit. PASS
- Manifest written: `autoresearch/results/cycle_273_manifest.txt` with all per-config/per-seed values, baselines, file SHA256s. Commit hash will be updated in bookkeeping commit. PASS
- exec_result written: `autoresearch/results/cycle_273_exec_result.txt`. PASS

### Plan text vs script disagreement
- Plan stated hetero trainable = "1,198,080" and agenda item 283 stated "1,200,768" — both are arithmetically incorrect. The correct value per the formula `3 × (512×768 + 768) + (22×768 + 768)` is **1,199,616**. The script computes this programmatically via summation and asserts consistency at runtime, so experiment integrity is preserved. Recording the discrepancy here but not flagging as an audit failure: both plan text numbers were computed by hand-off and the script computes its own expected value. PASS (informational)

## Issues found
None. Verdict logic, numerical consistency, per-block residual analysis, Pareto analysis, and bookkeeping all check out. REFUTED is the correct verdict per the agenda thresholds — the hetero configuration is 5.64 PPL above the REFUTED threshold of 47.0, and the operator-level residual on block 11 (MSE = 5.20 at seed 42 across both configs) is the dominant error source that joint Adam cannot absorb.

## Status
PASS
