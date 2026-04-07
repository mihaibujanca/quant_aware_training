# Cycle 283 Audit — Item 293

**Audit status: PASS**

## Scope
Audit item 293 execution: "1-block [2] K sweep — K ∈ {16, 32, 64}". Primary artifacts: `experiments/block2_k_sweep_cycle283.py`, `autoresearch/results/cycle_283_block2_k_sweep.json`, `autoresearch/results/cycle_283_summary.md`, `autoresearch/logs/cycle_283_run.log`, `autoresearch/results/cycle_283_manifest.txt`. Primary commit: `7e63138`.

## Verdict check (verbatim item 293 thresholds)
- CONFIRMED iff K=16 best_mean ≤ 52.0 OR K=32 best_mean ≤ 50.0
- REFUTED iff all of {K=16, K=32, K=64} best_mean ≥ 55.0
- INCONCLUSIVE otherwise

Measured (from JSON):
- K=16 best_mean = 49.46221 — clause 1 passes by 2.538 PPL
- K=32 best_mean = 47.89660 — clause 2 passes by 2.103 PPL
- K=64 best_mean = 47.51148
- All three <<55.0 — refuted clause not triggered

Verdict CONFIRMED (both clauses) is correct per the rule literally applied.

## Mechanical checks
1. **Trainable param count per K**: JSON records K*768+768 = {13056, 25344, 49920}. The script asserts this at runtime before training each fit. Per run log all 9 fits print "(expect K×768+768)". Match.
2. **n_seeds**: 3 for each K = 9 fits total, matches plan and matches aggregates.
3. **Seeds**: {42, 137, 2024} matches plan and matches k_results.
4. **Init PPL determinism**: init_mean == 57.4949 (K=16), 56.5899 (K=32), 56.4734 (K=64) — per-seed init all identical within K, as expected (lstsq init is seed-independent on identical calib). Correct.
5. **Float PPL in-run**: 53.89489152418256 matches the reference 53.895 used by cycles 271/276/280/281/282. Correct.
6. **Restore assert**: script has `assert abs(final_ppl_restored - best_ppl) < 1e-3` for every fit. Log shows Restored line for all 9 fits without crash. All passed.
7. **Best-step distribution**: 8 of 9 fits hit best at step 4000 (the budget cap); K=16 seed 137 hit best at step 3800 then final@4000 rose to 49.5764 — normal budget-bounded descent, final@4000 and best are aligned per-K (49.4786 vs 49.4622 for K=16 due to single overshoot; 47.8966 == 47.8966 for K=32 and 47.5115 == 47.5115 for K=64).
8. **Wall time**: 22.6 min ≤ 30-min cap. Seed drop logic was not triggered.

## Script fidelity to plan
- BLOCKS=[2], K_VALUES=[16, 32, 64] per script lines at top. Match item 293.
- All other config (LR, N_STEPS, EVAL_EVERY, SEEDS, N_CALIB, N_EVAL, SEQ_LEN, DEVICE, trainable subset W_out+b_out, topk scoring=act.std*W_proj.norm, lstsq init, best-state restore) is verbatim from cycle 276. Match.
- Verdict logic literally implements item 293's thresholds (`confirmed_clause = (k16_bm <= 52.0) or (k32_bm <= 50.0)`, `refuted_clause = all(bm >= 55.0 ...)`).

## Agenda bookkeeping
- Item 293 was moved from NEXT > P1 (deleted cleanly, no stub left) to PAST > Cycle 283.
- PAST entry is ≤4-line format (title + outcome line + artifacts line).
- No other items touched.

## TSV / manifest agreement
- TSV row 283 written with outcome=confirmed, commit=7e63138, matches manifest commit line.
- Manifest verdict CONFIRMED matches JSON verdict matches agenda PAST entry matches TSV outcome.

## Cross-cycle reading (audit verification, not new claims)
- Claim "block 2 beats block 1 at every K tested": K=32 [2] 47.897 vs cycle 276 [1] K=32 49.649 = -1.752; K=64 [2] 47.511 vs cycle 276 [1] K=64 48.045 = -0.534. Verified from both JSONs.
- Claim "[2] K=16 (13,056 params, 49.462) strictly dominates cycle 276 [1] K=32 (25,344 params, 49.649)": 13056 < 25344 AND 49.462 < 49.649 — strict Pareto domination on both axes. Verified.
- Claim "cycle 280 block 1 c_proj r=2 (2,304, 49.032) still dominates cycle 283 [2] K=16 (13,056, 49.462)": 2304 < 13056 AND 49.032 < 49.462. Verified — cycle 280 remains the ultra-low-param endpoint.
- Claim "cycle 271 [1] K=512 (393,984, 43.820) remains best-PPL 1-block anchor": cycle 283 best PPL is K=64 47.511 which is 3.69 PPL above 43.820. Verified — cycle 271 is not challenged.
- Claim "consistent with cycle 281 negative cascade finding": cycle 281 showed [1,2] is +0.52 worse than [1] alone at K=512; cycle 283 shows [2] alone at small K is useful, implying block 2 overlaps block 1's subspace (can substitute but can't add). Consistent reading.

## Numerical sanity
- best_mean descends monotonically in K: 49.462 (K=16) > 47.897 (K=32) > 47.511 (K=64). Expected — more capacity → lower PPL until saturation. Consistent with cycle 276's monotone K sweep at block 1.
- best_std tight: 0.047 / 0.066 / 0.061 — similar order-of-magnitude to cycle 276 K-sweep stds (0.026, 0.039, 0.065, 0.044). No outliers.
- Init PPL ordering: K=16 57.495 > K=32 56.590 > K=64 56.473 — larger topk → better init, as expected.

## Potential weaknesses (not grounds to downgrade)
1. All 8/9 K=32/K=64 fits hit best at step 4000 (budget-bounded). Longer training might push best_mean lower — this is a lower bound on achievable PPL for this K range, not an upper bound. Not audit-material.
2. The "block 2 beats block 1 at same K" claim is per the K-sweep family only; it does not extend to K=512 (cycle 271 [1] K=512 is untested at block 2). A [2] K=512 run is not included in this cycle and is not required by item 293.
3. Block 2 overlaps block 1 subspace hypothesis is inferred from cycle 281+283 consistency and cycle 278 eff_rank measurement, not directly measured (no cross-block subspace similarity computed this cycle). Correctly labeled as "reading" in summary, not as a measurement.

## Decision
PASS. The verdict CONFIRMED (both clauses) is correct. Script fidelity is clean. Trainable param asserts passed. Restore asserts passed. Float PPL matches reference. Agenda, TSV, manifest, summary all agree. No numerical or methodological red flags.

Status: PASS.
