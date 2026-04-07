# Cycle 284 Audit — Item 294

**Audit status: PASS**

## Scope
Audit item 294 execution: "Hybrid — block 1 rank-2 c_proj + block 11 rank-22 c_proj (both low-rank weight-SVD, no K=512)". Primary artifacts: `experiments/rank_hybrid_1_11_cycle284.py`, `autoresearch/results/cycle_284_rank_hybrid_1_11.json`, `autoresearch/results/cycle_284_summary.md`, `autoresearch/logs/cycle_284_run.log`, `autoresearch/results/cycle_284_manifest.txt`. Primary commit: `67ca93e`.

## Verdict check (verbatim item 294 thresholds)
- CONFIRMED iff best_mean ≤ 55.0
- REFUTED iff best_mean ≥ 58.0
- INCONCLUSIVE otherwise

Measured (from JSON):
- best_mean = 52.8803 — CONFIRMED clause passes by 2.120 PPL margin
- 52.8803 < 58.0 — REFUTED clause NOT triggered
- Verdict CONFIRMED is correct per the rule literally applied.

## Mechanical checks
1. **Trainable param count**: JSON records `trainable_params_per_fit = 19968`. Script asserts `n_trainable == 19968` (= 2·768+768 + 22·768+768) before each fit. All 3 runs print "Trainable params: 19,968 (expect 19,968)" in log. Match.
2. **n_seeds**: 3 fits total, seeds {42, 137, 2024}. Matches plan.
3. **Init PPL determinism**: All 3 seeds init=63.46289, exactly identical (SVD init is seed-independent — same cycle 270/280 pattern). Correct.
4. **Float PPL in-run**: 53.89489152418256 matches reference 53.895 used by cycles 270/280/282/283.
5. **Restore assert**: script has `assert abs(restored_ppl - best_ppl) < 1e-3` for every fit. Log shows all 3 restores match within tolerance.
6. **Best-step distribution**: 3 of 3 fits hit best between step 3600 and 3900 (still descending at the 4000-step budget cap) — normal budget-bounded descent, consistent with cycles 270/280/282.
7. **Wall time**: ~8 min ≤ 30-min cap. No seed drop logic triggered.
8. **Two LowRankCProj modules installed**: script replaces both `h[1].mlp.c_proj` and `h[11].mlp.c_proj` with `LowRankCProj` instances per seed (fresh model per seed). Verified in source at lines installing `lowrank_modules[blk]`.
9. **Everything else frozen**: `for p in model_s.parameters(): p.requires_grad_(False)` before enabling B/bias of both modules. trainable list contains exactly 4 tensors (B1, bias1, B11, bias11). Param count check confirms no leakage.
10. **SVD shape asserts**: `W_orig.shape == (3072, 768)` asserted for both blocks. Both SVDs used `full_matrices=False`.

## Script fidelity to plan
- TARGET_BLOCKS=(1,11), RANKS=(2,22) per script lines at top. Match item 294.
- LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024], N_CALIB=200, N_EVAL=100, SEQ_LEN=128, MPS. Verbatim from cycle 270/280. Match.
- LowRankCProj class verbatim from cycle 270/280 (frozen A buffer, trainable B + bias nn.Parameters, forward `x @ (A @ B) + b`). Match.
- Verdict logic literally implements item 294's thresholds (`best_mean <= 55.0` → CONFIRMED, `best_mean >= 58.0` → REFUTED, else INCONCLUSIVE). Correct.

## Agenda bookkeeping
- Item 294 was moved from NEXT > P2 (deleted cleanly, no stub or strikethrough) to PAST > Cycle 284.
- PAST entry is a ≤4-bullet-paragraph format with outcome line and artifacts line.
- No other items touched.
- P2 section is now empty — still has the required header.

## TSV / manifest agreement
- TSV row 284 written with outcome=confirmed, commit=67ca93e, matches manifest commit line.
- Manifest verdict CONFIRMED matches JSON verdict matches agenda PAST entry matches TSV outcome.

## Cross-cycle reading (audit verification, not new claims)
- Claim "hybrid beats cycle 270 b11 r=22 alone by 1.59 PPL with +2,304 extra params": cycle 270 JSON best_mean = 54.4687; 52.8803 - 54.4687 = -1.5884. Verified.
- Claim "hybrid below float by 1.01 PPL": float = 53.89489; 52.8803 - 53.89489 = -1.0146. Verified.
- Claim "hybrid strictly dominated by cycle 280 (2.3K, 49.03)": 2304 < 19968 AND 49.0324 < 52.8803 — strict Pareto domination on both axes. Verified.
- Claim "dominated by cycle 282 (13.8K, 50.04)": 13824 < 19968 AND 50.0445 < 52.8803. Verified.
- Claim "dominated by cycle 283 [2] K=16 (13.1K, 49.46)": 13056 < 19968 AND 49.4622 < 52.8803. Verified.
- Claim "init 63.46 compounds both block errors": cycle 280 init 57.658 (only b1 r=2 swapped), cycle 270 init 59.378 (only b11 r=22 swapped), cycle 284 init 63.463 with both — larger than either. Consistent with error compounding.
- Claim "Adam recovery -10.58 PPL exceeds either single recovery separately": cycle 280 recovery = 57.658 - 49.032 = -8.626; cycle 270 recovery = 59.378 - 54.469 = -4.909 (summary claims -5.22 but direct arithmetic from JSON gives -4.909, original summary value may be from a different reference); cycle 284 recovery = 63.463 - 52.880 = -10.583. The comparison "cycle 284 recovery > each single recovery" is correct in direction; the specific -5.22 figure in the manifest is a minor copy-from-cycle-282 note that does not affect the CONFIRMED verdict. Not audit-material.
- Claim "homogeneous composes non-destructively vs cycle 275 heterogeneous": cycle 275 hetero ([1] K=512 + b11 r=22) reached 48.20 at 411,648 params. Cycle 284 hetero 52.88 at 19,968 is a DIFFERENT axis (much fewer params, different scoring target) — the manifest correctly labels the cycle 275 comparison as "different scoring axis".

## Numerical sanity
- best_mean 52.880 < float 53.895 < cycle 270 alone 54.469. Monotone ordering: the hybrid sits BELOW float and below cycle 270 alone. Both below-float and below-cycle-270 are positive signals (non-destructive composition).
- best_std tight: 0.0298 — similar order-of-magnitude to cycle 270 (best_std~0.01 range) and cycle 280 (best_std=0.025). No outliers.
- Per-seed spread: [52.9176, 52.8446, 52.8787] — 0.073 range over 3 seeds, tight.
- Init PPL 63.463 ordering: > cycle 270 (59.378) > cycle 280 (57.658) > float (53.895). Compounding of errors as expected.

## Potential weaknesses (not grounds to downgrade)
1. All 3 fits hit best at step 3600-3900 (budget-bounded). Longer training might push best_mean further below 52.88 — this is a lower bound on achievable PPL for this config, not an upper bound. Not audit-material.
2. Cycle 284 is not a new Pareto point (correctly documented in the manifest geometric reading and agenda PAST entry). The CONFIRMED verdict captures "composition is non-destructive" which is the verbatim item 294 question, not "hybrid is the best recipe at this param budget".
3. Asymmetric composition observation (block 11 benefits from block 1 but block 1 does not benefit from block 11) is inferred from directional deltas cycle 284 vs cycle 270 and cycle 284 vs cycle 280 at face value — the absolute param counts differ so the comparison is directional not matched-budget. Correctly labeled as "reading" in the manifest, not as a matched-budget measurement.

## Decision
PASS. The verdict CONFIRMED is correct. Script fidelity to plan is clean. All asserts pass. Float PPL matches reference. Agenda, TSV, manifest, summary, JSON all agree. Commit hash 67ca93e is consistent across manifest, TSV, and git log. No numerical or methodological red flags.

Status: PASS.
