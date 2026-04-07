# Cycle 292 Audit — Item 302

**Audit status: PASS**

## Scope
Verify cycle 292 execution adheres to item 302's verbatim Hypothesis, verdict thresholds, and the cycle 280 template it derives from; verify arithmetic and cross-check against the raw JSON; cross-check the noretrain-r=1 measurement against cycle 280's reported reference.

## Verdict verification

Item 302 thresholds (verbatim):
- CONFIRMED iff rank-1 retrained best_mean ≤ 51.0 (still beats float by ≥2.9 PPL at 1,536 params — new lowest-param Pareto anchor)
- REFUTED iff rank-1 retrained best_mean ≥ 53.5 (rank-1 doesn't beat float meaningfully; r=2 is the floor for block 1)
- INCONCLUSIVE otherwise

Measured:
- r=1 retrain best_mean = 50.07942 (from JSON `retrain_aggregate.best_mean`)

Arithmetic:
- 50.07942 ≤ 51.0 → TRUE (margin = 0.92058) → CONFIRMED clause fires
- 50.07942 < 53.5 → REFUTED clause NOT triggered
- → **CONFIRMED** ✓ (matches manifest, exec_result, JSON)

## Per-seed and aggregate sanity

Seeds (JSON `retrain_seeds`):
- 42  : init 59.7513 → best 50.1471 @ step 3900 → final@4000 50.2724 → restored 50.1471
- 137 : init 59.7513 → best 50.0468 @ step 3900 → final@4000 50.1260 → restored 50.0468
- 2024: init 59.7513 → best 50.0444 @ step 4000 → final@4000 50.0444 → restored 50.0444

Aggregate (population std, recomputed from per-seed bests):
- mean(50.14708, 50.04679, 50.04439) = 150.23826/3 = 50.07942 ✓
- deviations: (+0.06766, −0.03263, −0.03503)
- squared: (0.0045779, 0.0010647, 0.0012271); sum = 0.0068697; /3 = 0.0022899
- std = sqrt(0.0022899) = 0.04785 ✓ (matches reported 0.04785 within 5dp)
- min = 50.04439 (seed 2024) ✓; max = 50.14708 (seed 42) ✓

All three seeds: init PPL = 59.75128 (deterministic, SVD init is seed-free) ✓
All three seeds: |restored − best| < 1e-3 (exact for all three) ✓
All three seeds: best step in {3900, 3900, 4000} — at or near the 4000-step cap, BUDGET-BOUNDED. Seed 2024 best=4000 with final==best (still descending). Seeds 42 and 137 best at 3900 with final > best by 0.125 / 0.079 — early-overfit signal but the descent had not yet found a clean floor. Convergence regime is qualitatively distinct from cycle 289 (best at 1500-1800, deeply floor-limited) — block 1 r=1 has more headroom for an 8000-step extension.

## No-retrain sweep sanity

| r | PPL     | Δfloat   | top-r energy |
|---|---------|----------|--------------|
| 1 | 59.7513 | +5.856   | 0.0459       |

- Cross-check vs cycle 280 reported r=1 no-retrain = 59.7513: measured 59.75128, |Δ| = 0.00000 (within 5dp — exact reproduction). The script's in-place `assert cycle_280_delta < 0.01` PASSES.
- After sweep, restore original c_proj.weight and assert PPL ≈ float: PASS per log ("Restored original c_proj.weight, PPL=53.895 (should match float 53.895)").

## Config / plan adherence

Plan (cycle_292.md) vs execution:
- TARGET_BLOCK=1 ✓
- RANKS_NORETRAIN=[1] ✓
- RETRAIN_RANK=1 ✓
- SEEDS=[42, 137, 2024] ✓
- N_STEPS=4000, LR=1e-4, EVAL_EVERY=100, best-state restore ✓
- N_CALIB=200, N_EVAL=100, SEQ_LEN=128, disjoint ✓
- MPS, GPT-2 small, wikitext-2 val ✓
- Trainable per fit = 1×768 + 768 = 1,536 asserted in-script ✓
- Float PPL 53.89489 within 0.5 of 53.895 reference ✓ (exact)
- Template: `experiments/block1_lowrank_cycle280.py` adapted verbatim with only `RANKS_NORETRAIN`, `RETRAIN_RANK`, thresholds, output paths, baseline constants, and a noretrain cross-check assert added. Same LowRankCProj class, same eval/retrain loop, same sanity asserts. ✓

## Baseline cross-checks (vs manifest)

| baseline                          | recorded | from JSON | match |
|-----------------------------------|----------|-----------|-------|
| float                             | 53.89500 | 53.89489  | ✓ (within 0.0001) |
| cycle 280 b1 r=2                  | 49.0324  | 49.0324   | ✓ |
| cycle 285 b2 r=2                  | 49.3290  | 49.3290   | ✓ |
| cycle 289 b10 r=5                 | 51.4659  | 51.4659   | ✓ |
| cycle 282 b10 r=17                | 50.0445  | 50.0445   | ✓ |
| cycle 288 b3 r=24                 | 50.5935  | 50.5935   | ✓ |
| cycle 270 b11 r=22                | 54.4687  | 54.4687   | ✓ |
| cycle 278 b1 eff_rank             | 1.9448   | 1.9448    | ✓ |
| cycle 278 b1 top-1 energy         | 0.9294   | 0.9294    | ✓ |
| cycle 280 b1 r=1 noretrain ref    | 59.7513  | 59.7513   | ✓ |

Delta cross-checks:
- 50.07942 − 53.89489 = −3.81547 ✓ (vs float — below float by 3.82 PPL)
- 50.07942 − 49.03240 = +1.04702 ✓ (vs cycle 280 — 1.05 PPL worse at 1.5× fewer params)
- 50.07942 − 51.46590 = −1.38648 ✓ (vs cycle 289 — 1.39 PPL better at 3× fewer params)
- 50.07942 − 49.32900 = +0.75042 ✓ (vs cycle 285 — 0.75 PPL worse at 1.5× fewer params)
- 50.07942 − 50.04450 = +0.03492 ✓ (vs cycle 282 — 0.03 PPL worse at 9× fewer params, statistical tie)
- 50.07942 − 50.59350 = −0.51408 ✓ (vs cycle 288 — 0.51 PPL better at 12.5× fewer params)
- 50.07942 − 54.46870 = −4.38928 ✓ (vs cycle 270 — 4.39 PPL better at 11.5× fewer params)

## Pareto reading — audit

Cycle 292 (1,536 params, 50.08 PPL) vs existing anchors:
- **NEW LOWEST-PARAM single-block c_proj recipe** by 33% (1,536 vs the previous lowest at 2,304 — cycles 280 and 285).
- **STRICTLY DOMINATES** cycle 289 b10 r=5 (4,608, 51.47): 3× fewer params AND 1.39 PPL better.
- **STRICTLY DOMINATES** cycle 288 b3 r=24 (19,200, 50.59): 12.5× fewer params AND 0.51 PPL better.
- **STRICTLY DOMINATES** cycle 270 b11 r=22 (17,664, 54.47): 11.5× fewer params AND 4.39 PPL better.
- **ESSENTIALLY TIES** cycle 282 b10 r=17 (13,824, 50.04) on PPL (+0.03) at 9× fewer params — cycle 282 dominated on PPL essentially free.
- vs cycle 280 b1 r=2 (2,304, 49.03): cycle 292 is +1.05 PPL worse at 33% fewer params — a NEW Pareto tile, NOT dominated (different param tier).
- **DOMINATED** by cycle 285 b2 r=2 (2,304, 49.33): 1.5× more params for 0.75 PPL — same general logic as cycle 280 (different tier).

Net: cycle 292 REMOVES THREE existing anchors (cycles 270, 288, 289) from the joint Pareto frontier and ESSENTIALLY-TIES cycle 282, adding a new lowest-param frontier tile at 1,536 params / 50.08 PPL. The single-block c_proj recipe landscape is now anchored by cycle 292 (1,536) and cycle 280 (2,304) at the small-param end. Major Pareto rewrite at the small-param end of the curve.

## Geometric reading — audit

The item's Hypothesis asks "does rank-1 weight-SVD + Adam retrain still beat float at 1,536 trainable params? Block 1 eff_rank=1.94 means r=1 is BELOW the eff_rank threshold (r/eff = 0.52)."

(i) **Adam absorbs the missing direction.** At r=1 < eff_rank=1.94 (r/eff_rank=0.514), Adam recovers −9.67 PPL from init 59.75 to best 50.08, landing 3.82 PPL below float. The single trainable B (1, 768) row realigns during training to span the dominant input-active direction of block 1's c_proj output — NOT necessarily the top-1 weight-SVD direction. The frozen A column = U[:,0]·Σ[0] is fixed; trainable B + bias do all the adaptation.

(ii) **Block 1 weight-SVD is NOT low-rank in the matrix sense.** Top-5 singular values 28.7, 23.1, 18.8, 13.8, 13.7 (S[0]/S[4] ≈ 2.1×) — flat profile, similar to block 10's flatness but with lower absolute values. Rank-1 captures only 4.59% of weight matrix squared-norm energy. The effectiveness of rank-1 is therefore NOT a property of the weight SVD; it is a property of the input distribution joint with the rank-1 post-training subspace found by Adam.

(iii) **Eff_rank does not set a hard floor — second confirmation.** Cycle 289 was the first sub-eff_rank success at r/eff=0.30 (block 10). Cycle 292 confirms the regime extends to the SMALLEST-eff_rank concentrator (block 1) at the LOWEST possible rank (r=1). Two data points (cycles 289, 292) now establish that the eff_rank threshold from cycle 278 is a SOFT prediction of minimum useful retrain rank, not a hard PPL floor.

(iv) **BUDGET-BOUNDED, not floor-limited — qualitatively distinct from cycle 289.** All 3 seeds hit best at step 3900-4000. Seed 2024 best=4000 with final==best (still descending at the cap). Seeds 42, 137 best at 3900 with final > best by 0.125 / 0.079 (early overfit signal). This is qualitatively DIFFERENT from cycle 289 block 10 r=5, which was deeply floor-limited (best at 1500-1800, then drift up by 0.18-0.24 PPL). Interpretation: cycle 292's 1,536-param landscape has more usable Adam slack than cycle 289's 4,608-param landscape — fewer parameters, more directions to explore per step. An 8000-step extension is the natural follow-up; expected gain ~0.3-0.8 PPL based on the budget-extension family ([1] K=512 4k→8k −1.04, [K=1] 5b 4k→16k −5.27, etc.).

(v) **Recipe landscape (7 data points now)**:
- block 1  r=1  (eff 1.94, r/eff **0.51**) → **50.08 @ 1,536**  ← cycle 292 NEW
- block 1  r=2  (eff 1.94, r/eff 1.03)→ 49.03 @ 2,304
- block 2  r=2  (eff 1.05, r/eff 1.90)→ 49.33 @ 2,304
- block 3  r=24 (eff 24.29, r/eff 0.99)→ 50.59 @ 19,200
- block 10 r=5  (eff 16.70, r/eff 0.30)→ 51.47 @ 4,608
- block 10 r=17 (eff 16.70, r/eff 1.02)→ 50.04 @ 13,824
- block 11 r=22 (eff 22.49, r/eff 0.98)→ 54.47 @ 17,664

The "r ≈ eff_rank + Adam retrain → below float by 3-5 PPL" pattern holds 6/7 (b11 still the soft outlier at 54.47); cycle 292 extends below-eff_rank coverage to r/eff=0.51 on block 1. Combined with cycle 289, two sub-eff_rank wins now demonstrate the regime is general, not block-specific.

(vi) **Single-block tradeoff curve at block 1**: r=1 (1,536, 50.08) → r=2 (2,304, 49.03) — 1.5× more params buys 1.05 PPL. Diminishing returns past r=1 in the same direction as cycle 289's within-block-10 curve (r=5 4,608 51.47 → r=17 13,824 50.04 — 3× more params buys 1.43 PPL).

## Artifact existence

- experiments/block1_rank1_cycle292.py — exists, ~446 lines
- autoresearch/plans/cycle_292.md — exists, 25 lines, 8 H2 headings + title
- autoresearch/briefs/cycle_292.md — exists (from brief stage)
- autoresearch/results/cycle_292_block1_rank1.json — exists, valid JSON, verdict="CONFIRMED"
- autoresearch/results/cycle_292_summary.md — exists
- autoresearch/results/cycle_292_manifest.txt — exists, commit=95be6f7
- autoresearch/results/cycle_292_exec_result.txt — exists, commit=95be6f7
- autoresearch/logs/cycle_292_run.log — exists
- Primary commit 95be6f7 exists in git log
- experiments.tsv row 292 appended with 8 columns ✓
- agenda.md: item 302 deleted from NEXT P0, added to PAST as Cycle 292 ✓

## Corrections

None. Provisional outcome holds.

## Final status

**PASS** — keep provisional outcome CONFIRMED. No agenda fix required beyond the move from NEXT to PAST already executed. Manifest, exec_result, JSON, summary, agenda, TSV, and audit all agree on verdict = CONFIRMED with best_mean = 50.0794 ± 0.0479.

Auditor note: cycle 292 is the SECOND successful sub-eff_rank single-block c_proj recipe (after cycle 289) and opens the LOWEST-PARAM tile in the entire single-block c_proj Pareto frontier (1,536 params). The budget-bounded convergence pattern (best at 3900-4000, still descending) is qualitatively distinct from cycle 289's floor-limited pattern and suggests an 8000-step extension is the natural follow-up — block 1 r=1 has not yet found its floor. Cycle 292 also REMOVES three existing anchors (270, 288, 289) from the joint Pareto frontier and ESSENTIALLY-TIES cycle 282 (13,824) on PPL at 9× fewer params, making it the most aggressive single-cycle Pareto rewrite in recent history. Recommended next items: (a) cycle 292 8k budget extension (rank-1 floor probe); (b) 2-block hybrids anchored on cycle 292 1,536-param tile to test composition with low-rank partners; (c) re-examine cycle 291's hybrid b1+b10 result with cycle 292's b1 r=1 instead of b1 r=2 (cheaper but fragile) — likely still REFUTED but worth a deterministic check.
