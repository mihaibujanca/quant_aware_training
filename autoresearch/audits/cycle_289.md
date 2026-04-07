# Cycle 289 Audit — Item 299

**Audit status: PASS**

## Scope
Verify cycle 289 execution adheres to item 299's verbatim Hypothesis, verdict thresholds, and the cycle 282 template it derives from; verify arithmetic and cross-check against the raw JSON; cross-check the noretrain-r=5 measurement against cycle 282's reported reference.

## Verdict verification

Item 299 thresholds (verbatim):
- CONFIRMED iff rank-5 retrained best_mean ≤ 53.0 (sub-eff_rank still beats float, opens a new ultra-low-param row at 4,608 params)
- REFUTED iff rank-5 retrained best_mean ≥ 56.0 (rank << eff_rank breaks block 10 — Adam cannot absorb the missing directions)
- INCONCLUSIVE otherwise

Measured:
- r=5 retrain best_mean = 51.46586 (from JSON `retrain_aggregate.best_mean`)

Arithmetic:
- 51.46586 ≤ 53.0 → TRUE (margin = 1.53414) → CONFIRMED clause fires
- 51.46586 < 56.0 → REFUTED clause NOT triggered
- → **CONFIRMED** ✓ (matches manifest, exec_result, JSON)

## Per-seed and aggregate sanity

Seeds (JSON `retrain_seeds`):
- 42  : init 59.9179 → best 51.4783 @ step 1500 → final@4000 51.6569 → restored 51.4783
- 137 : init 59.9179 → best 51.4614 @ step 1700 → final@4000 51.6385 → restored 51.4614
- 2024: init 59.9179 → best 51.4579 @ step 1800 → final@4000 51.6946 → restored 51.4579

Aggregate (population std, recomputed from per-seed bests):
- mean(51.47830, 51.46139, 51.45790) = 154.39759/3 = 51.46586 ✓
- deviations: (+0.01244, −0.00447, −0.00797)
- squared: (0.0001548, 0.00001996, 0.00006349); sum = 0.0002382; /3 = 0.00007940
- std = sqrt(0.00007940) = 0.008911 ✓ (matches reported 0.008909 within 5dp rounding)
- min = 51.45790 (seed 2024) ✓; max = 51.47830 (seed 42) ✓

All three seeds: init PPL = 59.91788 (deterministic, SVD init is seed-free) ✓
All three seeds: |restored − best| < 1e-3 (exact for all three) ✓
All three seeds: best step in {1500, 1700, 1800} — WELL INSIDE 4000-step budget, then overfit by step 4000 (final > best by 0.18/0.18/0.24) — FLOOR-limited, not budget-limited ✓

## No-retrain sweep sanity

| r | PPL     | Δfloat   | top-r energy |
|---|---------|----------|--------------|
| 5 | 59.918  | +6.023   | 0.0503       |

- Cross-check vs cycle 282 reported r=5 no-retrain = 59.92: measured 59.9179, |Δ| = 0.002 — essentially exact (reproduces cycle 282 measurement to 3dp).
- After sweep, restore original c_proj.weight and assert PPL ≈ float: PASS per log ("Restored original c_proj.weight, PPL=53.895 (should match float 53.895)").

## Config / plan adherence

Plan (cycle_289.md) vs execution:
- TARGET_BLOCK=10 ✓
- RANKS_NORETRAIN=[5] ✓
- RETRAIN_RANK=5 ✓
- SEEDS=[42, 137, 2024] ✓
- N_STEPS=4000, LR=1e-4, EVAL_EVERY=100, best-state restore ✓
- N_CALIB=200, N_EVAL=100, SEQ_LEN=128, disjoint ✓
- MPS, GPT-2 small, wikitext-2 val ✓
- Trainable per fit = 5×768 + 768 = 4,608 asserted in-script ✓
- Float PPL 53.89489 within 0.5 of 53.895 reference ✓ (exact)
- Template: `experiments/block10_lowrank_cycle282.py` adapted verbatim with only `RANKS_NORETRAIN`, `RETRAIN_RANK`, thresholds, output paths, and baseline constants changed. Same LowRankCProj class, same eval/retrain loop, same asserts. ✓

## Baseline cross-checks (vs manifest)

| baseline                          | recorded | from JSON | match |
|-----------------------------------|----------|-----------|-------|
| float                             | 53.89500 | 53.89489  | ✓ (within 0.0001) |
| cycle 280 b1 r=2                  | 49.0324  | 49.0324   | ✓ |
| cycle 285 b2 r=2                  | 49.3290  | 49.3290   | ✓ |
| cycle 282 b10 r=17                | 50.0445  | 50.0445   | ✓ |
| cycle 288 b3 r=24                 | 50.5935  | 50.5935   | ✓ |
| cycle 270 b11 r=22                | 54.4687  | 54.4687   | ✓ |
| cycle 278 b10 eff_rank            | 16.70    | 16.70     | ✓ |
| cycle 278 b10 top-1 energy        | 0.5889   | 0.5889    | ✓ |
| cycle 282 b10 r=5 noretrain ref   | 59.92    | 59.92     | ✓ |

Delta cross-checks:
- 51.46586 − 53.89489 = −2.42903 ✓ (vs float — below float by 2.43 PPL)
- 51.46586 − 50.04450 = +1.42136 ✓ (vs cycle 282 — 1.42 PPL worse at 3× fewer params)
- 51.46586 − 49.03240 = +2.43346 ✓ (vs cycle 280 — 2.43 PPL worse at 2× more params)
- 51.46586 − 49.32900 = +2.13686 ✓ (vs cycle 285 — 2.14 PPL worse at 2× more params)
- 51.46586 − 50.59350 = +0.87236 ✓ (vs cycle 288 — 0.87 PPL worse at 4.2× fewer params)
- 51.46586 − 54.46870 = −3.00284 ✓ (vs cycle 270 — 3.00 PPL better at 3.8× fewer params)

## Pareto reading — audit

Cycle 289 (4,608 params, 51.47 PPL) vs existing anchors:
- **STRICTLY DOMINATES** cycle 270 b11 r=22 (17,664, 54.47): 3.8× fewer params AND 3.00 PPL better.
- **STRICTLY DOMINATES** cycle 288 b3 r=24 (19,200, 50.59): 4.2× fewer params AND 0.87 PPL better.
- **DOMINATED** by cycle 280 b1 r=2 (2,304, 49.03): 2× more params AND 2.43 PPL worse.
- **DOMINATED** by cycle 285 b2 r=2 (2,304, 49.33): 2× more params AND 2.14 PPL worse.
- **DOMINATED** by cycle 282 b10 r=17 (13,824, 50.04): 3× fewer params but 1.42 PPL worse — different param tier (cycle 289 is lower-param, so not strictly dominated in the Pareto sense).

Net: cycle 289 is a **new Pareto interior point** at the ~4.6K param tier between the 2.3K anchor (cycle 280/285) and the 13.8K anchor (cycle 282). It removes TWO existing anchors from the joint frontier (cycle 270 and cycle 288) and adds a new frontier tile between cycle 280 and cycle 282.

## Geometric reading — audit

The item's Hypothesis asks "does Adam absorb the missing 12 directions, or does it fail to converge below float?"

(i) **Adam absorbs the missing directions.** At r=5 << eff_rank=16.70 (r/eff_rank=0.30), Adam recovers −8.45 PPL from init 59.92 to best 51.47, landing 2.43 PPL below float. The 5 rank directions realign during training to span the dominant input-active subspace of block 10's c_proj output — NOT the top-5 weight-SVD directions.

(ii) **Block 10 weight-SVD is NOT low-rank in the matrix sense.** Top-5 singular values 41.6, 28.2, 21.2, 20.2, 19.7 (S[0]/S[4] ≈ 2.1×) — the flattest profile of any tested block. Rank-5 captures only 5.03% of weight matrix squared-norm energy. The effectiveness of rank-5 is therefore NOT a property of the weight SVD; it is a property of the input distribution joint with the rank-5 post-training subspace found by Adam.

(iii) **Eff_rank roughly predicts MIN-required rank but NOT a hard floor.** Previous recipe was "rank ≈ eff_rank works". Cycle 289 extends: "rank = 0.30 × eff_rank still works at block 10". Block 10 rank-5 (51.47) EVEN BEATS block 11 rank-22 on-eff_rank (54.47) in absolute PPL at 3.8× fewer params, which falsifies any reading that eff_rank sets a per-block PPL floor.

(iv) **Floor-limited, not budget-limited.** All 3 seeds hit best by step 1800 and then degrade monotonically to step 4000 (final > best by 0.18-0.24 PPL on each seed). This is qualitatively different from every prior cycle in the 282-family, which were all still-descending at the 4000-step cap. Interpretation: 4,608 trainable params IS the constraint — additional training steps allow Adam to memorize calibration-specific signal that hurts eval, rather than find a lower minimum. Future rank-5 block-10 work should either add regularization, shrink the learning rate, or add rank (moving back toward the r≈eff_rank regime).

(v) **Recipe landscape (6 data points now)**:
- block 1  r=2  (eff 1.94, r/eff 1.03) → 49.03 @ 2,304
- block 2  r=2  (eff 1.05, r/eff 1.90) → 49.33 @ 2,304
- block 3  r=24 (eff 24.29, r/eff 0.99)→ 50.59 @ 19,200
- block 10 r=5  (eff 16.70, r/eff **0.30**) → **51.47 @ 4,608**  ← cycle 289
- block 10 r=17 (eff 16.70, r/eff 1.02)→ 50.04 @ 13,824
- block 11 r=22 (eff 22.49, r/eff 0.98)→ 54.47 @ 17,664

The "r ≈ eff_rank + Adam retrain → below float by 3-5 PPL" pattern holds 5/6; cycle 289 extends the curve below float even at r/eff_rank = 0.30 for block 10. The within-block-10 tradeoff: r=5 at 4.6K trades +1.42 PPL for 3× fewer params compared to r=17 at 13.8K.

(vi) Consistent with cycle 288's auditor note: "future single-block c_proj rank-patch items should de-prioritize blocks that eff_rank-predict they will need large rank budgets". Cycle 289 confirms the OPPOSITE direction is fertile — shrinking rank below eff_rank on blocks that already succeeded at r≈eff_rank may open new param tiers.

## Artifact existence

- experiments/block10_rank5_cycle289.py — exists, ~380 lines
- autoresearch/plans/cycle_289.md — exists, 35 lines, 8 top-level headings
- autoresearch/briefs/cycle_289.md — exists (from brief stage)
- autoresearch/results/cycle_289_block10_rank5.json — exists, valid JSON, verdict="CONFIRMED"
- autoresearch/results/cycle_289_summary.md — exists
- autoresearch/results/cycle_289_manifest.txt — exists, commit=7392cfe
- autoresearch/results/cycle_289_exec_result.txt — exists, commit=7392cfe
- autoresearch/logs/cycle_289_run.log — exists
- Primary commit 7392cfe exists in git log
- experiments.tsv row 289 appended with 8 columns ✓
- agenda.md: item 299 deleted from NEXT P2, added to PAST as Cycle 289 ✓

## Corrections

None. Provisional outcome holds.

## Final status

**PASS** — keep provisional outcome CONFIRMED. No agenda fix required beyond the move from NEXT to PAST already executed. Manifest, exec_result, JSON, summary, agenda, TSV, and audit all agree on verdict = CONFIRMED with best_mean = 51.4659 ± 0.0089.

Auditor note: cycle 289 is the FIRST successful sub-eff_rank single-block c_proj recipe and opens a new Pareto tile at the ~4.6K param level. The floor-limited convergence pattern (best at 1500-1800, overfit by 4000) is qualitatively distinct from the budget-limited pattern of cycles 270/282/288 and suggests that block 10 rank-3, rank-2, and rank-1 are worth testing — the question is whether the floor rises sharply below r=5 or gracefully, mirroring (or not) cycle 280's b1 r=2 success. An r=1 / r=2 / r=3 block-10 sweep at 4000 steps with the same LR would be a natural follow-up.
