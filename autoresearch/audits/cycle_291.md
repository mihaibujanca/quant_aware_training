# Cycle 291 Audit — Item 301

**Audit status: PASS**

## Scope
Verify cycle 291 execution adheres to item 301's verbatim Hypothesis, verdict thresholds, and the cycle 284 template it derives from. Verify arithmetic and cross-check against the raw JSON. Verify Pareto reading and the destructive composition claim against the two component anchors (cycles 280 and 289).

## Verdict verification

Item 301 thresholds (verbatim):
- CONFIRMED iff best_mean ≤ 49.5 (≤0.47 PPL above cycle 280, non-destructive composition; new Pareto tile at ~6.9K params)
- REFUTED iff best_mean ≥ 51.5 (composition damages cycle 280's anchor by ≥2.5 PPL — like cycle 275)
- INCONCLUSIVE otherwise

Measured:
- best_mean = 51.80080 (from JSON `retrain_aggregate.best_mean`)

Arithmetic:
- 51.80080 ≤ 49.5  → FALSE (over by 2.30080) → CONFIRMED clause NOT triggered
- 51.80080 ≥ 51.5  → TRUE  (over by 0.30080) → REFUTED clause TRIGGERED
- → **REFUTED** ✓ (matches manifest, exec_result, JSON, agenda)

## Per-seed and aggregate sanity

Seeds (JSON `retrain_seeds`):
- 42  : init 64.46495 → best 51.7511 @ step 3900 → restored 51.7511
- 137 : init 64.46495 → best 51.8509 @ step 3800 → restored 51.8509
- 2024: init 64.46495 → best 51.8004 @ step 4000 → restored 51.8004

Aggregate (population std, recomputed from per-seed bests):
- mean(51.7511, 51.8509, 51.8004) = 155.4024 / 3 = 51.80080 ✓
- deviations: (−0.04970, +0.05010, −0.00040)
- squared:    (0.0024701, 0.0025100, 1.6e-7); sum = 0.0049803; /3 = 0.0016601
- std = sqrt(0.0016601) = 0.04074 ✓ (matches reported 0.04073 to 4dp)
- min = 51.7511 (seed 42) ✓; max = 51.8509 (seed 137) ✓

All three seeds: init PPL = 64.46495 (deterministic; SVD seed-free at LOAD_SEED=42, both LowRankCProj modules construct from cached float SVD).
All three seeds: |restored − best| < 1e-3 ✓ (exact match — restore loop copies B and bias for both blocks).

best_step distribution: {3900, 3800, 4000} → 1/3 still BUDGET-BOUNDED at the 4000-step cap (seed 2024); 2/3 peaked before cap (seeds 42 and 137 overfit slightly past their best step). Aggregate is FLOOR-LIMITED with mild slack on one seed.

## Composition cross-check (UNIQUE to multi-anchor hybrids)

The brief asked for an init PPL "product" cross-check (cycle 280 r=2 init × cycle 289 r=5 init). This is mathematically incoherent — neither errors nor PPL multiply for stacked patches. The correct cross-check is that the joint init exceeds BOTH individual inits (the patches stack additively in perturbation magnitude, never cancel):

| anchor              | init PPL  | cycle 291 vs anchor init |
|---------------------|-----------|--------------------------|
| cycle 280 b1 r=2    | 57.65813  | +6.80682 (joint higher)  |
| cycle 289 b10 r=5   | 59.91788  | +4.54707 (joint higher)  |
| cycle 291 hybrid    | 64.46495  | —                        |

Both deltas are POSITIVE → confirms both LowRankCProj modules are simultaneously active in the forward pass and both perturbations contribute to the joint init PPL. The brief's "product" check is semantically wrong; the additivity-in-init check above is the correct test and PASSES.

## Config / plan adherence

Plan (cycle_291.md) vs execution:
- TARGET_BLOCKS=(1, 10) ✓
- RANKS=(2, 5) ✓
- N_STEPS=4000, EVAL_EVERY=100 ✓
- LR=1e-4 ✓
- SEEDS=[42, 137, 2024] ✓
- best-state restore ✓
- N_CALIB=200, N_EVAL=100, SEQ_LEN=128, disjoint ✓
- MPS, GPT-2 small, wikitext-2 val ✓
- Trainable per fit = (2·768 + 768) + (5·768 + 768) = 2,304 + 4,608 = 6,912 asserted in-script ✓
- Float PPL 53.895 within 0.5 of reference 53.895 ✓ (exact)
- Template: `experiments/rank_hybrid_1_11_cycle284.py` adapted verbatim. Diff vs cycle 284: TARGET_BLOCKS=(1,11)→(1,10), RANKS=(2,22)→(2,5), expected_trainable=19968→6912, OUTPUT_PATH/SUMMARY_PATH cycle_284→cycle_291, baseline constants (280/289/282/284/275 vs 270/280/282/275), thresholds (≤49.5 / ≥51.5 vs ≤55.0 / ≥58.0), JSON metadata ("cycle"/"item"/"verdict logic" labels). Same LowRankCProj, SVD loop, joint Adam loop, snapshot/restore. ✓

## Baseline cross-checks (vs JSON / manifest)

| baseline                            | recorded   | from JSON  | match |
|-------------------------------------|------------|------------|-------|
| float                               | 53.89500   | 53.89489   | ✓ (within 0.0002) |
| cycle 280 [1] r=2 retrain           | 49.03245   | 49.03245   | ✓ |
| cycle 289 [10] r=5 retrain          | 51.46586   | 51.46586   | ✓ |
| cycle 282 [10] r=17 retrain         | 50.04450   | 50.04450   | ✓ |
| cycle 284 hybrid b1r2+b11r22        | 52.88000   | 52.88000   | ✓ |
| cycle 275 heterogeneous hybrid      | 48.20040   | 48.20040   | ✓ (different scoring axis, recorded for context) |

Delta cross-checks (from manifest, recomputed):
- 51.80080 − 53.89489 =  −2.09410 → manifest reports −2.0941  ✓
- 51.80080 − 49.03245 =  +2.76835 → manifest reports +2.7684  ✓ (b1 anchor DAMAGED)
- 51.80080 − 51.46586 =  +0.33494 → manifest reports +0.3349  ✓ (b10 anchor DAMAGED)
- 51.80080 − 50.04450 =  +1.75630 → manifest reports +1.7563  ✓
- 51.80080 − 52.88000 =  −1.07920 → manifest reports −1.0792  ✓ (better than b1+b11 hybrid)

## Hypothesis falsification audit

Item 301 stated Hypothesis: "joint Adam over rank-2 (block 1) + rank-5 (block 10) c_proj patches reaches ≤49.5 PPL at ~6,912 trainable params — adds block 10 essentially for free on top of cycle 280's anchor and creates a new Pareto tile between cycle 280 (2.3K, 49.03) and cycle 282 (13.8K, 50.04)."

Three clauses to test:
1. **best_mean ≤ 49.5?** Measured 51.80, ABOVE 49.5 by 2.30 PPL. **FALSE**.
2. **Block 10 added "essentially for free" on top of b1?** Measured cost on b1 anchor is +2.77 PPL — this is the OPPOSITE of "free", and triggers exactly the REFUTED clause magnitude (≥2.5 PPL damage). **FALSE**.
3. **Lands between cycle 280 (49.03) and cycle 282 (50.04) on PPL axis?** Measured 51.80 lands ABOVE both, alongside cycle 289 (51.47, +0.33 worse). **FALSE**.

All three Hypothesis clauses FALSIFIED. The composition is destructive on the b1 anchor in the precise sense the REFUTED clause tested.

## Pareto reading — audit

Cycle 291 (6,912 params, 51.800 PPL) vs existing anchors:
- **STRICTLY DOMINATED** by cycle 280 b1 r=2 (2,304 params, 49.032 PPL): 3.0× MORE params AND 2.77 PPL WORSE. The hybrid's b1 anchor is itself crushed — combining with b10 r=5 destroys the cycle 280 result.
- **STRICTLY DOMINATED** by cycle 289 b10 r=5 (4,608 params, 51.466 PPL): 1.5× MORE params AND 0.33 PPL WORSE. Cycle 291's b10 anchor is also damaged (mildly), so the hybrid is worse than b10 alone too.
- **DOMINATED on PPL only** by cycle 282 b10 r=17 (13,824 params, 50.045 PPL): cycle 291 has 2.0× FEWER params but 1.76 PPL worse. Different tier.
- vs cycle 284 hybrid b1 r=2 + b11 r=22 (19,968 params, 52.880 PPL): cycle 291 has 2.9× FEWER params AND 1.08 PPL better. Cycle 291 dominates cycle 284 on BOTH axes — the only positive Pareto reading. But cycle 284 is itself dominated by other anchors, so this comparison is between two non-frontier hybrids and does not produce a frontier point.

Net: cycle 291 does NOT add a new Pareto tile. The hybrid is strictly Pareto-dominated by BOTH of its component anchors AND lies above cycle 282 in the same general region. The "between cycle 280 and cycle 282" target tile is unoccupied — cycle 291 lands NORTHEAST of both (more params AND worse PPL) than cycle 280, and ABOVE cycle 282 in PPL.

## Composition family pattern audit

2-block compositions involving block 1 (full history):

| cycle | composition                      | trainable | best_mean | verdict       |
|-------|----------------------------------|-----------|-----------|---------------|
| 273   | heterogeneous K-mix [1] + others | varies    | varies    | REFUTED       |
| 275   | [1] K=512 + b11 r=22 (hetero)   | 411,648   | 48.200    | REFUTED axis  |
| 281   | [1, 2] K=512 (channel-K)        | 787,968   | varies    | INCONCLUSIVE  |
| 284   | b1 r=2 + b11 r=22 (homo SVD)    | 19,968    | 52.880    | **CONFIRMED** |
| 291   | b1 r=2 + b10 r=5  (homo SVD)    | 6,912     | 51.801    | **REFUTED**   |

Pattern: only cycle 284 succeeded, and its partner block (b11) had rank 22 — high enough to absorb b1's perturbation. b10 r=5, b2 channel-K, and heterogeneous combinations all damage b1. The cycle 284 result was driven by b11's HIGH RANK SLACK, not by depth distance from b1. Cycle 291 disambiguates: (b1, b10) at non-adjacent depths with low partner rank still fails. The cycle 284 generalization is LIMITED to high-rank partners.

## Geometric reading — audit

(i) **Patches stack additively in init.** Hybrid init 64.465 > b10-only init 59.918 > b1-only init 57.658. The two patches contribute additively in the joint forward pass, with the joint init exceeding b10 alone by 4.55 PPL and b1 alone by 6.81 PPL. Neither patch is "free" in the joint pass.

(ii) **Joint Adam descent is LARGER but LANDING is HIGHER.** Joint recovery: 64.465 → 51.801 = −12.66 PPL. Solo recoveries: cycle 280 b1 r=2: 57.658 → 49.032 = −8.63 PPL; cycle 289 b10 r=5: 59.918 → 51.466 = −8.45 PPL. The joint optimizer descends 4 PPL FURTHER than either solo run, but lands 2.77 PPL ABOVE the b1 floor and 0.33 PPL ABOVE the b10 floor. More descent, worse destination. The 6,912-parameter joint surface is harder to navigate to BOTH anchors' floors than the 2,304-parameter and 4,608-parameter solo surfaces are to navigate to their respective floors.

(iii) **Recipe non-preservation under composition.** "Rank ≈ eff_rank + Adam retrain" works for SINGLE blocks (b1, b2, b3, b10, b11 all CONFIRMED in their solo cycles) but does NOT preserve under composition unless the partner has rank slack. The recipe is NOT compositional in the rank-budget sense.

(iv) **Depth hypothesis status.** The brief framed item 301 as "is the dependency depth-related" — testing whether non-adjacent depths (1, 10) would compose like adjacent depths (1, 11) did in cycle 284. Cycle 291 REFUTES the depth-distance reading: (1, 10) and (1, 11) have very similar depth distances (9 and 10 blocks apart) yet behave oppositely. The differentiator is partner RANK (b11 r=22 = 22 directions; b10 r=5 = 5 directions), not depth. Depth distance is NOT the right predictor.

(v) **Auditor recommendation for next agenda items**: cycle 291 closes the (b1, low-rank-partner) composition question with a REFUTED verdict and exposes RANK SLACK as the gating factor. Open follow-ups: (a) cycle 291's converse — b1 r=2 + b10 r=17 (use b10 at its eff_rank, not below it) to test whether the b10 r=5 sub-eff_rank regime is the cause of cycle 291's destruction; (b) b1 r=2 + b3 r=24 to test whether b3 (high-rank, mid-depth) absorbs b1 like b11 did; (c) reconsider whether ANY 2-block composition can beat cycle 280 alone on the Pareto frontier — current evidence (5 attempts, 1 winner, all dominated by single-block anchors) suggests not.

## Artifact existence

- experiments/rank_hybrid_1_10_cycle291.py — exists, ~370 lines, derived from cycle 284 verbatim
- autoresearch/plans/cycle_291.md — exists, 33 lines (≤35), 8 ## headings
- autoresearch/briefs/cycle_291.md — exists (from brief stage)
- autoresearch/results/cycle_291_rank_hybrid_1_10.json — exists, valid JSON, verdict="REFUTED"
- autoresearch/results/cycle_291_summary.md — exists
- autoresearch/results/cycle_291_manifest.txt — exists, commit=143d998
- autoresearch/results/cycle_291_exec_result.txt — exists, commit=143d998
- autoresearch/logs/cycle_291_run.log — exists
- Primary commit 143d998 exists in git log (HEAD)
- experiments.tsv row 291 appended with 8 columns ✓
- agenda.md: item 301 deleted from NEXT P0, added to PAST as Cycle 291 ✓

## Corrections

None. Provisional outcome holds.

## Final status

**PASS** — keep provisional outcome REFUTED. No agenda fix required beyond the move from NEXT to PAST already executed. Manifest, exec_result, JSON, summary, agenda, TSV, and audit all agree on verdict = REFUTED with best_mean = 51.8008 ± 0.0407 (over the 51.5 REFUTED threshold by 0.301 PPL).

Auditor note: cycle 291 closes item 301 by demonstrating that the cycle 284 "homogeneous low-rank weight-SVD hybrid composes non-destructively" result does NOT generalize to (b1, b10). The differentiator is partner RANK SLACK, not depth distance: cycle 284's b11 r=22 had 22 trainable directions to absorb b1's perturbation, while cycle 291's b10 r=5 has only 5. The 2-block composition family with b1 now stands at 5 attempts / 1 CONFIRMED, with the only winner requiring a high-rank partner. The hypothesis "block 10 added essentially for free on top of cycle 280's anchor" is FALSIFIED by a +2.77 PPL cost on the b1 anchor — exceeding the 2.5 PPL REFUTED threshold by 0.27 PPL. The next natural follow-up is b1 r=2 + b10 r=17 (use b10 at eff_rank, not below it) to isolate whether the cause is sub-eff_rank b10 or low partner rank generally.
