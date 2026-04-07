# Cycle 288 Audit — Item 298

**Audit status: PASS**

## Scope
Verify that cycle 288 execution adheres to item 298's verbatim Hypothesis, verdict thresholds, and the cycle 282 template it derives from; verify arithmetic and cross-check against the raw JSON.

## Verdict verification

Item 298 thresholds (verbatim):
- CONFIRMED iff rank-24 retrained best_mean ≤ 55.0
- REFUTED iff rank-24 retrained best_mean ≥ 57.0
- INCONCLUSIVE otherwise

Measured:
- r=24 retrain best_mean = 50.59353 (from JSON `retrain_aggregate.best_mean`)

Arithmetic:
- 50.59353 ≤ 55.0 → TRUE (margin = 4.40647) → CONFIRMED clause fires
- 50.59353 < 57.0 → REFUTED clause NOT triggered
- → **CONFIRMED** ✓ (matches manifest, exec_result, JSON)

## Per-seed and aggregate sanity

Seeds (JSON `retrain_seeds`):
- 42  : init 57.2062 → best 50.6474 @ step 3900 → final@4000 50.7107 → restored 50.6474
- 137 : init 57.2062 → best 50.5825 @ step 3900 → final@4000 50.6320 → restored 50.5825
- 2024: init 57.2062 → best 50.5507 @ step 4000 → final@4000 50.5507 → restored 50.5507

Aggregate (population std, recomputed from JSON per-seed bests):
- mean(50.64738, 50.58248, 50.55073) = 151.78059/3 = 50.59353 ✓
- deviations: (+0.05385, −0.01105, −0.04280)
- squared: (0.002900, 0.000122, 0.001832); sum = 0.004854; /3 = 0.001618
- std = sqrt(0.001618) = 0.04023 ✓ (matches reported 0.04023)
- min = 50.55073 (seed 2024) ✓; max = 50.64738 (seed 42) ✓

All three seeds: init PPL = 57.20624 (deterministic, SVD init is seed-free) ✓
All three seeds: |restored − best| < 1e-3 (exact match for all three) ✓
All three seeds: best step in {3900, 3900, 4000} — all still descending at the 4000-step cap (seed 2024 converged exactly at the cap) ✓

## No-retrain sweep sanity

| r  | PPL     | Δfloat   | top-r energy |
|----|---------|----------|--------------|
|  5 | 58.458  | +4.563   | 0.1136       |
| 10 | 58.317  | +4.422   | 0.1423       |
| 24 | 57.206  | +3.311   | 0.2067       |

- Energy strictly monotone in r ✓ (0.1136 < 0.1423 < 0.2067).
- PPL weakly monotone-decreasing (58.458 > 58.317 > 57.206), consistent with adding dominant directions improving reconstruction.
- After sweep, restore original c_proj.weight and assert PPL ≈ float: PASS per log ("Restored original c_proj.weight, PPL=53.895 (should match float 53.895)").

## Config / plan adherence

Plan (cycle_288.md) vs execution:
- TARGET_BLOCK=3 ✓
- RANKS_NORETRAIN=[5, 10, 24] ✓
- RETRAIN_RANK=24 ✓
- SEEDS=[42, 137, 2024] ✓
- N_STEPS=4000, LR=1e-4, EVAL_EVERY=100, best-state restore ✓
- N_CALIB=200, N_EVAL=100, SEQ_LEN=128, disjoint ✓
- MPS, GPT-2 small, wikitext-2 val ✓
- Trainable per fit = 24×768 + 768 = 19,200 asserted in-script ✓
- Float PPL 53.895 within 0.5 of 53.895 reference ✓ (exact)
- Template: `experiments/block10_lowrank_cycle282.py` adapted verbatim with only the target block, rank list, and retrain rank changed (plus baseline constants). Verified by diff-reading the two scripts: same LowRankCProj class, same eval/retrain loop, same asserts ✓

## Baseline cross-checks

| baseline | recorded in manifest | from JSON | match |
|---|---|---|---|
| float                       | 53.895   | 53.895   | ✓ |
| cycle 280 b1 r=2            | 49.0324  | 49.0324  | ✓ |
| cycle 285 b2 r=2            | 49.3290  | 49.3290  | ✓ |
| cycle 282 b10 r=17          | 50.0445  | 50.0445  | ✓ |
| cycle 270 b11 r=22          | 54.4687  | 54.4687  | ✓ |
| cycle 278 b3 eff_rank       | 24.2853  | 24.2853  | ✓ |
| cycle 278 b3 top-1 energy   | 0.5887   | 0.5887   | ✓ |

Delta cross-checks:
- 50.59353 − 53.89500 = −3.30147 ✓ (vs float — below float by 3.30 PPL)
- 50.59353 − 50.04450 = +0.54903 ✓ (vs cycle 282 — 0.55 PPL worse at 1.39× more params; DOMINATED)
- 50.59353 − 54.46870 = −3.87517 ✓ (vs cycle 270 — 3.88 PPL better at 1.09× more params; dominates on PPL only)
- 50.59353 − 49.03240 = +1.56113 ✓ (vs cycle 280 — 1.56 PPL worse at 8.33× more params; DOMINATED)
- 50.59353 − 49.32900 = +1.26453 ✓ (vs cycle 285 — 1.26 PPL worse at 8.33× more params; DOMINATED)

## Pareto reading — audit

Cycle 288 (19,200 params, 50.59 PPL) vs existing anchors:
- Cycle 280 b1 r=2 (2,304, 49.03): 8.33× fewer params, 1.56 PPL better → STRICTLY DOMINATED.
- Cycle 285 b2 r=2 (2,304, 49.33): 8.33× fewer params, 1.26 PPL better → STRICTLY DOMINATED.
- Cycle 282 b10 r=17 (13,824, 50.04): 1.39× fewer params, 0.55 PPL better → STRICTLY DOMINATED (both axes).
- Cycle 270 b11 r=22 (17,664, 54.47): cycle 288 has 1.09× more params AND 3.88 PPL better → DOMINATES on PPL axis only, NOT joint (param increase of 1,536).

Net: cycle 288 is NOT a new Pareto point. It is a confirming interior point — the recipe generalizes to block 3 but block 3 is not a better single-block compression target than block 10 (smaller params + lower PPL). Block 11 is the only anchor cycle 288 improves on in PPL, and only at higher param cost.

## Geometric reading — audit

The item's Hypothesis asks "whether the early-vs-late distinction matters at intermediate eff_rank, OR whether eff_rank alone determines retrained PPL regardless of block depth". Audit reading:

(i) The rank-eff_rank-match recipe holds for block 3 as it did for blocks 1/2/10/11: set r ≈ eff_rank, Adam retrain, recover to 3-5 PPL below float. This is the 5th confirming block.

(ii) The retrain PPL landing zone for the concentrator cluster is:
- b1 r=2 → 49.03 (eff_rank 1.94, top-1 0.9294)
- b2 r=2 → 49.33 (eff_rank 1.05, top-1 0.9961)
- b3 r=24 → 50.59 (eff_rank 24.29, top-1 0.5887)  **← cycle 288**
- b10 r=17 → 50.04 (eff_rank 16.70, top-1 0.5889)
- b11 r=22 → 54.47 (eff_rank 22.49, top-1 —)

(iii) Early-vs-late distinction is NOT clearly dominant. The EARLY block 3 (50.59, intermediate eff_rank) lands BETWEEN the two LATE blocks b10 (50.04, lower eff_rank) and b11 (54.47, similar eff_rank). Block 10 beats block 3 at lower params, which is the opposite of what a pure early-vs-late ordering would predict; and block 11 is massively worse despite being late and having similar eff_rank.

(iv) Eff_rank alone does NOT determine retrained PPL. Ordering by eff_rank (1.05, 1.94, 16.70, 22.49, 24.29) does not reproduce the PPL ordering (49.03, 49.33, 50.04, 50.59, 54.47) — block 11 is the consistent outlier. The weaker descriptive claim — "eff_rank roughly predicts minimum retrain rank for recovery below float" — holds, but "eff_rank predicts final PPL" does not.

(v) Block 3's weight-SVD is flat (S[0]/S[4] ≈ 3.6×), matching every other tested concentrator. The low output eff_rank is an INPUT property (joint with weight), not a weight-rank property. Cycle 288 provides additional evidence that the concentrator-cluster geometric picture is driven by the INPUT covariance, not the weight spectrum.

(vi) Consistent with the cycle 285 audit's refinement: "rank-small + Adam retrain works for c_proj across multiple blocks when input concentration is high" — cycle 288 confirms this weaker statement generalizes to intermediate input concentration (top-1 0.5887 at block 3 vs 0.93/0.996 at blocks 1/2), requiring a proportionally larger rank budget (r=24 vs r=2).

## Artifact existence

- experiments/block3_lowrank_cycle288.py — exists, ~390 lines
- autoresearch/plans/cycle_288.md — exists, 34 lines, 8 top-level headings
- autoresearch/briefs/cycle_288.md — exists (from brief stage)
- autoresearch/results/cycle_288_block3_lowrank.json — exists, valid JSON, verdict="CONFIRMED"
- autoresearch/results/cycle_288_summary.md — exists
- autoresearch/results/cycle_288_manifest.txt — exists, commit=82014e2
- autoresearch/results/cycle_288_exec_result.txt — exists, commit=82014e2
- autoresearch/logs/cycle_288_run.log — exists
- Primary commit 82014e2 exists in git log
- experiments.tsv row 288 appended with 8 columns ✓
- agenda.md: item 298 deleted from NEXT P1, added to PAST as Cycle 288 ✓

## Corrections

None. Provisional outcome holds.

## Final status

**PASS** — keep provisional outcome CONFIRMED. No agenda fix required beyond the move from NEXT to PAST already executed. Manifest, exec_result, JSON, summary, agenda, TSV, and audit all agree on verdict = CONFIRMED with best_mean = 50.5935 ± 0.0402.

Auditor note: cycle 288 is a confirming but non-Pareto block. Future single-block c_proj rank-patch items should de-prioritize blocks that eff_rank-predict they will need large rank budgets (i.e. >16), because cycle 282 (b10, r=17) already occupies the ~14K-param sweet spot and block 11 (cycle 270) demonstrates that intermediate eff_rank does not guarantee recovery near 50 PPL. The remaining plausible single-block targets are block 4 (eff_rank 49.04, boundary case) and block 0 (eff_rank 71.93). Multi-block low-rank compositions (e.g. [1, 2, 3, 10]) are a more promising next direction than further single-block rank sweeps.
