# Cycle 293 Audit — Item 303

**Audit status: PASS**

## Scope
Verify that cycle 293 execution adheres to item 303's verbatim Hypothesis, verdict thresholds, and the cycle 288 template it derives from; verify arithmetic and cross-check against the raw JSON.

## Verdict verification

Item 303 thresholds (verbatim from agenda.md):
- CONFIRMED iff retrained best_mean ≤ 55.0 (block 0 fits the rank-eff_rank recipe at the high end of the concentrator scale)
- REFUTED iff retrained best_mean ≥ 58.0 (block 0 does NOT fit — embedding-proximity or depth-0 specialness breaks the recipe)
- INCONCLUSIVE otherwise

Measured:
- r=72 retrain best_mean = 63.13265 (from JSON `retrain_aggregate.best_mean`)

Arithmetic:
- 63.13265 ≥ 58.0 → TRUE (margin = 5.13265 above REFUTED threshold) → REFUTED clause fires
- 63.13265 > 55.0 → CONFIRMED clause NOT triggered (gap 8.13265)
- → **REFUTED** ✓ (matches manifest, exec_result, JSON)

## Per-seed and aggregate sanity

Seeds (JSON `retrain_seeds`):
- 42  : init 396.25818 → best 63.05820 @ step 4000 → final@4000 63.05820 → restored 63.05820
- 137 : init 396.25818 → best 63.06452 @ step 4000 → final@4000 63.06452 → restored 63.06452
- 2024: init 396.25818 → best 63.27523 @ step 3900 → final@4000 63.55822 → restored 63.27523

Aggregate (population std, recomputed independently from per-seed bests):
- mean(63.05820, 63.06452, 63.27523) = 189.39795/3 = 63.13265 ✓ (matches reported 63.13265)
- deviations: (−0.07445, −0.06813, +0.14258)
- squared: (0.005543, 0.004642, 0.020330); sum = 0.030515; /3 = 0.010172
- std = sqrt(0.010172) = 0.10085 ✓ (matches reported 0.10085)
- min = 63.05820 (seed 42) ✓; max = 63.27523 (seed 2024) ✓

All three seeds: init PPL = 396.25818 (deterministic, SVD init is seed-free) ✓
All three seeds: |restored − best| = 0.0 (exact match for all three) ✓
Best step distribution: {4000, 4000, 3900} — all 3 BUDGET-BOUNDED at the 4000-step cap (still descending; seeds 42 and 137 reached best at the cap; seed 2024 at step 3900 with final@4000 worse, indicating slight overfit at the very end of its trajectory) ✓
Per-seed convergence trajectories (from history): monotone-ish descent from 79-80 PPL @ step 500 down through ~63 PPL by step 4000, consistent with full-budget Adam saturation ✓

## No-retrain sweep sanity

| r  | PPL       | Δfloat   | top-r energy |
|----|-----------|----------|--------------|
| 10 | 1939.006  | +1885.11 | 0.1495       |
| 30 | 913.701   | +859.81  | 0.2241       |
| 72 | 396.258   | +342.36  | 0.3271       |

- Energy strictly monotone in r ✓ (0.1495 < 0.2241 < 0.3271).
- PPL strictly monotone-decreasing in r ✓ (1939.0 > 913.7 > 396.3), consistent with adding dominant directions improving reconstruction.
- After sweep, restore original c_proj.weight and assert PPL ≈ float: PASS per log ("Restored original c_proj.weight, PPL=53.895 (should match float 53.895)").
- Init PPL at r=72 (396.25818) matches noretrain r=72 (396.25818) per-seed exactly ✓ — confirms LowRankCProj init reproduces the noretrain swap deterministically.

## Config / plan adherence

Plan (cycle_293.md) vs execution:
- TARGET_BLOCK=0 ✓
- RANKS_NORETRAIN=[10, 30, 72] ✓
- RETRAIN_RANK=72 ✓
- SEEDS=[42, 137, 2024] ✓
- N_STEPS=4000, LR=1e-4, EVAL_EVERY=100, best-state restore ✓
- N_CALIB=200, N_EVAL=100, SEQ_LEN=128, disjoint ✓
- MPS, GPT-2 small, wikitext-2 val ✓
- Trainable per fit = 72×768 + 768 = 56,064 asserted in-script ✓
- Float PPL 53.895 within 0.5 of 53.895 reference ✓ (exact)
- Template: `experiments/block3_lowrank_cycle288.py` adapted with only TARGET_BLOCK, RANKS_NORETRAIN, RETRAIN_RANK, baseline constants, and SVD index labels changed. Verified by structural diff: same LowRankCProj class, same eval/retrain loop, same asserts ✓

## Baseline cross-checks

| baseline | recorded in manifest | from JSON | match |
|---|---|---|---|
| float                       | 53.895   | 53.895   | ✓ |
| cycle 280 b1 r=2            | 49.0324  | 49.0324  | ✓ |
| cycle 285 b2 r=2            | 49.3290  | 49.3290  | ✓ |
| cycle 282 b10 r=17          | 50.0445  | 50.0445  | ✓ |
| cycle 288 b3 r=24           | 50.5935  | 50.5935  | ✓ |
| cycle 270 b11 r=22          | 54.4687  | 54.4687  | ✓ |
| cycle 292 b1 r=1            | 50.0794  | 50.0794  | ✓ |
| cycle 278 b0 eff_rank       | 71.93    | 71.93    | ✓ |

Delta cross-checks:
- 63.13265 − 53.89489 = +9.23776 ✓ (vs float — ABOVE float by 9.24 PPL — recipe failure)
- 63.13265 − 50.04450 = +13.08815 ✓ (vs cycle 282 b10 r=17 — 13.09 PPL worse at 4.06× more params)
- 63.13265 − 50.59350 = +12.53915 ✓ (vs cycle 288 b3 r=24 — 12.54 PPL worse at 2.92× more params)
- 63.13265 − 54.46870 = +8.66395 ✓ (vs cycle 270 b11 r=22 — 8.66 PPL worse at 3.17× more params)
- 63.13265 − 49.03240 = +14.10025 ✓ (vs cycle 280 b1 r=2 — 14.10 PPL worse at 24.3× more params)
- 63.13265 − 50.07940 = +13.05325 ✓ (vs cycle 292 b1 r=1 — 13.05 PPL worse at 36.5× more params)
- 63.13265 − 49.32900 = +13.80365 ✓ (vs cycle 285 b2 r=2 — 13.80 PPL worse at 24.3× more params)

## Pareto reading — audit

Cycle 293 (56,064 params, 63.13 PPL) vs all existing single-block c_proj recipe anchors:
- Cycle 280 b1 r=2 (2,304, 49.03): 24.3× fewer params, 14.10 PPL better → STRICTLY DOMINATES cycle 293.
- Cycle 285 b2 r=2 (2,304, 49.33): 24.3× fewer params, 13.80 PPL better → STRICTLY DOMINATES cycle 293.
- Cycle 282 b10 r=17 (13,824, 50.04): 4.06× fewer params, 13.09 PPL better → STRICTLY DOMINATES cycle 293.
- Cycle 288 b3 r=24 (19,200, 50.59): 2.92× fewer params, 12.54 PPL better → STRICTLY DOMINATES cycle 293.
- Cycle 270 b11 r=22 (17,664, 54.47): 3.17× fewer params, 8.66 PPL better → STRICTLY DOMINATES cycle 293.
- Cycle 289 b10 r=5 (4,608, 51.47): 12.17× fewer params, 11.66 PPL better → STRICTLY DOMINATES cycle 293.
- Cycle 292 b1 r=1 (1,536, 50.08): 36.5× fewer params, 13.05 PPL better → STRICTLY DOMINATES cycle 293.

Net: cycle 293 is the WORST single-block c_proj anchor on BOTH axes simultaneously. Highest param count (56,064) AND highest PPL (63.13). Not a Pareto point under any reading.

## Geometric reading — audit

The item's Hypothesis asks whether the rank-eff_rank recipe scales to ~72 dimensions or whether block 0's input proximity (LM embedding output, untrained-by-FFN) makes it qualitatively different. Audit reading:

(i) **The recipe REFUTES on block 0.** This is the FIRST single-block c_proj REFUTATION. Across cycles 280, 282, 285, 288, 289, 292 (5 prior blocks, 7 prior data points), every block tested produced a retrain best_mean BELOW float; cycle 293 is the first to land ABOVE float. The "rank ≈ eff_rank + Adam retrain → 3-5 PPL below float" pattern broke catastrophically.

(ii) **Block 0's no-retrain collapse is qualitatively distinct.** Noretrain r=72 PPL = 396.26 is roughly 7× worse than the worst noretrain init in any prior block (cycle 280 b1 r=1 = 59.75). Even at rank-72 (matching the cycle 278 eff_rank = 71.93), block 0 cannot be reconstructed via SVD truncation alone.

(iii) **Adam recovery is the largest measured (-333 PPL), but the floor lands above float.** Recovery 396.26 → 63.13 = -333.13 PPL is 35-40× larger than typical block recoveries (~8-12 PPL). Block 0 has the most slack to recover but the smallest cumulative ceiling — the slack is wasted because the floor is fixed at +9.24 PPL above float.

(iv) **Block 0 c_proj weight SVD is the FLATTEST tested.** Top-5 = (25.0, 21.1, 18.3, 17.4, 16.3); S[0]/S[4] ≈ 1.5×, even flatter than the prior flat blocks (b1, b2, b10 had S[0]/S[4] ≈ 2.1×; b3 had 3.6×). Top-72 captures only 32.7% of weight matrix energy. The cycle 278 eff_rank = 71.93 was an OUTPUT-side measurement; the WEIGHT side has no rank concentration to exploit.

(v) **The "early-vs-late distinction" question has a clean answer**: block 0 is QUALITATIVELY different from other early concentrators. Block 1 (depth 1, eff_rank 1.94) and block 2 (depth 2, eff_rank 1.05) are both in the recipe's confirmation set; block 0 (depth 0, eff_rank 71.93) is not. The discontinuity is at depth 0, not at the early-vs-late boundary.

(vi) **Mechanistic reading consistent with prior blocks**: cycle 198 found block 0 input rank = 269 (vs ~1 for other blocks), and cycle 187 found dropping block 0 catastrophic (PPL 1923). Block 0's c_proj reads from a 269-dimensional input subspace and writes a high-rank embedding-to-residual chart. It is NOT a low-rank concentrator and the cycle 278 OUTPUT eff_rank = 71.93 reflects this — a 36× higher eff_rank than the next-most-rank-rich block (b3 at 24.29). The recipe's working assumption (concentrator-like c_proj structure with top-1 OUTPUT energy 0.5-0.99) does not apply to block 0.

(vii) **Sub-eff_rank ratio**: r/eff_rank = 72/71.93 = 1.001 — block 0 was tested at exactly its eff_rank, the most generous configuration possible under the recipe. Even this fails. The recipe is not a sub-eff_rank stress test for block 0; it is the recipe's fairest possible test on block 0, and the recipe still loses by 9.24 PPL.

(viii) Block 0 r=72 effectively saturates the trainable budget: 56,064 params is the largest single-block c_proj fit in this family (3.17× cycle 270 b11, 2.92× cycle 288 b3, 4.06× cycle 282 b10, 24.3× cycle 280 b1). The failure is NOT a budget problem — it is a STRUCTURAL problem with block 0's role in the model.

## Artifact existence

- experiments/block0_lowrank_cycle293.py — exists, ~393 lines
- autoresearch/plans/cycle_293.md — exists, 30 lines, 8 H2 headings
- autoresearch/briefs/cycle_293.md — exists (from brief stage)
- autoresearch/results/cycle_293_block0_lowrank.json — exists, valid JSON, verdict="REFUTED"
- autoresearch/results/cycle_293_summary.md — exists
- autoresearch/results/cycle_293_manifest.txt — exists, commit=7368ff5
- autoresearch/results/cycle_293_exec_result.txt — exists, commit=7368ff5
- autoresearch/logs/cycle_293_run.log — exists
- Primary commit 7368ff5 exists in git log
- experiments.tsv row 293 appended with 8 columns ✓
- agenda.md: item 303 deleted from NEXT P1, added to PAST as Cycle 293 ✓
- STATUS.md: title bumped to 293 cycles, cycle 293 entry prepended ✓

## Corrections

None. Provisional outcome holds.

## Final status

**PASS** — keep provisional outcome REFUTED. No agenda fix required beyond the move from NEXT to PAST already executed. Manifest, exec_result, JSON, summary, agenda, TSV, STATUS, and audit all agree on verdict = REFUTED with best_mean = 63.1327 ± 0.1009 ≥ 58.0 by 5.13 PPL.

Auditor note: cycle 293 is the FIRST single-block c_proj recipe REFUTATION. Block 0 sits qualitatively outside the concentrator cluster — it is the model's embedding-to-residual chart, not a low-rank operator. Future single-block c_proj rank patches should treat blocks 0 and 5 (eff_rank 179.19 per cycle 278) as the "high-rank chart" cluster and not expect the rank-eff_rank recipe to apply. The remaining tested-or-untested blocks under the recipe are 4 (eff_rank 49.04), 6-9 (eff_rank 195-290+, all outside the recipe's working range), and the "high-rank chart" outliers 0 and 5. Block 4 is the only remaining plausible single-block test for the recipe at intermediate eff_rank (~49), but cycle 293's b0 result suggests the recipe's failure mode at high eff_rank is structural rather than gradient-budget. Multi-block cascades that include block 0 should NOT use rank-72 SVD for it — block 0 needs a different compression strategy (e.g., dense low-budget retrain or no compression at all).
