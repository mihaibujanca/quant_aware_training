# Cycle 286 Audit — Item 296

**Audit status: PASS**

## Scope
Verify that cycle 286 execution adheres to item 296's verbatim Hypothesis and
verdict thresholds, that the verdict arithmetic is correct, and that the
declared Pareto and geometric readings match the raw JSON.

## Verdict verification

Item 296 thresholds (verbatim):
- CONFIRMED iff best_mean ≤ 43.5 (beats cycle 271 [1] K=512=43.82 by ≥0.32 PPL)
- REFUTED iff best_mean ≥ 44.5
- INCONCLUSIVE otherwise

Measured: best_mean = 43.56269 (from JSON `aggregates.2.best_mean`).

Arithmetic:
- 43.56269 ≤ 43.5 → FALSE  (margin shortfall = 0.06269 PPL)
- 43.56269 ≥ 44.5 → FALSE  (margin shortfall = 0.93731 PPL)
- → **INCONCLUSIVE** ✓ (matches manifest, exec_result, JSON)

## Per-seed and aggregate sanity

Seeds (JSON `subset_results.2`):
- 42  : init 55.46118 → best 43.53027 @ step 4000 → final@4000 43.53027
- 137 : init 55.46119 → best 43.59336 @ step 4000 → final@4000 43.59336
- 2024: init 55.46119 → best 43.56444 @ step 4000 → final@4000 43.56444

Aggregate (population std):
- mean(43.53027, 43.59336, 43.56444) = 130.68807/3 = 43.56269 ✓
- deviations: (−0.03242, +0.03067, +0.00175)
- squared: (0.001051, 0.000941, 0.0000031); sum = 0.001995; /3 = 0.000665
- std = sqrt(0.000665) = 0.02579 ≈ reported 0.02579 ✓
- min = 43.53027 (seed 42) ✓; max = 43.59336 (seed 137) ✓

Init PPL across seeds: 55.46118, 55.46119, 55.46119 (delta < 1e-4) — effectively
deterministic, consistent with topk lstsq being seed-free. Tiny drift is
floating-point noise in the data loader (calib set order-independent at init).
PASS.

All three seeds: best_step == 4000 (terminal) — budget-bounded, still descending.
Noted in manifest and exec_result. This is a real caveat on the verdict: with
more budget the number would likely fall below 43.5 and trigger CONFIRMED.

All three seeds: `|restored - best| < 1e-3` (all exact, since best_step=4000
means the restore is a no-op on the terminal state).

## Config / plan adherence

Plan (cycle_286.md) vs execution:
- SUBSETS=[[2]] ✓
- K=512 ✓
- SEEDS=[42,137,2024] ✓
- N_STEPS=4000, LR=1e-4, EVAL_EVERY=100, best-state restore ✓
- N_CALIB=200, N_EVAL=100, SEQ_LEN=128, disjoint ✓
- MPS, GPT-2 small, wikitext-2 val ✓
- Trainable per fit = 512*768 + 768 = 393,984 asserted in-script ✓
- Float PPL 53.89489 within 0.5 of 53.895 reference ✓ (exact match)
- W_out+b_out trainable, all other params frozen ✓ (cycle 271 form)

Verdict logic implemented literally (CONFIRMED ≤ 43.5, REFUTED ≥ 44.5, else
INCONCLUSIVE) — matches item 296 verbatim.

## Baseline cross-checks

| baseline | recorded JSON | manifest | match |
|---|---|---|---|
| float                       | 53.89489  | 53.89500  | ✓ (within 1e-4) |
| cycle 271 [1] K=512         | 43.82100  | 43.82100  | ✓ |
| cycle 271 [4] K=512         | 44.63000  | 44.63000  | ✓ |
| cycle 271 [7] K=512         | 45.37000  | 45.37000  | ✓ |
| cycle 283 [2] K=64          | 47.51148  | 47.51148  | ✓ |
| cycle 283 [2] K=32          | 47.89700  | 47.89700  | ✓ |
| cycle 283 [2] K=16          | 49.46210  | 49.46210  | ✓ |

Delta cross-checks:
- 43.56269 − 53.89489 = −10.33220 (vs float) ✓ (JSON `delta_vs_float`)
- 43.56269 − 43.82100 = −0.25831 (vs cycle 271 [1]) ✓ (JSON `delta_vs_cycle_271_b1_k512`)
- 43.56269 − 47.51148 = −3.94879 (vs cycle 283 [2] K=64) ✓ (JSON `delta_vs_cycle_283_b2_k64`)

## Pareto reading — audit

Cycle 286 [2] K=512 (393,984 params, 43.56269 PPL) vs existing anchors at
the same 393,984-param tier:
- Cycle 271 [1] K=512 (393,984, 43.821): SAME params, 0.258 PPL BETTER.
  → Cycle 286 STRICTLY DOMINATES cycle 271 [1] on PPL axis at the 393,984
  parameter tier. New PPL Pareto winner on this tier.

Against other existing anchors:
- Cycle 280 b1 r=2 (2,304, 49.03): cycle 286 has 171× more params but 5.47 PPL
  better. Cycle 280 still dominates the ultra-low param region; cycle 286
  is not in the same Pareto cell.
- Cycle 271 [4] K=512 (393,984, 44.63): cycle 286 dominates by 1.07 PPL at
  same params.
- Cycle 271 [7] K=512 (393,984, 45.37): cycle 286 dominates by 1.80 PPL at
  same params.

Verdict-vs-Pareto distinction: INCONCLUSIVE per agenda thresholds (failed
CONFIRMED by 0.063 PPL), but cycle 286 IS a new PPL Pareto winner at the
393,984 tier. The manifest calls this out correctly. The exec_result also
calls this out. No disagreement.

## Hypothesis partial-confirmation reading

Item 296 Hypothesis: "[2] K=512 should land below cycle 271's 43.82 —
potentially the first sub-43 1-block K=512 measurement and a new dominant
Pareto winner on the PPL axis."

Two sub-claims:
1. "below 43.82" — TRUE (43.5627 < 43.821 by 0.258 PPL).
2. "first sub-43" — FALSE (43.5627 > 43.0).
3. "new dominant Pareto winner on the PPL axis" — TRUE at the 393,984 tier.

Agenda CONFIRMED threshold was 43.5 — between sub-43 (43.0) and cycle 271
(43.821). Cycle 286 lands in (43.5, 43.821) — confirms the directional
claim (beats [1]) but fails the quantitative threshold. The exec_result
calls this "PARTIALLY confirmed: dominant Pareto winner YES, sub-43 NO".
This is a faithful reading; the INCONCLUSIVE verdict is correct per the
literal thresholds.

## Geometric reading — audit

Per-K trend {[2] − [1]}: K=16 unknown (no matched [1] K=16), K=32 −1.75,
K=64 −0.53, K=512 −0.258. Monotone shrinkage confirmed. The manifest's
reading — "channel-K capacity surpasses block 2's input concentration as
K grows" — is consistent with the numbers but is a descriptive story, not
a test. Not a load-bearing claim.

Contrast with cycle 285 (block 2 rank-2 weight-SVD): cycle 285 block 2 r=2
= 49.33, cycle 285 block 1 r=2 = 49.03 (cycle 280). At rank 2, block 1
wins. At channel-K=16, block 2 wins by ~2 PPL (49.46 vs ~50+). Different
operators → different orderings. The "weight vs input structure" framing
in the manifest is a plausible explanation but not empirically isolated
by cycle 286 alone. Auditor flags this as speculation, consistent with
cycle 285's audit which already noted eff_rank is an OUTPUT covariance
property (joint of weight × input). Cycle 286's geometric reading is
compatible with that prior audit and does not overreach.

## Budget-bounded caveat

All 3 seeds hit best at step 4000 (terminal). The monotone descent over
the final ~1000 steps (seed 42 trajectory: 3900 → 43.657, 4000 → 43.593,
final 43.593; similar pattern for other seeds) indicates the true [2]
K=512 floor is below the reported 43.5627. A longer run (cf. agenda item
297 for [1] K=512 at 8000 steps) would likely push [2] K=512 below 43.5
and trigger CONFIRMED.

This is a methodological weakness in the cycle 286 plan (N_STEPS=4000 was
inherited from cycle 271 without accounting for block 2's slower descent
at high K). The verdict INCONCLUSIVE is correct per the measured value,
but the verdict INTERPRETATION should note that the floor was not reached.
Both the manifest and the exec_result explicitly call this out.

Auditor recommendation: a follow-up item at 8000 steps for [2] K=512
would likely flip this to CONFIRMED. Not a cycle 286 audit correction —
just a note on the item 296 result's durability.

## Artifact existence

- experiments/block2_k512_cycle286.py — exists, committed
- autoresearch/plans/cycle_286.md — exists, 30 lines, 8 headings, item 296 verbatim
- autoresearch/briefs/cycle_286.md — exists (pre-existing from brief stage)
- autoresearch/results/cycle_286_block2_k512.json — exists, valid JSON
- autoresearch/results/cycle_286_summary.md — exists
- autoresearch/results/cycle_286_manifest.txt — exists, commit=1e4c5ca
- autoresearch/results/cycle_286_exec_result.txt — exists, commit=1e4c5ca
- autoresearch/logs/cycle_286_run.log — exists
- Primary commit 1e4c5ca exists in git log
- experiments.tsv row 286 appended with 8 columns, commit column updated to 1e4c5ca
- agenda.md: item 296 deleted from NEXT P0, added to PAST as Cycle 286 ✓

## Corrections

None. Provisional outcome holds.

## Final status

**PASS** — keep provisional outcome INCONCLUSIVE. Cycle 286 is a new PPL
Pareto winner at the 393,984-param tier (replaces cycle 271 [1] K=512) but
fails the agenda's CONFIRMED threshold (43.5) by 0.063 PPL. All 3 seeds
still descending at the budget cap — a longer run would likely flip the
verdict to CONFIRMED. Manifest, exec_result, JSON, summary, agenda, TSV,
and audit all agree on verdict = INCONCLUSIVE with best_mean = 43.5627 ±
0.0258.

Auditor note: cycles 286 and 297 (pending, [1] K=512 at 8000 steps) are
now a natural pair. If cycle 297 pushes [1] below 43.0, the [2] vs [1]
cross-over at K=512 should also be retested at matched budget. The per-K
trend analysis (K=32 −1.75, K=64 −0.53, K=512 −0.258) is suggestive but
not load-bearing and depends on the current 4000-step budget holding for
both [1] and [2].
