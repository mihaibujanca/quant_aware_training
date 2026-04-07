# Cycle 285 Audit — Item 295

**Audit status: PASS**

## Scope
Verify that cycle 285 execution adheres to item 295's verbatim Hypothesis, verdict thresholds, and the cycle 280 template it derives from; verify arithmetic and cross-check against the raw JSON.

## Verdict verification

Item 295 thresholds (verbatim):
- CONFIRMED iff (r=2 retrained best_mean ≤ 50.0) OR (r=1 retrained best_mean ≤ 51.0)
- REFUTED iff r=2 retrained best_mean ≥ 53.0
- INCONCLUSIVE otherwise

Measured:
- r=2 retrain best_mean = 49.32900 (from JSON `retrain_r2_aggregate.best_mean`)
- r=1 retrain SKIPPED (noretrain r=1 PPL = 72.381 > in-script gate 55.0)

Arithmetic:
- 49.32900 ≤ 50.0 → TRUE (margin = 0.67100) → CONFIRMED clause fires
- 49.32900 < 53.0 → REFUTED clause NOT triggered
- → **CONFIRMED** ✓ (matches manifest, exec_result, JSON)

## Per-seed and aggregate sanity

Seeds (JSON `retrain_r2_seeds`):
- 42  : init 64.0814 → best 49.2446 @ step 3700 → final@4000 49.9992 → restored 49.2446
- 137 : init 64.0814 → best 49.2253 @ step 3900 → final@4000 49.7387 → restored 49.2253
- 2024: init 64.0814 → best 49.5171 @ step 3300 → final@4000 49.5760 → restored 49.5171

Aggregate (population std):
- mean(49.2446, 49.2253, 49.5171) = 146.9870/3 = 48.9957... let me recompute properly.
- (49.2446 + 49.2253 + 49.5171) = 147.9870; /3 = 49.3290 ✓
- deviations: (49.2446−49.3290, 49.2253−49.3290, 49.5171−49.3290)
             = (−0.0844, −0.1037, +0.1881)
- squared: (0.007123, 0.010754, 0.035382); sum = 0.053259; /3 = 0.017753
- std = sqrt(0.017753) = 0.13324... matches reported 0.13321 to 4 decimals ✓
- min = 49.2253 (seed 137) ✓; max = 49.5171 (seed 2024) ✓

All three seeds: init PPL = 64.0814 (deterministic, SVD init is seed-free) ✓
All three seeds: `|restored - best| < 1e-3` (0, 0, 0 exactly) ✓
All three seeds: best step ∈ [3300, 3900], still descending at 4000 cap ✓

## No-retrain sweep sanity

| r  | PPL     | Δfloat   | top-r energy |
|----|---------|----------|--------------|
| 1  | 72.3808 | +18.486  | 0.0716       |
| 2  | 64.0814 | +10.186  | 0.0828       |
| 5  | 64.0945 | +10.200  | 0.1049       |
| 10 | 62.1562 |  +8.261  | 0.1342       |

- Energy monotonic in r ✓ (0.0716 < 0.0828 < 0.1049 < 0.1342).
- PPL monotone-ish in r (r=2 and r=5 nearly identical, 64.08 vs 64.09 — energies
  differ by only 0.022, and the dominant singular directions are most of what
  matters at this budget; this is a weak inversion within noise, not a bug).
- r=1 noretrain 72.38 >> gate 55.0 → r=1 retrain correctly SKIPPED ✓.
- After sweep, restore original c_proj.weight and assert PPL ≈ float: PASS per
  log ("Restored original c_proj.weight, PPL=53.895 (should match float 53.895)").

## Config / plan adherence

Plan (cycle_285.md) vs execution:
- TARGET_BLOCK=2 ✓
- RANKS_NORETRAIN=[1,2,5,10] ✓
- RETRAIN_RANK=2 ✓
- SEEDS=[42,137,2024] ✓
- N_STEPS=4000, LR=1e-4, EVAL_EVERY=100, best-state restore ✓
- N_CALIB=200, N_EVAL=100, SEQ_LEN=128, disjoint ✓
- MPS, GPT-2 small, wikitext-2 val ✓
- Trainable per fit = 2×768 + 768 = 2,304 asserted in-script ✓
- Float PPL 53.895 within 0.5 of 53.895 reference ✓

The r=1 retrain gate was plan-sanctioned ("second retrain ~8 min if needed"
and "if no-retrain sweep shows r=1 is in striking distance"). Gate threshold
R1_GATE_PPL=55.0 is a legitimate operationalisation of "striking distance"
— the CONFIRMED-r=1 threshold is 51.0 and noretrain r=1 is 72.38, so no
realistic Adam recovery could close the 21.4 PPL gap. Skipping was correct.

## Baseline cross-checks

| baseline | recorded | in manifest | match |
|---|---|---|---|
| float                       | 53.895   | 53.895   | ✓ |
| cycle 280 b1 r=2            | 49.0324  | 49.0324  | ✓ |
| cycle 270 b11 r=22          | 54.4687  | 54.4687  | ✓ |
| cycle 282 b10 r=17          | 50.0445  | 50.0445  | ✓ |
| cycle 283 [2] K=16          | 49.4621  | 49.4621  | ✓ |
| cycle 278 eff_rank b2       | 1.0503   | 1.0503   | ✓ |
| cycle 278 top-1 energy b2   | 0.9961   | 0.9961   | ✓ |

Delta cross-checks:
- 49.3290 − 53.8950 = −4.5660 (vs float)  ✓ (manifest says −4.56589)
- 49.3290 − 49.0324 = +0.2966 (vs cycle 280) ✓ (dominated; 0.30 PPL worse at same params)
- 49.3290 − 49.4621 = −0.1331 (vs cycle 283 [2] K=16) ✓ (dominates on both axes: 5.7× fewer params)
- 49.3290 − 50.0445 = −0.7155 (vs cycle 282) ✓
- 49.3290 − 54.4687 = −5.1397 (vs cycle 270) ✓

## Pareto reading — audit

Cycle 285 (2,304 params, 49.33 PPL) vs existing anchors:
- Cycle 280 b1 r=2 (2,304, 49.03): same params, 0.30 PPL worse → STRICTLY DOMINATED.
- Cycle 283 [2] K=16 (13,056, 49.46): 5.7× fewer params, 0.13 PPL better → STRICTLY DOMINATES.
- Cycle 282 b10 r=17 (13,824, 50.04): 6.0× fewer params, 0.72 PPL better → STRICTLY DOMINATES.
- Cycle 270 b11 r=22 (17,664, 54.47): 7.7× fewer params, 5.14 PPL better → STRICTLY DOMINATES.

Cycle 285 is a new best-in-class for `[2]`-isolation but is NOT a new global
Pareto winner because cycle 280 (block 1, not block 2) still dominates the
2,304-param tier.

## Geometric reading — audit

The item's Hypothesis conflates cycle 278's `eff_rank=1.05` (an OUTPUT
covariance property) with the INTRINSIC rank of the weight matrix. Audit
confirms the execution report's correction: block 2 c_proj WEIGHT SVD is
essentially flat (top-5 SVs 38.25, 15.16, 13.08, 12.10, 11.58; ratio
S[0]/S[4] ≈ 3.3× — not a low-rank matrix by any reasonable criterion).
The cycle 278 eff_rank=1.05 captures that the INPUT to c_proj lives on
essentially 1 direction in the data-active subspace, so the c_proj output
covariance is dominated by one direction — but this is a JOINT property of
weight and input distribution, not a weight-rank property.

Consequence for the Hypothesis: rank-1 or rank-2 weight-SVD does not "match"
block 2's eff_rank structure; it destroys the flat weight and forces Adam
to realign the 2 surviving directions. That this still recovers to 49.33
(below float by 4.57) is evidence that 2 retrained directions are sufficient
when the input is one-dimensional in practice — consistent with the cycle
280 finding at block 1 (eff_rank=1.94) but NOT explained by matching eff_rank.

The "rank-eff_rank-match recipe" framing in item 295's Context paragraph is
therefore IMPRECISE. Cycle 285 neither confirms nor refutes it in its
literal form (eff_rank 1.05 ≠ retrain rank 2). What it confirms is the
weaker statement: "rank-2 weight-SVD + Adam retrain recovers ≤50 PPL at
both block 1 and block 2". It does NOT confirm the stronger statement that
the MOST concentrated block should give the BEST rank-2 retrain — block 1
still wins at the same rank.

## Artifact existence

- experiments/block2_lowrank_cycle285.py — exists, 446 lines
- autoresearch/plans/cycle_285.md — exists, 34 lines, 8 headings
- autoresearch/briefs/cycle_285.md — exists (pre-existing from brief stage)
- autoresearch/results/cycle_285_block2_lowrank.json — exists, valid JSON
- autoresearch/results/cycle_285_summary.md — exists
- autoresearch/results/cycle_285_manifest.txt — exists, commit=4c6edf5
- autoresearch/logs/cycle_285_run.log — exists
- Primary commit 4c6edf5 exists in git log
- experiments.tsv row 285 appended with 8 columns ✓
- agenda.md: item 295 deleted from NEXT P0, added to PAST as Cycle 285 ✓

## Corrections

None. Provisional outcome holds.

## Final status

**PASS** — keep provisional outcome CONFIRMED. No agenda fix required beyond
the move from NEXT to PAST already executed. Manifest, exec_result, JSON,
summary, agenda, TSV, and audit all agree on verdict = CONFIRMED with
best_mean = 49.3290 ± 0.1332.

Auditor note: future items should distinguish "output covariance eff_rank"
from "weight matrix rank" in their Hypothesis statements. The literal
"rank-eff_rank-match recipe" is not what cycles 270/280/282/285 established;
they established "rank-small + Adam retrain works for c_proj across multiple
blocks when input concentration is high". That is a weaker but more precise
statement.
