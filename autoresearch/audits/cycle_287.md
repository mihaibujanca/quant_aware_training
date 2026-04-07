# Cycle 287 Audit — Item 297

**Audit status: PASS**

## Scope
Verify that cycle 287 execution adheres to item 297's verbatim Hypothesis and
verdict thresholds, that the verdict arithmetic is correct, that the
matched-config sanity-check holds against cycle 271 [1] K=512 4k, and that
the declared Pareto and budget-extension claims match the raw JSON.

## Verdict verification

Item 297 thresholds (verbatim):
- CONFIRMED iff best_mean ≤ 43.0 (extension buys ≥0.82 PPL)
- REFUTED iff best_mean ≥ 43.7 (extension buys <0.12 PPL)
- INCONCLUSIVE otherwise

Measured: best_mean = 42.78529 (from JSON `aggregates.1.best_mean`).

Arithmetic:
- 42.78529 ≤ 43.0 → TRUE  (CONFIRMED clause passes by 0.21471 PPL margin)
- → **CONFIRMED** ✓ (matches manifest, exec_result, JSON)

## Per-seed and aggregate sanity

Seeds (JSON `subset_results.1`):
- 42  : init 54.68924816 → best 42.78651988 @ step 8000 → final@8000 42.78651988
- 137 : init 54.68924529 → best 42.80000092 @ step 8000 → final@8000 42.80000092
- 2024: init 54.68924529 → best 42.76934406 @ step 8000 → final@8000 42.76934406

Aggregate (population std):
- mean(42.78651988, 42.80000092, 42.76934406) = 128.35586486/3 = 42.78528829 ✓
- deviations: (+0.001231595, +0.014712631, −0.015944226)
- squared: (1.5168e-6, 2.1646e-4, 2.5422e-4); sum = 4.7220e-4; /3 = 1.5740e-4
- std = sqrt(1.5740e-4) = 0.012546 ≈ reported 0.012546 ✓
- min = 42.76934 (seed 2024) ✓; max = 42.80000 (seed 137) ✓

Init PPL across seeds: 54.68925, 54.68925, 54.68925 (delta < 1e-6 internally;
3.4e-9 between seeds 137/2024 and seed 42) — effectively deterministic,
consistent with topk lstsq being seed-free. PASS.

All three seeds: best_step == 8000 (terminal) — budget-bounded, still
descending. Noted in manifest and exec_result.

All three seeds: `|restored - best| < 1e-3` (all exact; restore is a no-op
on terminal state). PASS.

## Matched-config reproducibility vs cycle 271 [1] K=512 4k

Step-4000 eval per seed (cycle 287 vs cycle 271 — both use the same script
template, same scorer, same lstsq init, same Adam state, same dataloader
RNG seed):

| seed | cycle 287 step-4000 | cycle 271 step-4000 | delta |
|---|---|---|---|
| 42   | 43.82617220 | 43.82616280 | +9.4e-6 |
| 137  | 43.84588643 | 43.84588361 | +2.8e-6 |
| 2024 | 43.78931213 | 43.78931819 | -6.1e-6 |

All deltas < 1e-4. PASS — the only thing that differs between cycle 271 [1]
K=512 4k and cycle 287 [1] K=512 8k is the budget extension. The 1.04 PPL
improvement (43.821 → 42.785 on aggregate, −0.0019 to +0.011 on individual
seeds at step 4000) is unambiguously attributable to the additional 4000
Adam steps. This is the cleanest budget-extension reproduction in the
cascade family so far.

EVAL_EVERY changed from 100 (cycle 271) to 200 (cycle 287) to keep the
total eval-points count at 40. This does NOT affect optimizer state because
eval is in `model.eval()` mode under `torch.no_grad()` — no gradients are
computed during eval. The reproducibility check confirms this empirically.

## Config / plan adherence

Plan (cycle_287.md) vs execution:
- SUBSETS=[[1]] ✓
- K=512 ✓
- SEEDS=[42,137,2024] ✓
- N_STEPS=8000, LR=1e-4, EVAL_EVERY=200, best-state restore ✓
- N_CALIB=200, N_EVAL=100, SEQ_LEN=128, disjoint ✓
- MPS, GPT-2 small, wikitext-2 val ✓
- Trainable per fit = 512*768 + 768 = 393,984 asserted in-script ✓
- Float PPL 53.89489 within 0.5 of 53.895 reference ✓ (exact)
- W_out+b_out trainable, all other params frozen ✓ (cycle 271 form)

Verdict logic implemented literally (CONFIRMED ≤ 43.0, REFUTED ≥ 43.7,
else INCONCLUSIVE) — matches item 297 verbatim.

## Baseline cross-checks

| baseline | recorded JSON | manifest | match |
|---|---|---|---|
| float                       | 53.89489  | 53.89500  | ✓ (within 1e-4) |
| cycle 271 [1] K=512 4k      | 43.82045  | 43.82045  | ✓ |
| cycle 286 [2] K=512 4k      | 43.56269  | 43.56269  | ✓ |

Delta cross-checks:
- 42.78529 − 53.89489 = −11.10960 (vs float) ✓ (JSON `delta_vs_float`)
- 42.78529 − 43.82045 = −1.03517 (vs cycle 271 [1] 4k) ✓ (JSON `delta_vs_cycle_271_b1_k512_4k`)
- 42.78529 − 43.56269 = −0.77740 (vs cycle 286 [2] 4k) ✓ (JSON `delta_vs_cycle_286_b2_k512_4k`)

## Pareto reading — audit

Cycle 287 [1] K=512 8k (393,984 params, 42.78529 PPL) vs prior anchors at
the same 393,984-param tier:
- Cycle 286 [2] K=512 4k (393,984, 43.5627): SAME params, 0.7774 PPL BETTER.
  → Cycle 287 STRICTLY DOMINATES cycle 286 [2] on PPL axis. New PPL Pareto
  winner on this tier. Cycle 286 was the prior winner (replacing cycle 271
  [1] 4k); cycle 287 displaces both.
- Cycle 271 [1] K=512 4k (393,984, 43.8205): same params, 1.0352 PPL better.
  Trivially dominated (matched-config 4k counterpart of cycle 287).
- Cycle 271 [4] K=512 4k (393,984, 44.6330): cycle 287 dominates by 1.85 PPL.
- Cycle 271 [7] K=512 4k (393,984, 45.3745): cycle 287 dominates by 2.59 PPL.

Smaller-param-tier dominators (different cells):
- Cycle 280 b1 r=2 (2,304, 49.03): still dominates the ultra-low param region.
- Cycle 282 b10 r=17 (13,824, 50.04): dominates the mid-low param region.
- Cycle 283 [2] K=16 (13,056, 49.46): dominates the same mid-low region.
None of these are at 393,984 params, so cycle 287 takes the 393,984-tier
crown unconditionally.

## Hypothesis verification — audit

Item 297 Hypothesis (verbatim): "8000 steps drops cycle 271 [1] K=512 below
43.0 — establishes a new dominant PPL Pareto anchor."

Two sub-claims:
1. "below 43.0"                            → TRUE  (42.7853 < 43.0 by 0.215)
2. "new dominant PPL Pareto anchor"        → TRUE  at the 393,984 tier
Both confirmed. CONFIRMED is the literally correct verdict.

Per-seed margin: all 3 seeds individually below 43.0:
- seed 42  : 42.78652 < 43.0 by 0.21348
- seed 137 : 42.80000 < 43.0 by 0.20000
- seed 2024: 42.76934 < 43.0 by 0.23066
No seed marginally near the threshold. The verdict is robust.

## Budget-extension family — audit

Manifest update of the cascade extension family:
- cycle 269 K=1   6-block 4k→16k  bought −5.27 PPL  (4× extension, large)
- cycle 287 [1]   K=512 1-block 4k→8k bought −1.04 PPL  (this cycle, 2× extension)
- cycle 274 K=64  5-block 4k→8k   bought −0.80 PPL  (2× extension)
- cycle 254 K=128 5-block 4k→8k   bought −0.10 PPL  (2× extension, plateau)

Cross-check: cycle 269 was a 4× extension (4k→16k), so per-doubling rate
is roughly −2.6 PPL / doubling for that cell. Cycle 287's −1.04 PPL on a
single doubling places it firmly between cycle 274 (−0.80) and cycle 269
(−2.6 per-doubling). The "fewest trainable → most slack per param"
heuristic from item 297 is empirically consistent (394K is the fewest
trainable of any cascade tested at K=512; the per-doubling gain is on the
high end of the 2× extensions). NOT a load-bearing claim — just
qualitatively consistent.

## Geometric reading — audit

The matched-config reproducibility (<1e-4 per seed at step 4000) means the
[1] K=512 4k cap was BUDGET-LIMITED, not floor-limited. This is the
strongest possible empirical demonstration: same scorer, same lstsq init,
same optimizer state for 4000 steps; the only difference is whether the
optimizer continues for another 4000 steps. The 1.04 PPL gain is purely
from additional gradient descent on the same loss surface.

The manifest's reading — "the cycle 271 [1] K=512 4k cap was budget-
limited, not a true compression floor" — is faithful to the data and not
overreaching.

The auditor concurs with the manifest's recommendation that cycle 286 [2]
K=512 should be retested at 8000 steps. Cycle 286 was budget-bounded
(all 3 seeds best @ step 4000) and INCONCLUSIVE per agenda thresholds.
Cycle 287 demonstrates that the [1] K=512 4k cap was budget-limited; by
symmetry the [2] K=512 4k cap is almost certainly also budget-limited.
At matched 8k budgets the per-K [2]-vs-[1] cross-over may differ. This
cycle does NOT establish that [1] beats [2] in absolute terms — only that
at 8k vs cycle 286's 4k the [1] PPL is lower. This nuance is correctly
captured in both manifest and exec_result.

## Artifact existence

- experiments/one_block_1_k512_8k_cycle287.py — exists, committed
- autoresearch/plans/cycle_287.md — exists, 25 lines, 8 headings, item 297 verbatim
- autoresearch/briefs/cycle_287.md — exists (pre-existing from brief stage)
- autoresearch/results/cycle_287_one_block_1_k512_8k.json — exists, valid JSON
- autoresearch/results/cycle_287_summary.md — exists
- autoresearch/results/cycle_287_manifest.txt — exists, commit=920dcde
- autoresearch/results/cycle_287_exec_result.txt — exists, commit=920dcde
- autoresearch/logs/cycle_287_run.log — exists
- Primary commit 920dcde exists in git log
- experiments.tsv row 287 appended with 8 columns, commit column updated to 920dcde
- agenda.md: item 297 deleted from NEXT P0, added to PAST as Cycle 287 ✓

## Corrections

None. Provisional outcome holds.

## Final status

**PASS** — keep provisional outcome CONFIRMED. Cycle 287 is a new PPL
Pareto winner at the 393,984-param tier (replaces cycle 286 [2] K=512 4k)
at best_mean=42.7853 ± 0.0125, passing the CONFIRMED threshold (43.0) by
0.215 PPL with all 3 seeds individually below the line. The matched-config
step-4000 reproducibility against cycle 271 [1] K=512 4k is exact (<1e-4
per seed), making the 1.04 PPL improvement unambiguously attributable to
the additional 4000 Adam steps. All 3 seeds still descending at step 8000
— a 16k extension would likely push best_mean below 42.5. Manifest,
exec_result, JSON, summary, agenda, TSV, and audit all agree on
verdict = CONFIRMED with best_mean = 42.7853 ± 0.0125.

Auditor follow-up recommendation: a P0 cycle 286 [2] K=512 8k retest is
now strongly indicated. Cycle 286's auditor already flagged this; cycle
287's confirmation that the 4k cap was budget-limited (not floor-limited)
makes the case stronger. Until that retest is done, the cycle 287 vs cycle
286 comparison is at MISMATCHED budgets — not a clean apples-to-apples
result. Cycle 287 fairly takes the 393,984-tier PPL crown by the standard
of "what we have measured", but the underlying [1]-vs-[2] question at K=512
remains open at matched budgets.
