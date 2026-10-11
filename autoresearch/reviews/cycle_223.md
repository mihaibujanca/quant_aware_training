# Cycle 223 — Design Review

**Item:** 228 (P1) — Lower K with Adam correction — compression efficiency curve
**Verdict:** APPROVE

## Plan summary

Test K=256 and K=128 with Adam W_out+b_out correction on BLOCKS=[1,4,7],
4000 steps, EVAL_EVERY=100, early-stop / best-state restore. 3 seeds
(42, 137, 2024). Reuses cycle 221 machinery verbatim. Verdict criterion is
K=256-only (CONFIRMED ≤52.0; REFUTED >58.0; otherwise INCONCLUSIVE);
K=128 only contributes the efficiency curve. K=512 row of the curve uses
the cycle 210 baseline (46.556).

## Pre-flight

- **P1 duplicate:** clean. Cycle 199 K-sweep had no Adam; cycle 205 was a
  single-block K-sweep. K=256/K=128 with 3-block Adam correction is new.
- **P2 dependency:** none — cycle 221 machinery already merged.
- **P3 wall time:** 2 K × 3 seeds × 4000 steps. Roughly comparable to
  cycle 221 (6 blocks × 3 seeds × 4000 steps); 3-block setup is faster
  per step. Plan orders K=256 first per fallback so the verdict-relevant
  arm completes even if K=128 is dropped.
- **P4 program/prompts:** no changes needed.
- **3-seed hard rule:** satisfied.

## Pitfall scan

- **MP1–MP8:** none apply. PPL is the metric, varies in an informative
  range (~46–62 across the existing K=512 + cycle-205 K-sweep results);
  measurement is task accuracy not a geometric probe.
- **MP6 (verdict misclassification):** explicitly handled — K=128 has
  no pass/fail criterion, only feeds the curve.

## Hyperparameters

LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, N_CALIB=200, N_EVAL=100,
SEQ_LEN=128 — identical to cycles 210/221, so the K-axis is the only
moving variable vs the K=512 baseline.

## Validation criteria

Plan asserts the cycle-221 best-state restore round-trip and lists
expected init-PPL ordering (K=128 > K=256 > K=512). Trainable-param
arithmetic spelled out (197376/block @ K=256, 98688/block @ K=128).

## Approve.
