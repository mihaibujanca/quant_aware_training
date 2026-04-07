# Cycle 243 — Item 253: 6-block per-block PPL attribution

**Verdict: CONFIRMED**

- Criteria: Confirmed if any `ppl_if_X_restored ≤ 56.0`; Refuted if all 6 deltas in `[0.5, 1.5]`; else Inconclusive.
- Config: BLOCKS=[1, 2, 4, 5, 7, 8], K=512, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEED=42
- Phase 0 refit: init=86.686 → best=59.391 at step 1300
- Float baseline (held-out): 53.895
- Fully compressed (cycle 221 best-state, seed 42 refit): 59.391
- Cycle 221 seed-42 reference best: 59.391 (Δ vs refit: -0.000)
- Compression gap (fully − float): +5.496 PPL

## Per-block restoration scan (sorted by ΔPPL desc)

| rank | block | ppl_if_X_restored | ΔPPL (fully − ppl_if_X) | residual_mse |
|---|---|---|---|---|
| 1 | 5 | 54.2910 | +5.1000 | 3.2387e-01 |
| 2 | 4 | 54.4951 | +4.8959 | 2.1236e-01 |
| 3 | 7 | 55.2564 | +4.1346 | 5.6465e-01 |
| 4 | 8 | 55.6076 | +3.7834 | 9.2819e-01 |
| 5 | 2 | 55.8607 | +3.5303 | 2.2219e-01 |
| 6 | 1 | 56.6992 | +2.6917 | 1.0165e-01 |

## Notes

- Sum of deltas: +24.136 PPL (compare to +5.496 compression gap; non-additivity expected).
- Each restored ppl is in `[float, fully_compressed]` (monotone): min=54.291, max=56.699.
