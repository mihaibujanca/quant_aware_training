verdict: APPROVE with NON-BLOCKING notes
round: 1
issues: 0 blocking, 2 non-blocking
feedback:
  - NB1 (overfitting risk): 12 adapters = 1.19M params trained on 200 × 128 = 25.6K tokens (~46 params/token). c377 had 297K params on the same data (~12 params/token), so 4× the param density. The c378 protocol mitigates this (checkpoint selection on val, final eval on held-out test), so test PPL is honest. But watch for train PPL << val PPL divergence — if train PPL drops to single digits while test PPL stalls, the adapters are memorizing. Not blocking because the question is empirical: does 12-block adapter beat 3-block on test? Overfitting would just produce a REFUTED/INCONCLUSIVE verdict, which is a valid answer.
  - NB2 (2000 steps may underfit 12 adapters): 12 adapters have 4× more params but the same 2000 steps. Convergence is not guaranteed. The plan should log final training loss and val loss curves. If 12-block adapter shows PPL still decreasing at step 2000, the result is a lower bound, not a ceiling. Consider noting this in the verdict interpretation.
  - Split protocol: Correctly follows c378 — train on val (200 seqs), checkpoint select on val eval (50 seqs), final eval on test (200 seqs, touched once). Clean.
  - Thresholds: CONFIRMED < 36.0, REFUTED > 38.0. 3-block test ref is 39.73. Range is tight and meaningful.
  - Seeds: 3 seeds (42, 137, 2024), matching c377/c378. Sufficient for this measurement.
  - Pitfall scan: MP3 (underdetermined fitting) is relevant but mitigated by held-out test eval. No other pitfalls triggered.
