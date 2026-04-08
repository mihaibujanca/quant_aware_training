# Cycle 315 — Item 325: Calibrated SVD-LLM init + Adam retrain at block 1

**Verdict: REFUTED**

- Float PPL: 53.8949 (ref 53.895)
- Target: block 1 c_proj, rank 2, HF Conv1D (3072, 768)
- Trainable per fit: 2,304 = 2×768 (B) + 768 (bias); 2 variants × 3 seeds = 6 fits
- Wall: 15.7 min retrain

## Init PPLs (no retrain)

- Uncentered: 57.6581  (cycle 280 ref 57.664, |delta|=0.0059)
- Calibrated: 56.7030  (cycle 313 ref 56.7, |delta|=0.0030)
- Init delta (cal − unc): -0.9551

## Retrain (Adam LR=0.0001, 4000 steps, 3 seeds, best-state restore)

### Uncentered (cycle 280 recipe)

| seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|
| 42 | 57.6581 | 49.0603 | 3900 | 49.2498 | 49.0603 |
| 137 | 57.6581 | 49.0367 | 3900 | 49.0741 | 49.0367 |
| 2024 | 57.6581 | 49.0004 | 4000 | 49.0004 | 49.0004 |
| **mean** | — | **49.0324** | — | — | — |
| **std**  | — | 0.0246 | — | — | — |

### Calibrated (SVD-LLM)

| seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|
| 42 | 56.7030 | 48.9855 | 3900 | 49.1970 | 48.9855 |
| 137 | 56.7030 | 48.9944 | 3700 | 49.0707 | 48.9944 |
| 2024 | 56.7030 | 48.8848 | 4000 | 48.8848 | 48.8848 |
| **mean** | — | **48.9549** | — | — | — |
| **std**  | — | 0.0497 | — | — | — |

## Verdict logic (item 325 verbatim)

- delta_retrain = mean(calibrated) − mean(uncentered) = **-0.0775 PPL**
- CONFIRMED iff ≤ -1.0
- REFUTED iff |delta| ≤ 0.3
- VERDICT: **REFUTED**

## Comparisons

- Uncentered best_mean vs cycle 280 retrain anchor (49.03): +0.0024
- Calibrated best_mean vs cycle 280 retrain anchor (49.03): -0.0751
- Init gap closed by retrain (uncentered): +8.6257
- Init gap closed by retrain (calibrated): +7.7480

## Sanity

- Float PPL 53.8949 matches cycle ref 53.895 within 0.5
- Uncentered init PPL matches cycle 280 published 57.664 within 0.01 (|delta|=0.0059)
- Calibrated init PPL matches cycle 313 published 56.7 within 0.05 (|delta|=0.0030)
- SVD(W_r_cal) reconstruction ||A·B − W_r||/||W_r|| = 2.57e-06
- Σ symmetric err = 0.00e+00; ||A²−Σ||/||Σ|| = 1.44e-05; pinv rel err = 1.38e-03
- All 6 fits: trainable count == 2,304 asserted; |restored − best| < 1e-3 asserted
