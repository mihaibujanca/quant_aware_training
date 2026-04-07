# Cycle 259 — Per-token trajectory divergence GELU vs ReLU+b_leak (item 270)

**Verdict: CONFIRMED**

Single forward pass each for GPT-2 small with (A) GELU baseline and (B) ReLU+per-block
per-channel b_leak (cycle 258 recipe) on the standard 200x128 wikitext-2 calib slice
(seed 42). For each layer l in {0..12}, compute per-token
d_l(x) = ||x_G^l - x_R^l||_G under the pullback metric G = U^T U where U is the
unembedding matrix. Report median/p10/p90 of d_l and median per-token cos(d_l, d_{l+1})
under G.

- calib PPL A (GELU) = 45.0637
- calib PPL B (ReLU+b_leak) = 60.0816
- post-restore baseline PPL = 45.0637 (round-trip clean)

## Per-layer d_l (median / p10 / p90)

| layer l | median ||d_l||_G | p10 | p90 |
|---|---|---|---|
| 0 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| 1 | 2.2542e+02 | 1.5844e+02 | 4.0413e+02 |
| 2 | 2.9039e+02 | 2.0697e+02 | 5.1944e+02 |
| 3 | 3.9185e+02 | 2.8644e+02 | 6.8297e+02 |
| 4 | 4.6982e+02 | 3.6202e+02 | 7.8934e+02 |
| 5 | 5.4258e+02 | 4.2568e+02 | 8.8281e+02 |
| 6 | 6.6581e+02 | 4.9975e+02 | 1.1468e+03 |
| 7 | 7.9615e+02 | 5.8372e+02 | 1.3824e+03 |
| 8 | 1.0273e+03 | 7.0991e+02 | 1.8535e+03 |
| 9 | 1.2629e+03 | 8.6786e+02 | 2.1813e+03 |
| 10 | 1.7074e+03 | 1.1008e+03 | 3.0520e+03 |
| 11 | 2.6861e+03 | 1.4774e+03 | 4.9742e+03 |
| 12 | 1.0163e+04 | 3.3519e+03 | 1.9660e+04 |

## Per-layer per-token cos(d_l, d_{l+1}) under G

| l (pair l,l+1) | median | p10 | p90 |
|---|---|---|---|
| 0 | +0.0000 | +0.0000 | +0.0000 |
| 1 | +0.3718 | -0.3753 | +0.7960 |
| 2 | +0.5714 | +0.0074 | +0.8620 |
| 3 | +0.7032 | +0.2028 | +0.9080 |
| 4 | +0.7671 | +0.3879 | +0.9228 |
| 5 | +0.7814 | +0.4074 | +0.9283 |
| 6 | +0.8175 | +0.5198 | +0.9418 |
| 7 | +0.8388 | +0.5418 | +0.9528 |
| 8 | +0.8666 | +0.6308 | +0.9621 |
| 9 | +0.8855 | +0.6609 | +0.9686 |
| 10 | +0.9067 | +0.5934 | +0.9780 |
| 11 | +0.9061 | +0.3298 | +0.9755 |

## Curve fits (median d_l over l in [1, 12])

| fit | parameters | SSR |
|---|---|---|
| log-linear | a=+4.9763, b=+0.2771 | 3.8505e+07 |
| linear | a=-1.6241e+03, b=+5.0920e+02 | 4.6766e+07 |
| saturating | a=+2.2542e+02, c=+4.6572e+04, b=+0.0200 | 2.6704e+08 |

Best-fit: **log_linear**. Log-linear fit restricted to l in [1, 6] for the verdict
threshold: b_early = **+0.2135**.

## Verdict logic

- Clause 1: b_loglin on l in [1, 6] >= 0.20 -> **True** (value +0.2135).
- Clause 2: median per-token cos >= 0.6 in >= 4 of cos indices 1..6
  -> **True** (4 / 6 values >= 0.6).
- REFUTED growth clause: best-fit is linear or saturating AND early-band
  multiplicative ratio median_d[6]/median_d[1] = 2.95 < e^1.0 -> **False**.
- REFUTED cos clause: n(cos<=0.2) in idx 1..6 = 0 -> **False**.
- Verdict: **CONFIRMED**.

Notes:
- d_0 ~ 0.00e+00 by construction (both variants share transformer.drop output);
  log-linear fit uses l in [1, 12] and clause 1 uses l in [1, 6] restricted fit.
- cos index 0 = cos(d_0, d_1) is numerically undefined for d_0 ~ 0; the verdict
  counts cos indices 1..6 (6 values), not 0..5.

Plot: `autoresearch/results/cycle_259_divergence.png`. Per-layer aggregates and per-token arrays in JSON.
