# Cycle 299 — Item 309: Cycle 280 retrain decomposition (B drift)

**Verdict: INCONCLUSIVE**

- Float PPL: 53.8949 (ref 53.895)
- Block 1, rank 2 LowRankCProj, seed 42
- Init PPL: 57.6581 (cycle 280 ref 57.664)
- Best PPL: 49.0603 @ step 3900  (cycle 280 ref 49.032)
- Adam recovery: +8.598 PPL over 4000 steps
- Final rowspace cos(B(t), B(0)) = **0.9111**
- Final fro_rel = 0.4392
- Final bias delta norm = 1.9972
- corr(fro_rel, ppl_improvement) = 0.9584

## Trajectory (every 100 steps)

| step | eval_ppl | cos(B,B0) | fro_rel | W_sv0 | W_sv1 | top1_cos_Vh | top2_cos_Vh | bias_Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 57.658 | 1.0000 | 0.000e+00 | 28.706 | 23.104 | 1.0000 | 1.0000 | 0.000e+00 |
| 500 | 52.616 | 0.9855 | 1.706e-01 | 28.741 | 22.835 | 0.9870 | 0.9837 | 3.720e-01 |
| 1000 | 51.221 | 0.9691 | 2.508e-01 | 29.070 | 22.695 | 0.9715 | 0.9650 | 6.660e-01 |
| 1500 | 50.297 | 0.9555 | 3.034e-01 | 29.376 | 22.612 | 0.9577 | 0.9488 | 9.348e-01 |
| 2000 | 49.927 | 0.9446 | 3.405e-01 | 29.636 | 22.645 | 0.9455 | 0.9361 | 1.176e+00 |
| 2500 | 49.405 | 0.9340 | 3.742e-01 | 29.851 | 22.662 | 0.9321 | 0.9229 | 1.389e+00 |
| 3000 | 49.449 | 0.9241 | 4.032e-01 | 30.037 | 22.728 | 0.9190 | 0.9112 | 1.607e+00 |
| 3500 | 49.227 | 0.9179 | 4.213e-01 | 30.311 | 22.756 | 0.9099 | 0.9040 | 1.801e+00 |
| 4000 | 49.250 | 0.9111 | 4.392e-01 | 30.401 | 22.779 | 0.8999 | 0.8960 | 1.997e+00 |
| 4000 | 49.250 | 0.9111 | 4.392e-01 | 30.401 | 22.779 | 0.8999 | 0.8960 | 1.997e+00 |

## Verdict logic (item 309 verbatim)

- **CONFIRMED drift** iff rowspace_cos(B(4000), B(0)) ≤ 0.6 AND held-out PPL improvement tracks drift (corr(fro_rel, ppl_improvement) ≥ 0.5).
- **CONFIRMED stay-put** iff rowspace_cos(B(t), B(0)) ≥ 0.95 for all snapshots.
- **INCONCLUSIVE** otherwise.

- cos_final = 0.9111  → drift_cos_ok (≤ 0.6)? **False**
- all cos ≥ 0.95? **False**  (min=0.9111, max=1.0000)
- corr(fro_rel, ppl_improvement) = 0.9584  → tracking_ok? **True**
- VERDICT: **INCONCLUSIVE**

## Interpretation

Neither the drift nor the stay-put clause fires cleanly. The substantive reading depends on which quantities moved; see the trajectory table above.
