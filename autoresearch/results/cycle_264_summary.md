# Cycle 264 — Per-block partial swap κ heatmap (item 274)

**Verdict: INCONCLUSIVE**

For each k ∈ {0..11}, install ReLU + per-channel b_leak (cycle 258 recipe) ONLY at GPT-2 small block k, leave the other 11 blocks GELU, compute the κ profile under G = U^T U on 200×128 wikitext-2 calib (seed 42). κ via `aleph.pullback_geometry.compute_curvature(reduce='mean')` (11 values for layers 1..11).

- Baseline (all-GELU) PPL = 45.0637
- Post-restore PPL = 45.0637 (round-trip clean)
- max|Δκ|_diag = 8.0039e-04
- median|Δκ|_offdiag = 0.0000e+00
- diag/offdiag ratio = inf (need ≥3.0 for CONFIRMED clause 1)
- late-layer monotone cols (l∈{7..10}): 0/4 (need ≥3 for CONFIRMED clause 2)

**‖Δκ‖₁ per swap location k (descending):**

| rank | swap_at_k | ‖Δκ‖₁ | patched PPL |
|---|---|---|---|
| 1 | 2 | 1.7799e-03 | 46.0503 |
| 2 | 3 | 1.2242e-03 | 45.9652 |
| 3 | 1 | 5.9954e-04 | 45.6715 |
| 4 | 4 | 4.3588e-04 | 46.1116 |
| 5 | 0 | 4.1935e-04 | 45.0860 |
| 6 | 5 | 1.4890e-04 | 45.9834 |
| 7 | 6 | 4.3154e-05 | 45.8505 |
| 8 | 7 | 1.1357e-05 | 45.7750 |
| 9 | 8 | 6.1091e-06 | 45.6363 |
| 10 | 9 | 3.7940e-06 | 45.6122 |
| 11 | 11 | 2.4792e-06 | 45.4558 |
| 12 | 10 | 1.0422e-06 | 45.1888 |

**κ_baseline (layers 1..11):**

`+5.868e-05 +2.233e-03 +2.223e-03 +1.853e-03 +1.093e-03 +6.674e-04 +3.359e-04 +1.886e-04 +1.042e-04 +2.338e-05 +3.840e-06`

**Δκ matrix (rows = swap_at_k, columns = κ index l = layer l+1):**

| k\l | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | +3.35e-06 | +2.01e-04 | +1.53e-04 | -8.98e-06 | -3.51e-05 | -1.39e-05 | -1.94e-06 | -1.03e-06 | -3.59e-07 | +3.16e-08 | -6.51e-09 |
| 1 | +4.25e-06 | -5.15e-05 | +1.95e-04 | +2.50e-04 | +7.46e-05 | +1.58e-05 | -6.88e-06 | -1.45e-06 | -2.31e-07 | +1.33e-07 | -2.72e-07 |
| 2 | +0.00e+00 | +5.72e-04 | +8.00e-04 | +3.49e-04 | +3.35e-05 | -1.23e-05 | -9.39e-06 | -2.76e-06 | -1.40e-07 | +2.51e-07 | +1.59e-07 |
| 3 | +2.69e-04 | +0.00e+00 | +2.81e-04 | +3.99e-04 | +2.13e-04 | +5.12e-05 | -4.66e-06 | -4.73e-06 | -1.63e-06 | +9.91e-08 | +5.12e-08 |
| 4 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +1.15e-04 | +1.95e-04 | +1.03e-04 | +2.06e-05 | +1.88e-06 | -3.32e-07 | +1.35e-07 | +3.64e-07 |
| 5 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +4.66e-05 | +5.94e-05 | +3.25e-05 | +9.73e-06 | +1.98e-07 | +1.19e-07 | +2.40e-07 |
| 6 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +2.21e-05 | +1.68e-05 | +1.90e-06 | +2.10e-06 | +1.65e-07 | -8.55e-08 |
| 7 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +3.46e-06 | +4.79e-06 | +2.81e-06 | +2.02e-07 | -8.82e-08 |
| 8 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +2.76e-06 | +2.60e-06 | +6.95e-07 | -6.04e-08 |
| 9 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +2.73e-06 | +8.04e-07 | -2.56e-07 |
| 10 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +6.65e-07 | -3.78e-07 |
| 11 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +2.48e-06 |

**Verdict logic.** CONFIRMED requires (a) max|Δκ| on the diagonal band ≥ 3× the off-diagonal median AND (b) late-layer columns (l ∈ {7..10}) show |Δκ| monotonically non-decreasing as swap location k decreases through k=6..0. Diagonal band entries: for swap_at_k, the three κ indices {k-1, k, k+1} ∩ {0..10} (k=0 → {0,1}; k=11 → {10,11} ∩ valid = {10}).

- Clause 1 (diagonal dominance ≥ 3×): **PASS** (ratio=inf)
- Clause 2 (late-layer monotone): **FAIL** (0/4 cols pass)
- REFUTED triggers if `diag/offdiag<1.0` OR max_diag<median_offdiag.
- REFUTED structureless: False
- REFUTED antidiagonal:  False

**Result: INCONCLUSIVE**.

Heatmap: `autoresearch/results/cycle_264_swap_kappa_heatmap.png`. Full Δκ matrix and per-swap κ profiles saved in JSON.
