# Cycle 193 Brief: C3+Hybrid Combined Modification (Item 203)

## Item

203 — Superadditive interaction isolation — C3+Hybrid combined modification

## Context

Cycles 188, 190 established:
- C3 alone (constant replace negative regime): all-12 PPL=27.314 (+20.33%)
- Hybrid alone (ReLU for positive regime): all-12 PPL=28.980 (+27.67%)
- ReLU+bias (both regimes changed): all-12 PPL≈42.6 (+88%)

If the two error types were independent: C3+Hybrid expected ≈ +48% PPL (sum). But ReLU+bias
gives +88% — ~40% extra superadditivity. However, ReLU+bias ≠ C3+Hybrid: ReLU replaces
both regimes with max(z,0), while C3+Hybrid replaces negative with per-channel mean and
positive with ReLU. The question: is the superadditivity specific to zero-point for negative
regime, or is it general to "any simultaneous modification of both regimes"?

## Experiment

Apply BOTH C3 (negative→per-channel mean) and Hybrid (positive→ReLU) to all 12 blocks.
This is the combined activation:
  z < 0: mu_neg[b][c]  (per-channel GELU(z<0) mean from calibration)
  z >= 0: ReLU(z) = z

Measure all-12 PPL. Compare to:
- C3 alone: 27.314
- Hybrid alone: 28.980
- Simple sum prediction: ~50 (22.7 + 7.33% + 10.97% ≈ +20.33% + +27.67% = +48%)
- ReLU+bias: ~42.6

## Verdict (matching baseline=22.700)
- Confirmed if: C3+Hybrid PPL > 45 (+98%) — superadditivity is general
- Refuted if: C3+Hybrid PPL ≈ 28-35 (+23 to +54%) — no superadditivity without ReLU zero-point
- Inconclusive if: 35-45 range
