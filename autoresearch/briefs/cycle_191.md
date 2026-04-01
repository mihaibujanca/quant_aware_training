# Cycle 191 Brief: Mean Residual Error Correction (Item 201)

## Item

201 — Mean residual error correction — directional coherence-based bias (9216 params)

## Context

Cycle 189 confirmed that GELU→ReLU+bias cascade errors are directionally coherent
(mean cos=0.716, 11/11 layers). Despite being high-rank (rank_90=300-550/768), the
MEAN error vector per layer is nonzero and directly measurable.

Hypothesis: if we compute mean_error[l] = E[r_float[l] - r_swapped[l]] over calibration
tokens, we can add this as a constant bias to each swapped block's residual stream output.
This is the cheapest possible correction: 12 × 768 = 9,216 parameters, no training.

## Experiment Design

1. Build full-swap ReLU+bias model (same as cycles 181, 189)
2. Calibration pass (N=200 seqs, seed=1337): collect residuals from both float and swapped
   models, compute mean_correction[l] = mean(r_float[l] - r_swapped[l], dim=0) per layer
3. Register correction as hook on swapped model's transformer.h[l] output
4. Evaluate corrected PPL on analysis set (N=100 seqs, seed=42)
5. Compare: baseline (~35.9 at SEQ_LEN=128, using cycle 189 config), uncorrected full-swap
   (~44.8), corrected full-swap

## Verdict

- CONFIRMED: corrected PPL < 38 (saves >6 PPL from 44.8 uncorrected, >15% improvement)
- REFUTED: corrected PPL within 1 PPL of 44.8 (first moment is not useful)
- INCONCLUSIVE: 1-6 PPL improvement (partial benefit)

## Notes

Use same SEQ_LEN=128 and seeds as cycle 189 to ensure comparability of uncorrected PPL.
The correction vectors are computed from calibration data different from analysis data
(same split as cycle 189: calib seed=1337, analysis seed=42).
