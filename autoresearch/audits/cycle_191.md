# Audit — Cycle 191 (Item 201)

**Verdict:** REFUTED
**PPL change:** 44.842 → 56.781 (WORSENED by 11.938)

## Result Quality

All 12 layers had correction vectors computed. Layer 0 ||mean_corr||=0.000 (as expected —
the bias vectors added to c_proj.bias already compensate the first-order error at layer 0).
Layers 1-11: norms grow 1.09 → 35.9 (layer 11 largest), fraction_of_frob 0.13–0.35.

The experiment ran cleanly. Baseline matches cycle 189 (35.886). Uncorrected matches cycle 189
(44.842). Results are trustworthy.

## Why the Correction Fails

The mean correction computed from calibration data represents E_{calib}[r_float[l] - r_swapped[l]].
When added as a constant bias to the residual stream at layer l, it shifts the INPUT to layer l+1
by a constant vector. This means:
1. Layer l+1's layer norm sees a different distribution (shifted by the correction)
2. Layer l+1's attention keys/queries operate on shifted representations
3. These effects cascade, causing the 12 PPL degradation

This is the same mechanism as the cascade barrier itself: modifications to residual stream at
depth l compound through subsequent layers. Adding a first-moment correction trades the
ReLU-induced error for a correction-induced error that's just as harmful.

## Key Insight

The fraction_of_frob tells us the mean captures 13-35% of the per-token error norm. So the
mean is non-trivial and DOES represent real structure in the error. The problem is not that
the correction is zero — the problem is that the downstream layers cannot absorb the correction.

This is strong evidence that the cascade barrier is fundamental: any modification to the
residual stream (whether compression, correction, or bias) compounds through subsequent layers
and cannot be undone without downstream re-calibration.

## Implication for Next Steps

Item 202 (dead-channel pruning) is unaffected — it's an exact structural pruning, not a
residual stream modification. Proceed with that.
