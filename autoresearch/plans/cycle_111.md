# Cycle 111 Plan: Depth Gradient of GELU Operating Regime (GPT-2 FFN)

**Item:** 122
**Script:** `experiments/gelu_depth_gradient_gpt2.py` (new)

## Pre-flights

**P1 -- Null hypothesis:** Depth has no systematic effect on the GELU operating regime (negative fraction is flat across blocks).

**P2 -- Criteria (verbatim from agenda):**
- Confirmed if: negative fraction increases monotonically with depth (<=1 violation) AND slope is significant (>0.5% per block, R^2 > 0.7)
- Refuted if: no monotonic trend (>=4 violations) or slope < 0.2% per block
- Inconclusive if: trend exists but weak (R^2 0.3-0.7) or non-monotonic with clear pattern (e.g., U-shaped)

**P3 -- Novelty:** Cycle 103 reported 54%-64% as a side observation on bump-zone activations only. This cycle measures negative fraction on the FULL pre-activation distribution with formal monotonicity/regression analysis and adds E[GELU'(z)] and attention-effect measurements.

**P4 -- Confounds:** LayerNorm before FFN renormalizes to zero mean, so some negative fraction is structurally guaranteed (~50%). The signal is the DEVIATION from 50% and its depth dependence. W_in bias shifts the distribution post-LN. No circularity issues.

**P5 -- Cost:** Single forward pass on 4096 tokens, 12 blocks. <3 min on MPS.

**P6 -- Data:** `data/openwebtext_sample.txt`, same 4096 tokens as prior cycles.

**P7 -- Abort:** Baseline PPL outside [25, 65]. NaN in any hook output.

## Implementation

1. **Load GPT-2 + data.** 4096 tokens from OWT sample. Single deterministic pass.

2. **Hook pre-activations.** Hook `mlp.c_fc` output (pre-GELU) for all 12 blocks. Store z (N_tokens, 3072). Also hook residual stream BEFORE attention (`ln_1` input) and AFTER attention+LN (`ln_2` output, which is FFN input) per block.

3. **Per-block negative fraction.** For each block b:
   `neg_frac[b] = (z < 0).float().mean()` over all tokens and channels.

4. **Monotonicity test.** Count violations: `sum(neg_frac[b+1] < neg_frac[b] for b in range(11))`. <=1 violation = monotonic. >=4 = no trend.

5. **Linear regression.** OLS: neg_frac ~ block_index. Report slope, R^2, p-value.

6. **E[GELU'(z)] per block.** Compute GELU derivative: `gelu_prime = 0.5*(1+erf(z/sqrt(2))) + z*exp(-z^2/2)/sqrt(2*pi)`. Report mean per block. This captures how "linear" vs "gating" the regime is (1.0 = identity, 0.0 = fully gated).

7. **Pre-activation CDF analysis.** Per block, compute fraction of activations in bins: z < -2, -2 < z < -1, -1 < z < 0, 0 < z < 1, 1 < z < 2, z > 2. Report as 12x6 table.

8. **Attention effect.** Per block: compute neg_frac of `ln_2_output @ W_in + b_in` (actual FFN pre-act) vs a hypothetical `ln_1_input @ W_in + b_in` (residual stream without this block's attention). Difference = attention's contribution to negative drift.

9. **Verdict logic.**
   ```
   violations = sum(neg_frac[b+1] < neg_frac[b] for b in range(11))
   slope, r2 from OLS

   if violations <= 1 and slope > 0.005 and r2 > 0.7:  CONFIRMED
   elif violations >= 4 or slope < 0.002:                REFUTED
   else:                                                  INCONCLUSIVE
   ```

10. **Output:**
    - JSON: `autoresearch/results/cycle_111_gelu_depth_gradient.json`
      Per-block: neg_frac, mean_gelu_prime, CDF bins, attention_effect.
      Summary: violations, slope, r2, verdict.
    - Plot: `autoresearch/results/cycle_111_gelu_depth_gradient.png`
      2-panel: (top) neg_frac vs block with OLS line, (bottom) E[GELU'(z)] vs block.
