verdict: APPROVE with advisory
round: 1

## R1: Gaussian fit validity

The plan fits per-channel Gaussian N(μ_j, σ²_j) and uses relative excess to measure deviation. This is reasonable as a null model — the question is whether excess concentration *beyond* what variance explains exists. The plan does NOT claim pre-activations are Gaussian; it uses Gaussian as a baseline to detect anomalies. The zone-fraction comparison (step 4) is the real output; the KS test is supplementary. Sound framing.

## R2: KS test at n=4096

**Advisory (A1).** With n=4096 samples per channel, KS is extremely powerful — it will reject the Gaussian null for virtually every channel due to trivial deviations (e.g., slight skew or kurtosis). The KS p-values will be near-zero almost everywhere, providing no discrimination between "well-approximated by Gaussian" and "poorly approximated." The plan stores a KS p-values summary but doesn't use it in the verdict logic (verdict uses zone-fraction excess only), so this is non-blocking. Recommendation: report fraction of channels where KS statistic D > 0.05 (a practically meaningful deviation size) rather than p-values, or simply drop the KS test and rely on the zone-fraction comparison which is the more informative metric.

## R3: Importance correlation

**Advisory (A2).** `channel_scores` from `probe_ffn_geometry` is defined as `std_hidden * output_norms * (mean_hidden.abs() + mean_deriv.abs() + 1e-6)` where `std_hidden` and `mean_hidden` are computed on post-GELU activations `h`, not pre-activations `z`. The plan correlates σ_j (pre-activation spread) with this score via Spearman. This is well-defined but the correlation is partially tautological: channels with large pre-activation σ will tend to have large post-GELU std_hidden (monotonic relationship through GELU). A high Spearman ρ would mainly confirm that pre-activation spread drives post-activation spread, not that spread predicts task importance. Recommendation: note this interpretation caveat in the results, or additionally report Spearman(σ_j, output_norms_j) to isolate the output-weight contribution.

## R4: Known pitfalls

- **MP7 (post-LayerNorm):** Applicable and handled correctly. The plan hooks `mlp.c_fc` output, which is the linear projection of the post-LayerNorm input. The pre-activation z = W_in @ LN(x) + b includes LayerNorm normalization of the *input* but measures the *projected* distribution, which is the quantity of interest for GELU zone analysis. Not a pitfall here — the question is about the distribution entering GELU, not about the representation geometry.
- **MP6 (verdict misclassification):** The zone-fraction excess metric can vary continuously and the 5%/20% thresholds are reasonable. If empirical matches Gaussian within 5%, that genuinely confirms the null. Not at risk.
- **MP1-5, MP8:** Not applicable (no detection threshold, no activation regime mismatch, no fitting, no domain transfer, no capacity comparison, no dimension vs rank confusion).

## R5: Minor observations

- Step 2 says "hook `mlp.c_fc` output" to get pre-GELU values. This is correct for GPT-2's architecture (c_fc is the up-projection linear layer, GELU is applied after).
- The plan reuses the same 4096 token set as prior cycles (P4 item 1). Consistent and appropriate.
- Step 6 defines bump-dominated as σ < 0.5 AND |μ| < 0.5. These thresholds are specific to GELU's nonlinear zone. Sensible.
- Cost estimate (3 min, single forward pass) is realistic. No seed needed — deterministic analysis. Correct.
