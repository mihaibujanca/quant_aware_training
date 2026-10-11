# Cycle 222 audit — item 227 (Frobenius-weighted channel selection)

**Status: PASS** (old: refuted → new: refuted)

## Verdict check
- Claimed Δ = 0.111 PPL (61.7464 → 61.6355). Threshold for refuted is Δ < 0.3.
- 0.111 < 0.3 → REFUTED per literal item 227 criterion. Verdict matches.

## Quality
- **G1 (sanity)**: standard topk reproduces cycle 206 init within 0.0004 PPL — calibration and eval splits are parity-matched. Frobenius selection genuinely swaps 66/52/29 channels per block (out of 512), so the scoring path is exercised, but lstsq W_out absorbs the difference.
- **G1 (formula)**: variant B computes `||W_proj[c]||^2 * Σ_t gelu(z[t,c])^2` exactly as the plan specifies. Variant A uses cycle 206's `act_std × ||W_proj||`.
- **G2 (seeds)**: 3 seeds (42/137/2024) for variant A, std ≈ 7e-6 (deterministic given calibration sampling, as expected). Variant B is deterministic — single run is appropriate.
- **G3 (interpretation)**: cleanly extends cycle 207's single-block finding (item 218) to the 3-block cascade. Same conclusion: lstsq absorbs the selection rule.
- **G6 (genuine refutation)**: not a method failure. The variant produces a meaningfully different channel set, the lstsq fit succeeds, and the metric moves in the expected range — it just doesn't move enough. Cascade gap (61.75 vs float 53.89) is structural, not a selection problem.

## Recommendation
Close the channel-selection direction. Pursue Adam-side improvements (item 228 K-sweep, item 229 distillation). Selection is solved at K=512.
