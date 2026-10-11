# Cycle 228 Review — Item 238 (P1)

## Verdict: APPROVE

## Summary
Plan reruns the cycle 220 fair gelu_noneg patch with an offline-precomputed per-channel first-moment correction μ_{l,c} = E[GELU(z) - ReLU(GELU(z))] added as a post-activation bias. Tests whether closing the per-channel mean leak rescues PPL from 2093 toward float baseline 45.064.

## Soundness check
- **Sign convention**: GELU(z) for z<0 is small negative (max magnitude ~0.17). ReLU(GELU(z)) clamps that to 0. So `gelu - noneg` is negative where gelu<0 and 0 elsewhere → μ_l ≤ 0 elementwise. Adding μ_l (negative) to ReLU(GELU(z)) shifts the post-activation down to compensate for the missing negative leak. Correct.
- **Magnitude**: Validation step expects |μ_l| ≈ 0.17, consistent with cycle 184. Reasonable.
- **Linearity through c_proj**: Because μ_l is added BEFORE c_proj, the residual contribution becomes `c_proj @ (relu(gelu(z)) + μ_l) = c_proj @ relu(gelu(z)) + c_proj @ μ_l`. The second term is a constant per-block additive shift to the residual stream — equivalent to adjusting the c_proj output bias by `c_proj @ μ_l`. This is exactly the right first-moment compensation.
- **Calibration distribution match**: Calibration split == eval split (N=200, SEQ_LEN=128, seed 42). Eliminates distribution shift as a confound for this proof-of-concept.
- **Restore-and-eval sanity assertion**: Present (≤1e-4 vs baseline). Good.
- **Hook capture**: Pre-hook on c_fc captures LN2 output (input to c_fc). Plan recomputes preact via Conv1D matmul; fallback to capturing c_fc OUTPUT directly is documented.

## Concerns (none blocking)
- **Per-channel mean is only the first moment**. If the per-token leak distribution is heavy-tailed or strongly correlated with downstream attention/MLP nonlinearities, the mean correction will under-compensate. But that's exactly what this experiment tests; not a design flaw.
- **Calibration ≡ eval**: This is in-distribution best case. If verdict CONFIRMS, follow-up should test held-out calibration to rule out memorization. Plan is fine as a first pass.
- **Threshold structure**: low_thresh=49.57 (≈baseline+10%), refute_thresh=100. Reasonable for a P1 first-moment correction; gives clear CONFIRM/REFUTE/INCONCLUSIVE signal.

## Bookkeeping
- Novelty/dependency/budget/overwrite checks present and pass.
- Artifacts list complete; runner will write manifest.

Default APPROVE.
