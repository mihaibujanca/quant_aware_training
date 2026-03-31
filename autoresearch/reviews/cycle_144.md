# Cycle 144 Review — Layer-local Pullback Metric (Item 149)

Item 149. Reviewer: design-review agent.

## R1: Premise validity — OK

Cycle 136 showed global pullback G = U^T U is a constant rescaling of Frobenius at FFN output level (ratio ~34-35 across blocks 0-10). This is because U is applied identically to all blocks. A layer-local Jacobian through the actual downstream computation captures block-position-dependent amplification (later blocks have fewer downstream layers, attention interactions differ per position). The premise that this should produce non-constant scaling factors is well-motivated.

However, room to improve on Frobenius rho=0.797 is limited. Even a confirmed result (rho >= 0.947) would have uncertain practical value given that polish erases local quality differences and composition dominates.

## R2: Measurement soundness — OK

Null baseline: random error directions would give near-zero Spearman rho. Frobenius rho=0.797 is the active baseline. The plan correctly reuses PPL deltas from cycle 136 rather than recomputing, and includes a consistency check (Frobenius rho reproduction).

One subtlety: the plan proposes averaging e_l across tokens/sequences to get a single 768-dim error vector, then computing one JVP. This loses per-token variance in error direction. If error directions are diverse across tokens, the mean could underestimate true amplification. The fallback (per-token JVP, then average norms) is the correct approach if results are noisy. This is addressed in the plan and does not require revision.

## R3: Capacity and statistical adequacy — OK

3 seeds specified. 12 data points for Spearman is the standard for this program's per-block analyses. Cross-seed std threshold of 0.1 is appropriate.

## R4: Method-architecture alignment — OK

`torch.func.jvp` handles GELU, LayerNorm, and attention correctly through forward-mode AD. The rest-of-network map (layers l+1..11 + ln_f + lm_head) is well-defined. The JVP input is the compression error at the FFN output (post-GELU, post-down-projection), which is the residual stream perturbation — geometrically meaningful.

One note: the JVP should be computed with the model in eval mode and with the correct input context (the same probe sequences used for compression). The plan implies this but should make it explicit in the script.

## R5: Known pitfall scan — OK

- **MP7 (Post-LayerNorm):** The error e_l is measured at FFN output (before residual addition to the stream). The JVP propagates this through subsequent LayerNorms, which is correct — we want to measure how much the rest of the network amplifies this error, LayerNorm included.
- **MP8 (Representation vs functional rank):** Not applicable. We're measuring amplification of a specific error vector, not intrinsic dimension.
- **MP6 (Verdict misclassification):** If layer-local norms are nearly proportional to Frobenius norms (just with a non-constant but monotonic scaling), the metric could still fail to improve rho. This would correctly be classified as REFUTED per the criteria.
- No MP1-MP5 issues.

## Summary

No blocking issues. The plan is clean, well-scoped, and includes appropriate fallbacks.

**Verdict: APPROVE.**
