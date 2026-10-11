# Cycle 243 Review — Per-block PPL attribution under 6-block joint Adam fit

**Verdict: APPROVE**

## Summary

Analysis cycle (no new training beyond a single seed-42 refit). Reuses cycle 221's joint Adam recipe on BLOCKS=[1,2,4,5,7,8] K=512 W_out+b_out, then decomposes the 53.895 → 59.515 PPL gap by restoring each compressed block's float MLP one at a time.

## Strengths

- **Tight scope.** No new optimisation regime; one seed refit (~10 min) + 6 swap-and-eval rounds. Cheap probe with high diagnostic value.
- **Bookkeeping is honest about snapshot loss.** Pre-flight already confirmed cycle 221 logged trajectories but no `.pt` tensors, so refit is unavoidable and budgeted.
- **Restore roundtrip assert** (eval after un-swap matches `fully_compressed_ppl` to 1e-4) catches aliasing bugs that would silently corrupt the table.
- **Non-additivity acknowledged.** Plan explicitly notes that Σ deltas need not equal the 6 PPL gap — avoids the trap of treating per-block ablation as a linear decomposition.
- **Residual MSE in addition to PPL deltas** gives a second view (input-side reconstruction quality vs output-side task impact). Disagreement between the two rankings would itself be informative.
- **Verdict thresholds are pre-registered** (any block ≥3.5 PPL drop → CONFIRMED, all 6 in [0.5,1.5] → REFUTED, else INCONCLUSIVE). No post-hoc reframing.

## Concerns

- **Float MLP storage**: deepcopying 6 float MLPs while also holding the compressed model is fine on CPU/MPS for GPT-2 small (~few hundred MB), and the OOM fallback (move `model_float` to CPU) is in place.
- **Per-block restoration is order-independent by construction** (each swap is from the same fully-compressed baseline), so the six numbers are well-defined. Good.
- **Refit nondeterminism**: ±0.05 tolerance vs cycle 221's 59.391 is tight but defensible — same seed, same data order, same MPS. Fallback covers wider drift.
- Minor: the residual MSE measurement (Phase 3) re-runs `collect_mlp_io` twice (float + compressed). Cheap, but worth caching the calib activations once if Phase 0 already has them. Not blocking.

## Risks

- If the refit lands far from 59.5 (e.g. >60), the entire decomposition is being run on a different operating point than cycle 221 reported. Plan handles this by recording actual baseline — accept and proceed.
- The "bottleneck block" framing assumes one or two blocks dominate. If contributions are uniform, the result is still a valid (REFUTED) negative — but the headline becomes "no bottleneck, 6.0 PPL is distributed". Plan accepts this outcome.

## Approval

Approved as written. Proceed to execution.
