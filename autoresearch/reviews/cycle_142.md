# Cycle 142 Review — Correction Subspace Analysis (Item 154)

Item 154. Reviewer: design-review agent.

## R1: Premise validity — STRONG

The premise is well-motivated on two fronts. First, cycle 127 found correction-vs-error
alignment near null baseline (max 0.18 vs p99=0.13), leaving the "what do corrections
learn?" question genuinely open. Second, cycle 138 provided a key negative transfer
result: corrections trained on topk error actively worsened AE-compressed models
(76.7 vs 74.4). This combination makes the subspace decomposition question
scientifically sharp — if corrections learn a novel subspace specific to topk error
patterns, it explains both the low alignment AND the negative transfer.

## R2: Measurement soundness — APPROVE with one note

The projection + ablation design is methodologically clean. Key strengths:

1. **Orthogonal decomposition**: projecting C_attn onto the complement of C_error
   before measuring avoids double-counting. The plan explicitly specifies this
   sequential orthogonalization.

2. **Ablation as ground truth**: Frobenius norm fractions can be misleading
   (a low-norm component in a high-curvature direction could dominate PPL). The
   ablation conditions (zero-out + each-only) directly measure PPL impact,
   which is what matters.

3. **Data-weighted SVD**: the reviewer suggestion from cycle 127 is incorporated.
   SVD of C @ X^T captures which correction directions actually carry energy on
   real data, not just which are geometrically prominent.

**Note**: the plan ablates one block at a time. This is correct for isolation
but means the ablation results don't capture inter-block interactions. If block 4's
novel component compensates for block 1's error component, per-block ablation
won't reveal this. This is acceptable — the question is per-block subspace
characterization, not inter-block interaction. Flag it in the results as a
limitation.

## R3: Capacity and statistical adequacy — OK

3 seeds provides basic reproducibility. 4 blocks x 3 subspaces = 12 decompositions
per seed. The CONFIRMED criterion (>=60% from one subspace) is well-calibrated:
with 3 roughly equal subspaces, each contributes ~33%, so 60% requires clear
dominance. The INCONCLUSIVE criterion (varies by block/seed) is the right
escape hatch.

## R4: Method-architecture alignment — OK

The plan correctly hooks post-FFN, pre-LayerNorm (addressing cycle 127 review R2
Issue 1). The LowRankCorrection class from cycle 127 computes x + (x @ V^T) @ U^T,
so C = U @ V is the additive linear correction, matching the plan's extraction.

One subtlety: the correction is data-dependent (C @ x, not just C), so the SVD
of C characterizes the linear map but the *effective* correction on any given input
depends on x. The data-weighted SVD (step 2) handles this. Good.

## R5: Known pitfall scan — OK

- **MP7 (LayerNorm)**: addressed (hook placement specified).
- **Counterintuitive results**: if the novel subspace dominates, this is surprising
  but the cycle 138 negative transfer result provides independent corroboration.
  No flag needed.
- **Wall time**: 110 min estimate is realistic. Fallback to skip each-only
  conditions is sensible.

## Verdict: APPROVE

Clean design that directly extends cycle 127 with the right tools (projection +
ablation vs alignment-only). The cycle 138 negative transfer result adds weight
to the hypothesis and makes this timely. No blocking issues.
