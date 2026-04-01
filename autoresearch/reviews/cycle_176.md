# Cycle 176 Review — Sign-pattern-conditioned lstsq (Item 183)

## R1: Premise validity — WEAK BUT VALID

Cycle 167 refuted the premise that sign patterns cluster tractably. The plan acknowledges this directly and proceeds because the dependency is "not blocking." The experiment is still informative: it tests whether forced clustering via k-means helps even when natural clusters don't exist. This is a fair follow-up. Expected outcome: REFUTED.

## R2: Measurement soundness — OK

Baseline reproduction check (~59-60 PPL) catches pipeline regressions. Cluster size guard (merge if < 64 tokens) prevents degenerate lstsq. Per-block routing stats will expose pathological cluster distributions. Param accounting is explicit and honest about the 8x c_proj overhead.

## R3: Capacity and statistical adequacy — OK

3 seeds, 200 eval sequences, standard config. With K=16 and 4096 tokens, smallest expected cluster ~256 tokens, which is sufficient for lstsq with 512 features (overdetermined). The merge guard handles edge cases.

## R4: Method-architecture alignment — MINOR NOTE

The plan says "route to nearest centroid" at inference using full 3072-dim sign patterns. But the compressed block only has 512 topk channels. The sign pattern should be computed from the FULL pre-activation z (before topk selection), since the routing decision must be based on the same features used during training. The plan is correct about this (step 1 uses preactivation_sample which is full 3072-dim), but the inference routing in the CompressedMLP must also compute full z for routing, then apply topk for the actual computation. This adds overhead: a full c_fc forward pass just for routing, then a topk-only pass for the compressed output.

Alternative: compute sign patterns only on the topk channels. This is cheaper and tests whether the selected channels carry enough routing information. Recommend implementing this as the primary approach (topk-channel sign patterns for routing) with a note that full-dim routing is an option.

**Recommendation**: Use topk-channel sign patterns (512-dim) for k-means and routing. This is cheaper, avoids the full c_fc overhead, and tests the relevant question (do the selected channels partition the function space).

## R5: Known pitfall scan — OK

- MP3 (underdetermined): cluster size guard handles this.
- MP5 (insufficient capacity): K=16 has 16x params but same topk channels — capacity per cluster may be too low if clusters are unbalanced.
- No other pitfall flags.

## Summary

One recommendation (not blocking):
1. Use topk-channel sign patterns (512-dim) for clustering and routing instead of full 3072-dim. This is cheaper and more directly tests whether the compressed channels carry routing signal.

**Verdict: ACCEPT with recommendation.**
