# Cycle 167 Review — Sign-Pattern Census (Item 177)

Item 177. Reviewer: design-review agent.

## R1: Premise validity — OK

Cycles 165-166 established that ~91% of channels are locked negative and the
negative regime is functionally constant. Testing whether the ~9% flipping
channels create a tractable partition is the logical next step — it directly
informs item 183 (piecewise lstsq conditioned on sign patterns).

## R2: Measurement soundness — OK

Within-cluster FFN output variance / total variance is the right metric: it
measures whether tokens sharing a sign pattern also share similar FFN outputs
(which is exactly what piecewise-linear structure means). Hamming distances
provide a complementary view of pattern diversity. 4096 tokens is adequate
for clustering with k <= 256 (16:1 ratio at k=256).

## R3: Capacity and statistical adequacy — OK

4096 tokens, k-means up to k=256 → 16 samples/cluster at maximum k. For
variance estimation this is borderline but acceptable for a census. The plan
correctly includes k=256 as an upper bound to detect the REFUTED case.

## R4: Method-architecture alignment — OK

Hooking `mlp.c_fc` gives pre-GELU z (correct). Hooking `mlp.c_proj` gives
FFN output y (correct — this is the W_down @ GELU(z) + b_proj output).
k-means on float-cast boolean vectors with Euclidean distance is equivalent
to Hamming-distance clustering, which is the right metric for binary patterns.

## R5: Known pitfall scan — OK

- MP7 (Post-LayerNorm): hooking mlp.c_fc output is post-LayerNorm for the
  residual but pre-GELU for the FFN. The sign pattern is a property of the
  FFN computation, not the LayerNorm output. No violation.
- MP8 (Representation vs functional rank): sign patterns directly partition
  the piecewise-linear regions of the FFN — this IS functional structure, not
  representation dimension. No violation.
- No MP1-MP6 issues.

## Summary

No blocking or minor issues. Plan is well-scoped and measurements are sound.

**Verdict: APPROVE.**
