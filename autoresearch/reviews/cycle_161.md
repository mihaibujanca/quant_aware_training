# Cycle 161 Review (Round 2) — Centroid-affinity channel selection (Item 171)

Item 171. Reviewer: design-review agent. Round 2 of 2.

## Round 1 issues — resolution

1. **BLOCKING (R2): Method-definition mismatch.** Round 1 found the plan used
   activation-profile diversity (token-axis vectors) instead of centroid
   embeddings. The revised plan now correctly defines per-channel centroids as
   activation-weighted mean hidden states: `c_j = mean(a[:,j] * h)`, shape
   [d_model]. Farthest-point sampling runs in R^{768}. **Resolved.**

2. **Minor (R5/MP6): Jaccard > 0.75 verdict.** Round 1 recommended REFUTED
   instead of INCONCLUSIVE when methods select the same channels. The revised
   hypothesis now places Jaccard > 0.75 under the REFUTED condition.
   **Resolved.**

3. **Advisory (R1): Premise transfer unjustified.** Round 1 asked that the
   inter-block → intra-block transfer be flagged as exploratory. The revised
   plan includes a "Premise note" doing exactly this. **Resolved.**

## R1: Premise validity — OK

Premise note (lines 9-11) honestly frames this as exploratory. No overclaim.

## R2: Measurement soundness — OK

Centroid embeddings are properly defined as activation-weighted mean hidden
states in R^{d_model} (step 2). Farthest-point sampling in this space
maximizes coverage of hidden-state directions. Lstsq refit and 500-step Adam
polish match the baseline protocol exactly. Only the selection criterion
differs between methods — clean comparison.

## R3: Capacity and statistical adequacy — OK

3 seeds, 200 OWT sequences (~25K+ tokens), k=4 from 3072 channels. Lstsq
fitting well-determined. Calibration set sufficient for centroid estimation.

## R4: Method-architecture alignment — OK

Post-GELU intermediate activations and hidden states hooked at the correct
locations. Centroid computation uses both signals appropriately.

## R5: Known pitfall scan — OK

- MP1-MP3: Not applicable.
- MP4: Premise transfer acknowledged as exploratory. Acceptable.
- MP5: GPT-2 baseline well-established.
- MP6: Jaccard > 0.75 → REFUTED. Correct verdict mapping.
- MP7: Post-GELU activations, pre-LayerNorm. No issue.
- MP8: Centroid embeddings capture functional contribution (activation-weighted
  hidden states), not just representation geometry. No confusion.

## Summary

All three round 1 issues resolved. No new issues found. The plan is clean:
proper centroid embeddings, honest exploratory framing, correct verdict
boundaries, and a fair apples-to-apples comparison against topk.

**Verdict: APPROVED**
