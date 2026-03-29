verdict: APPROVE
round: 1

## R1: Criteria faithfulness
Criteria quoted verbatim in P2. Verdict logic (step 7) maps directly: c<=0.25 at r<=16 AND k=8 <=+30% -> CONFIRMED, c>=0.35 at all r -> REFUTED, else INCONCLUSIVE. The plan adds the k=8 PPL check for confirmation, which is stricter than just checking c (the agenda says "enabling k=8 at <=+30%"). This is correct — a low c that still gives bad k=8 PPL would be misleading. Pass.

## R2: Confound check
(1) LayerNorm redistribution: plan measures error pre-LN and acknowledges the concern. (2) Probe data matches prior cycles (OWT). (3) Signal destruction tracked via abort condition P7 (projection making PPL worse). (4) Projections are computed on same data used for eval — this is a mild train/eval overlap, but since the projection is a fixed SVD (no optimization), overfitting risk is low. Clean.

## R3: Known pitfalls
- MP7 (post-LayerNorm measurements): Explicitly addressed — plan hooks pre-LN residual stream. Pass.
- MP3 (underdetermined fitting): No fitting — SVD is a closed-form decomposition. N/A.
- MP8 (representation vs functional rank): The plan measures error rank (rank of the error matrix E_i), not representation rank. This is the right object — it's the rank of the perturbation, not the data manifold. Pass.
- MP1, MP2, MP4, MP5, MP6: Not applicable.

## R4: Cost and feasibility
Reuses existing compression infrastructure from cycle 089. SVD on (n_tokens, 768) matrices is cheap. 8 stacking evals at ~2 min each = ~15 min. Feasible.

## R5: Verdict logic soundness
Three-way verdict with clean thresholds. The confirmation criterion requires BOTH low c AND low k=8 PPL, which prevents false positives from a c measurement artifact. The refutation criterion (c>=0.35 at all r) allows modest improvement to land as INCONCLUSIVE rather than premature refutation. Sound.

## Advisory

**A1 (Calibration/eval split).** Projection vectors U_i are computed from the same 200 sequences used for PPL eval. For a no-training correction this is acceptable, but if results are borderline, consider a 100/100 cal/eval split to rule out data leakage. Not blocking.

**A2 (Pre-LN vs post-LN insertion point).** GPT-2 uses pre-norm architecture: LN is at the INPUT of each block, not the output. The plan says "after block i's FFN output, before LN" — verify this means after the residual addition (x + FFN(LN(x))) and before the next block's input LN. If the projection is inserted in the wrong spot, it will be undone by the next LN. Not blocking but worth a code comment.
