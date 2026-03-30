# Cycle 132 Design Review — Item 126: Attention Projection Pruning

## Verdict: REVISE

## R1: Hypothesis–Criteria Match — PARTIAL
Item 126 specifies "top-3 amplifying head positions" targeted pruning. Plan correctly notes cycle 113 refuted per-head concentration and pivots to all-heads-in-layer uniform SVD. This is a reasonable adaptation but changes the experiment meaning: it becomes "layer-wide rank reduction" not "targeted head pruning." The confirmation criteria ("targeted head pruning reduces cascade...") were written for the targeted version. The plan should explicitly restate the adapted hypothesis and note which criterion clauses still apply.

Group C test (step 6) partially addresses this — it tests top-3 vs all-12 on the one layer where concentration exists. Good.

## R2: Pre-flights — OK
P1 null reasoning is speculative ("if cascade is linear in rank") but acknowledges uncertainty. P3 cost reasonable. P4 abort adequate.

## R3: Known Pitfalls — FLAG MP7
The plan measures ||delta_downstream|| / ||delta_source|| for cascade amplification but does not specify hook points relative to LayerNorm. If deltas are captured post-LayerNorm, the renormalization erases the amplitude information and the ratio is meaningless (MP7). **Specify: hook pre-LayerNorm (i.e., at the residual stream before the next block's LN) for both source and downstream deltas.**

No MP8 issue: SVD of W_O is the correct functional rank measure for a linear projection.

## R4: Implementation — REVISE NEEDED
1. **W_O-only vs W_V@W_O:** The plan simplifies to rank-r SVD of W_O_h only (step 3), but item 126 targets "W_V W_O" — the composed map. W_V selects input directions; truncating W_O alone doesn't reduce the composed rank if V already saturates it. The experiment should either (a) SVD the composed 768→768 map and refactor, or (b) explicitly justify why W_O-only is sufficient. The confused derivation in step 2 suggests the correct approach was abandoned for convenience, not correctness.

2. **Dimension confusion in step 2:** The plan self-corrects three times on matrix dimensions. Final step 3 is correct, but step 2 should be deleted or marked as scratch — leaving it in risks confusing the implementation agent.

## R5: Verdict Logic — MINOR
Item 126 says "≥2% absolute." Plan says "≥2 absolute" (PPL points). For baseline PPL ~125, 2% = 2.5 PPL points. Small gap but should match the item's stated criterion. Use 2% of baseline PPL (i.e., ~2.5 points) to stay faithful to item 126.

## Required Revisions
1. Specify pre-LayerNorm hook points for cascade delta measurement (MP7).
2. Either use composed W_V@W_O SVD or justify W_O-only as sufficient.
3. Delete or collapse the confused step 2 derivation; keep only the final approach.
4. Align verdict threshold with item 126 ("≥2% absolute" → ~2.5 PPL points at baseline ~125).
