# Cycle 316 Design Review — Item 326

## Decision: APPROVE

## Strengths
- Plan implements all three tests verbatim from item 326 with concrete operational definitions:
  - M1: principal angle between the rank-matched right-singular subspaces of Δy_10 and Δy_11.
  - M2: sequential Adam (b10 phase A → freeze → b11 phase B), reusing the cycle 270/282/310 LowRankCProj template.
  - M3: two one-sided restores (b10 float + b11 r=22; b10 r=5 + b11 float).
- Verdict thresholds match the brief verbatim (≤55 → M3 CONFIRMED; ≥60 + angle < 30° → M1 CONFIRMED; ≥60 + angle > 60° → REFUTED-unknown; else INCONCLUSIVE).
- Reuses the cycle 282/310 LowRankCProj module exactly (3072×r frozen A, r×768 trainable B, 768 trainable bias) so the comparison to cycle 310's joint 60.7161 baseline is apples-to-apples.
- Runs entirely on the cycle 280/310 wikitext-2 100×128 eval slice with the same SEED=42; no infrastructure ambiguity.
- Wall ~10 min, well within the 30-min cap.

## Risks (acknowledged in plan)
- Single seed (item 326 verbatim) — verdict bands are wide (5 PPL gap between M3 and the upper bar) so a single seed is acceptable.
- M1's "principal angle" choice: plan defines it as the principal angles between rank-r subspaces (svdvals of Vh_10 @ Vh_11.T), which is the standard subspace angle measure and avoids ambiguity from per-token cosine sign noise.
- Δy as defined in item 326 ("Δy_10 = (GELU(z_10) − ReLU(z_10)) @ W_proj_10") is the cycle-302 curvature residual; the plan generalises this to "GELU(z_l) @ W_proj_l" to capture the FULL residual contribution of the b10 / b11 FFNs to the residual stream (since the question is "what does b_l write into the residual stream", not "what does the curvature residual write"). Both choices are defensible. Recording both would be safer; if budget allows, the executor should report both.

## Required execution-time addenda
1. Report **both** Δy variants (raw GELU(z)·W_proj AND curvature-residual (GELU − ReLU)·W_proj) for M1, since item 326 hypothesis literally says "Δy_l = the residual contribution of b_l's FFN" and parenthetically wires it via the curvature residual. Two principal-angle numbers, one verdict (use the larger of the two; if either crosses the 30° / 60° bar, report it).
2. Cross-check that ablation B's init PPL (b10 r=5 noretrain) matches cycle 310's diagonal cell 51.47 (after Adam fit; the no-retrain init will be larger).
3. Make sure phase B of M2 starts with a **fresh Adam optimizer** (no state carry-over from phase A's b10-only fit) — plan already says this; flag for the executor.

## Approval
APPROVE.
