# Cycle 088: Input-Space Token Clustering + Local Jacobian — GPT-2 FFN

## Item 102 Summary
Cycle 085 proved output-space GELU patterns are maximally diverse (2045/2048 unique tokens), killing cluster-based compression. **Pivot**: cluster tokens in *input* space instead, where geometry is lower-dimensional. For blocks [1, 4, 7, 11], collect FFN inputs x (4096 tokens, dim=768), run k-means with k ∈ {16, 32, 64, 128}. For each cluster, compute mean Jacobian J_k via `torch.func.jacrev` and reconstruct: y_approx = J_k @ (x - x_bar_k) + f(x_bar_k). Measure per-token reconstruction error ||y - y_approx|| / ||y||.

## Criteria
- **Confirmed**: K* ≤ 64 achieves < 5% error for ≥ 3/4 blocks
- **Refuted**: K* > 128 still > 10% error for majority (input space also too continuous)
- **Inconclusive**: works for some blocks but not others

## Context
- Cycle 085 (item 96): Output-space GELU clustering → REFUTED. K* > 256 for all 12 blocks.
- Cycle 087 (item 96): Mixed-corpus stacking eval → INCONCLUSIVE. Modest effect (+7.5% Wiki, below +10% threshold).
- Status.md: GELU fully nonlinear (cycle 083: 0.04% dead channels), K* > 256 patterns (cycle 085). Dead direction: GELU clustering.

## No Dependencies
No blocking items. Ready to start.
