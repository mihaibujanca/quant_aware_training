# Cycle 132 Design Review Round 2 — Item 126: Attention Projection Pruning

## Verdict: APPROVE

## Issue Resolution

1. **MP7 hook points** — Step 3 now specifies "post-FFN, pre-LayerNorm (raw residual stream before next block's LN1)." Correct.
2. **Composed W_V@W_O** — Step 2 now SVDs the composed M_h = V_h @ O_h and uses forward hooks to inject the rank-r replacement. Correct approach; avoids the dimension mismatch of trying to factor back into original shapes.
3. **Derivation cleanup** — Step 2 is now a single coherent procedure. No residual confusion.
4. **Verdict threshold** — Now reads "≥ 2% of baseline PPL (i.e., ≥ ~2.5 points at baseline ~125)." Matches item 126.

No new issues found. Plan is ready for implementation.
