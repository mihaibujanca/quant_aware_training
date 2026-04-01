# Design Review — Cycle 194 (Item 204): Self-Correction Depth Analysis

## Correctness

**R1 — Residual collection**: Same pattern as cycle 189 (which used identical GPT-2 block
output hooks). Float model is fixed; hybrid_k varies. Comparing residuals collected from the
same analysis sequences through both models gives e_l[k] = float_residual - hybrid_k_residual.
**CORRECT**.

**R2 — cos_vs_k7**: For each k, compare the per-layer error vectors e_l[k] to e_l[k=7].
This measures whether the error DIRECTION changes as more blocks are swapped.
If cos_vs_k7[k=10][l] < cos_vs_k7[k=7][l]=1.0, the errors rotated. **CORRECT**.

**R3 — Efficiency**: Collecting residuals for k=6..11 means 6 × (float + hybrid) = 12 forward
passes for residual collection, plus 12 for progressive PPL. ~24 passes × ~3s/pass on MPS ≈ 72s.
Well within 30 minutes. **OK**.

**R4 — Reference anchor**: cos_vs_k7 should use k=7 as anchor (the peak PPL). This makes
k=7 cos=1.0 (same error vectors), and k=8,9,10 show rotation if errors redirect. **CORRECT**.

## Potential Issues

**P1 — Memory**: Storing residuals for 6 configurations × 12 layers × 200×1023 tokens × 768 dims
= 6 × 12 × ~200K × 768 × 4B ≈ 6 × ~1.8 GB per config. This is too much memory to store all
at once. **MUST process one k at a time and only store the metrics, not the raw residuals.**

Fix: for each k, collect residuals, compute metrics, then discard residuals before moving to k+1.

**P2 — Float residuals reused**: Float residuals at k=7 serve as reference. Recompute float
residuals once and cache them (they don't change between k values). Then for each k=6..11,
collect only hybrid_k residuals and compute error vs float. **EFFICIENT**.

## Flags

**F1**: Do NOT store all residuals simultaneously — process one k at a time to avoid OOM.
Collect float residuals once, then loop k=6..11 collecting hybrid residuals one at a time.

**REVIEW RESULT: APPROVED with flag F1 (memory management).**
