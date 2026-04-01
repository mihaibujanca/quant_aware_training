# Cycle 202 Review: BLOCKS=[1,4,7] vs [1,4,7,11] Compression (Item 213)

## Verdict: CONFIRMED

Excluding block 11 from the compression set improves PPL by 6.4 points.

## Key Numbers

| Configuration    | PPL    | vs float |
|-----------------|--------|----------|
| Float           | 53.895 | —        |
| BLOCKS=[1,4,7]  | 61.746 | +14.6%   |
| BLOCKS=[1,4,7,11] | 68.138 | +26.4% |

Improvement from excluding block 11: 6.4 PPL (+26.4% → +14.6%).

## Interpretation

Cycle 201 confirmed block 11 is an output decompressor: it expands the rank-1/2 residual
stream to rank-231 for LM head projection. Compressing block 11 approximates this rank-expansion
with a topk+lstsq function that cannot faithfully reproduce rank-231 output from rank-2 input
(K=512 is insufficient to span 231 dimensions from a degenerate 2D input). Leaving block 11
uncompressed restores the LM head's expected rank-231 input, recovering 6.4 PPL.

The improvement is substantially larger than noise (6.4 PPL >> threshold 1 PPL).

## Revised Working Recipe

**Previous best**: BLOCKS=[1,4,7,11], K=512 → +26.4% above float (held-out eval)
**New best**: BLOCKS=[1,4,7], K=512 → +14.6% above float (held-out eval)

This improves the held-out-eval degradation by 11.8 percentage points with FEWER compressed
blocks (3 instead of 4).

## Impact on Architecture Picture

GPT-2 FFN architecture now understood as 4-zone:
1. Block 0: high-rank input compression (rank 244→ starts collapse) — DO NOT COMPRESS
2. Blocks 1-10: rank-1 residual processing — compressible; blocks 1,4,7 (and others) safe
3. Block 11: output decompression (rank-2→231) — DO NOT COMPRESS
4. LM head: expects rank-231 residual from final block

## Next Question

With BLOCKS=[1,4,7] as the new recipe, what's the limit? Can we compress more of blocks
2-10 (e.g., BLOCKS=[1,2,3,4,5,6,7,8,9,10] = all 10 compressible blocks) and still avoid
catastrophic cascade? Cycle 187 showed individual drops of blocks 1-10 all give PPL <40,
but multi-drop of 4 blocks already caused superadditive cascade. The new recipe [1,4,7] at
K=512 achieves +14.6% — can we reduce this further with a different block selection?
