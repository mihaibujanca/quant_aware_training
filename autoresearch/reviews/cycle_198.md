# Cycle 198 Review: Block 0 Criticality Analysis (Item 205)

## Verdict: CONFIRMED

Block 0 is geometrically unique: its input activation space spans 269 dimensions (rank_90=269/768)
vs median rank_90=1 for blocks 2-11. This 269× outlier identifies the source of block 0's
irreplaceability: it operates on raw token embeddings, which are high-diversity (one embedding
per distinct token type), whereas later blocks receive a residual stream that has collapsed to
near-rank-1 due to layer norm + attention + residual composition.

## Key Numbers

| Metric             | Block 0 | Blocks 1-11 median | Block 1 |
|--------------------|---------|--------------------|---------|
| input_rank_90      | 269     | 1                  | 207     |
| preact_rank_90     | 176     | 210                | 181     |
| mean_delta_norm    | 54.1    | 30.9               | 21.9    |
| channel_util_01    | 0.593   | 0.583              | 0.501   |

Block 1 also has high input_rank (207) — it immediately follows block 0 and still receives
a partly-diverse residual. By block 2, input rank has collapsed to 25; by block 3 it is 1.

## Interpretation

Raw token embeddings are one vector per token type (50257-way lookup). The embedding matrix has
high intrinsic dimensionality across the vocabulary. Block 0's MLP must map this high-diversity
input into the residual stream. After block 0+1, the residual stream stabilizes to a near-rank-1
direction (tokens cluster), and later blocks operate on that collapsed space.

When block 0 is dropped (cycle 187: PPL=1923), the model receives raw embeddings directly as
its "residual stream" without any initial transformation — all subsequent blocks are then
operating on the wrong subspace entirely. The other blocks' inputs are already in the collapsed
residual manifold and can be individually dropped with little damage because their transformations
are redundant at the local level.

## Original Hypothesis

"Block 0 may have more concentrated activation pattern (lower rank covariance)" — WRONG DIRECTION.
Block 0 has HIGHER input rank, not lower. Its criticality is from the diversity it must handle,
not from any specialization in a low-dimensional subspace.

## Impact

Finding closes item 205. The input rank collapse (269→207→25→1) across early blocks is a new
geometric characterization of GPT-2's residual stream. This confirms block 0 cannot be
compressed by the same approach used for blocks 1-11 (which operate on near-rank-1 input).
