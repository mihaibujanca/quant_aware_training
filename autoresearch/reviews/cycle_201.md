# Cycle 201 Review: Block 11 Geometry Deep-dive (Item 211)

## Verdict: CONFIRMED

Block 11 is geometrically anomalous in two ways that confirm a special "de-compression" role.

## Key Numbers

| Block | In rank | Out rank | Delta rank | Mean delta |
|-------|---------|----------|------------|------------|
| 3     | 1       | 1        | 207        | 19.4       |
| 7     | 1       | 1        | 401        | 30.9       |
| 10    | 1       | 2        | 335        | 101.6      |
| 11    | 2       | **231**  | **1**      | **335.7**  |

Outlier flags: output_rank_anomaly=True (231 vs median 1.5 = 154×), delta_token_concentration=True.
Delta direction coherent: False (mean |cos(δ_i, v₁)| = 0.245).

## Interpretation

**Block 11 output rank = 231** (from rank-2 input). This is the mirror of block 0: while
block 0 transforms high-rank input (244) into the residual stream and begins rank collapse,
block 11 receives the fully-collapsed rank-1/2 residual and expands it to rank-231 for the LM head.

**Block 11 delta rank = 1** — the block applies a nearly rank-1 (97% energy in top-3 SVs)
perturbation to each token. But the output_rank=231 suggests the delta contains a large shared
vector that, when added to the residual (which has a 2D span across tokens), produces a
high-dimensional output by constructive superposition.

**Token-type concentration**: top-20 tokens by delta norm have mean 848.7 (2.5× overall 335.7).
High-delta tokens include ' Hom', ' Returning', ' On', ' The', ' Head' — mostly common tokens
at start of common phrases. The variation across tokens is real but not obviously syntactic.

## Block 11 as "Output Decompressor"

Architecture picture:
- Blocks 0-1: transform high-rank embeddings into the rank-1/2 residual stream
- Blocks 2-10: operate on near-rank-1 residual (processing in 1D manifold)
- Block 11: expand rank-1/2 residual → rank-231 for the LM head (final projection prep)

This explains block 11's borderline droppability (41.4 PPL, cycle 187): when dropped, the LM
head receives a rank-2 residual stream instead of a rank-231 stream. Prediction is still
possible (rank-2 encodes most token identity) but PPL is high because the LM head was trained
to receive rank-231 input.

It also explains the "self-correction" in Frobenius divergence at blocks 10-11 (cycle 200):
block 11's rank expansion operation is robust and actively restructures the residual,
partially absorbing upstream compression errors.

## Impact

Block 11 is a "bookend" block along with block 0. Compressing block 11 with the same
topk+lstsq approach requires approximating a rank-expansion operation (rank-2 → rank-231),
which is fundamentally different from the rank-1 processing in blocks 2-10. This may explain
why [1,4,7,11] is a suboptimal block set for compression — block 11's output structure
matters more than any other compressed block.

Recommendation: exclude block 11 from the compression set (alongside block 0). Test
BLOCKS=[1,4,7] compression to see if removing block 11 from the compression set reduces PPL.
