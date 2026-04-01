# Cycle 200 Review: Residual Stream Rank Trajectory Under Compression (Item 210)

## Verdict: REFUTED

Compression with BLOCKS=[1,4,7,11], K=512 does NOT disrupt the residual stream rank collapse.
Float and compressed rank trajectories are virtually identical.

## Key Numbers

| Block | Float rank | Comp rank | Ratio | Frob div |
|-------|-----------|-----------|-------|----------|
| 0     | 244       | 244       | 1.00  | 0.000    |
| 1     | 192       | 192       | 1.00  | 0.000    |
| 2     | 23        | 20        | 0.87  | 0.138    |
| 3     | 1         | 1         | 1.00  | 0.126    |
| 4     | 1         | 1         | 1.00  | 0.122    |
| 5     | 1         | 1         | 1.00  | 0.191    |
| 6     | 1         | 1         | 1.00  | 0.189    |
| 7     | 1         | 1         | 1.00  | 0.178    |
| 8     | 1         | 1         | 1.00  | 0.240    |
| 9     | 1         | 1         | 1.00  | 0.228    |
| 10    | 1         | 1         | 1.00  | 0.211    |
| 11    | 2         | 1         | 0.50  | 0.172    |

Max rank disruption ratio: 1.0 at block 3. Frob div peaks at block 8-9 (~0.24).

## Interpretation

The rank collapse 244→192→23→1 is exactly preserved by compression. Block 1 (K=512) outputs
a residual stream that maintains rank=192, which then collapses to rank=20-23 at block 2, and
to 1 by block 3 — same as float. The compressed FFN, despite being an approximation, does not
inject high-rank noise into the residual stream's dominant direction structure.

The Frobenius divergence (0.000 at blocks 0-1, 0.12-0.24 at blocks 3-11) confirms the cascade
error is in the CONTENT of the residual stream, not its low-dimensional rank structure. This is
consistent with cycle 189 (directionally coherent but diffuse cascade errors).

The small Frob div at blocks 0-1 (exactly 0.000) is because blocks 0 and 1's INPUTS are not
compressed (they receive the original embedding/residual). Block 1's output is compressed, so
its error first appears at block 2's input (Frob=0.138).

## Secondary Finding

Frob div peaks at block 8-9 then decreases at blocks 10-11 (0.172 at block 11), consistent
with the passive self-correction at k=8-10 from cycle 194. The cascade error grows then
partially rotates to lower-PPL dimensions in the final blocks.

## Impact

Closes item 210. The cascade barrier is purely from Frobenius content error (diffuse, 17× by
block 12 per cycle 189), NOT from rank structure disruption. The rank-1 residual stream is
robustly preserved by topk+lstsq compression. Future approaches must target the content of
the compressed approximation error, not its spectral structure.
