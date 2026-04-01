# Cycle 197 Review: Block-Pair Joint Compression (Item 208)

## Verdict: INCONCLUSIVE

Sequential progressive fitting (each block fitted on compressed predecessors' output)
gains only 0.135 PPL over all-float fitting (30.089→29.954). Within measurement noise.

## Key Results

| Strategy | PPL | Description |
|----------|-----|-------------|
| Float | 25.082 | Baseline |
| A: all-float | 30.089 | All 4 blocks fitted on float model I/O |
| B: sequential | 29.954 | Each block fitted on compressed predecessor's I/O |
| Improvement | 0.135 | B better than A |

## Interpretation

- The distribution shift between float and compressed inputs to each block's MLP is
  SMALL for 4-block compression (only blocks 1,4,7,11 modified)
- The compressed model for 4 blocks is close enough to float that fitting on float
  data works nearly as well as fitting on compressed data
- The 5 PPL gap (25→30) between float and compressed is intrinsic to the compression
  itself (information loss from K=512 channels), not from distribution mismatch
- Sequential fitting doesn't help because the FORM of the compressed function
  (limited to K=512 channels) limits approximation quality, not training distribution

## What This Implies for Cascade

The cascade barrier is NOT primarily from:
- LN distributional mismatch (cycle 196: LN recal didn't help)
- Training/inference distribution mismatch (this experiment: 0.135 PPL gain)

The cascade barrier IS from:
- Fundamental information loss from compressing each block (each block contributes
  ~1.25 PPL per block = 5 PPL / 4 blocks)
- This compounds nonlinearly when many blocks are compressed (12 blocks → catastrophic PPL)
- The information loss is intrinsic: K=512 channels cannot faithfully represent 3072

## Methodological Note

Block-pair JOINT compression (compressing two blocks as a single unit) was not tested —
only sequential fitting. True joint compression would require fitting a combined student
on f_{l+1}(f_l(x)) simultaneously, which may be different from sequential.
