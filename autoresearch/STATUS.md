# Research Status — 21 Cycles Complete (009-029)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 4 (010, 019, 026, 027) |
| Refuted | 9 (009, 011, 012, 014, 020, 021, 025, 028, 029) |
| Inconclusive | 8 (013, 015, 016, 017, 018, 022, 023, 024) |

## What Works for No-SGD FFN Compression
1. **Data-driven channel selection** beats SVD and weight-based methods (cycle 026)
2. **Centroid activation similarity** is the best merge signal (ρ=0.85, cycle 027)
3. **Single-block identity removal** viable for easy blocks (+3.9% ppl, cycle 018)
4. **Global analytic method** is the best no-SGD approach tested

## What Doesn't Work
- Single-affine collapse (GPT-2), structural pruning (GELU), k-means/spectral on maps, local-chart compression (budget too thin), GELU→PWL, intrinsic dim, curvature, grokking

## Hard Ceiling
Compounding limits no-SGD compression to ~1 block + 5 heads. Deeper compression requires SGD.

## Next
P1: 18 (learned compressor generalization). Last P1 item before the agenda is fully processed.
