# Research Status — 36 Cycles (009-043)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 9 (010, 019, 026, 027, 033, 039, 040, 041, 042) |
| Refuted | 13 |
| Inconclusive | 13 (+ 043) |

## Best Compression Recipe (GPT-2 proxy, FINAL)
1. Select 4 easiest blocks by single-block damage ranking [7,5,4,1]
2. Compress each at 6-10× via topk channel selection + lstsq output refit
3. Fit single affine correction at block 8 input via lstsq on ~25K probe tokens
4. **Result: ~28-30% FFN param reduction at +8-9% PPL, zero extra inference params**

## Key Results Summary
| Finding | Cycle | Status |
|---------|-------|--------|
| Topk + lstsq is the viable no-SGD method | 026-027 | CONFIRMED |
| Single-block 10× is free (regularization) | 040 | CONFIRMED |
| 4-block stacking at 10× + correction = +8.45% | 042 | CONFIRMED |
| Residual correction recovers 2-3.6pp | 039, 042 | CONFIRMED |
| Method generalizes to GPT-2-medium | 041 | CONFIRMED |
| Block count wall is hard (5 blocks >15%) | 037, 038 | REFUTED items |
| Multi-point corrections interfere | 043 | INCONCLUSIVE |
| Single end-point correction is optimal | 039, 043 | Established |

## Eligible Next
- **P0 Item 62**: Domain robustness (non-Shakespeare eval)
- **P0 Item 63**: Practical recipe consolidation
- **P1 Items 50, 51, 55**: Secondary experiments
