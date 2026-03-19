# Research Status — 35 Cycles (009-042)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 9 (010, 019, 026, 027, 033, 039, 040, 041, 042) |
| Refuted | 13 |
| Inconclusive | 12 |

## Latest (cycle 042) — KEY RESULT
k=4@10× corrected: **+8.45% PPL at ~30% FFN param reduction** (3 seeds). Matches k=4@6× corrected. Per-block ratio doesn't matter once correction is applied — the binding constraint is block count.

## Best Compression Recipe (GPT-2 proxy)
1. Select blocks [7,5,4,1] (4 easiest by single-block damage)
2. Compress each at 10× via topk channel selection + lstsq output refit
3. Fit affine correction at block 8 input via lstsq on ~25K probe tokens
4. **Result: ~30% FFN param reduction at +8.45% PPL, zero extra inference params**

## Method Generalization
- GPT-2-small: confirmed (all cycles)
- GPT-2-medium: confirmed (cycle 041, 5/6 blocks pass at 4×)
- Block 0 universally fails across model sizes

## Eligible Next Items
- **P0 Item 61**: Multi-point correction — can k=5 become viable?
- **P0 Item 62**: Domain robustness — does it work beyond Shakespeare?
- **P0 Item 63**: Practical recipe consolidation
