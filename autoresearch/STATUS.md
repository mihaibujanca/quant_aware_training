# Research Status — 96 Cycles (001–096)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 27 |
| Inconclusive | 45 |
| Crash | 1 |

## Latest Cycle
- **Cycle 096** (item 111): Cascade attribution — plan rejected (verdict logic issue)

## Key Findings Summary

### Structural Stacking Limit
- c≈0.41/block cascade rate is ratio/allocation/order-independent (089, 093, 094)
- Same pattern in attention: rank-16 ≈ rank-11 in multi-block (095)
- Bottleneck = inter-block error propagation, not per-block quality

### FFN Compression
- Topk + lstsq + 500 Adam → ~+5% PPL at k≤4 blocks 6×
- Basis irrelevant, channel heuristic sufficient, GELU fully active
- Piecewise-linear dead (both input and output space)

### Attention Compression
- 5/12 blocks viable at 5.4× QKV SVD (skip block 0)
- Block 0 catastrophically sensitive (+102%)

## Remaining NEXT Items
- **P0**: Items 103 (low-rank Jacobian), 112 (inter-block residual correction)
- **P1**: Items 113 (joint FFN+attention), 114 (block-0-skip stacking)
- **P2**: Items 87-88 (Procrustes, learned bottleneck AE)
