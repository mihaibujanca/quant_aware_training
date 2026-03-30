# Research Status — 115 Cycles (001–115)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 19 |
| Refuted | 36 |
| Inconclusive | 51 |
| Crash | 1 |

## Latest Cycle
- **Cycle 115** (item 118): Cascade corrections — CONFIRMED
  - Per-block LoRA-style corrections (I + UV) on residual stream after compressed FFN
  - r=4: 73% degradation recovery (24K params), r=8: 74% (49K params), r=16: 74% (saturated)
  - PPL: 80.68 → 59.11 (from +56% to +14% over baseline 51.66)
  - **Audit flags**: LoRA-equivalent, needs extended-polish ablation, per-block vs e2e polish gap

## Key Results (consolidated)

### Cascade Mechanism
- c≈0.41 is attention-driven, uniform across heads (cycles 104, 113)
- **LoRA cascade corrections recover 74% of degradation** with 49K params (cycle 115)
- But: may be equivalent to extended polish — ablation needed

### Compression Recipe (updated)
1. **FFN**: Topk + lstsq + 500 Adam, k≤4 at 6× → PPL ~80 (stacked)
2. **+ LoRA corrections** (r=8, 49K params): PPL ~59 (+14% over float)
3. **FFN k=5**: +22-24% polished (cycle 106)
4. **Attention**: Per-head rank-11 SVD on QKV (skip block 0)

### What Doesn't Work
- Allocation optimization (cycles 094, 114)
- Subspace projection (099), Jacobian factorization (100), GELU clustering (101)
- Zero-shot GELU→ReLU (102), head-level targeting (113)

### GELU Structure
- Negative leak = 87-99% constant bias (109)
- Regime depth-invariant ~85% gated (111)
- Block 1 anomalously linear (111)

## Remaining NEXT Items
- **P2**: Items 87, 88, 126 (blocked by 124 refuted), 127
- 3 eligible items — AGENDA REFRESH NEEDED
