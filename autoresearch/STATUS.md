# Research Status — 114 Cycles (001–114)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 18 |
| Refuted | 36 |
| Inconclusive | 51 |
| Crash | 1 |

## Latest Cycle
- **Cycle 114** (item 125): Linearity-adaptive compression ratios — REFUTED
  - Uniform vs best non-uniform: 0.90% delta (< 1% threshold)
  - Middle-heavy allocation does not help despite linearity gradient
  - Second independent refutation of allocation-based improvement (cycle 094)
  - Capacity distribution is not a lever at fixed total budget

## Key Structural Findings (consolidated)

### Cascade Mechanism (cycles 104, 113)
- **c≈0.41 is attention-driven**, uniform across heads (no surgical targets)
- GELU attenuates error in early layers

### Compression: What Doesn't Work
- Allocation optimization: irrelevant after polish (cycles 094, 114)
- Subspace projection: error spans 85% of dims (099)
- Jacobian factorization: full-rank (100)
- GELU clustering: K*>256 all blocks (101)
- Zero-shot GELU→ReLU: catastrophic (102)
- Head-level cascade targeting: uniform amplification (113)

### Compression: What Works
1. **FFN**: Topk + lstsq + 500 Adam, k≤4 at 6× → ~+5% polished
2. **FFN k=5**: +22-24% polished, seed-dependent (cycle 106)
3. **Attention**: Per-head rank-11 SVD on QKV (skip block 0)

## Remaining NEXT Items
- **P1**: Item 100 (blocked on 99)
- **P2**: Items 87, 88, 118, 126, 127
- 5 eligible items — approaching refresh threshold
