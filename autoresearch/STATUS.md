# Research Status — 127 Cycles (001–127)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 23 |
| Refuted | 37 |
| Inconclusive | 57 |
| Crash | 2 |

## Latest Cycles
- **Cycle 127** (item 144): Correction SVD alignment — INCONCLUSIVE
  - Corrections learn a partially novel subspace; moderate error overlap (cos 0.44-0.46 in middle blocks)
  - Attention alignment weak (max 0.25); block 7 has largest correction but lowest local error alignment
  - Suggests corrections compensate propagated (cross-block) error, not just local
- **Cycle 126** (item 143): PLAN INVALID — predecessor 142 refuted
- **Cycle 125** (item 142): Placement sweep — REFUTED (P1 near-optimal)

## Key Results (consolidated)

### Cascade Mechanism & Correction
- c≈0.41 is attention-driven, uniform across heads (cycles 104, 113)
- **LoRA cascade corrections recover 74% of degradation** with 49K params (cycle 115, confirmed 116)
- **Placement near-optimal at residual stream** (cycle 125)
- Corrections learn partially novel subspace — moderate error overlap, weak attention alignment (cycle 127)

### Compression Recipe
1. **FFN k≤4**: Topk + lstsq + 500 Adam at 6× → PPL ~80 (stacked)
2. **+ LoRA corrections** (r=8, 49K params): PPL ~59 (+14% over float 51.66)
3. **FFN k=5**: +22-24% polished (cycle 106)

### What Doesn't Work
- Allocation optimization (114), Jacobian methods (100, 103), GELU clustering (101)
- Correction placement optimization (125): residual stream already near-optimal

## Next Up
- **134**: Dead/always-active neuron census + exact pruning (P1, no deps)
- **100**: Is FFN isotropy architectural or learned? (P1, depends on 99)
