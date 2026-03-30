# Research Status — 125 Cycles (001–125)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 23 |
| Refuted | 37 |
| Inconclusive | 55 |
| Crash | 2 |

## Latest Cycle
- **Cycle 125** (item 142): Correction placement sweep — REFUTED
  - P1 (after FFN, residual stream) PPL=69.52, best of 5 placements
  - P5 (dual residual) PPL=70.18, P2 (after attn) 70.97, P4 (V+O) 71.65, P3 (Q+K) 80.85
  - P1 within 2% of best → placement is near-optimal, not worth optimizing further

## Previous Cycles
- **Cycle 124** (item 141): Attention entropy shift — INCONCLUSIVE
- **Cycle 123** (item 140): QK vs V error decomposition — INCONCLUSIVE
- **Cycle 121** (item 138): MoE training — INCONCLUSIVE (audit-corrected)
- **Cycle 120** (item 137): Cascade corrections k=8 — CONFIRMED

## Key Results (consolidated)

### Cascade Mechanism & Correction
- c≈0.41 is attention-driven, uniform across heads (cycles 104, 113)
- **LoRA cascade corrections recover 74% of degradation** with 49K params (cycle 115, confirmed 116)
- **Placement is near-optimal at residual stream** (cycle 125): P1 best, attention-internal corrections worse
- **Extended polish cannot replicate corrections** — ablation confirmed (cycle 116)

### Compression Recipe
1. **FFN k≤4**: Topk + lstsq + 500 Adam at 6× → PPL ~80 (stacked)
2. **+ LoRA corrections** (r=8, 49K params): PPL ~59 (+14% over float 51.66)
3. **FFN k=5**: +22-24% polished (cycle 106, confirmed 116)

### What Doesn't Work
- Allocation optimization (114), Jacobian methods (100, 103), GELU clustering (101)
- Zero-shot GELU→ReLU (102), head-level amplification targeting (113)
- Correction placement optimization (125): residual stream already near-optimal

## Next Up
- **134**: Dead/always-active neuron census + exact pruning (P1)
- **100**: Is FFN isotropy architectural or learned? (P1, depends on 99)
- **143**: Best correction placement at scale (P0, depends on 142 — deprioritized by cycle 125)
