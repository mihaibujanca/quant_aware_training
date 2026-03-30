# Research Status — 124 Cycles (001–124)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 23 |
| Refuted | 36 |
| Inconclusive | 55 |
| Crash | 2 |

## Latest Cycle
- **Cycle 124** (item 141): Attention entropy shift — INCONCLUSIVE
  - Only 2/12 layers exceed 0.1 nats |delta_H| (need >=8 for CONFIRMED)
  - Mean |delta_H| = 0.059 nats — shift is real but small, not REFUTED either
  - Negative correlation with amplification (rho=-0.55, p=0.08) — fragile, inflated by trivial L1
  - **Interesting**: signed analysis shows attention sharpening; signed rho=+0.72 (p=0.013) significant

## Previous Cycles
- **Cycle 123** (item 140): QK vs V error decomposition — INCONCLUSIVE
  - V dominates early layers, QK dominates late; no component >70% anywhere
- **Cycle 121** (item 138): MoE training — INCONCLUSIVE (audit-corrected)
- **Cycle 120** (item 137): Cascade corrections k=8 — CONFIRMED WITH CAVEAT
- **Cycle 119** (item 136): Cascade corrections k=5 — CONFIRMED

## Key Results (consolidated)

### Cascade Mechanism & Correction
- c≈0.41 is attention-driven, uniform across heads (cycles 104, 113)
- **LoRA cascade corrections recover 74% of degradation** with 49K params (cycle 115, confirmed 116)
- **Extended polish cannot replicate corrections** — ablation confirmed (cycle 116)

### Compression Recipe
1. **FFN k≤4**: Topk + lstsq + 500 Adam at 6× → PPL ~80 (stacked)
2. **+ LoRA corrections** (r=8, 49K params): PPL ~59 (+14% over float 51.66)
3. **FFN k=5**: +22-24% polished (cycle 106, confirmed 116)
4. **Attention**: Per-head rank-11 SVD on QKV (skip block 0)

### What Doesn't Work
- Allocation optimization (114), Jacobian methods (100, 103), GELU clustering (101)
- Zero-shot GELU→ReLU (102), head-level amplification targeting (113)

### GELU Structure
- Negative leak = 87-99% constant bias (cycle 109)
- Regime depth-invariant ~85% gated (cycle 111)

## Next Up
- **142**: Correction placement sweep — GPT-2 FFN k=4 (P0)
- **134**: Dead/always-active neuron census + exact pruning (P1)
- **100**: Is FFN isotropy architectural or learned? (P1, depends on 99)
