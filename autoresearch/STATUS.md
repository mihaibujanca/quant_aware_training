# Research Status — 120 Cycles (001–120)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 23 |
| Refuted | 36 |
| Inconclusive | 52 |
| Crash | 1 |

## Latest Cycle
- **Cycle 120** (item 137): Cascade corrections k=8 — CONFIRMED WITH CAVEAT
  - 8-block stacking at 6×, r=8 LoRA corrections (98K params), seed=42
  - **Recovery: 91.4%** by r=8 metric, BUT corrected PPL 143.6 vs baseline 57.7 (+149% above baseline)
  - Progression: k=4 PPL 59.1 → k=5 PPL 70.1 → k=8 PPL 143.6 (corrected)
  - **Interpretation**: Corrections help but scaling stalls; compound error overwhelms recovery at 8 blocks
  - **Key finding**: k-scaling has hard limit; corrections necessary but insufficient for deep stacking

## Previous Cycles
- **Cycle 119** (item 136): Cascade corrections k=5 — CONFIRMED
  - Recovery 73.8% (seeds 42/123/7: 72.9/73.9/75.2%), 61K params, matches k=4 baseline 74.3%
- **Cycle 118** (item 128): MoE REAP pruning baseline — INCONCLUSIVE
  - Tiny Mixtral under-trained; REAP infrastructure confirmed

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
- **136**: Cascade corrections k=5 only (P0 priority)
- **137**: Cascade corrections k=8 only (P0 priority)
- **129**: Geometric decomposition of expert pruning (MoE follow-on, P0)
