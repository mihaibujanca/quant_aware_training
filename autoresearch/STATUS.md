# Research Status — 119 Cycles (001–119)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 22 |
| Refuted | 36 |
| Inconclusive | 52 |
| Crash | 1 |

## Latest Cycle
- **Cycle 119** (item 136): Cascade corrections k=5 — CONFIRMED
  - Blocks [7,5,4,1,6] at 6× (topk+lstsq+500 Adam), r=8 LoRA corrections 2000 CE steps
  - Recovery: 73.8% ± varied (seeds 42/123/7: 72.9/73.9/75.2%), matches k=4 baseline 74.3%
  - Corrections scale gracefully 4→5 blocks, 61K params, stable recovery despite 70% more degradation
  - **Key finding**: k-scaling is robust; corrections preserve efficiency across compression depth

## Previous Cycle
- **Cycle 118** (item 128): MoE REAP pruning baseline — INCONCLUSIVE
  - Tiny Mixtral (15.5M, 4-layer) trained 20M tokens WikiText-2, final PPL=78.65 (>50 threshold)
  - REAP infrastructure confirmed; under-training responsible for PPL threshold miss

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
