# Research Status — 118 Cycles (001–118)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 21 |
| Refuted | 36 |
| Inconclusive | 52 |
| Crash | 1 |

## Latest Cycle
- **Cycle 118** (item 128): MoE REAP pruning baseline — INCONCLUSIVE
  - Tiny Mixtral (15.5M, 4-layer) trained 20M tokens WikiText-2, final PPL=78.65 (>50 threshold)
  - REAP saliency scoring works: Gini [0.481, 0.351, 0.345, 0.377], monotone degradation
  - REAP 50% = 103.25 PPL vs random 50% = 299.58±57.65 (>3σ better)
  - Outcome: under-training, but infrastructure confirmed working for follow-on studies (items 129-131)

## Previous Cycle
- **Cycle 117** (item 133): Cascade corrections k=5+k=8 — CRASH
  - Killed after 140+ min (exceeded 30-min cap)
  - Superseded by split items 136 (k=5) and 137 (k=8)

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
