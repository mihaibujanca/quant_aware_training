# Research Status — 123 Cycles (001–123)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 23 |
| Refuted | 36 |
| Inconclusive | 54 |
| Crash | 2 |

## Latest Cycle
- **Cycle 123** (item 140): QK vs V error decomposition — INCONCLUSIVE
  - V dominates early layers (L2-L4: V_frac ~0.66), QK dominates late (L9-L11: QK_frac ~0.57)
  - No component exceeds 70% threshold in any layer; dominant source varies by position
  - Cross-seed stability excellent (std < 0.02). Interaction terms small (<0.1)
  - **Finding**: no single surgical target — correction must be layer-adaptive (V-focused early, QK-focused late)

## Previous Cycles
- **Cycle 121** (item 138): MoE training — INCONCLUSIVE (audit-corrected)
  - Tiny Mixtral under-trained without dropout; PPL 65.08 best at 15M tokens
- **Cycle 120** (item 137): Cascade corrections k=8 — CONFIRMED WITH CAVEAT
  - Recovery 91.4%, 98K params; corrected PPL 143.6 vs baseline 57.7 (+149%)
- **Cycle 119** (item 136): Cascade corrections k=5 — CONFIRMED
  - Recovery 73.8% (seeds 42/123/7: 72.9/73.9/75.2%), 61K params, matches k=4 baseline 74.3%

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
- **141**: Attention entropy shift under FFN compression (P0)
- **142**: Correction placement sweep (P0, informed by cycle 123)
- **134**: Dead/always-active neuron census + exact pruning (P1)
