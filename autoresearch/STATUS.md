# Research Status — 66 Cycles (001–066) — AGENDA COMPLETE

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 13 |
| Refuted | 19 |
| Inconclusive | 25 |
| Crash | 1 |

## Final Answer

**Research question**: "Given a trained transformer LM and a corpus drawn from the data distribution it was trained on, how do we construct a much smaller FFN reparameterization that preserves geometry without SGD?"

**Answer**: Pure no-SGD compression works per-block but fails when stacking. The practical recipe requires minimal gradient polish:

### The Recipe (cycle 061)
1. **topk + lstsq** → analytic single-block FFN compression (sub-baseline PPL per block)
2. **Stack k=4 blocks at 6×** → +30% PPL unpolished
3. **500 Adam steps** (lr=1e-4, 4096 tokens, W_down only) → **+4.9% PPL on OpenWebText**

### Key Supporting Findings
- **Per-block is essentially free**: single-block 6× gives sub-baseline PPL (lstsq regularization)
- **Stacking is the sole bottleneck**: +30% at k=4 vs sub-baseline per-block (cycles 057-058)
- **Cascaded re-probing doesn't help**: error propagates through attention/LN (cycle 060)
- **Boundary corrections partially help**: +5pp gain, but far less than gradient polish (cycle 063)
- **Gradient polish is the answer**: 500 steps = 84% relative reduction of stacking error (cycle 061)
- **Compression-difficulty oracle**: P1 Frobenius std rho=0.84, corpus-invariant rho=0.94 (cycles 048, 055)
- **Channel selection is corpus-invariant**: Jaccard=0.5 but zero PPL impact (cycle 057)

### Dead Ends
- Topology-based methods (PH, Betti, topological distillation): 4 experiments, 0 useful results
- Learned/distilled compressors: geometric baseline dominates (cycles 030, 049)
- Cascaded re-probing: negligible improvement (cycle 060)
- Variable per-block ratios: breaks correction step (cycle 050)

## Remaining NEXT Items
None. All agenda items exhausted.
