# Research Status — 72 Cycles (001–072)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 14 |
| Refuted | 21 |
| Inconclusive | 28 |
| Crash | 1 |

## Latest Cycle
- **Cycle 072** (item 83): Stacking error spectral decomposition — GPT-2 FFN — REFUTED
  - Error energy in retained spectrum: 77–91% (vs 63–82% random baseline)
  - Stacking shift: < 0.01 — stacking amplifies magnitude (1.4–3.6×) but does NOT shift spectral profile
  - Compression error lives where the model is most sensitive, not where info was discarded
  - Gradient polish = retained-spectrum denoising, not information recovery

## Final Answer (from cycle 061, updated with cycle 072 insight)

**Research question**: "Given a trained transformer LM and a corpus drawn from the data distribution it was trained on, how do we construct a much smaller FFN reparameterization that preserves geometry without SGD?"

**Answer**: Pure no-SGD compression works per-block but fails when stacking. The practical recipe requires minimal gradient polish:

### The Recipe (cycle 061)
1. **topk + lstsq** → analytic single-block FFN compression (sub-baseline PPL per block)
2. **Stack k=4 blocks at 6×** → +30% PPL unpolished
3. **500 Adam steps** (lr=1e-4, 4096 tokens, W_down only) → **+4.9% PPL on OpenWebText**

### Key Supporting Findings
- **Per-block is essentially free**: single-block 6× gives sub-baseline PPL
- **Stacking is the sole bottleneck**: +30% at k=4 vs sub-baseline per-block
- **Gradient polish is retained-spectrum denoising**: error lives in sensitive directions (77-91%), stacking amplifies but doesn't shift spectrum (cycle 072)
- **Rotation+lstsq > topk+lstsq**: any orthogonal projection + refit beats axis-aligned topk (cycle 071)
- **Compression-difficulty oracle**: P1 Frobenius std rho=0.84, corpus-invariant rho=0.94
- **Channel selection partially corpus-specific**: Jaccard 0.34–0.58 across domains (cycle 070)

### Dead Ends
- Topology-based methods: 4 experiments, 0 useful results
- Learned/distilled compressors: geometric baseline dominates
- Cascaded re-probing: negligible improvement
- Variable per-block ratios: breaks correction step
- Jacobian SVD basis at k/n=0.67: no better than random basis

## Remaining NEXT Items
- **P0**: Item 84 (rank-matched adaptive compression)
- **P1**: Item 66 (mixed-corpus refit) — unblocked
- **P1**: Item 28 (persistent homology GPT-2) — unblocked
- **P1**: Item 35 (region volume Pareto GPT-2) — unblocked
- **P2**: Items 85–88 (moonshots) — unblocked
- **Note**: Item 81 needs re-run with correct hook methodology
