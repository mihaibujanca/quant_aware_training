# Research Status — 71 Cycles (001–071)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 14 |
| Refuted | 20 |
| Inconclusive | 28 |
| Crash | 1 |

## Latest Cycle
- **Cycle 071** (item 82): Jacobian SVD compression vs topk — GPT-2 FFN — CONFIRMED (with caveat)
  - SVD+lstsq beats topk+lstsq by ≥3% in 2/4 blocks (meets criterion)
  - BUT random+lstsq (122.31) ≈ SVD+lstsq (121.84) — Jacobian SVD basis not special at k/n=0.67
  - Real mechanism: any rotation+refit >> axis-aligned topk. lstsq refit dominates basis choice.
  - Implication: skip expensive Jacobian SVD, use random projection + lstsq instead at this ratio

## Final Answer (from cycle 061, still valid)

**Research question**: "Given a trained transformer LM and a corpus drawn from the data distribution it was trained on, how do we construct a much smaller FFN reparameterization that preserves geometry without SGD?"

**Answer**: Pure no-SGD compression works per-block but fails when stacking. The practical recipe requires minimal gradient polish:

### The Recipe (cycle 061)
1. **topk + lstsq** → analytic single-block FFN compression (sub-baseline PPL per block)
2. **Stack k=4 blocks at 6×** → +30% PPL unpolished
3. **500 Adam steps** (lr=1e-4, 4096 tokens, W_down only) → **+4.9% PPL on OpenWebText**

### Key Supporting Findings
- **Per-block is essentially free**: single-block 6× gives sub-baseline PPL (lstsq regularization)
- **Stacking is the sole bottleneck**: +30% at k=4 vs sub-baseline per-block (cycles 057-058)
- **Gradient polish is the answer**: 500 steps = 84% relative reduction of stacking error (cycle 061)
- **Rotation+lstsq > topk+lstsq**: any orthogonal projection + lstsq refit beats axis-aligned topk (cycle 071)
- **Compression-difficulty oracle**: P1 Frobenius std rho=0.84, corpus-invariant rho=0.94 (cycles 048, 055)
- **Channel selection partially corpus-specific**: Jaccard 0.34–0.58 across domains (cycle 070)

### Dead Ends
- Topology-based methods: 4 experiments, 0 useful results
- Learned/distilled compressors: geometric baseline dominates (cycles 030, 049)
- Cascaded re-probing: negligible improvement (cycle 060)
- Variable per-block ratios: breaks correction step (cycle 050)
- Jacobian SVD basis at k/n=0.67: no better than random basis (cycle 071)

## Remaining NEXT Items
- **P0**: Items 83–84 (stacking error decomposition, rank-matched compression)
- **P1**: Item 66 (mixed-corpus refit) — unblocked (64 complete)
- **P1**: Item 28 (persistent homology GPT-2) — unblocked
- **P1**: Item 35 (region volume Pareto GPT-2) — unblocked
- **P2**: Items 85–88 (moonshots) — unblocked
- **Note**: Item 81 needs re-run with correct hook methodology
