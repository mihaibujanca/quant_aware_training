# Research Status — 74 Cycles (001–074)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 14 |
| Refuted | 21 |
| Inconclusive | 30 |
| Crash | 1 |

## Latest Cycle
- **Cycle 074** (item 28): Persistent homology GPT-2 — INCONCLUSIVE
  - PCA variance 72.4% for blocks 0-1 (below 80% threshold)
  - TP monotonically INCREASES through depth (opposite of hypothesis)
  - Null TP > block TP for 11/12 blocks (spherical baseline artifact)
  - **Topology direction definitively closed**: 5 cycles (047, 064, 065, 066, 074), 0 useful results

## Final Answer (from cycle 061, updated through cycle 074)

**Research question**: "Given a trained transformer LM and a corpus drawn from the data distribution it was trained on, how do we construct a much smaller FFN reparameterization that preserves geometry without SGD?"

**Answer**: Pure no-SGD compression works per-block but fails when stacking. The practical recipe requires minimal gradient polish:

### The Recipe (cycle 061)
1. **topk + lstsq** → analytic single-block FFN compression (sub-baseline PPL per block)
2. **Stack k=4 blocks at 6×** → +30% PPL unpolished
3. **500 Adam steps** (lr=1e-4, 4096 tokens, W_down only) → **+4.9% PPL on OpenWebText**

### Key Findings (spectral analysis phase)
- **FFN Jacobian rank**: 10/12 blocks rank_99 ≤ 500 (genuinely low-rank)
- **Stacking error = retained-spectrum corruption**: 77–91% of error in sensitive directions (cycle 072)
- **Rotation+lstsq > topk+lstsq**: any orthogonal projection + refit beats axis-aligned topk (cycle 071)
- **Uniform allocation near-optimal** for GPT-2 (blocks too homogeneous for rank-matching)
- **Channel selection partially corpus-specific**: Jaccard 0.34–0.58 across domains (cycle 070)

### Definitively Dead Directions
- **Topology-based methods**: 5 experiments (047, 064, 065, 066, 074), 0 useful results. Scale confound never resolved.
- Learned/distilled compressors: geometric baseline dominates
- Cascaded re-probing: negligible improvement
- Jacobian SVD basis at k/n=0.67: no better than random basis

## Remaining NEXT Items
- **P1**: Item 66 (mixed-corpus refit) — unblocked
- **P1**: Item 35 (region volume Pareto GPT-2) — unblocked
- **P2**: Items 85–88 (moonshots) — unblocked
- **Note**: Item 81 needs re-run with correct hook methodology
- **⚠ Agenda refresh needed soon**: only 2 P1 + 4 P2 items remain (below 4-item threshold when excluding moonshots)
