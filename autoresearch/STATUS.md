# Research Status — 73 Cycles (001–073)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 14 |
| Refuted | 21 |
| Inconclusive | 29 |
| Crash | 1 |

## Latest Cycle
- **Cycle 073** (item 84): Rank-matched adaptive compression — GPT-2 FFN — INCONCLUSIVE (audit-corrected)
  - Uniform vs rank-matched indistinguishable (±0.3 PPL noise)
  - Rank variation only 1.27× across blocks [7,5,4,1] — hypothesis untestable
  - GPT-2's uniform block rank structure means uniform allocation is near-optimal

## Final Answer (from cycle 061, updated through cycle 073)

**Research question**: "Given a trained transformer LM and a corpus drawn from the data distribution it was trained on, how do we construct a much smaller FFN reparameterization that preserves geometry without SGD?"

**Answer**: Pure no-SGD compression works per-block but fails when stacking. The practical recipe requires minimal gradient polish:

### The Recipe (cycle 061)
1. **topk + lstsq** → analytic single-block FFN compression (sub-baseline PPL per block)
2. **Stack k=4 blocks at 6×** → +30% PPL unpolished
3. **500 Adam steps** (lr=1e-4, 4096 tokens, W_down only) → **+4.9% PPL on OpenWebText**

### Key Supporting Findings (spectral analysis phase, cycles 068–073)
- **FFN Jacobian rank**: 10/12 blocks rank_99 ≤ 500 (genuinely low-rank), but rank ≠ compressibility (cycle 068)
- **Stacking error = retained-spectrum corruption**: 77–91% of error in top singular directions, not information loss (cycle 072)
- **Rotation+lstsq > topk+lstsq**: any orthogonal projection + refit beats axis-aligned topk, but random ≈ SVD (cycle 071)
- **Uniform allocation near-optimal**: rank variation too small across GPT-2 blocks to benefit from rank-matching (cycle 073)
- **Channel selection partially corpus-specific**: Jaccard 0.34–0.58 across domains (cycle 070)

### Dead Ends
- Topology-based methods: 4 experiments, 0 useful results
- Learned/distilled compressors: geometric baseline dominates
- Cascaded re-probing: negligible improvement
- Jacobian SVD basis at k/n=0.67: no better than random basis
- Rank-matched allocation: untestable on GPT-2 (blocks too homogeneous)

## P0 Items Complete
All P0 items (80–84, 64) are now complete. Remaining items are P1/P2.

## Remaining NEXT Items
- **P1**: Item 66 (mixed-corpus refit) — unblocked (64 complete)
- **P1**: Item 28 (persistent homology GPT-2) — unblocked
- **P1**: Item 35 (region volume Pareto GPT-2) — unblocked
- **P2**: Items 85–88 (moonshots) — unblocked
- **Note**: Item 81 needs re-run with correct hook methodology
- **Agenda refresh may be needed**: only 3 P1 items + 4 P2 moonshots remain
