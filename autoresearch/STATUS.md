# Research Status — 75 Cycles (001–075)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 14 |
| Refuted | 21 |
| Inconclusive | 31 |
| Crash | 1 |

## Latest Cycle
- **Cycle 075** (item 35): Region volume Pareto GPT-2 — INCONCLUSIVE
  - Gini structure real (0.316 vs null 0.179, 12/12 blocks above null)
  - Top-20% coverage only 5-26% (need 70%) — Pareto fails decisively
  - Diffuse cluster structure reinforces uniform compression strategy

## Final Answer (from cycle 061, still valid)

**Answer**: Pure no-SGD compression works per-block but fails when stacking. The practical recipe requires minimal gradient polish:

### The Recipe (cycle 061)
1. **topk + lstsq** → analytic single-block FFN compression (sub-baseline PPL per block)
2. **Stack k=4 blocks at 6×** → +30% PPL unpolished
3. **500 Adam steps** (lr=1e-4, 4096 tokens, W_down only) → **+4.9% PPL on OpenWebText**

### Key Findings (spectral phase + recent)
- FFN Jacobian genuinely low-rank (10/12 blocks rank_99 ≤ 500)
- Stacking error = retained-spectrum corruption (77-91%)
- Rotation+lstsq > topk+lstsq (but random ≈ SVD at k/n=0.67)
- Jacobian cluster structure diffuse, not Pareto (cycle 075)
- Channel selection partially corpus-specific (Jaccard 0.34-0.58)

### Dead Directions
- Topology (5 experiments, 0 results)
- Learned compressors, cascaded re-probing, variable ratios
- Jacobian SVD basis at high retention
- Rank-matching on GPT-2 (blocks too homogeneous)

## Remaining NEXT Items — ⚠ REFRESH NEEDED
- **P1**: Item 66 (mixed-corpus refit) — only remaining P1
- **P2**: Items 85-88 (moonshots)
- **Note**: Item 81 needs re-run with correct hook methodology
- **Only 1 P1 + 4 P2 = 5 items left** → agenda refresh needed (threshold: <4 non-moonshot items)
