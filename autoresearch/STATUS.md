# Research Status — 76 Cycles (001–076)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 14 |
| Refuted | 21 |
| Inconclusive | 32 |
| Crash | 1 |

## Latest Cycle
- **Cycle 076** (item 66): Mixed-corpus lstsq refit — GPT-2 FFN — INCONCLUSIVE (audit-corrected)
  - Condition A (Shakespeare 512 tokens) has lstsq degeneracy (n=512, p=513) → PPL 3.7M
  - Condition C (mixed 2048 tokens) works: PPL 112.9 (+61%)
  - Confound: improvement is sample size, not corpus diversity. Needs rerun with matched n=2048.

## Final Answer (from cycle 061, still valid)

### The Recipe (cycle 061)
1. **topk + lstsq** → analytic single-block FFN compression (sub-baseline PPL per block)
2. **Stack k=4 blocks at 6×** → +30% PPL unpolished
3. **500 Adam steps** (lr=1e-4, 4096 tokens, W_down only) → **+4.9% PPL on OpenWebText**

### Key Findings
- FFN Jacobian genuinely low-rank (10/12 blocks rank_99 ≤ 500)
- Stacking error = retained-spectrum corruption (77-91%)
- Rotation+lstsq > topk+lstsq (but random ≈ SVD at k/n=0.67)
- Channel selection partially corpus-specific (Jaccard 0.34-0.58)
- Mixed-corpus refit: needs matched sample sizes to isolate diversity effect

### Dead Directions
- Topology (5 experiments, 0 results) — definitively closed
- Learned compressors, cascaded re-probing, variable ratios
- Jacobian SVD basis at high retention, rank-matching on GPT-2

## Remaining NEXT Items — ⚠ AGENDA REFRESH NEEDED
- **P2**: Items 85-88 (moonshots only)
- **Note**: Item 81 needs re-run, item 66 needs rerun with matched sizes
- All P0 and P1 items exhausted. Only moonshots remain → agenda refresh required.
