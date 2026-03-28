# Research Status — 77 Cycles (001–077)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 14 |
| Refuted | 22 |
| Inconclusive | 32 |
| Crash | 1 |

## Latest Cycle
- **Cycle 077** (item 89): SVD vs random at aggressive compression — GPT-2 FFN — REFUTED
  - SVD+lstsq ≈ random+lstsq at ALL k/n ratios (0.1, 0.2, 0.3) — all diffs < 2%
  - lstsq refit completely dominates basis choice. FFN output is isotropic.
  - topk+lstsq much worse (126-165 vs ~123) — axis-aligned selection is the real bottleneck
  - ⚠ Single-block PPL is misleading: lstsq acts as domain regularizer (shakespeare→shakespeare)

## Final Answer (from cycle 061, consolidated through cycle 077)

**Answer**: Pure no-SGD compression works per-block but fails when stacking. The practical recipe requires minimal gradient polish:

### The Recipe
1. **Random projection + lstsq** → analytic single-block compression (rotation-invariant, no Jacobian needed)
2. **Stack k=4 blocks at 6×** → +30% PPL unpolished
3. **500 Adam steps** (lr=1e-4, W_down only) → **+4.9% PPL**

### Key Findings (now definitive)
- **Basis is irrelevant**: SVD = random at all compression ratios (cycles 071, 077). Only k/n matters.
- **Stacking error = retained-spectrum corruption** (77-91%, cycle 072)
- **Gradient polish = retained-spectrum denoising**
- **Channel selection partially corpus-specific** (Jaccard 0.34-0.58, cycle 070)
- **Single-block metrics are misleading**: lstsq regularization inflates single-block quality

### Dead Directions
- Topology (5 experiments, 0 results)
- Jacobian SVD basis (any rotation is equivalent)
- Rank-matching (GPT-2 blocks too homogeneous)
- Learned compressors, cascaded re-probing

## Remaining NEXT Items
- **P0**: Item 90 (mixed-corpus refit matched sizes)
- **P1**: Item 91 (subspace alignment re-run)
- **P2**: Items 85-88, 92 (moonshots)
