# Research Status — 61 Cycles (001–061)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 13 |
| Refuted | 18 |
| Inconclusive | 21 |
| Crash | 1 |

## Latest Cycle
Cycle 061 — item 77 (gradient polish scaling) — **CONFIRMED**. 500 Adam steps reduce stacking degradation from +30.2% to +4.9% (independent) / +3.8% (joint). The stacking penalty is essentially eliminated with minimal gradient polish. No plateau at 500 steps.

## Key Result: The Complete Compression Recipe

**Analytic init + 500 gradient steps achieves practical FFN compression on broad text:**
1. Select top-k channels per block using topk + centroid affinity (no training needed)
2. Fit lstsq output projection (analytic, no training)
3. Stack k=4 blocks at 6× compression
4. Polish: 500 Adam steps (lr=1e-4) on 4096 held-out tokens, updating only W_down
5. Result: +4.9% PPL on OpenWebText (GPT-2's training distribution)

This answers the program's central question: **no-training FFN compression IS viable when combined with a tiny amount of gradient polish (500 steps << full training).**

## Remaining NEXT Items

| Item | Priority | Description | Dependencies | Status |
|------|----------|-------------|-------------|--------|
| 78 | P1 | Reduced ratio for stacking (k=4@2-3x) | none | Eligible |
| 79 | P1 | Stacking-aware boundary correction | none | Eligible |
| 42 | P2 | Topology-preserving distillation | none | Eligible |
| 43 | P2 | Topological truncation | none | Eligible |
| 45 | P2 | Weight-space diffusion | none | Eligible |
