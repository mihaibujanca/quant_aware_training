# Research Status — 62 Cycles (001–062)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 13 |
| Refuted | 18 |
| Inconclusive | 22 |
| Crash | 1 |

## Latest Cycle
Cycle 062 — item 78 (reduced ratio stacking) — **INCONCLUSIVE**. k=4@3× = +18.5% mean (inconclusive band). Audit warns: ~15pp std is mostly baseline mismatch across eval splits, not method variance.

## Key Result: The Complete Recipe (updated)

**Analytic init + 500 gradient steps = practical FFN compression on broad text:**
1. topk + lstsq → analytic FFN compression (no training)
2. Stack k=4 blocks at 6× compression
3. Polish: 500 Adam steps (lr=1e-4) on 4096 held-out tokens
4. Result: **+4.9% PPL on OpenWebText** (cycle 061)

Without polish, stacking gives +30% (6×) or +18% (3×). Cascaded re-probing does NOT help (cycle 060). Gradient polish is the only effective stacking mitigation found.

## Remaining NEXT Items

| Item | Priority | Description | Dependencies | Status |
|------|----------|-------------|-------------|--------|
| 79 | P1 | Stacking-aware boundary correction | none | Eligible |
| 42 | P2 | Topology-preserving distillation | none | Eligible |
| 43 | P2 | Topological truncation | none | Eligible |
| 45 | P2 | Weight-space diffusion | none | Eligible |
