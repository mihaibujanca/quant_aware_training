# Research Status — 59 Cycles (001–059)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 12 |
| Refuted | 17 |
| Inconclusive | 21 |
| Crash | 1 |

## Latest Cycle
Cycle 059 — item 74 (gradient-polished lstsq) — **CONFIRMED**. 50 Adam steps reduce stacking degradation from +30.2% to +18.4% (38.9% relative improvement). The lstsq solution is a good starting point for gradient refinement.

## Consolidated Findings

1. **Per-block method works**: Single-block 6× on OWT = sub-baseline PPL (cycles 056-057).
2. **Stacking is the sole bottleneck**: +30% at k=4 vs sub-baseline per-block (cycles 057-058).
3. **Gradient polish helps**: 50 Adam steps reduce stacking degradation by 39% relative (cycle 059). Still +18.4% though.
4. **Compression-difficulty oracle**: P1 rho=0.84, corpus-invariant rho=0.94 (cycles 048, 055).
5. **Channel selection is corpus-invariant**: Jaccard=0.5 but zero PPL impact (cycle 057).

## Remaining NEXT Items

| Item | Priority | Description | Dependencies | Status |
|------|----------|-------------|-------------|--------|
| 42 | P2 | Topology-preserving distillation | none | Eligible |
| 43 | P2 | Topological truncation | none | Eligible |
| 45 | P2 | Weight-space diffusion | none | Eligible |

## Note
All P0/P1 items exhausted. Only P2 moonshots remain. Agenda refresh needed — new items should focus on improving stacking (iterative re-probing, cascaded correction, more gradient steps, residual-aware compression).
