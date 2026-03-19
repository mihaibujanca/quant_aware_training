# Research Status — 58 Cycles (001–058)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 11 |
| Refuted | 17 |
| Inconclusive | 21 |
| Crash | 1 |

## Latest Cycle
Cycle 058 — item 75 (stacking error budget) — **INCONCLUSIVE**. Single blocks improve PPL (lstsq regularization), but stacking produces +30% at k=4. Marginals: [-3.3, +10.4, +13.1, +10.0]pp — hard nonlinear jump at k=2. 51pp gap between additive prediction and actual. Stacking is unambiguously super-additive.

## Key Findings (consolidated)

1. **Per-block method works**: Single-block 6× on OWT = sub-baseline PPL (cycles 056-057).
2. **Stacking is the sole bottleneck**: +30% at k=4 vs sub-baseline per-block (cycle 057-058).
3. **Super-additive error compounding**: 51pp gap between sum-of-singles and stacked actual.
4. **Compression-difficulty oracle**: P1 rho=0.84, corpus-invariant rho=0.94 (cycles 048, 055).
5. **Channel selection is corpus-invariant**: Jaccard=0.5 but zero PPL impact (cycle 057).

## Remaining NEXT Items

| Item | Priority | Description | Dependencies | Status |
|------|----------|-------------|-------------|--------|
| 74 | P1 | Gradient-polished lstsq | 72 (done) | Eligible |
| 42 | P2 | Topology-preserving distillation | none | Eligible |
| 43 | P2 | Topological truncation | none | Eligible |
| 45 | P2 | Weight-space diffusion | none | Eligible |

## Note
Only 1 P1 item left (74). After that, only P2 moonshots. Agenda refresh may be needed — new items should focus on stacking mitigation (iterative re-probing, per-block correction, cascaded compression).
