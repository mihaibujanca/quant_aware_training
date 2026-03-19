# Research Status — 57 Cycles (001–057)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 11 |
| Refuted | 17 |
| Inconclusive | 20 |
| Crash | 1 |

## Latest Cycle
Cycle 057 — item 73 (failure decomposition) — **INCONCLUSIVE** per criteria (single-block <10% → stacking is the sole problem). Channel selection is corpus-invariant (Jaccard=0.5 but zero PPL impact). The entire +48% failure on OpenWebText is from error compounding across stacked blocks, not per-block method quality.

## Updated Research Picture

**The per-block method works.** Single-block 6× topk+lstsq on OpenWebText gives sub-baseline PPL (lstsq regularization). Channel selection is corpus-invariant. The problem is exclusively **multi-block stacking** — error compounds through the residual stream.

This reframes the research question: the next experiments should target stacking mitigation, not per-block improvements.

## Remaining NEXT Items

| Item | Priority | Description | Dependencies | Status |
|------|----------|-------------|-------------|--------|
| 74 | P1 | Gradient-polished lstsq | 72 (done) | Eligible |
| 75 | P1 | Multi-block stacking error budget | 72 (done) | Eligible |
| 42 | P2 | Topology-preserving distillation | none | Eligible |
| 43 | P2 | Topological truncation | none | Eligible |
| 45 | P2 | Weight-space diffusion | none | Eligible |
