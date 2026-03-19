# Research Status — 54 Cycles (001–054)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 10 |
| Refuted | 17 |
| Inconclusive | 18 |
| Crash | 1 |

## Latest Cycle
Cycle 054 — item 70 (OpenWebText training-distribution probing) — **REFUTED**. OpenWebText-probed, OpenWebText-evaluated k=4@6× corrected gives +48.21% ±5.03 PPL degradation. Even training-distribution data cannot save the method at this compression ratio. The Shakespeare +8.5% result was a narrow-distribution artifact.

## Critical Update to Research Question
The program asks: "Given a trained transformer LM and a corpus drawn from the data distribution it was trained on, how do we construct a much smaller FFN reparameterization?"

**Answer update**: The no-SGD geometric compression method (topk + lstsq + correction) does NOT work on training-distribution data at k=4@6×. It only appears to work on narrow distributions (Shakespeare) where the lstsq refit acts as regularization. On broad text (OpenWebText +48%, WikiText-2 +44%), the method fundamentally fails. The compression ratio k=4@6× is too aggressive for broad distributions.

## Remaining NEXT Items (eligible)

| Item | Priority | Description | Dependencies | Status |
|------|----------|-------------|-------------|--------|
| 71 | P1 | Jacobian Frobenius std corpus invariance | none | Eligible |
| 42 | P2 | Topology-preserving distillation | none | Eligible |
| 43 | P2 | Topological truncation | none | Eligible |
| 45 | P2 | Weight-space diffusion | none | Eligible |
