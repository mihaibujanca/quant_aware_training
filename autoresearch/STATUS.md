# Research Status — 63 Cycles (001–063)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 13 |
| Refuted | 18 |
| Inconclusive | 23 |
| Crash | 1 |

## Latest Cycle
Cycle 063 — item 79 (stacking-aware correction) — **INCONCLUSIVE**. Cascaded corrections: +25% vs uncorrected +30% (5pp gain). Naive corrections: +51% (interference persists). Audit flagged implementation mismatch in naive variant.

## The Answer (consolidated across 63 cycles)

**Research question**: "Given a trained transformer LM, how do we construct a smaller FFN reparameterization that preserves geometry without SGD?"

**Answer**: Pure no-SGD compression works per-block but fails when stacking multiple blocks. The practical recipe requires a small amount of gradient-based polish:

1. **topk + lstsq** → analytic single-block compression (no training, sub-baseline PPL per block)
2. **Stack k=4 blocks at 6×** → +30% PPL (super-additive error compounding)
3. **500 Adam steps** (lr=1e-4, 4096 held-out tokens) → **+4.9% PPL** on OpenWebText

Key supporting findings:
- Compression-difficulty oracle (P1 Frobenius std): rho=0.84, corpus-invariant rho=0.94
- Channel selection is corpus-invariant: zero PPL impact from cross-corpus channels
- Cascaded re-probing doesn't help (error propagates through attention/LN)
- Boundary corrections: marginal improvement (+5pp) but far less effective than gradient polish

## Remaining NEXT Items

| Item | Priority | Description | Dependencies | Status |
|------|----------|-------------|-------------|--------|
| 42 | P2 | Topology-preserving distillation | none | Eligible |
| 43 | P2 | Topological truncation | none | Eligible |
| 45 | P2 | Weight-space diffusion | none | Eligible |

## Note
All P0/P1 items exhausted. The central question is substantially answered. Remaining items are P2 moonshots on toy models.
