# Cycle 059 Brief — Gradient-Polished lstsq

**Item**: 74 — Gradient-polished lstsq: 10-50 Adam steps after analytic refit — GPT-2 FFN

## Objective
Test whether gradient-based fine-tuning on top of analytic lstsq solution can narrow the broad-text gap. After topk+lstsq compression on OpenWebText probes, apply {10, 25, 50} Adam steps (lr=1e-4) minimizing cross-entropy loss on 512 OpenWebText tokens, updating ONLY kept FFN channels' W_down. Compare to pure lstsq baseline (cycle 057 variant A).

## Precedent
- **Cycle 057** (item 73): Single-block lstsq achieves ~-4% PPL (lstsq implicit regularization). Channel selection and lstsq quality ruled out as bottlenecks.
- **Cycle 058** (item 75): Stacking is super-additive: +30% at k=4 vs additive prediction of -21%. MSE compounds through residual stream.
- **Prior**: Item 72 (implicit regularization mechanism) — provides theoretical foundation.

## Method
Blocks [7, 5, 4] at 6× compression ratio. Baseline: topk+lstsq only. Polish variants: 10/25/50 Adam steps on held-out 512 OpenWebText tokens with pre-tuned learning rate. Single-block evaluation (no stacking yet).

## Success Criteria
- **Confirmed**: ≥30% relative improvement (e.g., +15pp → ≤+10.5pp)
- **Refuted**: <5% relative improvement or PPL increase
- **Inconclusive**: 5–30% relative improvement

## Risk Flag
Polish may overfit to 512-token probe. Validate on full OpenWebText validation set to confirm generalization.
