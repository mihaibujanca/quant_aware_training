# Audit — Cycle 061: Gradient polish scaling (item 77)

**Status**: PASS with caveats

## Guardrail flags

- **G1 (Implementation)**: WARN — Polish tokens (`probe_holdout`) are disjoint from eval (`eval_val`, separate document split). Freeze is correct (all params frozen, only `c_proj.weight`/bias unfrozen). However, `baseline_ppl` is computed on seed=42's eval split while per-seed deltas use seed-specific eval splits — delta_pct values are contaminated by eval-set variance (~15pp std, dwarfing the polish effect).
- **G2 (Reproducibility)**: WARN — 3 seeds run, but per-seed 0-step PPLs range from 67.6 to 84.9 due to different eval sets (not model variance). Within-seed improvement is consistent (~14–16 PPL points at 500 steps), confirming the polish effect is real.
- **G3 (Overfitting)**: LOW RISK — 1.6M params polished on 25.6K tokens (~10 epochs at 500 steps). Eval on held-out val_docs mitigates overfitting concern. No plateau at 500 steps (200→500 = 4.7pp gain) suggests diminishing returns not yet saturated.
- **G4 (Counterintuitive)**: NO FLAG — Gradient refinement reducing lstsq approximation error is expected behavior; magnitude (+30% → +5%) is noteworthy but not suspicious.

## Key issue

The eval-set mismatch inflates `delta_pct_std` to ~15pp, making the mean +4.9% statistically indistinguishable from 0% or +13% at 95% CI. The CONFIRMED verdict is directionally correct (within-seed improvements are consistent) but the absolute threshold check (<=+8%) is not statistically rigorous. Fix: compute per-seed baseline PPL on the same eval split used for that seed, or use a single fixed eval split across all seeds.

## Recommendations

1. Recompute deltas using per-seed baseline PPL (teacher evaluated on same seed's eval_val) to eliminate eval-set variance from the metric.
2. The trend is trustworthy: polish consistently reduces PPL by ~25% relative within each seed. Verdict stands directionally.
