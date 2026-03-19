# Cycle 061: Gradient Polish Scaling

**Item 77** — Gradient polish scaling: 100–500 Adam steps on stacked blocks — GPT-2 FFN

## Context

Cycle 059 (CONFIRMED): Gradient polish on independent blocks [7,5,4,1]@6× with 50 Adam steps reduced stacking degradation from +30.2% to +18.4% (39% relative improvement).

Trajectory: 0 steps → +30.2% | 10 → +26.0% | 25 → +22.3% | 50 → +18.4% (baseline 57.65 PPL).

## Hypothesis

Is the improvement curve still descending? Run polish at {100, 200, 500} steps to identify convergence plateau. Secondary hypothesis: joint polish (all 4 blocks simultaneously) outperforms independent polish by ≥5pp.

## Experiment

- Stacked k=4 blocks [7,5,4,1] at 6× compression on OpenWebText
- Gradient polish configs: {100, 200, 500} Adam steps, lr=1e-4
- Independent polish (primary) + joint polish (secondary)
- Track per-step loss curve; measure convergence
- Same holdout setup as cycle 059, 3 seeds

## Success Criteria

- **Confirmed**: 500 steps achieves ≤+8% PPL degradation OR joint polish beats independent by ≥5pp
- **Refuted**: 200 steps shows <2pp gain vs 50 steps (plateau reached)
- **Inconclusive**: Continues improving but >+12% at 500 steps + no loss plateau

## Key Metrics

Per-seed stacking degradation (PPL % above baseline), relative improvement vs 50-step baseline.
