# Cycle 162 Brief

## Item
**172. GELU-aware dead and near-saturated channel census — GPT-2 FFN**
- R2 (exact white-box compression) analysis baseline.
- Analysis type, not training.

## Context
- Cycle 28 refuted with **flawed methodology**: used ReLU thresholds on GELU.
  - Dead channel: `max|post-GELU act| < 1e-5` → wrongly assumes GELU reaches zero.
  - GELU(-3)≈-0.004, GELU(-5)≈-0.00002 — both above threshold.
  - Cycle 28 found 0.00% dead, 0.008% near-linear → structural pruning impossible.
- **Correct approach**: Check **pre-GELU** values.
  - Dead: `max(h_j) < -T` for T∈{2,3,4}.
  - Near-linear: `min(h_j) > T`.
  - Also: output-contribution score `||W_down[:,j]|| × rms(act_j)`.

## Dependencies
- None. Analysis only, no prior cycles required.

## Key Files
- `experiments/exact_prune_linear_collapse_gpt2.py` (cycle 28 artifact for reference).
- Existing GELU census: `cycle_083_gelu_census_v2.*`, `cycle_101_gelu_output_clustering.*`.
- Docs: `docs/gelu_nonlinear_zone_analysis.md` (if present).

## Constraints
- Use **pre-GELU** activations, not post-GELU.
- Test T∈{2,3,4} for saturation thresholds.
- Compute output-contribution score and plot distribution.
- Confirmed if: ≥5% channels dead/saturated AND removable with <0.5% PPL increase (single-block swap).
- Refuted if: <1% channels qualify at any threshold.

## Handoff
Search codebase for GPT-2 FFN GELU analysis code or use exact_prune cycle 28 as template. Instrument pre-GELU activation tracking. Prioritize output-contribution metric over activation magnitude alone.
