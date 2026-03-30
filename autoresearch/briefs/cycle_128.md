# Cycle 128 Brief — Item 134

## Item
**Dead/always-active neuron census + exact pruning — GPT-2 FFN (GELU-correct)**

## Context
R2 baseline: exact white-box compression floor before any approximation. Prior attempt (item 47, cycle 032) was flawed:
- Threshold calibrated for ReLU, not GELU (post-GELU |activation| < 1e-5 is not meaningful)
- Contribution-based metric used 0.05% threshold, exceeding equal-share baseline (1/3072 = 0.033%)
- Found ~0% dead channels, not credible

## Task
For each of 12 FFN blocks, scan 256 OWT sequences (seq_len=256):
1. **Dead neurons**: |GELU(W_up @ x + b)_i| < 1e-6 for ALL tokens → can be removed (zero error)
2. **Always-active neurons**: GELU(z_i) ≈ z_i (saturated) → fold into bias correction
3. **Near-linear neurons**: max|GELU(z_i) - z_i| < 0.01 for all tokens

Report per-block counts. Construct exact-pruned model (remove dead, fold always-active). Measure single-block and k=4 stacked PPL.

## Dependencies
None (P1)

## Key Files
- Compression recipe: `docs/gelu_nonlinear_zone_analysis.md` (block-level GELU saturation analysis)
- Reference (flawed): `autoresearch/agenda.md` line 656 (item 47, cycle 032)
- Data: OWT 256×256 sequences (load via existing pipeline)
- Baselines: topk+lstsq+500 Adam (PPL ~80 stacked), float=51.66

## Constraints
- **Pre-GELU thresholds only**: detect z_i stuck in tail (z < -3 or z > 4), not post-GELU magnitude
- **Contribution metric**: ||W_down[:,i]|| × RMS(activation_i) must be well below 1/3072 to be meaningful
- **PPL criterion**: <0.1% change at single-block level confirms exact pruning is truly zero-cost
- **Minimum bar**: ≥5% removable neurons/block on average, else REFUTED

## Handoff
Ready for implementation. No blocking dependencies. Use GELU (not ReLU) activation thresholds. Verify thresholds don't exceed equal-share baseline before reporting results.
