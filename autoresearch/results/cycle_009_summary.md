# Cycle 009 — Jacobian Variance Atlas

## Item
19 — Jacobian variance atlas: which FFN blocks are near-linear on probe data?

## Setup
- Teacher: `gpt2` (proxy), Shakespeare corpus, 256 probe tokens per block
- Metric: linearity index L = 1 − mean_x(||J(x) − J̄||_F / ||J̄||_F) per block
- Correlation target: Spearman between L and (1 − relative ppl increase from cycle 005)

## Results

| Block | L | mean_ratio | J̄ norm | damage |
|-------|------|------------|---------|--------|
| 0 | 0.070 | 0.930 | 102.0 | 0.331 |
| 1 | 0.382 | 0.618 | 156.5 | 0.007 |
| 2 | 0.085 | 0.915 | 118.7 | 0.011 |
| 3 | −0.352 | 1.352 | 81.0 | 0.010 |
| 4 | −0.306 | 1.306 | 89.5 | 0.006 |
| 5 | −0.369 | 1.369 | 89.6 | 0.001 |
| 6 | −0.377 | 1.377 | 100.7 | 0.010 |
| 7 | −0.521 | 1.521 | 106.1 | −0.006 |
| 8 | −0.565 | 1.565 | 118.4 | 0.020 |
| 9 | −0.615 | 1.615 | 133.2 | 0.016 |
| 10 | −0.310 | 1.310 | 194.0 | 0.018 |
| 11 | −0.012 | 1.012 | 302.1 | 0.065 |

- **Blocks with L ≥ 0.85**: 0 of 12
- **Blocks with L < 0.6**: 12 of 12 (all)
- **Spearman ρ(L, 1−damage)**: −0.14, p = 0.66

## Outcome: REFUTED

All blocks have L < 0.6. Most blocks have negative L (mean Jacobian deviation exceeds mean Jacobian norm), indicating substantial per-token Jacobian variation. No GPT-2 FFN block is near-linear on the probe distribution.

The Spearman correlation is weak and non-significant, so linearity index does not predict compression difficulty.

## Interpretation
- Single-affine collapse is not viable for any GPT-2 FFN block on this corpus.
- The high mean_ratio (>1.0 for blocks 3–11) means the average deviation from J̄ is larger than J̄ itself — each token activates a genuinely different local map.
- Block 1 is the most linear (L=0.38) but still far from the 0.85 threshold.
- This rules out the "some blocks are basically linear" hypothesis and points toward multi-region approaches (items 22–25) as the necessary path.

## Artifacts
- `experiments/jacobian_variance_atlas.py`
- `autoresearch/results/cycle_009_jacobian_atlas.json`
- `autoresearch/results/cycle_009_jacobian_atlas.png`
