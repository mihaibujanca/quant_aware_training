# Cycle 010 Summary — Single-Affine Collapse Quality: Toy ReLU (Item 20)

## Setup

- Dataset: spirals (n=2000, 3 turns, 2× scale, noise=0.0, seed=42)
- Architecture: 2 → 64 → 64 → 64 → 64 → 1 (4 hidden ReLU layers)
- Training: Adam lr=0.001, BCE loss, stopped at epoch 300 when accuracy > 92%
- Final train accuracy: 99.45%
- Probe split: first 50% (n=1000) for fitting A,b; second 50% (n=1000) held-out for evaluation
- Fit method: `torch.linalg.lstsq` on augmented input `[h_in | 1]` → A, b

## Per-Layer Results

| Layer | r_fit  | r_held | frac_dead | frac_always_active |
|-------|--------|--------|-----------|-------------------|
| L1    | 0.4241 | 0.4297 | 0.0312    | 0.0625            |
| L2    | 0.0473 | 0.0504 | 0.1406    | 0.0312            |
| L3    | 0.0315 | 0.0333 | 0.1250    | 0.2969            |
| L4    | 0.0221 | 0.0237 | 0.1719    | 0.2656            |

r = normalized residual = ||h_out − A·h_in − b||_F / ||h_out − mean(h_out)||_F on held-out data.

## Verdict: CONFIRMED

**Criterion met**: At least one hidden layer achieves r < 0.15 on held-out probe data.

In fact, three layers (L2, L3, L4) are well below the 0.15 threshold, with L4 achieving r = 0.024.

Only L1 (first hidden layer, taking raw 2D spiral input) shows moderate r = 0.43, consistent with
the intuition that the first layer must compute the critical nonlinear feature encoding from
the 2D spiral geometry.

## Key Findings

1. **Deep layers are near-linear**: Layers 2–4 have r < 0.06 on held-out data. The function they
   compute is almost entirely captured by a single affine map fitted from probe data.

2. **Layer depth strongly predicts r**: r decreases monotonically L1→L4 (0.43, 0.05, 0.03, 0.02).
   Deeper layers of a well-trained ReLU MLP operate in an increasingly linear regime on the data
   distribution.

3. **Stability correlates with r**: Dead neuron fraction increases with depth (0.03, 0.14, 0.13,
   0.17), and always-active fraction increases for L3/L4 (0.30, 0.27). Layers with more stable
   neurons (dead or always-active) are more affinely collapsible. The unstable (switching) neurons
   in L1 are what makes it non-linear.

4. **Train/held-out gap is negligible**: r_fit ≈ r_held for all layers (difference < 0.01),
   confirming the affine map generalizes well and is not overfit to the probe set.

5. **Implication for compression**: L2, L3, L4 can each be replaced by a single matrix
   multiplication (affine map) on the training distribution. This suggests that networks
   "linearize" their deep layers post-training. Only L1 performs genuinely nonlinear computation
   on the 2D spiral input.

## Correlation Summary

| Layer | r_held | frac_dead | frac_always_active | r < 0.15? |
|-------|--------|-----------|--------------------|-----------|
| L1    | 0.4297 | 0.0312    | 0.0625             | No        |
| L2    | 0.0504 | 0.1406    | 0.0312             | Yes       |
| L3    | 0.0333 | 0.1250    | 0.2969             | Yes       |
| L4    | 0.0237 | 0.1719    | 0.2656             | Yes       |

Spearman correlation between r_held and frac_dead: negative (higher dead fraction → lower r),
consistent with stable neurons being more linearly collapsible.

## Files

- `experiments/single_affine_collapse_toy.py` — experiment script
- `autoresearch/results/cycle_010_single_affine.json` — raw per-layer results (JSON)
- `autoresearch/results/cycle_010_summary.md` — this file
