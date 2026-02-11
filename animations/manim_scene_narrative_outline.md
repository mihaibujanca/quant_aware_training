# Narrative Outline (4-Step Structure)

This outline matches the split scene files in:
`/Users/mihai/Projects/quant_aware_training/animations/quantization_story/`.

## 1) `LayerProgressionScene`

- Start with a manifold.
- Apply linear map `W,b`: manifold stretches/rotates.
- Apply ReLU explicitly: clip negative pre-activations (red guides at axes).
- Repeat per layer.
- Core message: network = alternating affine transforms and nonlinear clipping.

## 2) `QuantizationComparisonScene`

- Run the same manifold through float and quantized weights.
- Show the two trajectories after each operation.
- Divergence appears and compounds across layers.
- Core message: quantization perturbs each affine step, so the path drifts.

## 3) `LinearCorrectionScene`

- In output space, fit a full-precision affine map from quantized path to float path
  on points that stayed in the same ReLU regions.
- Apply that map to quantized outputs.
- Show large reduction in error (especially on same-region points).
- Core message: a lot of quantization error looks like the wrong affine chart.

## 4) `TopologyFailureScene`

- Reuse the same affine correction.
- Color points by region agreement:
  - green = same ReLU regions
  - red = changed regions (topology changed)
- Draw residual arrows for red points.
- Core message: once points cross region boundaries, one linear map cannot recover them.

## Stitching

- `FullNarrativeScene` runs all 4 scenes in order.
