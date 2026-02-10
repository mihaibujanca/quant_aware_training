# Manim: Geometric Weight Quantization Story

This animation track follows `docs/geometric_error_correction_brief.md` with a 2->2->2 ReLU toy model.

## File

- `animations/geometric_weight_quantization.py`

## Scenes

1. `OpeningScene`
   - Introduces quantization and piecewise-affine form.
2. `PartitionToTransportScene` (recommended)
   - Single narrative scene: partition shift in input space -> mapped output geometry -> affine transport correction -> remaining topological residual.
3. `PartitionShiftScene` (legacy split)
   - Shows float vs quantized ReLU boundaries and the changed region.
4. `GridDistortionScene` (legacy split)
   - Compares mapped grids (input, float output, quantized output).
5. `ActivationSpaceComparisonScene`
   - Shows why raw layer-wise L2 on activations can be misleading, then compares against affine-aligned residuals on the same manifold samples.
6. `ErrorDecompositionScene`
   - Decomposes output error into metric vs topological components.
7. `CorrectionCascadeScene`
   - Visualizes layer-wise correction and the first-order inverse formula.
8. `FullNarrativeScene`
   - Convenience scene that stitches all segments.

## Render commands

```bash
manim -pqh animations/geometric_weight_quantization.py OpeningScene
manim -pqh animations/geometric_weight_quantization.py PartitionToTransportScene
manim -pqh animations/geometric_weight_quantization.py PartitionShiftScene
manim -pqh animations/geometric_weight_quantization.py GridDistortionScene
manim -pqh animations/geometric_weight_quantization.py ActivationSpaceComparisonScene
manim -pqh animations/geometric_weight_quantization.py ErrorDecompositionScene
manim -pqh animations/geometric_weight_quantization.py CorrectionCascadeScene
manim -pqh animations/geometric_weight_quantization.py FullNarrativeScene
```

Use `-pql` while iterating quickly and `-pqh` for final quality.

## Main knobs to edit

Inside `animations/geometric_weight_quantization.py`:

- `StoryConfig.bits`: quantization bit-width.
- `StoryConfig.extent`: visible coordinate range.
- `StoryConfig.circle_radius` and `StoryConfig.circle_samples`: manifold sampling.
- `W1`, `b1`, `W2`, `b2`: network geometry.

## Next extension (activation quantization)

Keep this file weight-only for now. For activation quantization, add a second module and inject activation rounding between layer outputs while preserving the same scene structure so comparisons stay direct.
