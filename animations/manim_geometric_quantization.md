# Manim: Geometric Weight Quantization Story

This track is now split into focused scene modules under:

- `/Users/mihai/Projects/quant_aware_training/animations/quantization_story/`

Entrypoint (for rendering):

- `/Users/mihai/Projects/quant_aware_training/animations/geometric_weight_quantization.py`

Legacy monolith backup:

- `/Users/mihai/Projects/quant_aware_training/animations/legacy_geometric_weight_quantization.py`

## Scene structure (current)

1. `LayerProgressionScene`
   - What a simple ReLU network does step by step: `W,b` then `ReLU` per layer.
2. `QuantizationComparisonScene`
   - Add quantization and show float vs quantized trajectories diverge.
3. `LinearCorrectionScene`
   - Fit a full-precision affine correction on same-region points.
4. `TopologyFailureScene`
   - Show where linear correction fails because topology changed.
5. `FullNarrativeScene`
   - Convenience stitch of the 4 scenes.

## Render commands

```bash
manim -pqh animations/geometric_weight_quantization.py LayerProgressionScene
manim -pqh animations/geometric_weight_quantization.py QuantizationComparisonScene
manim -pqh animations/geometric_weight_quantization.py LinearCorrectionScene
manim -pqh animations/geometric_weight_quantization.py TopologyFailureScene
manim -pqh animations/geometric_weight_quantization.py OneDErrorGrowthScene
manim -pqh animations/geometric_weight_quantization.py FullNarrativeScene
```

Use `-pql` while iterating.

## Files to edit

- Shared math/helpers/constants: `/Users/mihai/Projects/quant_aware_training/animations/quantization_story/common.py`
- Scene 1: `/Users/mihai/Projects/quant_aware_training/animations/quantization_story/scene_01_layer_progression.py`
- Scene 2: `/Users/mihai/Projects/quant_aware_training/animations/quantization_story/scene_02_quantization.py`
- Scene 3: `/Users/mihai/Projects/quant_aware_training/animations/quantization_story/scene_03_linear_correction.py`
- Scene 4: `/Users/mihai/Projects/quant_aware_training/animations/quantization_story/scene_04_topology_failure.py`
