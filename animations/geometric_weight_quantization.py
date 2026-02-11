"""Entry point for the geometric weight-quantization Manim story.

This file is intentionally short. Scene implementations live in
`animations/quantization_story/`:

1. LayerProgressionScene
2. QuantizationComparisonScene
3. LinearCorrectionScene
4. TopologyFailureScene
5. OneDErrorGrowthScene

Legacy monolithic version is preserved at:
`animations/legacy_geometric_weight_quantization.py`
"""

from __future__ import annotations

from manim import Scene

from animations.quantization_story import (
    LayerProgressionScene as _LayerProgressionScene,
    LinearCorrectionScene as _LinearCorrectionScene,
    OneDErrorGrowthScene as _OneDErrorGrowthScene,
    QuantizationComparisonScene as _QuantizationComparisonScene,
    TopologyFailureScene as _TopologyFailureScene,
)


class LayerProgressionScene(_LayerProgressionScene):
    pass


class QuantizationComparisonScene(_QuantizationComparisonScene):
    pass


class LinearCorrectionScene(_LinearCorrectionScene):
    pass


class TopologyFailureScene(_TopologyFailureScene):
    pass


class OneDErrorGrowthScene(_OneDErrorGrowthScene):
    pass


class FullNarrativeScene(Scene):
    """Convenience stitch of the 4-step narrative."""

    def construct(self) -> None:
        for segment in (
            LayerProgressionScene.construct,
            QuantizationComparisonScene.construct,
            LinearCorrectionScene.construct,
            TopologyFailureScene.construct,
        ):
            segment(self)
            self.wait(0.3)
            self.clear()


__all__ = [
    "LayerProgressionScene",
    "QuantizationComparisonScene",
    "LinearCorrectionScene",
    "TopologyFailureScene",
    "OneDErrorGrowthScene",
    "FullNarrativeScene",
]
