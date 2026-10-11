# Watch the shape, then name the error

A weight matrix moves an entire patch of data. A square becomes a parallelogram. A circle becomes an ellipse. Every point keeps its identity while the shape stretches, tilts, or flattens. Quantization changes that motion by snapping the weights to a small set of values.

The useful question is: **what would we have to do to the quantized shape to recover the original one?**

This guide follows the visual language in `animations/geometric_weight_quantization.py`: keep the same points on screen, move the whole shape, and show correction as another motion. The grid makes shear and collapse visible; the circle makes changes in shape easy to follow. A and B are the same two named samples in every panel. Axes stay fixed across a sequence, so a changing camera cannot disguise a changing map.

## A different shape is not necessarily lost information

![A grid and circle move through full precision, quantization and correction](figures/quant_geometry/geometry_deformation.png)

Read the panels in order. The original map tilts the grid and turns the circle into an ellipse. The very coarse quantizer happens to snap this example to the identity: its output is the original square and circle again. That is a substantial difference from the FP output, but nothing has collapsed. A and B remain distinct, and the quantized representation still tells us where each input was.

Now apply a second linear map to that quantized output. The entire grid and circle move onto the FP reference, not just the two highlighted samples. The change was reversible.

That correcting motion is **R**. In this example, **U is zero**: we need no additional information from the input. Large pointwise displacement was a change of map, not information destruction.

**Uncompensated error** is the displacement you leave in place by stopping before that correction. It measures an available repair that has not been applied. Once ReLU is involved, applying R to its output may be insufficient; that is the next picture.

## ReLU can remove the coordinate the correction needed

ReLU does something visibly different from a linear map. It clips negative coordinates to zero. An edge lying to the left of the vertical axis gets pressed onto that axis; points in the negative quadrant can pile up at the origin. It does not stretch the shape into another invertible shape.

![Correct before clipping versus clip before correcting](figures/quant_geometry/geometry_relu.png)

The upper-right panel takes the quantized preactivation, corrects its map, **then** applies ReLU. It recovers the FP activated shape.

The lower-left panel changes the order: **ReLU first**. Watch A and B. They started at different points on the two negative axes, but ReLU sends both to the origin. In the FP route, A would have produced a positive second coordinate; B would not. The quantized activated representation no longer tells us which of those inputs we had.

The lower-right panel applies the same R after this clipping. A linear map sends the origin to the origin, so it cannot separate A from B. The gap to the dashed FP reference remains.

This is **nonlinearity-induced error**: the two orders of operation give different shapes. It is measured by comparing “correct, then activate” with “activate, then correct.” In this particular example, the A/B collision is also direct evidence that no decoder using only this activated representation could recover both FP outputs. In other examples, a nonzero nonlinear component only shows that this chosen correction and activation do not commute; it does not establish universal irrecoverability.

Notice where the loss happened: the quantized linear map was invertible. ReLU discarded the distinguishing information afterwards. That is different from a quantized matrix already losing an input direction.

## A collapsed direction cannot be restored by moving the line

![A two-dimensional patch collapses onto a line; correcting the line does not restore its width](figures/quant_geometry/geometry_loss.png)

Here the FP map gives the patch area. Quantization wipes out the weaker input row, and the whole patch collapses onto a line. A and B have the same first input coordinate and different second coordinates. After quantization they occupy exactly the same output point.

R tilts the line toward the FP patch. It can repair the part of the map still represented in the quantized output. But the image remains a line: moving or stretching a line cannot recover the missing width. A and B remain indistinguishable.

The last panel restores that width by adding a displacement that depends on the original input. A and B receive different displacements and separate again. This additional map is **U**, and the missing contribution for an input x is **xU**. It is a research oracle contribution; Q alone did not retain what is needed to compute it.

**Range error** is the remaining output difference between the R-recovered route and the FP route once both have been activated. The storyboard deliberately shows the collapse before activation, where the cause is easiest to see. ReLU can hide or expose its effect afterwards.

This example has a genuine zero singular direction. Merely choosing to ignore a tiny nonzero singular value is different: its information is weakly represented and might be recoverable with a large gain. The [technical reference](quant_geometry_foundations.md#mathematical-audit-numerical-rank-and-attribution-conventions) keeps that distinction explicit.

## What the three error components compare

We now have three visual questions, rather than three unexplained names:

- **Uncompensated:** how far does applying the available correction move the activated Q state?
- **Nonlinear:** how far apart are the two outcomes when correction and activation swap order?
- **Range:** after restoring the recoverable map, what difference remains from the missing input contribution?

These are signed differences along a declared sequence of counterfactual states. They can oppose one another. They are not three independent piles of damage, nor does each have a natural additive percentage of total error. The [reference's signed example](quant_geometry_foundations.md#exactness-and-interaction) shows a cancellation explicitly.

For zero bias, the names translate into this path:

`Q after ReLU → apply R → instead apply R before ReLU → restore xU before ReLU → FP output`

The three ledger vectors point backwards along successive edges of that path. Their sum is Q minus FP. That is why the ledger telescopes. **The geometry supplies the meaning; the identity checks that the accounting closes.** With a shared float bias, use the bias-preserving affine correction described in the reference instead of R alone.

## What to watch in a network

Follow the same samples through the next layer. Does their separation grow? Does another ReLU collapse them? Does a downstream projection make their discrepancy disappear? Those are questions about the *fate* of an existing error, separate from the layer that created it.

An immediate oracle puts the samples back on the FP trajectory before the next layer consumes them. This gives us a clean experiment: measure the current layer's deformation, then prevent that deformation from contaminating every later measurement. Delaying that reset lets us watch propagation. A useful next animation should preserve point identities while switching between those two trajectories.

## Reproduce and extend the visual story

```sh
MPLCONFIGDIR=/tmp/qgeom-mpl .venv/bin/python -m experiments.quant_geometry_visual_story
```

The reusable geometry lives in `aleph/qgeom/geometry_story.py`. It uses the same weight quantizer, decomposition, and ledger as the tests. `experiments/quant_geometry_visual_story.py` produces these three storyboards and the numeric states for the moving version; `animations/quant_geometry_story.html` is its animation template. This adds the new ledger story alongside the existing Manim work, without replacing its scenes.

All three examples use a bounded symmetric 2-bit quantizer and zero bias. Exact matrices, R/U and point coordinates are recorded in [the animation state data](figures/quant_geometry/geometry_story_states.json). Playback interpolates corresponding points between these exact endpoints to make the transformations visible; intermediate frames are not additional quantization bit widths or claimed inference states. These examples deliberately separate the mechanisms so their geometry is visible; they are not a claim about the typical frequency of each mechanism in a trained model. The [technical reference](quant_geometry_foundations.md) contains the bias extension, numerical-rank caveats, API and acceptance tests.

The new contributions are the three controlled geometric examples, the grid/circle storyboards, the moving point-correspondence view, and this picture-first narrative. Tests verify the claims behind the pictures: reversible deformation with U=0; distinct FP outputs after a Q-ReLU collision; and restoration of lost width only when the input-dependent residual is added. No quantization or oracle algorithm was changed in this explanatory pass.
