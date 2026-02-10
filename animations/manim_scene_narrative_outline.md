# Manim Scene Narrative Outline (Weight Quantization)

This is a rough narrative scaffold for the scenes in
`/Users/mihai/Projects/quant_aware_training/animations/geometric_weight_quantization.py`.

It is intentionally unsynced and high-level: use it as talking points while iterating visuals.

## `OpeningScene`

- Start from the core idea: a ReLU network is piecewise-affine.
- For each activation pattern, the network acts like one affine map.
- Weight quantization replaces each weight matrix with a grid-snapped version.
- So the geometric question is: how does that perturb piecewise-affine geometry?
- Frame the rest of the video as a geometry story, not just a numeric error story.

## `PartitionToTransportScene` (recommended main transition)

- First, show input-space partition boundaries for float vs quantized weights.
- Point out that these boundary shifts are real, but they are only part of the story.
- Transition: correction does not happen by comparing points directly in raw input space.
- After network mapping, float and quantized outputs live on related but distorted coordinate systems.
- In stable regions, this distortion is approximately an affine transport.
- So correction is an inverse transport operation, not naive subtraction.
- Show that best affine transport dramatically reduces metric distortion.
- Then emphasize the residual red/topological error: points that crossed regions cannot be fixed by one linear map.
- Final punchline of this scene: metric error is linearly invertible; topological error is routing error.

## `PartitionShiftScene` (legacy standalone)

- Isolate the ReLU hyperplane shift under quantization.
- Show changed-region bands where activation patterns disagree.
- Explain these are topological perturbations of the partition.
- Stress that this view alone can over-emphasize nonlinear boundaries unless paired with output-space transport.

## `GridDistortionScene` (legacy standalone)

- Push a regular grid through float and quantized networks.
- In output space, compare smooth deformation differences (metric) vs tearing near kinks (topological).
- Green arrows correspond to mostly within-region affine mismatch.
- Red arrows correspond to boundary-crossing/topological mismatch.
- Reinforce that most correction power comes from undoing the smooth affine mismatch.

## `ActivationSpaceComparisonScene`

- Sample a small set of points from one manifold and track their layer-1 activations.
- First comparison: naive pointwise `L2` between quantized and float activations.
- Explain the implicit assumption in raw `L2`: identity correspondence between coordinate charts.
- If quantization applies a near-affine transport (rotation/shear/scale), raw `L2` includes chart mismatch.
- Second comparison: fit best affine alignment from quantized activations to float activations.
- Show metric points collapse much closer after alignment.
- Keep topological points highlighted as residual outliers that affine transport still cannot match.
- Punchline: raw `L2` is still meaningful, but its meaning is “error under identity chart assumption,” not “intrinsic geometric mismatch.”

## `ErrorDecompositionScene`

- Sample a simple manifold (circle) and map it through both models.
- Visualize pointwise error vectors between float and quantized outputs.
- Decompose total error into metric component and topological component.
- Narrate that metric errors tend to align with a smooth low-dimensional structure.
- Narrate that topological errors appear as sparse outliers relative to that structure.
- Connect this to low-rank correction intuition.

## `CorrectionCascadeScene`

- Introduce layer-wise correction as inverse distortion.
- Use the first-order decomposition:
  - local term from current layer quantization
  - propagated term from earlier layers transformed forward
- Explain why this is compositional across depth, not simple additive noise.
- In the toy setup, local pre-activation correction already matches float closely.
- In deeper/variable-width settings, propagated components become more important.

## `FullNarrativeScene`

- Use as a stitched demo pass, not the final edited video.
- Treat it as a rehearsal timeline for voiceover and pacing decisions.

## Optional bridge to next phase (activation quantization)

- Keep this current story weight-only.
- Next extension: introduce activation rounding as additional state-space discretization.
- Reuse the same decomposition language (metric transport vs topological/routing effects).
