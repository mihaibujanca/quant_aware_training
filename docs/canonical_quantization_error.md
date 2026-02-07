# Canonical Quantization Error Analysis

## What this is

When we quantize a neural network's weights, each layer introduces a small
error. These errors compound through the network. This document reports
experiments measuring that compounding, decomposing it into sources, and
testing whether an oracle correction can eliminate it.

The primary configuration uses MLPs trained on the spirals dataset (2000
samples, 3 turns, 2× scaled) with width-32 hidden layers, depth 12, 4-bit
delta quantization, Adam optimizer (lr=0.001, 5000 epochs). We also test
embedding the same 2D manifold into 100D to verify findings generalize.

The analysis code lives in `aleph/qgeom/canonical.py` and the notebook in
`notebooks/canonical_error_correction.py`.

## Quantization method

We use **delta (grid) quantization**: snap each weight to the nearest
multiple of a fixed step size δ.

$$W_q = \text{round}(W / \delta) \cdot \delta$$

With 4 bits, δ = 1/(2³) = 0.125. The maximum per-weight error is δ/2 =
0.0625. The quantization error matrix is E_L = W_q - W.

We chose delta quantization over bit-width quantization (which scales
per-tensor or per-channel) because it's more geometric: the error is a
uniform grid displacement, making it easier to reason about spatially. The
framework itself is quantization-agnostic — all classes take W and W_q as
inputs and never call `quantize()`. The same analysis applies to any
quantization method; we just need to swap how W_q is produced.

Biases are kept at full precision. They cancel in the error formula:
ẑ - z = (W_q·â + b) - (W·a + b) = E·â + W·ε.

## The framework

### Error decomposition

At each layer, the pre-activation error is:

$$\hat{z}_L - z_L = \underbrace{E_L \cdot \hat{a}_{L-1}}_{\text{local}} + \underbrace{W_L \cdot \varepsilon_{L-1}}_{\text{propagated}}$$

- **Local**: this layer's quantization error E_L applied to the (quantized)
  input activations. Present even if all previous layers were perfect.
- **Propagated**: the float weight matrix W_L amplifying accumulated error
  ε_{L-1} from all previous layers.
- Computed exactly as `propagated = total - local` (no approximations).

### Perfect (oracle) correction

At each layer, apply: C_L = -E_L · â_{L-1} - W_L · ε_{L-1}

This exactly undoes both local and propagated error, recovering the float
pre-activation: ẑ_L + C_L = z_L. It's an oracle because it needs the float
activations to compute ε_{L-1}.

### Canonical space

Errors at different layers have different dimensions (2D input, 32D hidden,
1D output). To compare them, we map everything back to input space via the
pseudoinverse of the cumulative transform T_L = W_L ⋯ W_1:

$$\text{canonical error} = T_L^+ \cdot \text{output error}$$

This is a linear approximation — it ignores ReLU nonlinearities, so it's
exact for the linear component and approximate for the rest. Condition
numbers grow with depth (reaching 126 at depth 12 for the 2D model), making
canonical comparisons approximate but still informative for moderate depths.

### ReLU disagreement

Where sign(z_float) ≠ sign(z_quant), the quantized and float networks make
different on/off decisions. This is the only source of nonlinear error — and
the one thing perfect correction can't fix in general (though it does fix it
when we correct at every layer, since then pre-activations match exactly).

## Primary results: spirals, depth 12, width 32

**Config**: 2→32×12→1, spirals (n=2000, 3 turns, 2× scale), 4-bit, lr=0.001

**Float accuracy**: 92.8%

### Error attribution

| Layer | Shape   | Local   | Propagated | Total    | % Propagated |
|-------|---------|---------|------------|----------|--------------|
| L0    | (32,2)  | 0.1994  | 0.0000     | 0.1994   | 0%           |
| L1    | (32,32) | 0.5616  | 0.1703     | 0.6212   | 23%          |
| L2    | (32,32) | 0.4028  | 0.3981     | 0.5935   | 50%          |
| L3    | (32,32) | 0.3519  | 0.6077     | 0.7556   | 63%          |
| L4    | (32,32) | 0.4635  | 1.2373     | 1.3795   | 73%          |
| L5    | (32,32) | 0.5050  | 2.6709     | 2.8062   | 84%          |
| L6    | (32,32) | 0.6795  | 5.0720     | 5.0859   | 88%          |
| L7    | (32,32) | 1.0150  | 6.6069     | 6.7137   | 87%          |
| L8    | (32,32) | 1.6730  | 13.9324    | 14.1236  | 89%          |
| L9    | (32,32) | 3.5488  | 28.7663    | 29.5667  | 89%          |
| L10   | (32,32) | 8.3475  | 61.2720    | 63.0692  | 88%          |
| L11   | (32,32) | 14.6569 | 124.1130   | 126.3674 | 89%          |
| L12   | (1,32)  | 5.9420  | 87.8461    | 93.4747  | 94%          |

Error grows from 0.20 at L0 to 93.47 at the output (~470× amplification
across 13 layers). Propagated error dominates by L3 (63%) and reaches 94%
at the output. This is dramatically more error than the old moons/width-8
configuration (~8.5 at depth 8), because the wider network has more weights
to quantize and the spirals dataset requires more complex decision boundaries.

### Perfect correction

Correction residual ~10⁻⁶ at every layer (float32 precision), confirming
the math holds for deeper, wider networks.

### Partial correction

| Strategy | Output error |
|----------|-------------|
| All layers | 0.0000 |
| Layer 0 only | 71.29 |
| Layer 6 only (middle) | 4.90 |
| Output layer only (L12) | 0.0000 |
| No correction | 93.47 |

Output-layer-only correction still works perfectly — the (1, 32) bottleneck
discards 31 of 32 error dimensions. Single interior layer correction at L6
reduces error from 93.47 to 4.90 (95% reduction) — much more effective than
at width-8, because the wider layers give correction more degrees of freedom.

### ReLU disagreement

Ranges from 2% (L0) to 11.5% (L7). More uniform across layers than at
width-8, where disagreement spiked dramatically at specific layers. The
wider representation means individual quantization errors are smaller
relative to the activation magnitudes.

### Geometric metrics

| Layer | ||E||₂ | ||W||₂ | cond(T_L) |
|-------|--------|--------|-----------|
| L0    | 0.2154 | 2.7246 | 1.2       |
| L4    | 0.3779 | 6.5994 | 26.0      |
| L8    | 0.3805 | 3.7165 | 129.9     |
| L11   | 0.3949 | 5.7325 | 125.6     |
| L12   | 0.1914 | 2.2094 | 1.0       |

Condition numbers stay moderate (max 130 at the hidden layers) — much better
than width-8 depth-8 (cond 5089). The wider hidden layers provide more
numerical stability for the canonical transform.

## 2D vs 100D comparison

We embed the same 2D spirals manifold into 100D via a random affine
projection (X_high = X_2d @ W_embed + b_embed) and train a
100→32×12→1 network with the same hyperparameters.

### Side-by-side comparison

| Metric | 2D | 100D |
|--------|-----|------|
| Float accuracy | 92.8% | 93.3% |
| Output error (uncorrected) | 93.47 | 8.82 |
| % propagated at output | 94% | 99% |
| Perfect correction residual | ~10⁻⁶ | ~10⁻⁷ |
| Output-layer-only correction | 0.0000 | 0.0000 |
| Max ReLU disagreement | 11.5% | 18.0% |
| cond(T₁₁) | 126 | 564,000,000 |

### What changes

1. **Lower absolute error**: the 100D model has 8.82 total output error vs
   93.47 for 2D. The 100D first layer (32, 100) distributes quantization
   error across more dimensions, and the subsequent layers see a different
   weight distribution. The error compounding pattern differs quantitatively.

2. **Higher propagated fraction**: 99% propagated at output (vs 94% for 2D).
   The 100→32 first layer's larger quantization error propagates more
   aggressively relative to subsequent local errors.

3. **Condition numbers explode**: cond(T₁₁) = 564M in 100D vs 126 in 2D.
   The 100→32 projection introduces severe ill-conditioning from the start.
   Canonical space is numerically meaningless for 100D; the correction
   formula is unaffected.

4. **ReLU disagreement is slightly higher**: max 18% vs 11.5%. The different
   weight landscape in 100D produces more neurons near zero.

### What stays the same

- Error compounds through layers in both settings.
- Perfect correction works to float32 precision in both.
- Output-layer-only correction achieves zero error in both (bottleneck
  absorption).
- Error is spatially structured in both (concentrates away from training
  data on the 2D manifold).

**Bottom line**: the 2D analysis is a valid model for understanding
quantization error dynamics. The high-dimensional ambient space changes
absolute magnitudes and conditioning but not the qualitative behavior.

## How to read the plots

### Decision boundary

Three panels: float model, quantized model, oracle-corrected model. The
color shows P(class 1) from blue (0) to red (1), with the decision boundary
(P=0.5) as a black line. Training data is overlaid. Quantization visibly
distorts the boundary; correction restores it exactly.

### Attribution bar chart

Stacked bars showing mean pre-ReLU error (output space) at each layer.
**Blue** = local error (this layer's quantization). **Orange** = propagated
error (upstream quantization errors amplified forward). The percentage label
shows the propagated fraction. A growing orange fraction means earlier
quantization mistakes are increasingly dominating.

### Error heatmap

Each panel shows one layer. The x/y axes are input space coordinates; the
color is **output-space** error magnitude ||â_L - a_L|| at that layer.
White dots are training data. Error concentrates away from the training data
because the network extrapolates large activations outside the data
manifold, and larger activations → larger quantization error. The color
scale grows across panels (check the colorbars) — this is the compounding.

### ReLU disagreement map

Red regions = at least one neuron at that layer made a different on/off
decision in the quantized vs float network. These are inputs where
sign(z_float) ≠ sign(z_quant) for some neuron. They cluster near the
network's decision boundaries where neurons are close to zero.

## Key findings

### 1. Error compounds, driven by weight spectral norms

Propagated error grows from 0% at L0 to 89–99% at the output. Total error
grows roughly exponentially with depth. The rate depends on ||W_L||₂:
layers with large spectral norms cause sharp jumps, while layers with
||W||₂ ≈ 1.5 grow slowly. At width-32 depth-12, total output error reaches
93.5 (~470× amplification) — enough to visibly distort the decision boundary.

### 2. Perfect correction works exactly

The correction formula C_L = -E_L·â - W_L·ε recovers float pre-activations
to float32 machine precision (~10⁻⁶ to 10⁻⁷) at every layer, for every
configuration tested. This confirms the math: quantization error is fully
characterized by the local + propagated decomposition. The corrected
decision boundary is visually indistinguishable from the float boundary.

### 3. Bottleneck layers absorb upstream error

Correcting only the output layer achieves near-zero output error in every
configuration tested — width-8 through width-32, depth 2 through 12, both
2D and 100D inputs. The rank-1 projection discards all but one dimension of
accumulated error.

At width-32, single-interior-layer correction is much more effective than at
width-8: correcting L6 alone reduces error from 93.47 to 4.90 (95%
reduction). The wider layers give correction more degrees of freedom to
absorb error.

### 4. Error is spatially structured, not random noise

The error heatmaps show that quantization error concentrates in specific
spatial regions (large activations, typically away from training data). The
canonical-space PCA shows error variance concentrating along fewer directions
with depth (100% in one component by the later layers). This structure is
good news for learned corrections: a correction network doesn't need to
handle arbitrary noise, just a structured, low-rank perturbation.

### 5. ReLU disagreement grows with depth but stays manageable

At width-32 depth-12, disagreement ranges from 2–12% per layer — more
uniform than width-8 networks where it spiked to 30%+ at specific layers.
Wider networks distribute quantization error across more neurons, keeping
individual perturbations small relative to activations. These disagreement
regions are the hard cases for any non-oracle correction.

### 6. Findings generalize to high-dimensional inputs

The 2D analysis provides a valid model for understanding quantization error
dynamics. Embedding the same 2D manifold in 100D changes absolute
magnitudes and conditioning but not the qualitative behavior: error
compounds, correction works, bottleneck absorption holds.

## What's next

The oracle correction proves the theoretical ceiling — 100% of quantization
error is correctable (where ReLU agrees), in both 2D and 100D settings. The
open question is efficiency: can we replace the oracle (which needs float
activations) with a learned correction that achieves most of the benefit
using only quantized activations? The canonical space framework identifies
where to focus that correction (bottleneck layers, high-error spatial
regions).
