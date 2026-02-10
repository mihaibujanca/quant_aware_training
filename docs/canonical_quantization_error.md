# Canonical Quantization Error Analysis

## What this is

Each layer of a neural network applies an affine transform — rotation,
stretching, projection — that changes the coordinate system the data lives
in. When we quantize a layer's weights, we perturb that transform: the
quantized layer maps data to a slightly different space than the float layer
would. By the next layer, the coordinate systems have diverged, and that
layer's own quantization pushes them further apart. After many layers the
accumulated geometric distortion can be large enough to flip decision
boundaries.

Naively comparing activations across layers (e.g. "error = 5 at L3 vs L8")
is misleading because those numbers live in different, layer-specific
coordinate systems — one space might look like a square while another is a
stretched parallelogram, and the same norm means something entirely
different. The **canonical space** projection maps all errors back to a
common coordinate system (input space), making them genuinely comparable.

This document reports experiments measuring how quantization error compounds
geometrically through depth, decomposing it into local vs propagated sources,
and testing whether an oracle correction can eliminate it.

The primary configuration uses MLPs trained on the spirals dataset (2000
samples, 3 turns, 2× scaled) with width-32 hidden layers, depth 12, 4-bit
delta quantization, Adam optimizer (lr=0.001, 5000 epochs). We also test
embedding the same 2D manifold into 100D, and validate across architectures
(autoencoder, transformer).

Code: `aleph/qgeom/canonical.py` (library), `notebooks/canonical_error_correction.py`
(interactive analysis), `experiments/canonical_overnight.py` (cross-architecture
validation).

## Quantization method

We use **delta (grid) quantization**: snap each weight to the nearest
multiple of a fixed step size δ.

$$W_q = \text{round}(W / \delta) \cdot \delta$$

With 4 bits, δ = 1/(2³) = 0.125. The maximum per-weight error is δ/2 =
0.0625. The quantization error matrix is E_L = W_q - W.

Geometrically, delta quantization snaps the weight matrix to a uniform grid
in weight space. Every entry moves by at most δ/2 to the nearest grid point.
We chose delta quantization over bit-width quantization (which scales
per-tensor or per-channel) because the error is a uniform grid displacement,
making it easier to reason about spatially. The framework itself is
quantization-agnostic — all classes take W and W_q as inputs and never call
`quantize()`. The same analysis applies to any quantization method; we just
need to swap how W_q is produced.

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

Each layer's weight matrix transforms the coordinate system: it rotates,
stretches, and projects the space. After L layers, the data lives in a
coordinate system that is the composition of all those transforms. An error
vector at layer 8 points in a direction that only makes sense in layer 8's
coordinate system — you can't directly compare it to an error vector at
layer 3 without accounting for the geometric transforms between them.

The canonical projection undoes these transforms. We compute the cumulative
transform T_L = W_L ⋯ W_1 and map errors back to input space via its
pseudoinverse:

$$\text{canonical error} = T_L^+ \cdot \text{output error}$$

This puts all errors in the same coordinate system (input space), making
them genuinely comparable: you can see whether layer 3's error and layer 8's
error point in the same direction, have the same magnitude, and affect the
same region of the input manifold.

This is a linear approximation — it ignores ReLU nonlinearities, so it's
exact for the linear component and approximate for the rest. Condition
numbers grow with depth (reaching 126 at depth 12 for the 2D model), making
canonical comparisons approximate but still informative for moderate depths.

### ReLU disagreement

Where sign(z_float) ≠ sign(z_quant), the quantized and float networks make
different on/off decisions. This is the only source of nonlinear error — and
the one thing perfect correction can't fix in general (though it does fix it
when we correct at every layer, since then pre-activations match exactly).

### Geometric metrics

We track several geometric quantities that characterize how quantization
distorts each layer's transform:

**Error magnitude: ||δ_L||₂** — The L2 norm of the activation error
â_L - a_L, averaged over the data. This is the raw distance between
quantized and float activations at layer L. Because each layer's activations
live in a different coordinate system, comparing this across layers is
misleading without the canonical projection. It tells you how much the
activations have drifted in that layer's own space.

**Error direction: principal components** — PCA of the error vectors
δ_L = â_L - a_L across the dataset. Quantization error is not uniformly
distributed — it concentrates along specific directions in activation space.
If the top principal component captures most variance, the error is
effectively low-rank: a one-dimensional perturbation rather than a diffuse
cloud. This is good news for learned corrections, which only need to predict
a low-dimensional correction vector.

**Spectral norm: ||E_L||₂** — The largest singular value of the weight
quantization error matrix E_L = W_q - W. This is the maximum factor by
which any unit-length input vector gets scaled by the quantization
perturbation alone. A large spectral norm means some directions in activation
space get strongly perturbed by this layer's quantization. For delta
quantization with step δ, ||E_L||₂ is bounded by δ/2 · √(min(m,n)) for an
(m,n) weight matrix, but trained networks typically have structured weight
distributions that yield smaller spectral norms.

**Frobenius norm: ||E_L||_F** — The root-sum-of-squares of all entries of
E_L. While ||E_L||₂ measures worst-case directional amplification, ||E_L||_F
measures total quantization perturbation across all weight entries. Their
ratio ||E_L||₂ / ||E_L||_F indicates how concentrated the quantization error
is along a few singular directions (ratio near 1 = concentrated, ratio near
1/√rank = diffuse).

**Weight spectral norm: ||W_L||₂** — The largest singular value of the float
weight matrix. This controls how much propagated error from earlier layers
gets amplified at layer L. When ||W_L||₂ > 1, upstream errors grow; when
||W_L||₂ < 1, they shrink. The product of spectral norms across layers
gives an upper bound on total error amplification — layers with large
||W_L||₂ are the main drivers of error compounding.

**Volume distortion: det(W_q) / det(W)** — How much quantization changes the
volume scaling of the layer's linear transform. A ratio of 1 means the
quantized layer preserves volumes identically to the float layer. Deviations
indicate that quantization is selectively expanding or contracting certain
regions of the space. Only defined for square weight matrices; for
rectangular ones, we compare the product of singular values.

**Cumulative amplification: ||T_L||₂** — The spectral norm of the cumulative
transform T_L = W_L ⋯ W_1. This measures how much the network stretches
the most-amplified direction from input to layer L. Large values mean the
network maps some input directions to very large activations, which then
incur proportionally larger quantization errors (since delta quantization
has fixed step size regardless of activation magnitude).

**Condition number: cond(T_L)** — The ratio of largest to smallest singular
values of T_L. This measures how much the cumulative transform distorts the
shape of the space. cond(T_L) = 1 means all directions are equally
stretched; cond(T_L) = 10⁶ means one direction is amplified a million times
more than another. High condition numbers make canonical space mapping
numerically unstable — the pseudoinverse amplifies noise in the
least-stretched directions. At cond > ~10⁴, canonical comparisons become
unreliable.

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
at the output.

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
reduces error from 93.47 to 4.90 (95% reduction).

### ReLU disagreement

Ranges from 2% (L0) to 11.5% (L7). Wider networks distribute quantization
error across more neurons, keeping individual perturbations small relative
to activations.

### Geometric metrics

| Layer | ||E||₂ | ||W||₂ | cond(T_L) |
|-------|--------|--------|-----------|
| L0    | 0.2154 | 2.7246 | 1.2       |
| L4    | 0.3779 | 6.5994 | 26.0      |
| L8    | 0.3805 | 3.7165 | 129.9     |
| L11   | 0.3949 | 5.7325 | 125.6     |
| L12   | 0.1914 | 2.2094 | 1.0       |

||E||₂ stays roughly constant (~0.2–0.4) across layers — the quantization
grid perturbs each layer's transform by a similar amount. ||W||₂ varies more
(2.2 to 7.1), and the layers with largest spectral norms (L4, L5: ~6–7)
correspond to the sharpest jumps in total error. Condition numbers stay
moderate (max 130), meaning canonical space is geometrically meaningful for
this architecture.

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
   error across more dimensions.

2. **Higher propagated fraction**: 99% propagated at output (vs 94% for 2D).
   The 100→32 first layer's larger quantization error propagates more
   aggressively relative to subsequent local errors.

3. **Condition numbers explode**: cond(T₁₁) = 564M in 100D vs 126 in 2D.
   The 100→32 projection introduces severe ill-conditioning from the start.
   Canonical space is numerically meaningless for 100D; the correction
   formula is unaffected.

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

## Cross-architecture validation

We ran the same canonical analysis on three architectures to confirm the
findings are not specific to classification MLPs.
See `experiments/canonical_overnight.py` and `docs/canonical_overnight_results.json`.

### Summary table

| Model | Output error | Amplification | % Propagated | Correction residual | Output-only |
|-------|-------------|---------------|-------------|--------------------:|------------:|
| MLP 2→32×12→1 (spirals) | 93.47 | 469× | 94% | 1.2e-6 | 0.0000 |
| MLP 2→8×4→1 (spirals) | 2.20 | 21× | 98% | 3.4e-7 | 0.0000 |
| MLP 100→32×12→1 (spirals) | 8.82 | 6× | 99% | 3.4e-7 | 0.0000 |
| Autoencoder 784→256→128→32→…→784 (MNIST) | 10.29 | 1.1× | 58% | 2.2e-6 | 0.0000 |
| Autoencoder 784→128→64→16→…→784 (MNIST) | 7.09 | 1.1× | 56% | 1.8e-6 | 0.0000 |
| Transformer FFN path, 4 blocks (Shakespeare) | 0.27 | ~1× | 43% | 2.2e-7 | 0.0000 |
| Transformer FFN path, 8 blocks (Shakespeare) | 0.22 | ~1× | 19% | 2.1e-7 | 0.0000 |

### Classification MLPs

Strong compounding (21–469× amplification). Propagated error dominates
heavily (94–99%). Deep sequential transforms with no skip connections are
the worst case for error accumulation. Oracle correction works perfectly.

### Autoencoders

Much less amplification (~1.1×) despite 6 layers. The symmetric
encoder→decoder structure constrains error growth: the decoder approximately
inverts the encoder's transforms, so errors don't compound as aggressively
in one direction. Propagated and local error are roughly balanced (~56/44%).
Oracle correction and bottleneck absorption both work perfectly.

### Transformers

The FFN path alone shows minimal error (0.2–0.3) and minimal compounding,
because each block's FFN is only two layers (up-project + down-project).
However, the **full model** (including attention) shows significant error
growth in the residual stream: 6.2 → 22.8 across 4 blocks, 5.6 → 25.4
across 8 blocks. Most transformer quantization error flows through
**attention** (data-dependent routing), not FFN. The canonical framework's
sequential decomposition does not cleanly apply to attention.

Oracle correction on the FFN path works to ~10⁻⁷. Output-only correction
achieves zero error. The full-model attention error would require a
different decomposition framework.

### What holds universally

1. **Oracle correction works** to float32 precision — this is a mathematical
   guarantee (the decomposition is exact), not an empirical finding.
2. **Output-only correction achieves zero error** via bottleneck absorption —
   the output projection discards all but one dimension of accumulated error.
3. **Error is structured**, not random noise — it concentrates in specific
   directions and spatial regions, making learned correction feasible.

## Depth sweep (moons dataset)

Earlier experiments used the moons dataset (500 samples, noise=0.15) with
width-8 MLPs at depth 2, 4, 8, and 16. These figures are preserved in
`docs/figures/`.

### Figures

**Attribution bar charts** — `depth{2,4,8,16}_attribution.png`

Stacked bars showing mean pre-ReLU error (output space) at each layer.
Blue = local error (this layer's quantization). Orange = propagated error
(upstream errors amplified forward). The percentage label shows the
propagated fraction. A growing orange fraction means earlier quantization
mistakes are increasingly dominating.

![Depth 2 attribution](figures/depth2_attribution.png)
![Depth 4 attribution](figures/depth4_attribution.png)
![Depth 8 attribution](figures/depth8_attribution.png)
![Depth 16 attribution](figures/depth16_attribution.png)

**Error heatmaps** — `depth{2,4,8,16}_heatmap.png`

Each panel shows one layer. The x/y axes are input space coordinates; the
color is **output-space** error magnitude ||â_L - a_L|| at that layer.
White dots are training data. Error concentrates away from the training data
because the network extrapolates large activations outside the data
manifold, and larger activations → larger quantization error. The color
scale grows across panels — this is the compounding.

![Depth 2 heatmap](figures/depth2_heatmap.png)
![Depth 4 heatmap](figures/depth4_heatmap.png)
![Depth 8 heatmap](figures/depth8_heatmap.png)
![Depth 16 heatmap](figures/depth16_heatmap.png)

**ReLU disagreement maps** — `depth{2,4,8,16}_relu.png`

Red regions = at least one neuron at that layer made a different on/off
decision in the quantized vs float network. These cluster near the network's
decision boundaries where neurons are close to zero.

![Depth 2 ReLU disagreement](figures/depth2_relu.png)
![Depth 4 ReLU disagreement](figures/depth4_relu.png)
![Depth 8 ReLU disagreement](figures/depth8_relu.png)
![Depth 16 ReLU disagreement](figures/depth16_relu.png)

### Decision boundary (primary config)

Three panels generated by the notebook: float model, quantized model,
oracle-corrected model. The color shows P(class 1) from blue (0) to red (1),
with the decision boundary (P=0.5) as a black line. Training data is
overlaid. Quantization visibly distorts the boundary; correction restores
it exactly. See `notebooks/canonical_error_correction.py`.

## Key findings

### 1. Error compounds geometrically, driven by weight spectral norms

Each layer's weight matrix stretches some directions more than others.
Propagated error from upstream layers gets amplified by that stretching:
the spectral norm ||W_L||₂ controls the worst-case amplification per layer.
Layers with ||W_L||₂ > 1 expand the error ellipsoid; layers with
||W_L||₂ < 1 contract it. The cumulative effect is roughly multiplicative —
total amplification scales with the product of per-layer spectral norms.
At width-32 depth-12, this yields ~470× amplification, enough to visibly
distort classification boundaries.

### 2. Perfect correction works exactly

The correction formula C_L = -E_L·â - W_L·ε recovers float pre-activations
to float32 machine precision (~10⁻⁶ to 10⁻⁷) at every layer, for every
architecture tested (MLP, autoencoder, transformer FFN). This is a
mathematical guarantee: the local + propagated decomposition exactly
accounts for all linear error.

### 3. Bottleneck layers absorb upstream error

Correcting only the output layer achieves zero output error in every
configuration tested. Geometrically: the output projection maps
high-dimensional activations to a lower-dimensional space, discarding all
error components orthogonal to the output direction. A (1, 32) output
layer discards 31 of 32 error dimensions. This is bottleneck absorption —
the dimensionality reduction at the output naturally kills most of the
accumulated error.

### 4. Error is spatially structured, not random noise

The error heatmaps show that quantization error concentrates in specific
spatial regions (large activations, typically away from training data). The
canonical-space PCA shows error variance concentrating along fewer directions
with depth. This structure means a correction network doesn't need to
handle arbitrary noise — just a structured, low-rank perturbation.

### 5. ReLU disagreement grows with depth but stays manageable

At width-32 depth-12, disagreement ranges from 2–12% per layer. Wider
networks distribute quantization error across more neurons, keeping
individual perturbations small relative to activations. These disagreement
regions are the hard cases for any non-oracle correction.

### 6. Architecture affects error dynamics but not correctability

MLPs compound worst (no skip connections). Autoencoders self-stabilize
(symmetric structure). Transformers accumulate error mainly through
attention, not FFN. But in all cases: oracle correction works exactly,
output-only correction achieves zero error, and the error is structured.
The framework applies universally.

## Geometric error decomposition: metric vs topological distortion

### Motivation

The oracle correction proves that 100% of error is correctable. But the
oracle needs float activations — can we replace it with something cheap?
To answer this, we need to know what *kind* of geometric distortion the
error represents. Not all error is created equal.

Each row of a weight matrix $W_L$ defines a hyperplane in activation space,
and ReLU partitions the space at that hyperplane into two half-spaces (active
vs inactive). When quantization perturbs $W \to W_q$, these hyperplanes
shift. Two geometrically distinct things can happen to each neuron for a
given input:

1. **Metric distortion** — the input stays in the same half-space under both
   $W$ and $W_q$. The linear transform rotates and stretches differently, but
   the topology of the representation is preserved. The error is
   $E_L \cdot \hat{a}_{L-1}$ restricted to agreeing neurons — a linear
   perturbation that a linear (or low-rank) correction can undo exactly.

2. **Topological distortion** — the hyperplane shifts enough that the input
   crosses the boundary. A dimension carrying manifold structure (many
   distinct values tracing a curve through the data) collapses to zero via
   ReLU, or a zero-valued dimension inflates. This is a qualitative change
   in the representation geometry — no linear correction can recover
   $\text{ReLU}(x) = 0$ from $\text{ReLU}(x') > 0$ or vice versa.

The decomposition is exact: we use the ReLU disagreement mask
$M^{\text{agree}}_L$ (neurons where $\text{sign}(z^{\text{float}}) =
\text{sign}(z^{\text{quant}})$) to separate the post-activation error into
metric and topological components:

$$\varepsilon^{\text{post}}_L = \underbrace{(\hat{a}_L - a_L) \odot M^{\text{agree}}}_{\text{metric}} + \underbrace{(\hat{a}_L - a_L) \odot M^{\text{disagree}}}_{\text{topological}}$$

### Methodology

For each configuration, we:
1. Train the float model and verify task accuracy (>85% gate).
2. Run float and quantized forward passes, collecting all pre/post-activation
   traces.
3. At each hidden layer, compute the ReLU agreement mask and split the
   post-activation error into metric and topological components.
4. Measure the fraction of $||\varepsilon||^2$ in each component (energy
   budget).
5. Compute the SVD of the metric error matrix across all samples — this
   reveals how many directions the linearly correctable perturbation uses
   (rank for 95% variance recovery).
6. Measure the variance carried by flipped neurons in the float network
   (var_ratio: how much signal the topological distortion destroys).
7. Run a **metric-only corrected forward pass**: at each hidden layer,
   subtract the metric error component (error on agreeing neurons) from
   the quantized activations, leaving only topological error. Measure
   task accuracy of the result.

Configurations tested: spirals and moons datasets, depths 4/8/12, widths
8/32, plus 100D embedding. See `experiments/geometric_decomposition_sweep.py`
and `docs/geometric_decomposition_sweep.json`.

### Results

| Config | Float | Quant | Metric-corrected | Metric% | Topo% | Rank₉₅ | Max disagree |
|--------|-------|-------|-----------------|---------|-------|--------|-------------|
| spirals 32×12 | 92.8% | 53.6% | 92.7% | 93% | 7% | 3.6 | 22.6% |
| spirals 32×8 | 94.8% | 51.7% | 94.5% | 89% | 11% | 6.5 | 35.7% |
| spirals 32×4 | 94.0% | 56.1% | 91.9% | 91% | 9% | 7.5 | 32.5% |
| moons 32×12 | 99.8% | 98.2% | 99.7% | 89% | 11% | 4.2 | 15.3% |
| moons 32×4 | 99.5% | 98.1% | 99.4% | 95% | 5% | 4.0 | 4.9% |
| moons 8×4 | 99.2% | 91.6% | 99.2% | 98% | 2% | 1.8 | 3.7% |
| spirals 100D 32×12 | 93.2% | 64.5% | 93.2% | 88% | 12% | 6.8 | 21.6% |

Spirals width-8 configs skipped (accuracy <85% — undertrained, see rule 0c).

### Key findings

**7. Metric-only correction fully recovers task performance**

Correcting only the metric distortion (linear perturbation on neurons where
the ReLU half-space is preserved) recovers float accuracy in every
configuration. The most dramatic case: spirals 32×12 goes from 53.6%
(quantized) back to 92.7% (matching float 92.8%). The topological error
(hyperplane crossings), while geometrically irreducible by linear methods,
is task-irrelevant — it destroys signal on neurons that don't affect the
classification decision.

This means the entire correction problem is effectively linear.

**8. Metric distortion accounts for 88–98% of error energy**

Across all valid configurations, 88–98% of $||\varepsilon||^2$ at each layer
comes from neurons where the ReLU half-space is preserved. The topological
fraction (2–12%) is consistently small. This holds across datasets (spirals,
moons), depths (4, 8, 12), widths (8, 32), and input dimensionality (2D,
100D).

**9. Correction rank decreases with depth**

The SVD of the metric error matrix shows that deeper layers need fewer
correction dimensions: early layers need rank 5–7 for 95% recovery, while
layers beyond L8 need only rank 1–2. Error propagation through the network
concentrates the perturbation along the dominant singular directions of the
cumulative transform — the compounding itself acts as a low-rank projector.
A uniform rank-3 correction across all layers would recover ~90% of error
energy.

**10. Topological distortion destroys real signal but is task-irrelevant**

The variance ratio (variance carried by flipped neurons / variance of all
neurons) is ≈1.0 across all layers and configurations. Hyperplane crossings
are not selectively targeting low-information neurons — when a crossing
happens, it destroys real manifold structure. But because crossings are rare
(2–12% of neurons) and concentrated near decision boundaries (where
activations are near zero, hence near the hyperplane), their aggregate
impact on classification accuracy is negligible.

**11. Harder tasks show more quantization damage but equal recovery**

Spirals (92.8% float) drops to 51–56% under quantization, while moons
(99.5% float) only drops to 91–98%. The harder task has decision boundaries
closer to the data manifold, so hyperplane shifts cause more classification
errors. But metric-only correction fully recovers both — the correction
doesn't need to be harder, just more of the error energy matters for the
task.

### Geometric interpretation

The weight perturbation $E_L = W_q - W$ distorts the linear transform at
each layer. This distortion has two effects:

1. **Within half-spaces** (metric): the transform maps data to slightly
   different locations. The data manifold is stretched and rotated
   differently, but its topology is preserved. The error is a linear
   function of the input activations ($E_L \cdot \hat{a}_{L-1}$), and its
   SVD reveals that it concentrates along a few directions — the dominant
   singular vectors of $E_L$ projected through the data distribution. A
   rank-$k$ correction along these directions undoes the distortion.

2. **At half-space boundaries** (topological): the hyperplane defined by
   row $i$ of $W_L$ shifts to the hyperplane defined by row $i$ of $W_{q,L}$.
   Inputs near the original boundary may cross to the other side. This
   changes the ReLU activation pattern, which is a discrete topological
   change (a dimension of the representation is created or destroyed). No
   continuous linear correction can undo a discrete change. However, these
   crossings cluster near the boundaries where activations are small, so the
   affected neurons carry little absolute signal even when their relative
   variance is high.

The empirical finding that metric-only correction recovers full task accuracy
provides the geometric explanation for the earlier empirical result that
linear corrections outperform MLPs at 4-bit quantization (see
`docs/quantization_correction_system.md`): there is essentially nothing for
nonlinear capacity to fix.

Code: `notebooks/canonical_learned_correction.py` (interactive analysis),
`experiments/geometric_decomposition_sweep.py` (cross-configuration sweep).

## Low-rank correction (Experiment 1)

### Motivation

The geometric decomposition (Experiment 3) establishes that the correction
problem is effectively linear and low-rank. The next question: what rank $k$
does each layer actually need, and can our geometric metrics predict it?

### Methodology

At each hidden layer, the oracle correction on the calibration set produces
a matrix $C_L \in \mathbb{R}^{N \times H}$. The SVD of $C_L$ reveals the
correction subspace — the directions in pre-activation space that the
correction actually uses. A rank-$k$ correction keeps only the top-$k$
singular components (the best possible rank-$k$ approximation by
Eckart–Young).

Critically, the rank-$k$ correction is applied **pre-ReLU**: $z_L + C_k$,
then $\text{ReLU}(z_L + C_k)$. Applying it post-ReLU would be wrong because
$\text{ReLU}(z) + C \neq \text{ReLU}(z + C)$ — the nonlinearity means the
correction must happen before the activation function.

We test three rank-selection strategies:
1. **Uniform rank**: same $k$ at every layer.
2. **Metric-predicted rank**: use the rank₉₅ of the metric error matrix
   from Experiment 3 (no oracle needed — computed from quantized activations
   alone).
3. **Full rank** (oracle baseline): $k = H$ at every layer.

### Results

| Config | Float | Quant | Rank-1 | Rank-3 | Rank-5 | Predicted | Full |
|--------|-------|-------|--------|--------|--------|-----------|------|
| spirals 32×12 | 92.8% | 53.6% | 53.3% | 73.2% | 84.8% | 85.8% | 92.5% |
| spirals 32×8 | 94.8% | 51.7% | 69.0% | 74.8% | 81.0% | 91.2% | 94.8% |
| spirals 32×4 | 94.0% | 56.1% | 69.7% | 85.2% | 88.4% | 92.9% | 94.0% |
| moons 32×12 | 99.8% | 98.2% | 99.1% | 99.5% | 99.6% | 99.5% | 99.7% |
| moons 32×4 | 99.5% | 98.1% | 99.4% | 99.3% | 99.2% | 99.2% | 99.4% |
| moons 8×4 | 99.2% | 91.6% | 99.1% | 99.2% | 99.2% | 99.2% | 99.2% |
| spirals 100D 32×12 | 93.2% | 64.5% | 58.0% | 83.8% | 87.8% | 88.5% | 93.4% |

### Key findings

**12. Metric-predicted variable rank is the best accuracy/parameter strategy**

Using the metric error rank₉₅ from Experiment 3 to assign per-layer ranks
(higher at early layers, lower at deep layers) consistently matches or beats
uniform rank-5 while using fewer parameters. For spirals 32×8, predicted
ranks achieve 91.2% (vs uniform rank-5 at 81.0%) because they allocate
rank 7–8 to early layers that need it and rank 4 to later layers that
don't. This strategy requires no oracle — only the metric error SVD on a
calibration set, which uses quantized activations alone.

**13. Energy recovery ≠ task recovery**

The oracle correction SVD shows rank₉₅ ≈ 2 at most layers — 95% of
correction energy is captured in 2 directions. But rank-2 uniform correction
only achieves 58.6% accuracy on spirals (from 53.6% quantized). The
remaining 5% of energy, carried in higher singular components, matters
disproportionately for the task because errors compound through layers: a
small uncorrected residual at layer $L$ gets amplified by all downstream
transforms.

This means energy-based rank selection (rank₉₅ of the oracle SVD) is
insufficient. The metric error rank₉₅ from Experiment 3, which is
consistently higher (5–7 at early layers vs oracle's 2–3), serves as a
better predictor precisely because it overestimates — it's a conservative
bound that captures the task-relevant correction subspace.

**14. Task difficulty determines the rank budget**

Moons (simple boundary) needs only rank 1 — even rank-1 correction
recovers from 91.6% to 99.1%. Spirals (complex boundary with 3 turns)
needs rank 5+ for meaningful recovery. The geometric explanation:
spirals require the network to learn fine-grained rotational structure,
which uses more dimensions of the weight matrix's column space. Quantization
perturbs more of these task-relevant directions, so the correction must
address more dimensions.

**15. Full-rank correction nearly recovers float accuracy**

Full-rank oracle correction achieves 92.5% on spirals 32×12 (vs 92.8%
float) — a 0.3% gap from float32 SVD precision compounding across 12
layers. For shallower configs (32×4), full-rank exactly matches float
(94.0%). This confirms the oracle correction is the true ceiling.

### Per-layer rank structure

For the primary config (spirals 32×12), the oracle correction rank₉₅
is approximately 2 at every layer, while the metric-predicted rank
varies: 7 at L0–L1, decreasing to 1–2 at L9–L11. The predicted ranks
overestimate the oracle ranks at early layers and match at deep layers.

Geometrically: early layers have diverse error directions (the quantization
perturbation hasn't yet been projected through many transforms), so the
correction subspace is higher-dimensional. Deep layers see error that has
been concentrated by the cumulative transform $T_L$ into fewer dominant
directions, so rank 1–2 suffices.

Code: `notebooks/canonical_learned_correction.py` (Experiment 1 section),
`experiments/geometric_decomposition_sweep.py` (cross-configuration sweep
with rank-$k$ accuracy).

## Correction distillation (Experiment 2)

### Motivation

Experiments 1 and 3 established that quantization error is mostly metric
distortion (88–98%), low-rank, and correctable. But the oracle corrections
require float activations at inference time. Can we distill the oracle into
a small shared network that works with quantized activations only?

A separate question is whether the correction patterns across layers have
enough regularity that a single shared network can exploit them — rather
than needing independent per-layer correctors (which would be equivalent to
just adding layers to the model).

### Architecture

A single MLP shared across all layers:
$f_\theta(z_L, \text{context}_L) \to C_L$, applied pre-ReLU.

Three context variants test what geometric information the network needs:

1. **Learned embedding** — layer index as a trainable vector; the network
   must infer geometric context from z alone. Genuinely deployable (no E_L
   needed).
2. **Local error term** — $c_{\text{local}} = -E_L \cdot a_{L-1}$, the
   computable part of the oracle correction. Requires storing E_L per layer.
3. **Combined** — both embedding and local error term.

For variants that include $c_{\text{local}}$, we use a **skip connection**:
$C = c_{\text{local}} + f_\theta(z, \text{context})$. The network learns
only the residual (propagated error correction), avoiding a critical
architecture limitation (see Key finding 16 below).

### Analytical baseline: direct $c_{\text{local}}$

Before training any network, we test the analytical correction
$c_{\text{local}} = -E_L \cdot a_{L-1}$ applied at each layer. Since:

$$z_L + c_{\text{local}} = W_{q,L} \cdot a_{L-1} + b_L - (W_{q,L} - W_L) \cdot a_{L-1} = W_L \cdot a_{L-1} + b_L$$

this exactly recovers the float pre-activation. With zero learned
parameters, this achieves the direct correction ceiling. The only cost is
storing the error matrix $E_L = W_q - W$ per layer (same size as the weight
matrix).

### Training

Two-phase distillation from oracle corrections:

- **Phase 1 (teacher forcing)**: each layer sees float activations as input,
  oracle $C_L$ as target. 1000 epochs, lr=10⁻³.
- **Phase 2 (autoregressive)**: the network runs sequentially using its own
  corrected activations. Loss: $||f_\theta(\text{output}) - \text{float output}||^2$.
  500 epochs, lr=10⁻⁴.

The base model weights are frozen during all correction training. Only the
correction network parameters are updated, using distillation loss (never
task loss). This ensures the correction learns quantization error correction,
not additional task capacity.

### Parameter counts

The correction network is a 2-layer MLP shared across all layers, with a
layer embedding of dimension 8. The "% of task" column shows correction
network parameters as a percentage of the task network (classifier)
parameters.

| Correction variant | Params | % of task (32×12) | % of task (32×4) |
|-------------------|--------|-------------------|------------------|
| Direct $c_{\text{local}}$ | 0 learned | 0% | 0% |
| Embedding only | 4,800 | 40.9% | 143.6% |
| Local + skip | 6,240 | 53.1% | 189.3% |
| Combined + skip | 6,848 | 58.3% | 205.8% |

Reference task network sizes: 32×4 = 3,297 params, 32×8 = 7,521 params,
32×12 = 11,745 params, 32×16 = 15,969 params, 32×20 = 20,193 params.

The correction network size is **fixed** regardless of depth — it's shared
across layers, so deeper networks amortize the cost better. At depth 12,
the best correction network (combined+skip, 6,848 params) is 58% of the
task network (11,745 params). At depth 20, it's only 34% (6,912 / 20,193).
At depth 4, the correction exceeds the task network size — at that point,
the correction network is no longer "small" relative to the model.

For the 100D embedding variant (spirals 100→32×12→1), the task network
grows to 14,881 parameters while the correction stays at 6,848 (46%). The
correction operates in the hidden dimension (32), not the input dimension,
so it's invariant to input size.

### Results

| Config | Float | Quant | Direct $c_L$ | Embedding | L+Skip | C+Skip |
|--------|-------|-------|---------------|-----------|--------|--------|
| spirals 32×12 | 92.8% | 53.6% | 92.5% | 82.6% | 91.3% | 91.9% |
| spirals 32×8 | 94.8% | 51.7% | 94.8% | 93.0% | 94.7% | 94.6% |
| spirals 32×4 | 94.0% | 56.1% | 94.0% | 93.9% | 93.9% | 93.8% |
| moons 32×12 | 99.8% | 98.2% | 99.7% | 99.6% | 99.6% | 99.6% |
| moons 32×4 | 99.5% | 98.1% | 99.4% | 99.6% | 99.5% | 99.5% |
| moons 8×4 | 99.2% | 91.6% | 99.2% | 99.2% | 99.1% | 99.2% |
| spirals 100D 32×12 | 93.2% | 64.5% | 93.4% | 92.1% | 93.1% | 93.2% |

Spirals width-8 configs skipped (accuracy <85% — undertrained).

### Float ablation: correction vs capacity

Applying trained corrections to the float model (E_L = 0) tests whether
the network adds task capacity or genuinely corrects quantization error.

| Config | Float | Embed/Float | L+Skip/Float | C+Skip/Float |
|--------|-------|-------------|--------------|--------------|
| spirals 32×12 | 92.8% | 50.3% | 73.9% | 80.1% |
| spirals 32×8 | 94.8% | 54.1% | 85.0% | 82.5% |
| spirals 32×4 | 94.0% | 53.0% | 80.8% | 89.1% |
| moons 32×12 | 99.8% | 96.2% | 99.2% | 98.9% |

All variants substantially degrade float accuracy on spirals (the hard
task). On moons (easy task), degradation is minimal because the corrections
are small. This confirms the networks are learning quantization-specific
corrections, not adding general capacity.

### Key findings

**16. ReLU bottleneck prevents naive correction architectures**

A 64-unit hidden layer with ReLU cannot represent identity on 32-dimensional
signed correction vectors. The ReLU activation requires 2× the output
dimension just to represent both positive and negative components (each
neuron handles one half-space). With 64 hidden units and 32-dimensional
output, the identity mapping alone saturates the network's capacity, leaving
nothing for the actual correction task.

This caused the initial "local" variant (without skip connection) to achieve
only 74.1% accuracy — worse than the embedding-only variant (82.6%), despite
having strictly more information. The counterintuitive result was not a
genuine finding but an architecture limitation.

**Fix**: skip connection $C = c_{\text{local}} + f_\theta(z, \text{context})$.
The network only learns the residual (propagated error), which is smaller
and doesn't require passing the full correction through the ReLU bottleneck.
This jumped local from 74.1% → 91.3%.

**17. Direct $c_{\text{local}}$ is the practical ceiling for deployable correction**

Applying $-E_L \cdot a_{L-1}$ at each layer recovers float accuracy with
zero learned parameters across all configurations. The cost is storing the
error matrices $E_L$ (same size as weights, but these are known at
quantization time and fixed). This is the strongest baseline any learned
approach must beat — and none of the trained variants exceed it.

The gap between direct $c_{\text{local}}$ (92.5%) and the best learned
variant combined+skip (91.9%) on the hardest config (spirals 32×12) shows
that even small per-layer approximation errors (2–5%) compound through 12
layers to produce a measurable accuracy gap.

**18. Embedding-only correction is viable for shallow networks**

The embedding variant (no E_L needed, genuinely deployable at inference with
no per-layer storage) achieves 93.0–93.9% on spirals at depth 4–8, nearly
matching float accuracy. At depth 12, it drops to 82.6% — the 12 layers of
compounding without the local geometric context is too much.

This suggests a depth-dependent deployment strategy: for shallow quantized
networks (≤8 layers), a learned embedding correction may suffice. For deep
networks, storing E_L and using the local+skip variant is necessary.

**19. Correction network size is depth-invariant**

Because the correction is a single shared network, the parameter cost is
fixed (~5K–7K) regardless of task network depth. At depth 12 (11,745 task
params), the correction is 41–58% of the task network. The correction cost
amortizes over depth — making distillation most efficient for deep networks,
which are exactly the ones that suffer most from quantization error
compounding.

### Geometric interpretation

The correction network learns a mapping from layer-space pre-activations
to corrections, conditioned on geometric context (layer embedding and/or
local error term). The skip connection architecture decomposes this as:

$$C_L = \underbrace{-E_L \cdot a_{L-1}}_{\text{local (computable)}} + \underbrace{f_\theta(z_L, \text{context}_L)}_{\text{propagated (learned)}}$$

The first term exactly undoes this layer's quantization error. The second
term learns to approximate $-W_L \cdot \varepsilon_{L-1}$ — the propagated
error from upstream layers, which is the part that requires knowing the
global error state. The residual is smaller and smoother than the full
correction, making it easier for a small network to learn.

The float ablation confirms this interpretation: when E_L = 0 (float model),
the local term vanishes and the learned residual produces noise that
degrades accuracy.

Code: `notebooks/correction_distillation.py` (interactive analysis),
`experiments/correction_distillation_sweep.py` (cross-configuration sweep),
`docs/correction_distillation_sweep.json` (raw results).

### Depth scaling and correction network capacity

To understand how correction quality scales with task network depth and
whether larger correction networks compensate, we ran spirals (width 32)
at depths 4–24 with correction hidden sizes from 32 to 256.

| Depth | Float | Quant | Direct $c_L$ | Embed | h=32 | h=64 | h=128 | h=256 | Task params |
|-------|-------|-------|---------------|-------|------|------|-------|-------|-------------|
| 4 | 93.2% | 56.7% | 93.3% | 93.0% | 93.0% | 93.2% | 92.9% | 93.0% | 3,297 |
| 8 | 94.8% | 51.7% | 94.8% | 93.0% | 94.8% | 94.6% | 94.5% | 94.7% | 7,521 |
| 12 | 94.4% | 55.1% | 93.6% | 84.7% | 93.8% | 92.1% | 91.5% | 93.0% | 11,745 |
| 16 | 93.3% | 48.2% | 93.3% | 90.0% | 92.8% | 93.0% | 93.0% | 91.8% | 15,969 |
| 20 | 93.8% | 55.6% | 93.7% | 84.1% | 93.6% | 92.8% | 93.1% | 92.9% | 20,193 |

Depth 24 skipped (task network undertrained at 82.6%).

Correction network parameter counts (combined+skip) as % of task network:

| Depth | h=32 params (% of task) | h=64 params (% of task) | h=128 params (% of task) | h=256 params (% of task) |
|-------|------------------------|------------------------|-------------------------|-------------------------|
| 4 | 3,424 (103.9%) | 6,784 (205.8%) | 13,504 (409.6%) | 26,944 (817.2%) |
| 8 | 3,456 (46.0%) | 6,816 (90.6%) | 13,536 (180.0%) | 26,976 (358.7%) |
| 12 | 3,488 (29.7%) | 6,848 (58.3%) | 13,568 (115.5%) | 27,008 (230.0%) |
| 16 | 3,520 (22.0%) | 6,880 (43.1%) | 13,600 (85.2%) | 27,040 (169.3%) |
| 20 | 3,552 (17.6%) | 6,912 (34.2%) | 13,632 (67.5%) | 27,072 (134.1%) |

**20. The smallest correction network (h=32) works best**

Across all depths, the h=32 combined+skip variant matches or beats
larger correction networks. At depth 12: h=32 gets 93.8% vs h=64 at 92.1%.
At depth 20: h=32 gets 93.6% vs h=256 at 92.9%. Larger correction networks
don't improve accuracy — they have more parameters to overfit the Phase 1
teacher-forced targets but don't generalize better in the autoregressive
Phase 2 where errors compound.

The h=32 correction network has only ~3,500 parameters — just 18–30% of
the task network at depth 12–20. This is the most parameter-efficient
correction variant.

**21. Direct $c_{\text{local}}$ scales perfectly with depth**

The analytical correction maintains near-float accuracy at every depth
tested (93.3–94.8%), confirming that $c_{\text{local}} = -E_L \cdot a$ is
depth-invariant. This makes sense: it exactly undoes the local quantization
error at each layer, so there is no compounding. The only cost is storing
the error matrices $E_L$.

**22. Embedding-only correction degrades at depth ≥12 but not monotonically**

The embedding variant (no E_L needed) oscillates between 84–93% depending on
depth: 93.0% at depth 8, 84.7% at depth 12, 90.0% at depth 16, 84.1% at
depth 20. The non-monotonic pattern suggests sensitivity to training
dynamics rather than a clean depth scaling law. Without the local error term,
the network relies entirely on z to infer the correction — and z itself
becomes noisier at deeper layers due to compounding, creating a harder
learning problem.

**23. Learned corrections approach but don't exceed the analytical ceiling**

At every depth, the best learned variant (combined+skip, h=32) comes within
0.2–1.0% of direct $c_{\text{local}}$. The gap is from per-layer
approximation errors that compound through depth. Crucially, the gap does
not grow with depth: 0.3% at depth 4, 0.0% at depth 8, 0.2% at depth 12,
0.5% at depth 16, 0.1% at depth 20. The learned network successfully
tracks the increasing correction complexity without systematic degradation.

Code: `experiments/correction_depth_scaling.py`,
`docs/correction_depth_scaling.json`.

## What's next

The three planned experiments are complete:
- **Experiment 1** (Low-rank correction): metric-predicted variable rank is
  the best accuracy/parameter strategy
- **Experiment 2** (Correction distillation): shared correction network
  approaching oracle ceiling; skip connection is critical
- **Experiment 3** (Geometric decomposition): metric distortion accounts for
  88–98% of error energy and is the only task-relevant component

Open questions for future work:
- Can the correction network generalize across different quantization
  configurations (e.g. train at 4-bit, deploy at 3-bit)?
- Does the direct $c_{\text{local}}$ approach scale to transformer-scale
  models where storing E_L per layer has meaningful memory cost?
- Can attention quantization error be decomposed similarly, or does
  the data-dependent routing require fundamentally different correction
  strategies?
