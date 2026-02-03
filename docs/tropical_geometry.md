# Tropical Geometry for Quantization

> You're already doing tropical geometry. You just don't know it yet.

The Minkowski sum error regions, the fate classification (survive/dead/flipped),
the piecewise-linear error structure that is perfectly correctable within a region
but discontinuous at boundaries — all of this **is** tropical geometry, without the
name.

This document makes the connection explicit. It introduces the relevant tropical
theory (no proofs, lots of examples), maps it onto what `aleph.qgeom` already
computes, and proposes experiments that use the framework to push further.

**Key references:**

- Brugallé & Shaw, "A Bit of Tropical Geometry" (2013) — expository introduction
  to tropical geometry: the tropical semiring, tropical polynomials, tropical
  curves, dual subdivisions, Maslov dequantization.
- Zhang, Naitzat & Lim, "Tropical Geometry of Deep Neural Networks" (ICML 2018) —
  proves that ReLU networks with integer weights are exactly tropical rational
  maps. Counts linear regions via dual subdivisions. Decision boundaries are
  subsets of tropical hypersurfaces.
- Balestriero & Baraniuk, "A Spline Theory of Deep Learning" (ICML 2018) — proves
  that DNs with piecewise-affine activations are compositions of max-affine spline
  operators (MASOs). The output is a signal-dependent affine transformation of
  the input. Connects DNs to vector quantization and matched filtering.
- You, Balestriero et al., "Max-Affine Spline Insights Into Deep Network Pruning"
  (TMLR 2022) — uses the spline partition to analyze how pruning (and by extension,
  quantization) affects the DN's piecewise-affine mapping and input space partition.
- Humayun, Balestriero & Baraniuk, "Deep Networks Always Grok and Here is Why"
  (ICML 2024) — introduces "local complexity" as the density of spline partition
  regions around a point. Shows that partition regions migrate during training
  toward decision boundaries and away from data, explaining grokking.
- Budd, Ideami, Rynne, Duggar & Balestriero, "SplInterp" (NeurIPS 2025) —
  characterizes sparse autoencoders as piecewise-affine splines whose partition
  cells form power diagrams (weighted Voronoi diagrams). Connects SAEs to
  k-means + local PCA.

---

## 1. Tropical Algebra in 5 Minutes

### The tropical semiring

Replace the usual arithmetic with:

| Operation | Classical | Tropical |
|-----------|-----------|----------|
| Addition  | a + b     | max(a, b) |
| Multiplication | a × b | a + b |

The identity for tropical addition is -∞ (since max(a, -∞) = a).
The identity for tropical multiplication is 0 (since a + 0 = a).

This is the **(max, +) semiring**. It's a valid algebraic structure — associative,
commutative, distributive — but there are no additive inverses (you can't "undo" a
max).

### Tropical polynomials

A classical polynomial in one variable looks like:

    p(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ

Replace + with max and × with +:

    p_trop(x) = max(c₀, c₁ + x, c₂ + 2x, ..., cₙ + nx)

This is a **piecewise-linear convex function**: the pointwise maximum of finitely
many affine functions. Each term `cᵢ + ix` is a line with slope `i` and intercept
`cᵢ`.

**1D example.** Take p_trop(x) = max(0, 1 + x, 3 + 2x):

- For x << 0: the term `3 + 2x` dominates (steepest negative slope... wait, no).
  Let's be concrete. At x = -5: terms are 0, -4, -7. Winner: 0.
  At x = -1: terms are 0, 0, 1. Winner: 3 + 2(-1) = 1.
  At x = 0: terms are 0, 1, 3. Winner: 3.
  At x = 2: terms are 0, 3, 7. Winner: 7.

The function is convex and piecewise-linear with kinks where two terms tie.

### Tropical hypersurfaces

The **tropical hypersurface** of a tropical polynomial is the set of points where
the maximum is achieved by **two or more** terms simultaneously. These are the kinks
— the points of non-differentiability.

In 1D: isolated points (the kinks).
In 2D: a graph (piecewise-linear curves).
In nD: a codimension-1 piecewise-linear complex.

**2D example.** Take p_trop(x, y) = max(0, 1 + x, 2 + y, 3 + x + y):

Each term is an affine function of (x, y). The tropical hypersurface is the set of
(x, y) where at least two terms tie for the max. This produces a network of line
segments partitioning R² into regions, one per "winning" term.

### Tropical rational functions

A tropical **rational** function is a difference of two tropical polynomials:

    r(x) = f(x) - g(x)

where f and g are both tropical polynomials (piecewise-linear convex). The
difference of two convex PL functions is still piecewise-linear, but **not**
necessarily convex. This is a **DC function** (difference of convex).

This distinction matters: neural networks compute tropical *rational* functions,
not tropical polynomials.

---

## 2. The Punchline: ReLU Networks = Tropical Rational Maps

Zhang, Naitzat & Lim (2018) prove:

> **Theorem (informal).** A feedforward neural network with ReLU activations and
> integer weights computes a tropical rational map. Conversely, any tropical
> rational map can be represented by such a network.

### Why this is true

A single ReLU neuron computes:

    max(0, w·x + b)

This is already a tropical polynomial in x — the max of the zero function and one
affine function. Two terms, one kink.

**Composing layers** builds up more complex tropical polynomials. The key insight:
when you compose piecewise-linear convex functions, the result is piecewise-linear
convex. Each layer increases the number of affine pieces.

**Negative weights** create the "rational" part. When a negative weight multiplies
a ReLU output, you get subtraction of a tropical polynomial, yielding f - g rather
than just f. This is why the full network is a tropical *rational* function, not
merely a tropical polynomial.

### Worked example: a 2→2→1 network

Consider a network R² → R² → R with:
- Layer 1: W₁ = [[1, 0], [0, 1]], b₁ = [0, 0] (identity + ReLU)
- Layer 2: w₂ = [1, -1], b₂ = 0

Layer 1 output: (max(0, x₁), max(0, x₂))

Layer 2 output:

    y = 1·max(0, x₁) + (-1)·max(0, x₂) + 0
      = max(0, x₁) - max(0, x₂)

This is a tropical rational function: f(x) - g(x) where:
- f(x) = max(0, x₁) — a tropical polynomial
- g(x) = max(0, x₂) — another tropical polynomial

The output is piecewise-linear with 4 regions (the four quadrants, determined by
signs of x₁ and x₂):

| Region | x₁ | x₂ | y |
|--------|----|----|---|
| Q1 | + | + | x₁ - x₂ |
| Q2 | - | + | -x₂ |
| Q3 | - | - | 0 |
| Q4 | + | - | x₁ |

The boundaries between regions are exactly the tropical hypersurfaces of f and g —
the lines x₁ = 0 and x₂ = 0.

---

## 3. Spline Microscope, Tropical Telescope

The tropical geometry framework (Zhang et al.) and the affine spline framework
(Balestriero & Baraniuk) describe the **same partition** of input space into
convex polytopes where the network is affine. They are not complementary theories
about different things — they are the same theorem, arrived at from different
starting points.

The value of having both is not that they describe different objects. It's that
they answer different **questions** well.

### The shared object

Both frameworks establish the same fact: a ReLU network partitions its input
space into regions, and within each region the network computes an affine function.

- Tropical: the output is a tropical rational function (DC = difference of convex
  PL functions). Each "cell" of the dual subdivision is one affine region.
- Spline: the output is f(x) = A[x]·x + b[x], where A[x] and b[x] are the slope
  matrix and offset for whichever region x falls into. The region is identified
  by the selection matrices T^(ℓ) (one-hot per neuron, recording which branch
  of each ReLU was taken).

These are the same partition. A "dual subdivision cell" and a "spline partition
region ω" are the same convex polytope, described differently.

### Where they differ: unit of analysis

The frameworks diverge in what they treat as the primitive building block.

**Spline = microscope (bottom-up, per-neuron).** The selection matrix T^(ℓ) records
each individual neuron's on/off state. You can trace a causal chain:

    weight w_{k,j} → neuron k's hyperplane → which side of the hyperplane → T^(ℓ)_k → A[x]

This gives you surgical access. When weight w_{k,j} gets quantized to w_q, exactly
one hyperplane rotates, and you can compute which points crossed it.

**Tropical = telescope (top-down, global-algebraic).** The Newton polytope and its
dual subdivision encode the partition's global combinatorial structure. A weight
perturbation changes the coefficients of the tropical polynomial. The zonotope
structure (Minkowski sum of generators, one per neuron) bounds the total error
region's shape and volume.

This gives you bounds and invariants. You can say "the error region is a zonotope
with at most N vertices" without tracking individual neurons.

### Which framework answers which question

| Question | Better framework | Why |
|---|---|---|
| Which neuron's boundary did this point cross? | **Spline** (T^(ℓ)) | T^(ℓ) is per-neuron; check which row changed |
| How does perturbing weight w_{k,j} move one hyperplane? | **Spline** (direct) | Normal vector rotates by δw; displacement is computable |
| What is the exact affine map at this input? | **Spline** (Jacobian) | A[x] = ∂f/∂x via one backward pass |
| What correction fixes the error within a stable region? | **Spline** (A[x] - A_q[x]) | Affine difference is exact and computable |
| What's the shape of accumulated error across layers? | **Tropical** (zonotope) | Minkowski sum of line segments = zonotope; direct from generators |
| How many regions can change under bounded perturbation? | **Tropical** (Newton polytope) | Coefficient perturbation → dual subdivision change → region count |
| What structural constraints exist on decision boundaries? | **Tropical** (DC decomposition) | f - g structure limits boundary topology |
| What is the volume of the error region? | **Tropical** (zonotope volume) | Closed-form from generator matrix determinants |

The pattern: **spline for "what happened to this specific point/neuron" and
"what's the exact local correction"; tropical for "what's the global shape/bound
of the error" and "how does the partition's combinatorics change."**

### Concretely: quantization error analysis

Consider quantizing a single layer's weights. Both frameworks see the same event:

**Spline microscope view:**
1. Weight matrix W → W_q. Each row k changes by δw_k.
2. Neuron k's activation boundary (a hyperplane in input space) rotates and shifts.
3. For each input x, check T^(ℓ)_k: did neuron k's on/off state change?
   - No change → error is exactly (A[x] - A_q[x])·x + (b[x] - b_q[x]).
     This is what `linear_error_norm` measures.
   - Change (flip) → error is discontinuous. This is what `relu_flip_rate` counts.
4. The correction for stable points is the affine residual — computable, correctable.

**Tropical telescope view:**
1. The tropical polynomial's coefficients change by amounts bounded by ||δW||.
2. The error region at this layer is a zonotope: Minkowski sum of segments
   {δw_k · x_k : x_k ∈ [0, 1]} for each neuron k that's active.
3. The zonotope's volume bounds the total error. Its shape (aspect ratio,
   principal axes via SVD of generators) reveals directional sensitivity.
4. The number of dual subdivision cells that changed bounds the fraction of
   input space where flips occur.

Neither view is redundant. The spline view tells you *which* neuron flipped and
*what* the correction is. The tropical view tells you *how big* the total error
region is and *how many* regions changed. In practice, you use the microscope
to diagnose and fix, and the telescope to bound and predict.

### Formal connection

For completeness, here is the dictionary:

| Tropical Geometry | Affine Spline | Notes |
|---|---|---|
| Tropical polynomial max_α(c_α + α·x) | Max-affine spline max_r(<a_r,x> + b_r) | Identical objects |
| Tropical hypersurface (kink locus) | Partition boundary (argmax switch) | Same set |
| Dual subdivision cell | Spline partition region ω | Same polytope |
| Activation pattern | Selection matrix T^(ℓ) | Same info; T^(ℓ) is per-neuron |
| Newton polytope | (No direct analog) | Combinatorial; tropical-only |
| Tropical rational function f - g | Composition of MASOs | Same DC structure |
| Zonotope (Minkowski sum) | (Implicit in layer composition) | Tropical names the geometry |
| Integer weight assumption | No weight constraint | Spline view is more general |

### What the spline view adds for this project

Three capabilities that matter for quantization and that are easier in spline
language:

**1. Exact per-point error computation.** For any input x, A[x] = ∂f/∂x via one
backward pass. The quantization error within a stable region is exactly
(A[x] - A_q[x])·x + (b[x] - b_q[x]). No approximation.

**2. The VQ interpretation.** Each spline partition region is a cluster. The
selection matrices T^(ℓ) are one-hot encodings — the network performs K-means-like
clustering at each layer. Quantizing weights perturbs the cluster boundaries.
Points near boundaries get reassigned — this is `relu_flip_rate`.

**3. Signal-dependent templates.** For a classifier:

    z^(L)(x) = W^(L) · (A[x] · x + B[x]) + b_W

Row c of W^(L)·A[x] is the "template" for class c, adapted to input x.
Classification = choosing the class whose template best matches x. Quantization
perturbs A[x] → A_q[x], changing the templates. Within a stable region this is
smooth; at partition boundaries (flips) the template changes discontinuously.

---

## 4. Newton Polytopes and Dual Subdivisions

### The geometry of activation patterns

For a tropical polynomial p(x₁, ..., xₙ) = max_α (cα + α·x), each term is
indexed by an exponent vector α ∈ Zⁿ.

The **Newton polytope** is the convex hull of all exponent vectors:

    Newt(p) = conv{α : cα ≠ -∞}

To get the **dual subdivision**: lift each exponent vector α to height cα in one
higher dimension, take the upper convex hull, and project the upper faces back
down. This produces a subdivision of the Newton polytope into cells.

Each cell corresponds to **one activation pattern** — a specific set of ReLU
neurons being on vs off. Within each cell, the network output is a single affine
function. The boundaries between cells are exactly the tropical hypersurface.

This is the geometric object that encodes the linear regions of the network.

### Why this matters for quantization

When you quantize the weights, you perturb the coefficients cα of the tropical
polynomial. This:
1. Shifts the boundaries of the dual subdivision (hypersurface moves)
2. Changes the affine function within each cell (slope/intercept change)
3. May merge or split cells (topology change)

Effects 1 and 2 are **linear** perturbations — correctable. Effect 3 is a
**combinatorial** change — the source of non-correctable error.

---

## 5. Mapping to What We Already Compute

The following table maps `aleph.qgeom` concepts to their names in both frameworks.

| Project Concept | Tropical Name | Spline Name | Where in Code |
|---|---|---|---|
| Minkowski sum of error boxes | Minkowski sum of zonotopes | (Same — zonotope structure) | `minkowski_sum_2d`, `geometry_2d.py` |
| Error parallelogram / polytope | Perturbation of tropical polytope | Perturbation of MASO parameters | `compute_manifold_errors()` |
| Fate: survive / dead / flipped | Activation pattern agreement | Same/different spline partition region | `compute_fate_metrics` in `metrics.py` |
| Error is linear within a region | Affine on dual subdivision cell | Affine on spline partition region ω | `linear_error_norm` in `LayerGeometryReport` |
| Flips are rare but discontinuous | Tropical hypersurface crossing | Selection matrix T^(ℓ) changed | `relu_flip_rate` in `compute_fate_metrics` |
| Condition number / anisotropy | Zonotope shape (SVD of generators) | Template norm ratios | `anisotropy_ratio` in `compute_geometry_metrics` |
| Error concentrates near boundaries | Points near tropical hypersurface | High local complexity regions | Visible in classification boundary plots |
| Correctability score | DC-decomposition regularity | Partition stability under perturbation | `compute_correctability_score()` |
| Collapse mass (dead + flip) | Lost mass under map perturbation | Fraction of VQ re-assignments | `collapse_mass` in `compute_fate_metrics` |
| Per-layer error box | Zonotope generator | MASO parameter perturbation | `get_box_vertices_2d` in `geometry_2d.py` |
| (Not yet computed) | — | Local complexity (partition density) | Could use Humayun et al. method |
| (Not yet computed) | — | Signal-dependent template A[x] | Jacobian via backpropagation |

### The zonotope connection, spelled out

In `aleph.qgeom`, error accumulates through layers as:

    E_cumulative = E₁ ⊕ (W₁⁻¹ · E₂) ⊕ (W₁⁻¹ · W₂⁻¹ · E₃) ⊕ ...

where ⊕ is Minkowski sum and each Eᵢ is an axis-aligned box (the per-layer
error region).

An axis-aligned box in Rⁿ is a zonotope with n generators aligned to the
coordinate axes. When you transform it by a weight matrix W, you get a zonotope
with n generators pointing in the directions of W's columns. The Minkowski sum of
zonotopes is again a zonotope.

So the cumulative error region is **exactly a zonotope** — not an approximation.
Its generators are the transformed per-layer error segments. The zonotope's
properties (number of vertices, face structure, volume) give tight bounds on the
complexity and magnitude of the error region.

This is what `minkowski_sum_2d` and `minkowski_sum_3d` compute. The functions work
with convex hulls of vertex sets, which is correct because zonotopes are convex.

---

## 6. What's NOT Useful — Rabbit Holes to Avoid

### 1. "Dequantization" ≠ neural network quantization

Maslov dequantization is a mathematical procedure: take classical algebra, apply
log base t, and let t → ∞. In the limit, (+, ×) becomes (max, +). This is how
tropical geometry emerges from classical algebraic geometry.

The word "quantization" here means "discretization of a continuous algebra."
Neural network quantization means "representing weights/activations with fewer
bits." **These are completely different uses of the same word.** Don't try to
connect them — it leads nowhere.

### 2. Amoebas, Viro's patchworking, Hilbert's 16th problem

Amoebas are images of algebraic varieties under the log-absolute-value map. They
converge to tropical varieties as t → ∞ in Maslov dequantization. Viro used
tropical geometry to construct real algebraic curves with prescribed topology
(relevant to Hilbert's 16th problem).

Beautiful pure mathematics. Zero relevance to ML quantization.

### 3. Tropical intersection theory

Bézout's theorem for tropical curves, stable intersections, intersection
multiplicities. This counts how tropical curves meet in tropical projective space.

Not useful for quantization error analysis. We care about how curves *move* under
perturbation, not how they intersect each other.

### 4. Integer weight assumption

Zhang et al.'s theorem technically requires integer weights. The proof works by
constructing explicit tropical monomials from the weight entries.

In practice: any real-valued weights can be approximated by rationals, and clearing
denominators gives integers (with a different tropical polynomial). **Do not try to
constrain your weights to be integers.** The geometric intuition — piecewise-linear
structure, activation pattern regions, boundary crossings — transfers to real
weights without any loss.

And conveniently, quantized weights *are* integers (on a grid), so the theorem
applies exactly to the quantized network.

### 5. Exact Newton polytope computation for real networks

For a network with layers of width d, the number of monomials in the tropical
polynomial can grow as O(∏ dₗ) — exponentially in depth. For the 2→8→8→1
classifier in `classification_quantization_geometry`, the tropical polynomial could
have up to ~2⁸ × 2⁸ = 65,536 monomials.

The value of the tropical framework is in the **geometric intuition and structural
theorems**, not in literally computing Newton polytopes for production networks.

### 6. Linear region counting for quantization

Zhang et al. derive bounds like O(n^{d(L-1)}) on the number of linear regions a
ReLU network can represent. This is about **expressiveness** — how complex a
function the network *can* compute.

It's tangential to quantization. We don't care how many regions the network has; we
care about how many regions **change** under quantization. The total count is an
upper bound on the changed count, but a very loose one.

---

## 7. What IS Useful — Actual Insights for Quantization

### Insight 1: Quantization error = perturbation of a tropical rational map

The float network computes a tropical rational function f - g. The quantized
network computes f' - g'. The error is:

    (f - g) - (f' - g') = (f - f') - (g - g')

This is itself a tropical rational function (difference of two PL convex
functions). Within each linear region where both the float and quant networks share
the same activation pattern, the error is **exactly affine** — a single linear map
plus a constant offset.

This is why `linear_error_norm` in `LayerGeometryReport` captures most of the
error: within each activation-pattern cell, the error IS linear. The nonlinear
component (`nonlinear_error_norm` from saturation, `relu_flip_rate` from flips)
comes from the boundaries.

### Insight 2: Flips = tropical hypersurface crossings

A neuron "flips" when quantization changes its ReLU decision — the float
pre-activation is positive but the quantized one is negative, or vice versa.

In tropical terms: the input point was near the tropical hypersurface (the ReLU
kink boundary), and the weight perturbation pushed the boundary across the point.
Or equivalently, the perturbation effectively moved the point across the boundary.

This is the **only** source of non-correctable error. `compute_fate_metrics`
already measures this as `relu_flip_rate`. The tropical framework tells us *why*
it's the critical metric: flips are combinatorial changes to the dual subdivision,
while everything else is a continuous perturbation within a fixed cell.

### Insight 3: Zonotopes explain the Minkowski sum structure

The error region at layer l is a zonotope — a Minkowski sum of line segments, one
per neuron. This is not an approximation; it's the exact geometry.

Key zonotope properties:
- **Vertex count**: a zonotope with n generators in Rᵈ has at most 2·C(n-1, d-1)
  vertices. For the 2D case (`geometry_2d.py`), this means O(n) vertices — linear
  in the number of neurons.
- **Volume**: determined by the generators' lengths and relative angles. Parallel
  generators don't increase volume (degenerate direction). Orthogonal generators
  maximize volume.
- **Face structure**: each face of a zonotope corresponds to a subset of generators.
  The face lattice encodes which neurons contribute to each "extreme" error
  direction.

The `anisotropy_ratio` metric already captures the zonotope's elongation (ratio of
longest to shortest axis). The tropical framework says this is the ratio of the
largest to smallest generator projections.

### Insight 4: Decision boundaries are tropical hypersurfaces

For binary classification (the 2→8→8→1 networks in
`classification_quantization_geometry`), the decision boundary is:

    {x : network(x) = threshold}

Since network(x) is a tropical rational function, this level set is a subset of a
tropical hypersurface. It's a piecewise-linear curve (in 2D input space) or
surface (in higher dimensions).

Quantization perturbs the coefficients of the tropical polynomial, which shifts
the hypersurface. The maximum shift is bounded by the coefficient perturbation —
which is directly related to the quantization step size δ and the weight
magnitudes.

### Insight 5: DC structure constrains decision regions

Since the network output = f(x) - g(x) where f, g are convex PL, on each cell
where f is affine and g is affine, the output is affine. But at a coarser level:
within each convex region of f's subdivision, -g is concave, so the output is
concave there.

This means: within each "f-cell," the set {x : output ≥ 0} is convex. Decision
regions are unions of convex pieces. This constrains the possible shapes of
decision boundaries and limits how badly quantization can distort them.

---

## 8. Paper-by-Paper Insights

### Paper 3: "A Spline Theory of Deep Learning" — Balestriero & Baraniuk (2018)

**arXiv 1805.06576 | ICML 2018**

This is the foundational paper for the affine spline view. Key results for us:

**Result 1: DNs are compositions of MASOs → the output is piecewise affine.**
Every standard DN (CNN, ResNet, fully connected, etc.) with piecewise-affine
activations can be written as f(x) = A[x]·x + b[x], where the matrix A[x] and
vector b[x] depend on which partition region x falls in. This is not an
approximation — it's an identity.

**Result 2: The partition is implicitly defined by the parameters.**
Unlike classical splines where you must specify the knots, a max-affine spline's
partition is determined entirely by its slope and offset parameters. Changing
weights → changing partition. This is exactly what quantization does: perturbing
weights simultaneously perturbs both the per-region affine maps AND the partition
boundaries.

**Result 3: Two MASOs suffice for universal approximation.**
A composition of just two MASOs (one hidden layer + one linear layer) can
approximate any continuous operator arbitrarily closely. The approximation error
scales as O(1/D^(1)) where D^(1) is the width of the hidden layer. This has a
direct quantization implication: wider networks have more partition regions and
finer-grained affine approximations, so each individual region is less critical
— making quantization less destructive.

**Result 4: The Lipschitz constant decomposes per-layer.**
κ = ∏ κ^(ℓ), where each κ^(ℓ) depends on the operator norms of the weight
matrices and activation functions. This gives a quantitative bound on how much
output can change per unit of weight perturbation — directly applicable to
bounding quantization error.

**Result 5: Partition regions can be empty.**
In their toy experiments (2D input, 45 hidden units), most of the theoretically
possible 2^45 partition regions have zero volume, and many non-trivial regions
contain no training data. This explains why pruning (and by extension,
quantization) can be so aggressive without destroying accuracy: the network isn't
using most of its combinatorial capacity.

**Relevance to our work:** The MASO framework gives us the **computational**
counterpart to the tropical framework's **algebraic** characterization. We can
compute A[x] via backpropagation, compute the selection matrices T^(ℓ) to
identify partition regions, and use the VQ distance to measure how quantization
changes the clustering of inputs.

### Paper 4: "Max-Affine Spline Insights Into Deep Network Pruning" — You, Balestriero et al. (2022)

**arXiv 2101.02338 | TMLR 2022**

This paper applies the spline framework specifically to network compression
(pruning), which is the closest existing work to our quantization geometry
analysis.

**Key insight 1: Pruning affects both the partition AND the per-region maps.**
A pruning mask Q applied to weights W^(ℓ) simultaneously changes:
- The per-region affine mappings A_ω, b_ω (directly, through zeroed weights)
- The partition Ω itself (indirectly, because partition boundaries depend on W)

This is identical to what quantization does. The paper formalizes what our
`compute_fate_metrics` measures empirically: weight perturbation simultaneously
changes the function within each region AND which region each point falls in.

**Key insight 2: Node pruning removes partition boundaries; weight pruning rotates them.**
- **Node (unit) pruning** removes entire "subdivision lines" from the partition.
  If you prune neuron k, you remove the hyperplane {x : w_k·x + b_k = 0} from
  the partition entirely. This is a coarse, structural change.
- **Weight pruning** alters the orientations of existing partition boundaries
  without removing them (unless an entire row is zeroed). It "rotates" the
  subdivision lines, potentially aligning them with coordinate axes.

**Quantization analog:** Weight quantization is more like weight pruning than node
pruning — it perturbs individual entries of W, rotating the partition boundaries
by small amounts. The flip rate measures how many data points cross a rotated
boundary. Saturation (clipping to quantization range) is more like a soft version
of node pruning, effectively zeroing the excess signal.

**Key insight 3: Partition convergence happens early in training (the "early bird" phenomenon).**
The spline partition stabilizes early in training, long before the weights
converge. This means the combinatorial structure (which neurons are on/off for
each input) is established early, and later training only refines the affine maps
within fixed regions. For quantization, this suggests that the partition structure
of a trained network is robust — most of the "important" partition boundaries are
well-established and have margin, making flips under small perturbation rare.

**Key insight 4: The number of regions bounds what pruning can preserve.**
For unit pruning that removes p of K neurons, the maximum number of partition
regions drops from 2^K to 2^(K-p). This is a tight bound. For weight pruning, the
bound is looser — you can zero many individual weights without losing any regions,
as long as no complete row is zeroed.

**Relevance to our work:** This paper's analysis of pruning-as-partition-perturbation
maps directly onto quantization-as-partition-perturbation. The key differences:
quantization perturbs all weights simultaneously (not selectively), and the
perturbation magnitude is bounded by δ/2 (the quantization step). Their
visualization of "subdivision lines" in 2D input space is exactly what our
Experiments 2 and 4 in the roadmap below would produce.

### Paper 5: "Deep Networks Always Grok and Here is Why" — Humayun, Balestriero & Baraniuk (2024)

**arXiv 2402.15555 | ICML 2024**

This paper introduces a spatial measure of the spline partition's density —
**local complexity** — and discovers that partition regions migrate during
training. While not about quantization directly, the insights are highly relevant.

**Key insight 1: Local complexity = partition region density around a point.**
Given an input x, sample directions around it and count how many neuron
hyperplanes intersect a local neighborhood. More hyperplane crossings = higher
local complexity = the network's function is more nonlinear near x.

This is directly related to flip vulnerability: points in high local complexity
regions are surrounded by many partition boundaries, so small weight perturbations
(quantization) are more likely to push a boundary across them. `relu_flip_rate`
should correlate with local complexity.

**Key insight 2: Region migration — partition boundaries move toward decision boundaries.**
During training, partition regions undergo a phase transition:
1. **Memorization phase:** partition regions are dense everywhere, including around
   training points. The network is "locally nonlinear" near the data.
2. **Migration phase:** regions migrate away from training data and concentrate
   near the decision boundary. The network becomes "locally linear" near data
   points and "locally nonlinear" near the boundary.
3. **Grokking phase:** once regions have migrated, the network generalizes (and
   becomes robust to adversarial examples).

**Quantization implication:** After region migration, training points live in large,
stable partition regions where the network is locally linear. This is exactly the
condition for quantization to be benign — within a large stable region, the error
is purely affine and correctable. The only vulnerable points are near the decision
boundary, where partition regions are dense and small — and these are exactly the
points where flips would cause misclassification anyway.

This explains why well-trained networks tend to quantize better than undertrained
ones: the training process itself creates the conditions (large stable regions
around data) that make quantization safe.

**Key insight 3: Batch normalization prevents grokking by keeping the partition uniform.**
BN acts as a regularizer on the partition structure, preventing the concentration
of regions near the decision boundary. This keeps the local complexity uniform
across the input space. For quantization: BN may make the partition more uniform
but also more fragile (more small regions near data), potentially increasing flip
rates. This is testable.

### Paper 6: "SplInterp: Sparse Autoencoders via Spline Theory" — Budd et al. (2025)

**arXiv 2505.11836 | NeurIPS 2025**

This paper characterizes sparse autoencoders (SAEs) as piecewise-affine splines
and connects their partition geometry to power diagrams (weighted Voronoi
diagrams).

**Key insight 1: TopK SAE partitions are K-th order power diagrams.**
For a TopK SAE with encoder W_enc and bias b_enc, the partition of input space
(which K of d neurons are active) forms a K-th order power diagram with centroids
μ_i = W_enc^T e_i and weights α_i = 2(b_enc)_i + ||W_enc^T e_i||². The
centroids are the rows of the encoder weight matrix.

This is a direct generalization of the K-means / Voronoi connection from
Balestriero & Baraniuk (2018) — and it extends to *weighted* Voronoi diagrams
where the bias terms act as weights, shifting the partition boundaries.

**Quantization implication:** When we quantize an SAE's encoder weights, we perturb
the power diagram centroids by at most δ/2 per coordinate. The partition cells
shift accordingly. Points near cell boundaries (where two sets of K neurons are
close to tied for the top-K) are most vulnerable. The analysis framework for
predicting flips (open question 1 in Section 9) could be made concrete using
power diagram geometry: the flip region is exactly the set of points within
distance O(δ) of a power diagram cell boundary.

**Key insight 2: SAEs generalize k-means autoencoders to piecewise affine.**
A k-means autoencoder assigns each input to a cluster and reconstructs with the
cluster centroid (constant per region). An SAE does the same but reconstructs
with an affine function per region (using the active neurons' decoder columns).
The SAE sacrifices the optimality of k-means centroids for the expressiveness
of per-region affine maps.

This is precisely the MASO structure applied to autoencoders rather than
classifiers. The quantization geometry of SAEs follows the same rules: within
each partition region, error is affine; at boundaries, error is discontinuous.

---

## 9. Experimental Roadmap

Ordered from simplest to most realistic. Each builds on the previous.

### Experiment 1: Single neuron tropical polynomial (1D)

**Setup:** One neuron computing max(0, wx + b).

**Steps:**
1. Plot the function for w = 1.7, b = -0.5 (a tropical polynomial in one variable)
2. Quantize w to grid δ = 0.5: w_q = 1.5. Plot the new function.
3. The kink moves from x = 0.5/1.7 ≈ 0.294 to x = 0.5/1.5 ≈ 0.333
4. Overlay both functions. The error is zero for x < 0.294, linear for x > 0.333,
   and a flip region in [0.294, 0.333].

**Tools needed:** matplotlib only. No `aleph.qgeom` required.

**Payoff:** Visceral understanding of "tropical hypersurface = ReLU kink" and
"quantization moves the kink."

### Experiment 2: Single layer, 2D input (R² → R²)

**Setup:** A 2→2 network with one ReLU layer.

**Steps:**
1. Choose W = [[1.7, 0.3], [-0.4, 1.2]], b = [0, 0]
2. Each output neuron defines a tropical polynomial in (x₁, x₂)
3. Draw the tropical hypersurfaces: the lines in R² where each neuron's
   pre-activation equals zero (the ReLU decision boundaries)
4. Quantize W with δ = 0.5. Draw the new hypersurfaces.
5. Color input points: green = same activation pattern, red = different (flipped)
6. The red region is exactly the strip between the old and new hypersurfaces.

**Tools needed:** `aleph.qgeom.quantize`, matplotlib.

**Payoff:** Flipped points are exactly those that crossed a tropical hypersurface.

### Experiment 3: Two layers, 2D input (R² → R² → R¹)

**Setup:** Compose two layers to get a tropical rational function.

**Steps:**
1. Layer 1: W₁ (2×2), ReLU. Layer 2: w₂ (1×2), no activation.
2. The output is f(x) - g(x) where f, g are tropical polynomials
   (positive and negative weight contributions)
3. Draw both tropical hypersurfaces (from f and from g)
4. Draw the decision boundary (where f = g, if used for classification)
5. Quantize both layers. Show how all three curves shift.
6. Identify the "flip strip" between old and new boundaries.

**Tools needed:** `aleph.qgeom.quantize`, `aleph.qgeom.compute_fate_metrics`.

**Payoff:** See the DC decomposition in action. The decision boundary is where two
tropical hypersurfaces interact.

### Experiment 4: Dual subdivision visualization

**Setup:** For the 2-layer network from Experiment 3, visualize the activation
pattern partition of input space.

**Steps:**
1. Sample a dense grid in input space
2. For each point, record which neurons are active (the activation pattern)
3. Color the grid by activation pattern — this IS the dual subdivision
4. Side-by-side: float network tiling vs quantized network tiling
5. Highlight cells that changed (merged, split, appeared, disappeared)
6. Overlay the tropical hypersurfaces — they should be exactly the cell boundaries

**Tools needed:** `aleph.qgeom.compute_fate_metrics`, dense grid evaluation.

**Payoff:** Activation pattern changes become visible as combinatorial changes in a
tiling. The tiling IS the dual subdivision.

### Experiment 5: Connect to existing weight_quantization_geometry analysis

**Setup:** Use the existing `run_experiment` and `run_all_manifolds` infrastructure
from `weight_quantization_geometry`.

**Steps:**
1. Take the existing Minkowski sum error computation
2. Label the error polytopes as zonotopes (they already are)
3. Compute zonotope generators explicitly: for each neuron in each layer, the
   generator is the column of the weight matrix scaled by δ/2 · |input|
4. Show that error volume = zonotope volume (from `compute_polygon_area`)
5. Relate anisotropy to the generator angle distribution: more parallel generators
   → higher anisotropy → more directional error

**Tools needed:** `aleph.qgeom.run_experiment`, `aleph.qgeom.run_all_manifolds`,
existing notebook infrastructure.

**Payoff:** The existing analysis gets a precise mathematical name. "Error region is
a zonotope" is not a metaphor — it's a theorem. And zonotope theory gives us tools
(generator analysis, face enumeration) that go beyond what we currently compute.

### Experiment 6: Tropical view of classification geometry

**Setup:** Use the trained 2→8→8→1 classifier from
`classification_quantization_geometry`.

**Steps:**
1. For each point on a dense grid, record the full activation pattern (which of the
   16 hidden neurons are active)
2. Draw the activation pattern map in input space — a piecewise-linear tiling
3. Compare float vs quantized tilings
4. The disagreement regions (where patterns differ) = where flips happen
5. Compute flip rate as the area fraction of disagreement regions
6. Overlay the decision boundaries. Flips near the decision boundary cause
   misclassification; flips far from it don't.

**Tools needed:** `aleph.qgeom.quantize`, `aleph.qgeom.compute_fate_metrics`, the
trained models from `classification_quantization_geometry`.

**Payoff:** The `relu_flip_rate` metric gets a geometric interpretation: it's the
fractional area of input space where the dual subdivision changed. And we can see
that not all flips are equal — only flips near the decision boundary affect
accuracy.

### Experiment 7: Local complexity as flip predictor

**Setup:** Use the local complexity approximation from Humayun et al. (2024) on
the 2→8→8→1 classifier.

**Steps:**
1. For each point on a dense grid, approximate local complexity: sample P random
   directions, form a local polytope, count how many neuron hyperplanes intersect it
2. Produce a heatmap of local complexity across input space
3. Overlay with the flip map from Experiment 6 (which points flipped under quant)
4. Measure the correlation: do high-LC points flip more often?
5. Compare local complexity with distance-to-decision-boundary as flip predictors

**Tools needed:** `aleph.qgeom.quantize`, `aleph.qgeom.compute_fate_metrics`, custom
local complexity computation (straightforward: forward pass + sign counting).

**Payoff:** Validates whether local complexity is a useful proxy for quantization
vulnerability. If correlated, LC could replace Monte Carlo flip rate estimation with
an analytical (or cheaper) measure.

### Experiment 8: Jacobian-based quantization error decomposition

**Setup:** Use the spline view to decompose quantization error into partition-stable
vs partition-changing components.

**Steps:**
1. For the 2→8→8→1 classifier, compute A[x] = ∂f/∂x (Jacobian) for float and
   quantized networks at each point on a grid
2. For points where the activation pattern is identical (no flip): error is exactly
   (A[x] - A_q[x])·x + (b[x] - b_q[x]). Verify this.
3. For points where a flip occurred: measure the discontinuity in A[x] — the
   difference between the affine map on each side of the boundary
4. Plot the Jacobian difference ||A[x] - A_q[x]||_F as a function of input position

**Tools needed:** PyTorch autograd for Jacobian computation, existing classifier.

**Payoff:** Direct visualization of the spline perturbation theory. Confirms that
within-region error is exactly affine and that flips are the only source of
nonlinear error — the central claim of both frameworks.

---

## 10. Open Questions

Things tropical geometry might help answer, but we don't know yet:

1. **Predicting which points will flip.** Can we identify the "vulnerable" region
   near each tropical hypersurface *before* quantizing, using only the float
   weights and the quantization grid? The hypersurface location is known (it's
   where the pre-activation equals zero); the perturbation magnitude is bounded by
   δ. So the flip region should be a strip of width O(δ) around each hypersurface.

2. **Zonotope-optimal quantization.** Current quantization minimizes weight MSE
   (or some proxy). But the actual error region is a zonotope whose shape depends
   on the weight perturbation direction, not just magnitude. Could we quantize to
   minimize zonotope volume (or some other geometric measure) instead?

3. **Analytical flip volume estimates.** The fraction of input space that flips
   should be related to: (a) the total "surface area" of tropical hypersurfaces in
   the input region, times (b) the strip width O(δ). This would give an analytical
   formula for flip rate as a function of bit-width, without needing Monte Carlo
   estimation.

4. **Tropical characterization of quantizability.** Is there a property of the
   tropical polynomial (number of monomials, Newton polytope shape, coefficient
   distribution) that predicts how well a network quantizes? Networks whose
   tropical hypersurfaces have large "margin" (distance from data to nearest kink)
   should be more robust to quantization.

5. **Layer-wise tropical complexity.** Each layer increases the number of linear
   regions (monomials in the tropical polynomial) multiplicatively. Does this mean
   deeper networks are inherently harder to quantize, or does the actual complexity
   (number of *active* regions on the data manifold) grow much more slowly?

6. **Does region migration make networks quantization-friendly?** Humayun et al.
   show that training pushes partition boundaries away from data and toward the
   decision boundary. Does this mean that longer training (past generalization)
   improves quantization robustness? If so, there might be a "grokking for
   quantization" phenomenon where quantized accuracy improves long after float
   accuracy has converged.

7. **Can we use the Jacobian A[x] to design better correction modules?** The
   spline view gives us the exact linear map within each region. If we know A[x]
   and A_q[x], we know the exact correction needed: multiply by
   A[x]·A_q[x]^(-1). Is this more efficient than learning the correction from
   data?

8. **Power diagram analysis of quantization in SAEs.** SAE partition cells are
   power diagram cells with explicit centroids and weights (Budd et al., 2025).
   Quantizing the encoder shifts centroids by at most δ/2. Can we compute the
   flip volume analytically from the power diagram geometry?

---

## 11. Summary

The connection between this project and the tropical/spline frameworks is not
a metaphor or an analogy. It is a mathematical identity, visible from two
complementary angles:

**From tropical geometry (Zhang et al., 2018):**
- ReLU networks compute tropical rational functions
- Quantization perturbs the coefficients of these tropical functions
- The error regions (Minkowski sums of boxes through layers) are zonotopes
- Activation pattern changes (flips) are crossings of tropical hypersurfaces
- Decision boundaries are subsets of tropical hypersurfaces

**From spline theory (Balestriero et al., 2018):**
- ReLU networks are compositions of max-affine spline operators (MASOs)
- The network output is a signal-dependent affine transformation f(x) = A[x]·x + b[x]
- The input space is partitioned into convex regions where f is affine
- Quantization simultaneously perturbs both the per-region maps and the partition
- The partition acts as a vector quantization; flips are VQ re-assignments

**From the follow-up work:**
- Pruning and quantization have the same geometric effect: perturbing the spline
  partition and its per-region affine maps (You et al., 2022)
- Well-trained networks develop large stable partition regions around training
  data, making quantization naturally safe (Humayun et al., 2024)
- Local complexity (partition density) predicts vulnerability to perturbation
- SAE partitions are power diagrams, giving explicit geometric tools for
  analyzing quantization of sparse representations (Budd et al., 2025)

The `aleph.qgeom` library already computes the right objects. The tropical
framework gives them algebraic names and structural theorems. The spline framework
gives them computational tools and connects them to the broader theory of
approximation, VQ, and matched filtering. Together, they provide a complete
geometric theory of quantization error.
