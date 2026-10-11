# The Geometry of a Neural Network

## 1. Statement of the Question

What does it mean to talk about **the geometry of a neural network**?

Not the geometry of a single weight matrix (its singular spectrum), or a single
activation (its eff_rank), or a single input (its pullback curvature). Those are
local probes — they describe a *neighbourhood*, not the *function*. The geometry
of a neural network is the global object: the partition of the input domain into
regions where the network behaves coherently, the smooth metric on the embedding
space induced by the function, the boundaries between regions, and the way these
objects interact with the data the network was trained on.

A weight matrix W ∈ ℝ^{m×n} has a singular value decomposition. That tells you
about one linear map. But a neural network f: ℝ^d → ℝ^V is not a linear map —
it is a composition of affine maps interleaved with nonlinearities. The
nonlinearities create *structure*: they partition the input space into regions
where the network computes a fixed affine function, or (for smooth activations)
where the network is *approximately* affine. The totality of these regions —
their count, their sizes, their shapes, their adjacency, the affine maps they
carry — is the geometry of the network.

This document formalizes two complementary ways of looking at this geometry,
reviews what has already been measured in 366 autoresearch cycles on GPT-2, and
identifies the formal questions that remain open. It is a thinking document, not
a plan. The goal is to commit to a definition of what we mean by "the geometry
of this network" before running another experiment.

### Why local probes are insufficient

This project has run 366 cycles of automated experiments. The strongest
empirical findings are:

- Salience-curvature anticorrelation ρ = −0.745 (cycle 135). A *per-block*
  measurement: it tells you which blocks bend the residual stream the most, but
  not *what they bend it around*.
- Block 11 rank expansion 2 → 231 (cycle 201). A *per-block* measurement: it
  tells you block 11 decompresses, but not what the decompressed signal looks
  like in the space of possible outputs.
- Error inflection at b10–b11 accounting for 77% of total compression error
  (cycle 366, 3 seeds). A *per-block* measurement: it tells you where error
  accumulates under Frobenius, but not whether the error is geometrically
  meaningful (output-probability-shifting) or geometrically trivial (orthogonal
  to the output manifold).
- Centred Δ̃ alignment to downstream readers at 11/11 blocks (cycle 218). A
  *pairwise-block* measurement: it tells you that the nonlinear contribution of
  each FFN is structurally aligned with the next block's reader, but not what
  that alignment looks like globally.

Each of these is a projection of the full geometric picture onto one or two
scalar quantities. We have been studying shadows, not the object.

---

## 2. Two Complementary Formalisms

The geometry of a neural network can be formalized in two ways. They are not
competing theories — they are two views of the same object, related by a
limiting argument. Modern transformers (GELU FFNs + softmax attention) require
both.

### 2a. The Combinatorial View: MASO Partitions (Balestriero)

**Reference:** Balestriero & Baraniuk 2018 ("Mad Max: Affine Spline Insights
into Deep Learning"), Balestriero 2020 ("A Spline Theory of Deep Learning").
Survey: `autoresearch/surveys/balestriero_spline_theory.md`.

A ReLU network is a continuous piecewise-linear (CPWL) function. It partitions
its input space into convex polyhedral cells — **linear regions** — each
carrying a fixed affine map. Formally:

For a depth-L ReLU network with weights W^1, ..., W^L and biases b^1, ..., b^L,
the **activation code** at input x is the binary vector:

    Q(x) = (q^1(x), q^2(x), ..., q^L(x))  ∈ {0,1}^{Σ_l n_l}

where q^l_i(x) = 1 iff the i-th neuron at layer l fires (pre-activation > 0).

Each distinct activation code Q defines a **cell** ω_Q — the set of all inputs
x that produce code Q. Inside cell ω_Q the network computes:

    f(x) = A_ω x + c_ω
    A_ω = W^L · diag(q^L) · W^{L-1} · diag(q^{L-1}) · ... · W^1 · diag(q^1)

The **geometry of the network** under this view is the **cell complex**: the set
of all cells {ω_Q}, their volumes vol(ω_Q), the adjacency graph (which cells
share a boundary), and the affine maps {(A_ω, c_ω)} they carry.

**SplineCam** (Balestriero et al.) enumerates this cell complex exactly in 2D.
For Fashion-MNIST MLP networks, it found ~1,364 active cells out of a
theoretical maximum of >> 10^100. The ratio (active / possible) is a measure of
the network's geometric complexity.

**For smooth activations (GELU, SwiGLU):** the partition picture is approximate
but not vacuous. Pre-activation sign patterns still partition input space, and
inside each sign-pattern region the network is *approximately* affine, with
approximation error controlled by the curvature residual δ_σ (see §3). The cell
boundaries are no longer hyperplanes but smooth hypersurfaces where the
pre-activation crosses zero. The combinatorial structure (cell count, adjacency)
still exists; the affine maps become smooth maps that vary within each cell.

**What cycles 001–025 found, and what they got wrong:**

Cycles 001–025 attempted MASO-style analysis on GPT-2. Cycle 134 reported:
"99.99% unique activation patterns, no Jacobian clustering." This was taken as
evidence that the MASO framework is vacuous for transformers. But the
measurement was wrong — not the framework. The probing methodology was:

> For each token in a corpus, compute the activation code at each block. Count
> distinct codes.

This probes the function at *discrete token endpoints* — a finite sample from
the input manifold. It answers "how many distinct cells does the corpus visit?"
(answer: almost all cells are singletons). It does NOT answer the structural
question: "what does the cell complex look like *between* tokens?" The right
probe is along **curves** in the embedding space: interpolate between two token
embeddings, walk the interpolation path, count how many cell boundaries the path
crosses, and ask whether the crossings are structured or random. This was never
done.

The fact that each token lands in its own cell is expected — the embedding
dimension is 768, the number of neurons per block is 3072, and the capacity for
distinct cells is astronomical. The interesting question is not "how many cells
are visited?" but "what is the *local* structure of the cell complex around each
token?" — i.e., how many cells are *reachable* by small perturbations, how are
they arranged, and what happens to the affine map when you cross a boundary.

### 2b. The Smooth View: Pullback Metric (Manson)

**Reference:** Manson 2025 (arXiv 2507.21107, "Curved Inference: Geometric
Interpretability of Residual Stream Trajectories"). Implementation:
`aleph/pullback_geometry.py`.

The network induces a metric on the residual stream space via its output map.
For an LLM with unembedding matrix U ∈ ℝ^{V×d}, the **pullback metric** is:

    G = U^T U  ∈ ℝ^{d×d}

Under this metric, distances in the residual stream correspond to shifts in
output token probabilities. If two residual vectors differ by δ, the induced
distance is:

    ||δ||_G = ||U δ||_2

which measures how much the output logit vector shifts — a semantically
meaningful quantity.

The **salience** of block l is the per-layer velocity under G:

    S_l = ||x^{l+1} − x^{l}||_G

where x^{l} is the residual stream after block l. Salience measures how much
each block changes the output distribution.

The **curvature** κ_l is computed via 3-point central differences:

    v_l = (x^{l+1} − x^{l−1}) / 2          (tangent)
    a_l = x^{l+1} − 2x^{l} + x^{l−1}       (acceleration)
    κ_l = √(||a||²_G ||v||²_G − ⟨a,v⟩²_G) / ||v||³_G

Curvature measures how much the residual stream trajectory *bends* at block l.
Cycle 135 found that salience and curvature are anticorrelated across blocks
(ρ = −0.745, p = 0.0085, n = 11): high-salience blocks tend to have low
curvature (they move the output distribution a lot, but in a straight line),
while high-curvature blocks have low salience (they bend the trajectory sharply
but move the output distribution little).

**What this metric is, and what it is not:**

G = U^T U is *one* metric choice, not *the* metric. Other viable choices that
this project has not explored:

- The **Fisher information metric** on the parameter manifold (related to G but
  accounts for the probability simplex structure).
- The **input-Jacobian metric** J^T J where J = ∂f/∂x (measures sensitivity to
  input perturbations, not output shifts).
- The **per-block FFN-Jacobian metric** J_l^T J_l where J_l = ∂(block_l output)
  / ∂(block_l input) (measures per-block stretching).

Each metric tells you something different about the network's geometry. The
project has so far committed exclusively to G = U^T U. This choice is well-
motivated (output probabilities are what matter for language modelling) but it is
not the only game in town.

**Cycle 136's refutation, and its limits:**

Cycle 136 tested whether G-weighted least squares produces better compression
than Frobenius least squares. It found: REFUTED. The ratio ||e||_G / ||e||_F was
34–35× and nearly constant across blocks (ρ(Frobenius, PPL) = 0.797 vs
ρ(G-weighted, PPL) = 0.734). The pullback metric is isotropic at the FFN output
level — Frobenius is sufficient for *per-block lstsq regression*.

But this refutation is scoped narrowly. It says: "at the level of a single
block's linear regression, G and Frobenius give the same ranking of solutions."
It does NOT say: "G and Frobenius give the same ranking of *multi-block*
compression errors." The isotropic scaling might break at the multi-block level,
where error propagates through nonlinearities that can concentrate energy in
specific G-directions. This has never been tested.

### 2c. Their Relationship

The smooth Riemannian picture is the *limit* of the MASO picture as cells become
small and dense.

**For a strict ReLU network:** MASO is exact. The network is piecewise-affine;
inside each cell the Jacobian is constant (= A_ω) and the Riemannian curvature
is exactly zero. All curvature lives on cell boundaries — it is a distribution
(delta function) supported on the hyperplane arrangement {w_i^l · z = 0}. The
pullback metric G = U^T U applied to A_ω gives the per-cell metric tensor
A_ω^T G A_ω, which measures how much the cell's affine map distorts output
probabilities. The Riemannian picture is degenerate (zero curvature except on
boundaries) but the pullback magnitude ||A_ω||_G is a per-cell scalar that
connects the two views.

**For a smooth-activation network (GELU, SwiGLU):** the Riemannian picture is
the natural one. The Jacobian varies smoothly everywhere; curvature is a smooth
field. The MASO partition is approximate: sign-pattern regions still exist, and
inside each region the map is *close to* affine (with error δ_σ), but the
boundaries are smooth hypersurfaces, not flat hyperplanes. The cell complex
degrades into a "fuzzy" partition where each cell has a "core" (where the map is
nearly affine) and a "boundary zone" (where the curvature residual δ_σ is
large). The Riemannian curvature κ at a point x is related to the density of
cell boundaries near x — high κ means many nearby regions with different affine
maps; low κ means the function is locally linear over a large neighbourhood.

**For modern transformers (GELU FFNs + softmax attention):** both views matter.
The FFN blocks are smooth nonlinearities (GELU) whose geometry is captured by
κ, S, and the δ_σ framework. The attention blocks are a different geometric
object — they implement a data-dependent routing that changes the *topology* of
the computation graph (which residual positions communicate). The residual stream
itself is a shared linear space through which all blocks communicate. The
geometry of the full network is the interplay of FFN curvature, attention
routing, and residual stream topology. This project has so far studied only the
FFN component (blocks 0–11 of GPT-2, ignoring attention). The formalization must
acknowledge this gap.

---

## 3. The Curvature Residual: δ_σ

**Reference:** `docs/gelu_bump_audit.md` §12 (postscript, 2026-04-08). Cycle
302 unified module: `experiments/unified_curvature_cycle302.py`.

The 200-cycle "GELU bump" corpus studied a phenomenon that was given the wrong
name. The actual geometric object is the **curvature residual function**:

    δ_σ(z) = σ(z) − σ_lin(z)

where σ is the activation function and σ_lin is its linear approximation (the
identity for GELU→ReLU; the ReLU for ReLU²→ReLU; etc.). The pair (GELU, ReLU)
is one instance of a function family parameterized by (σ, σ_lin).

For an FFN block with weights W_fc, W_proj and activation σ:

    f_block(x) = W_proj · σ(W_fc · x + b)

The contribution of the nonlinearity relative to the linear reference is:

    Δy(x) := f_σ(x) − f_{σ_lin}(x) = W_proj · δ_σ(z(x))

where z(x) = W_fc · x + b is the pre-activation. Δy(x) is the per-token
residual-stream perturbation due to the curvature of the activation function.

**Connection to the MASO picture:** δ_σ is the *local linearization defect*.
Where δ_σ(z) ≈ 0, the FFN is locally affine — we are in the interior of a MASO
cell (in the smooth limit). Where δ_σ(z) is large, we are near a cell boundary
— the function is bending, the Jacobian is changing, and the MASO approximation
breaks down. The magnitude ||Δy(x)||_G measures, in output-probability units,
how much nonlinearity the block applies to token x.

**Five geometric measurements of Δy** (from gelu_bump_audit.md §12):

1. **Rank:** SVD of [Δy(x_t)]_t over a calibration corpus. The rank_95 value
   measures the intrinsic dimensionality of the nonlinear contribution. Cycle
   217: rank_95 ≤ 64 in 9/12 blocks for (GELU, ReLU); cycle 302 reproduced
   this. However, the audit downgraded to INCONCLUSIVE because raw rank_95 was
   dominated by the token-independent mean (TI/TD ratio 0.87–0.99) — the
   token-*dependent* structure is lower-rank.

2. **Subspace alignment:** Principal cosines between the column space of
   [Δy(x_t)]_t and the row space of downstream readers (W_fc of the next block,
   Q/K/V heads, unembedding U). Cycle 218: c_fc alignment significant at 11/11
   blocks (top-5 cos_sum ≥ 2.0, σ-excess +17σ to +56σ above random baseline).
   The nonlinear contribution is structurally pre-aligned with what the next
   block will read.

3. **Per-token magnitude under G:** Distribution of ||Δy(x)||_G across tokens.
   Measures how much "curvature" each token experiences. This has been partially
   measured (sections 4–5 of gelu_bump_audit.md used scalar norms) but never
   fully characterized under the pullback metric.

4. **Pullback κ contribution:** Patch the block with σ_lin (replacing σ with its
   linear reference), recompute the residual trajectory under G, and measure the
   per-layer κ shift. Cycle 258: +47% κ kink at layer 4 for (GELU, ReLU+b_leak).
   This measures how much each block's nonlinearity contributes to the global
   curvature of the residual stream.

5. **Energy decomposition vs frozen subspace:** Project Δy onto a fixed low-rank
   subspace and measure kept vs discarded energy. This connects the curvature
   residual to low-rank compression — if Δy lives in a rank-r subspace, the
   nonlinear contribution can be represented with r vectors.

**Cycle 302** unified measurements 1–5 into a single framework parameterized by
(σ, σ_lin). It reproduced cycles 217, 218, and 258 within numerical precision
(rank_95 bit-identical, c_fc cos_sum within ±0.005, κ shift 0.4772 vs 0.477).
The unified module (`experiments/unified_curvature_cycle302.py`, 532 lines,
using `experiments/curvature_residual_geometric.py`, 308 lines) exposes class
`UnifiedCurvatureFramework` with methods for all 5 panels on any (σ, σ_lin)
pair.

**What δ_σ does NOT tell you:** δ_σ is a *per-block, per-token* quantity. It
does not describe how blocks interact, how error propagates, or what the global
cell complex looks like. It is a bridge between the smooth and combinatorial
views — the "amount of curvature at a point" — but not the full geometry.

---

## 4. The Empirical Record

Everything below was measured on GPT-2 (12 blocks, 768 hidden, 3072 FFN,
50257 vocab) with ~200 OpenWebText calibration sequences (25,600 tokens) unless
otherwise noted. All cycle numbers reference `autoresearch/`.

### 4a. MASO-Style Measurements

| Cycle | Measurement | Verdict | Finding |
|-------|-------------|---------|---------|
| 001 | Balestriero literature survey | CONFIRMED (survey) | 6 techniques; SplineCam finds ~1,364 regions on Fashion-MNIST MLP |
| 003–025 | Activation census, Jacobian clustering, region enumeration on GPT-2 | REFUTED / INCONCLUSIVE | 99.99% unique activation patterns per token; no Jacobian clustering |
| 134 | "No Jacobian clustering" | CONFIRMED | Per-token codes are unique |

**Verdict:** MASO-as-discrete-enumeration was tested and found vacuous. But the
methodology was wrong (see §2a): probing at discrete token endpoints tells you
nothing about the cell complex between tokens. The interesting questions —
boundary density, cell volume distribution, interpolation-path crossings — were
never asked.

**Do not repeat these measurements as written.** The next attempt must probe
along *curves* in embedding space, not at isolated points.

### 4b. Pullback / Riemannian Measurements

| Cycle | Measurement | Verdict | Effect Size |
|-------|-------------|---------|-------------|
| 135 | κ-S anticorrelation under G = U^T U | CONFIRMED | ρ = −0.745, p = 0.0085, n = 11, 3 seeds within ±0.09 |
| 136 | G-weighted lstsq vs Frobenius as compression criterion | REFUTED | G/Frob ratio 34–35× constant; ρ(Frob,PPL) = 0.797 > ρ(G,PPL) = 0.734 |
| 144 | Local linearity under G | — | (measurement details in cycle 144 audit) |
| 258 | κ shift under (GELU → ReLU+b_leak) linearization | CONFIRMED | +47% κ kink at layer 4; κ shift = 0.477 |
| 259 | Per-token velocity distribution | — | (measurement details in cycle 259 audit) |
| 263 | κ + salience vs activation swap errors | — | (correlational) |
| 264 | κ per layer after progressive block swaps | — | (correlational) |
| 302 | Unified (σ, σ_lin) framework reproducing 217/218/258 | CONFIRMED | Reproduces within numerical precision (4 decimal places) |

**Verdict:** Pullback measurements are real and reproducible. The κ-S
anticorrelation is the strongest geometric finding in the project. But it has
not been connected to either the MASO partition or the compression barrier.
Cycle 136's refutation is *narrowly scoped* — it tested G vs Frobenius for
*per-block lstsq regression*, not for multi-block error ranking.

### 4c. Spectral / Eff_rank Measurements

| Cycle | Measurement | Verdict | Effect Size |
|-------|-------------|---------|-------------|
| 217 | Rank of Δy = W_proj · δ_σ(z) over tokens | INCONCLUSIVE | rank_95 ≤ 64 in 9/12 blocks, but mean-dominated (TI/TD = 0.87–0.99) |
| 218 | Centred Δ̃ alignment to downstream readers | CONFIRMED | 11/11 blocks significant, top-5 cos_sum ≥ 2.0, σ-excess +17σ to +56σ |
| 270 | eff_rank(c_proj_b11) = 22.49 → rank-22 SVD+retrain | CONFIRMED | PPL = 54.469 ± 0.024 (below float 53.89 by 0.58) |
| 358 | topk error eff_rank vs LASER error eff_rank | CONFIRMED | topk eff_rank = 463 (flat), LASER = 6.14 (concentrated) |

**Verdict:** Spectral measurements describe the *local linearization* of the
FFN-as-linear-map. They are Jacobian-flavoured probes — single-point, single-
block. Cycle 270 showed that eff_rank can predict a viable compression rank for
a single block. Cycle 358 showed that spectral flatness of compression error
explains why topk composes better than LASER (flat errors cancel across tokens;
concentrated errors accumulate). Neither finding connects to the global cell
complex.

### 4d. Composition / Cascade Measurements

| Cycle | Measurement | Verdict | Effect Size |
|-------|-------------|---------|-------------|
| 189 | Cascade error directional coherence (adjacent layers) | CONFIRMED | cos(e_l, e_{l-1}) = 0.373 → 0.834 across 11 layers; 100 seqs, 12,800 tokens |
| 201 | Block 11 rank expansion (decompressor) | CONFIRMED | input_rank_90 = 2 → output_rank_90 = 231, delta_rank = 1 |
| 362 | Cross-block error decorrelation (per-block error similarity) | REFUTED | mean |cosine| = 0.0162, max = 0.0703 across 6 blocks |
| 366 | Residual error inflection at b10–b11 | CONFIRMED | b10 + b11 = 77% of total error, inflection at b9→b10, 3/3 seeds |

**The c189 ↔ c362 reconciliation:**

These two measurements appear to contradict each other but measure different
things:

- **Cycle 189** measures the cosine between the error vector at layer l and the
  error vector at layer l−1, **at the same token position, across adjacent
  layers**. It asks: "does the error at token t propagate in a consistent
  direction through depth?" Answer: yes, increasingly so (cos 0.37 at early
  layers → 0.83 at late layers). This is a *propagation* measurement.

- **Cycle 362** measures the cosine between the **flattened error matrices** of
  different blocks (E_b ∈ ℝ^{25600×768}). It asks: "does block 2's error pattern
  across all tokens look like block 7's error pattern across all tokens?" Answer:
  no, they are nearly orthogonal (cos ≈ 0.016). This is a *similarity*
  measurement between blocks' contributions.

**Reconciliation:** Errors propagate coherently *through* the stack (each block
amplifies the error from previous blocks in a consistent per-token direction)
but each block *adds* its own orthogonal local error contribution. The net effect
is superlinear growth in a propagated direction plus small orthogonal
perturbations from each block. This explains the b10–b11 explosion: blocks 10–11
have the highest Frobenius norms (4.5× variance across blocks), and the
propagated error from blocks 0–9 (already directionally coherent, cos ≈ 0.83)
gets amplified the most through these high-norm blocks.

**What this means geometrically:** The residual stream trajectory under
compression is a *perturbation* of the float trajectory. The perturbation grows
coherently in a specific direction (determined by the geometry of the early
blocks) and accelerates at blocks 10–11 (where the network's "decompression"
geometry amplifies any directional error). This is a *global* phenomenon — it
involves the interaction of all 12 blocks — and it has never been studied as a
global geometric object (under either MASO or pullback formalism).

---

## 5. Open Formal Questions

The following are questions about the geometry of GPT-2, ordered by how
load-bearing the answer would be for understanding the network's shape. Each
question is stated formally, with what would count as an answer and what
existing evidence is partial.

### Q1. Cell Count and Traversal

**Formal question:** For GPT-2 block l, let C_l(x) denote the activation cell
(sign-pattern region) containing input x. For a sequence of T tokens with
embeddings x_1, ..., x_T, define:

    N_l(T) = |{C_l(x_t) : t = 1, ..., T}|

What is the scaling of N_l(T) with T? Does it grow linearly (each token is in
its own cell), sub-linearly (tokens share cells), or saturate (the network uses
a fixed number of cells regardless of corpus size)?

**What counts as an answer:** A plot of N_l(T) for T = 100, 500, 2000, 10000,
stratified by block l. If sub-linear, the exponent α in N ~ T^α. If saturating,
the saturation count N_∞.

**What about cell volume?** In ℝ^768, "volume" is not directly measurable, but
the number of tokens in each cell is a proxy for the probability mass of the
cell under the input distribution. The distribution P(|cell|) — histogram of
tokens per cell — is a measure of the network's geometric concentration. A
power-law distribution (few large cells, many small cells) means the network
is "geometrically simple"; a uniform distribution means it is maximally complex.

**Partial evidence:** Cycle 134 found "99.99% unique patterns" — i.e., N_l(T) ≈
T. But this was measured per-token, not along interpolation paths. The scaling
question is open.

**What's missing:** An activation-pattern fingerprint function that can be run
across a corpus and across continuous paths.

### Q2. Cell Semantics

**Formal question:** For tokens x_i and x_j that belong to the same semantic
category (e.g., both are code tokens, or both are proper nouns, or both are
function words), is it more likely that C_l(x_i) = C_l(x_j) or that
C_l(x_i) is *adjacent to* C_l(x_j) in the cell complex, compared to random
token pairs?

**What counts as an answer:** A measure of semantic-geometric association. For
example: compute the mutual information I(semantic_class; cell_id) across a
corpus. If I > 0 significantly above random permutation, the cell complex is
semantically structured.

**Partial evidence:** Cycle 218 found that the *nonlinear contribution* Δy is
aligned with downstream readers at 11/11 blocks. This is suggestive — if the
nonlinear contribution is structured, the cells that differ by "nonlinear vs
linear" should correspond to semantically distinct groups. But this was not
directly tested.

**What's missing:** A semantic labelling of tokens (coarse: POS tags, code vs
prose, named entities) paired with cell assignments.

### Q3. Interpolation Paths

**Formal question:** Given two token embeddings x_a and x_b, define the linear
interpolation path γ(t) = (1−t)x_a + t x_b for t ∈ [0, 1]. Let B_l(x_a, x_b)
be the number of cell boundaries crossed by γ in block l:

    B_l(x_a, x_b) = |{t ∈ (0,1) : C_l(γ(t−ε)) ≠ C_l(γ(t+ε))}|

What is the distribution of B_l across random token pairs? How does it depend
on the semantic similarity of x_a and x_b? What happens to the block's output
f_l(γ(t)) at the crossing points — is there a discontinuity in the Jacobian, a
sharp change in the output, or a smooth transition (for GELU, the latter is
expected)?

**What counts as an answer:** A histogram of B_l for, say, 1000 random token
pairs, stratified by block. Plus: a plot of f_l(γ(t)) along a few representative
paths, showing the behaviour at crossings.

**Why this is the right first probe:** This is how you study MASO geometry on a
high-dimensional input space. You cannot enumerate cells in ℝ^768. But you can
*walk curves through the cell complex* and measure the boundary density along
those curves. The boundary density is a 1D projection of the cell complex
geometry — a computable object that connects the combinatorial and smooth views
(boundary density ↔ Riemannian curvature, approximately).

**Partial evidence:** None. This has never been attempted on GPT-2.

**What's missing:** A curve sampler and a cell-boundary detector (for GELU: sign
changes in the pre-activation z_i(γ(t)) at each neuron).

### Q4. Local ↔ Global Bridge

**Formal question:** Does the pullback curvature κ_l (cycle 135) correlate with
the cell boundary density B_l (Q3) at the same block, on the same inputs?

Specifically: for a set of inputs {x_t}, compute both κ_l(x_t) (from
`pullback_geometry.py`) and B_l(x_t, x_{t+1}) (the number of crossings between
adjacent tokens in the residual stream). Is ρ(κ_l, mean(B_l)) significant?

**What counts as an answer:** A correlation coefficient across blocks l = 0..11,
analogous to cycle 135's ρ = −0.745 for κ-S. If significant, the smooth and
combinatorial pictures are measuring the same object; if not, they are
independent and we need both.

**Partial evidence:** The theoretical argument (§2c) predicts they should
correlate: high κ means the trajectory is bending, which means the function is
crossing many cell boundaries. But this is a limit argument for strict ReLU; for
GELU the relationship might be weaker.

**What's missing:** Cell boundary density measurements on the same inputs where
κ was computed.

### Q5. The b10–b11 Phenomenon

**Formal question:** Cycle 366 confirms that compression error explodes at
blocks 10–11 under the working topk+lstsq recipe. Is this a *shape property* of
the network — do blocks 10–11 have qualitatively different cell geometry (count,
boundary density, cell size distribution, curvature profile) than the easy
blocks (1, 2, 7) — or is it an artifact of the compression method being
misaligned with the right error metric?

**Two testable sub-questions:**

(a) **Cell geometry test:** Compute B_l (Q3) at blocks 1, 7, 10, 11. If
B_{10}, B_{11} are significantly higher than B_1, B_7, the late blocks are
"geometrically harder" — they have denser cell structure, meaning the function
varies more rapidly and is harder to approximate linearly (which is what
topk+lstsq does).

(b) **Metric test:** Re-measure cycle 366's error trace under the pullback
metric G instead of Frobenius. If ||e_l||_G / ||e_l||_F is no longer constant at
l = 10, 11 (unlike cycle 136's finding at the per-block level), the multi-block
error is concentrating in G-significant directions, and the explosion is real in
output-probability space. If the ratio stays constant, the explosion is
Frobenius-real but G-irrelevant.

**What counts as an answer:** (a) A per-block cell density profile; (b) a
per-block ratio ||e_l||_G / ||e_l||_F for the multi-block working recipe.

**Partial evidence:** Block 11 is known to be a rank-expansion decompressor
(cycle 201: rank 2 → 231). This is consistent with "geometrically harder" — a
decompressor that expands rank is likely to have denser cell structure (more
neurons participating means more potential boundary crossings). But this is
inference, not measurement.

### Q6. Shape vs Failure Modes

**Formal question:** Do tokens where the model has high perplexity (PPL > some
threshold) occupy distinguishable positions in cell space? Specifically:

- Are high-PPL tokens on cell boundaries (near a pre-activation zero crossing)?
- Are they in small cells (low frequency in the calibration corpus)?
- Are they in cells with high local curvature (high ||Δy||_G)?

**What counts as an answer:** A classifier that predicts "this token will have
high PPL" from its cell features (boundary distance, cell frequency, local
curvature), with AUC significantly above random. If AUC ≈ 0.5, cell geometry
does not predict failure modes. If AUC > 0.7, cell geometry is a meaningful
failure indicator.

**Partial evidence:** None directly. The relationship between cell geometry and
model failures has not been studied in this project.

**Why this matters:** If the geometry predicts failure modes, it predicts *what
the network doesn't know*. That is intrinsically valuable for interpretability,
robustness, and — eventually — for knowing where compression is safe (compress
the high-confidence cells, protect the boundary cells).

### Q7. Shape vs Scale

**Formal question:** "Would you know about that shape if the input space were
much larger?"

This is the user's question, and it has a precise formulation: if we trained
GPT-2 on k× more data (k = 2, 5, 10), would the cell complex (a) stay roughly
the same (the geometry is saturated at training-set size), (b) become denser
(more cells, smaller cells — the network is learning finer structure), or
(c) change qualitatively (different topology, different boundary arrangement)?

**What counts as an answer:** Cell density profiles (Q1, Q3) at multiple
training checkpoints or training set sizes.

**Partial evidence:** None. The KAN / implicit-representation thread (cycles
003–022) touched on this indirectly (asking whether the network's "functional
complexity" is bounded) but produced no measurements.

**What's missing:** Access to GPT-2 training checkpoints at different stages, or
training a small model from scratch with controlled dataset scaling. This is
probably out of scope for the current branch and is noted as a longer-term
question.

---

## 6. What This Formalization Explicitly Drops

The following directions are **not part of the geometric research program** going
forward. They may have been productive; they are not what we are investigating
now.

1. **Compression-first framing.** We are not optimizing PPL. If understanding
   the geometry suggests a compression strategy, we will test it. But the
   deliverable is the geometric picture, not a compressed model.

2. **Adapter sweeps and k-allocation heuristics.** Cycles 350–366 tested adapter
   hidden sizes, importance pruning, adaptive k per block, KD polish. These are
   parameter-space searches that do not increase geometric understanding. Paused.

3. **The composition barrier as the central problem.** The b10–b11 error
   explosion (cycle 366) is a *symptom* of the network having interesting global
   structure — structure that the working per-block compression recipe ignores.
   It is not the disease. We study it (Q5) as a geometric phenomenon, not as an
   engineering target to be patched with better adapters.

4. **Per-block isolation.** The working recipe compresses blocks independently
   and aggregates. This treats the network as 12 independent modules. The
   geometry of the network — the cell complex, the residual stream trajectory,
   the cascade coherence — is inherently *inter-block*. The formalization commits
   to studying the network as a whole, not block by block.

5. **Frobenius as the default error metric.** Cycle 136 established that
   Frobenius and G-weighted give the same *per-block* ranking. But the
   interesting geometric questions (Q5, cascade coherence) are *multi-block*
   phenomena where Frobenius may not be the right norm. We will measure under
   both norms and compare.

---

## 7. Candidate Probing Experiments

These are *not* committed cycles. They are descriptions of what would need to be
built and measured to answer the questions in §5. Each becomes an autoresearch
item only after explicit approval.

### Probe A: Activation-Pattern Fingerprinting (serves Q1, Q2, Q3, Q5, Q6)

**What:** A module that, given a GPT-2 checkpoint, a set of inputs, and a block
index l, returns the activation-pattern fingerprint (sign vector of FFN pre-
activation) for each input at each block. For GELU, the "sign" is sign(z_i) —
the same discretization that defines MASO cells for ReLU, applied approximately
to the smooth case.

**Output:** A tensor of shape (n_inputs, n_blocks, n_neurons) with values in
{−1, +1}. Plus: a hash function that maps each fingerprint to a cell ID for
efficient comparison.

**Where it would live:** `aleph/maso_geometry.py` (new module, counterpart to
`aleph/pullback_geometry.py`).

**Estimated complexity:** ~100–150 lines of PyTorch. No training, no
optimization — pure measurement.

### Probe B: Curve Sampler + Boundary Detector (serves Q3, Q4, Q5)

**What:** Given two token embeddings x_a, x_b, sample N points along the linear
interpolation γ(t) = (1−t)x_a + t·x_b, pass each through the model, and count
how many activation-pattern boundaries are crossed (where any neuron's pre-
activation changes sign). Report: boundary count B_l per block, the t-values
of crossings, and the model's output logits at each sample point.

**Output:** Per-path: B_l for each block, crossing locations, logit traces.
Aggregated: histogram of B_l over many random paths, correlation with κ_l.

**Where it would live:** Same `aleph/maso_geometry.py` module.

**Estimated complexity:** ~80–120 lines. The main challenge is efficient batched
forward passes along interpolation paths (can reuse the model's standard forward
pass).

### Probe C: Per-Block Cell Census (serves Q1, Q2, Q6)

**What:** Run fingerprinting (Probe A) over a calibration corpus (200 OWT
sequences, ~25,600 tokens). At each block, count distinct cell IDs, compute the
frequency distribution, and classify cells by their content (using POS tags or a
simple code-vs-prose classifier on the tokens they contain).

**Output:** Per-block: N_l, frequency histogram, top-k cell sizes, semantic
composition of the 10 largest cells. Global: scaling of N_l with corpus size T
(subsample at T = 100, 500, 2000, 10000, 25600).

**Estimated complexity:** ~50–80 lines on top of Probe A. Requires a POS tagger
or content classifier for the semantic analysis (can use spaCy or a simple
heuristic).

### Probe D: Multi-Block Error Under G (serves Q5)

**What:** Re-run cycle 366's error trace (residual stream error after
progressive multi-block compression) but measure error under both Frobenius and
the pullback metric G = U^T U. Compare the per-block ratio ||e_l||_G / ||e_l||_F
to cycle 136's per-block ratio (which was 34–35× constant). If the ratio is no
longer constant at the multi-block level, the b10–b11 explosion has a G-specific
component.

**Output:** A per-block trace of ||e_l||_F and ||e_l||_G for the working recipe
at 1, 3, 6, 12 compressed blocks. Plus: the ratio trace.

**Estimated complexity:** ~30–50 lines. Reuses `aleph/pullback_geometry.py`
(`pullback_norm`) and the existing compression evaluation pipeline.

---

## Appendix: Key File Locations

| File | Role |
|------|------|
| `aleph/pullback_geometry.py` | Smooth-view: G = U^T U, salience, curvature, G-weighted lstsq |
| `experiments/unified_curvature_cycle302.py` | δ_σ unified framework (5 panels) |
| `experiments/curvature_residual_geometric.py` | `UnifiedCurvatureFramework` class |
| `autoresearch/surveys/balestriero_spline_theory.md` | MASO / SplineCam survey |
| `docs/gelu_bump_audit.md` | §12: δ_σ reframing; §10: superadditive interaction |
| `autoresearch/STATUS.md` | Working recipe + confirmed findings + open questions |
| `autoresearch/briefs/cycle_135.md` | κ-S anticorrelation |
| `autoresearch/briefs/cycle_136.md` | G-weighted lstsq refutation (narrowly scoped) |
| `autoresearch/briefs/cycle_189.md` | Cascade error propagation coherence |
| `autoresearch/briefs/cycle_201.md` | Block 11 rank expansion |
| `autoresearch/briefs/cycle_218.md` | Centred Δ̃ downstream alignment |
| `autoresearch/briefs/cycle_362.md` | Cross-block error decorrelation |
| `autoresearch/briefs/cycle_366.md` | b10–b11 error explosion |
