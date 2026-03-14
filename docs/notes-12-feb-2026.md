# Geometric Interpretation Project: Session Notes

## Date: February 21, 2026

---

## Project Goals and Principles

### The Core Problem

The best architecture for *learning* a function (via gradient-based optimization) is not the best architecture for *representing* that function. Trained neural networks carry representational overhead — parameters that exist because the training process needed them, not because the final function requires them. This overhead is not removable by standard compression methods (pruning, quantization, low-rank decomposition) because those methods stay within the same representational format.

### The Scaffolding Hypothesis

Many neurons in a trained network serve as scaffolding: they exist to enable smooth gradient flow during training and to incrementally approximate curved functions with piecewise linear (or piecewise smooth) segments. In ReLU networks, this manifests as dense clusters of partition boundaries in high-curvature regions — many hyperplanes packed together to approximate a smooth curve. The hypothesis predicts that these clusters could be replaced by a compact smooth representation (e.g., splines) with far fewer parameters, if you change the representational format rather than just pruning within it.

### Stated Principles

1. **The best architecture for learning is not the best for solving the task.** ReLU is great for deriving a gradient but poor at representing curved space. Many neurons likely exist only to allow smooth gradient flow (lottery ticket hypothesis connection).

2. **Backpropagation inherently relies on scaffolding.** Therefore distillation is also subject to the same problem — distilling into the same architecture just reproduces the scaffolding in compressed form.

3. **Current architectures don't allow clean separation between scaffolding and representation.** The representation is distributed. But at the same time, trained networks are empirically sparse.

4. **Geometry provides the right lens, via duality with linear algebra.** Linear algebra gives local transforms. Geometry gives global shape. Data consists of samples on an unknown manifold we're trying to model inside the network to perform a task. The goal is to capture and manipulate shape.

### What We're Looking For

Reparameterization. Given a trained network that solves a task, approximate the network while minimizing parameters — not by pruning or rank reduction within the existing format, but by finding a representation that natively captures the global geometry implicit in the trained network.

---

## KAN Work: What Was Tried on High-Dimensional Embedding and Why It Didn't Work

### Implicit Surface Fitting in 3D

Fitted 3D meshes as implicit surfaces using both MLPs and KANs. Both worked fine. This is expected — in 3D, the input is already low-dimensional and the coordinates are naturally axis-aligned, which is exactly KAN's sweet spot.

### Embedding in 100 Dimensions

Took the same 3D mesh, embedded it in 100-dimensional space, and tried to fit the implicit surface with a KAN. Result: horribly slow training and poor quality.

**Why this failed (geometric explanation):** KANs operate by applying independent univariate spline functions along each coordinate axis, then summing. When the 3D surface is embedded in 100D, the relevant 3D subspace is rotated relative to the coordinate axes. No single coordinate axis aligns with the relevant directions. The KAN has to use compositions of axis-aligned nonlinear functions to reconstruct what is fundamentally a rotation — which is a linear operation that KANs have no native mechanism for. This is like trying to draw a diagonal line on an Etch-a-Sketch: technically possible but enormously inefficient.

**The general lesson:** KANs cannot discover relevant subspaces when those subspaces are not aligned with the input coordinate axes. They require the relevant directions to already be aligned with their input axes. This makes them unsuitable as a learning architecture for high-dimensional problems where the relevant structure is rotated relative to coordinate axes (which is almost all real problems with high-dimensional inputs). Their strength is representing smooth functions on low-dimensional domains where the input coordinates are already the right coordinates — either naturally (e.g., physics problems with known variables) or because a prior processing step has aligned them.

### KAN Geometry: What They Actually Do

A KAN is NOT "an MLP with spline activation functions." The architecture is fundamentally different:

- **MLP neuron:** σ(Σ w_i x_i) — linear mix, then fixed nonlinearity
- **KAN node:** Σ φ_i(x_i) — learned univariate splines, then sum. No weight matrices at all.

Geometric picture per layer:
- **MLP:** Defines oriented hyperplane folds. Weights choose the fold directions. Level sets are parallel hyperplanes.
- **KAN:** Warps each coordinate axis independently via splines, then sums. Level sets are curved but axis-decomposable. A single KAN layer cannot represent interactions between variables (e.g., x₁·x₂), but depth introduces interactions — the second layer's splines operate on sums of warped inputs from the first layer, entangling the variables.

Depth in KANs entangles variables — the second layer applies splines to sums of warped inputs, breaking the axis-alignment. The Kolmogorov-Arnold theorem guarantees that 2 additive layers suffice for universality. But the practical issue remains: when the relevant coordinates are rotated relative to input axes, KANs must use depth to approximate what would be a simple linear rotation, making learning slow and parameter-inefficient. Representing smooth functions on coordinates that are already correct is what they excel at.

---

## Interpretability Critique

### Neurons as Boundary Conditions, Not Semantic Units

The project has a specific stance on interpretability: **semantics lives in the partitions (regions), not in the neurons (boundaries).** A neuron in a ReLU network defines a hyperplane — a boundary between regions. The meaning is in the regions, which are defined by the intersection of many boundaries.

This explains why "polysemantic" neurons are the norm, not the exception. A hyperplane in high-dimensional space borders many regions. As you traverse the hyperplane, the regions on either side change. It's like saying a national border is "polysemantic" because it's the Spain-France border in one place and the France-Germany border elsewhere.

The "maximally activating input" for a neuron (moving along the normal vector) crosses through many regions with entirely different semantics — the Russia problem. Maximally far from the Spain-France border heading northeast isn't "more France," it's a completely different place.

**Implications for KAN interpretability claims:** KANs are marketed as more interpretable because you can inspect learned spline functions on each edge. This is still looking at individual components. KANs are genuinely more interpretable only when the problem has natural additive/axis-aligned structure (physics, symbolic regression) — where the decomposition matches the problem structure. On problems where semantics aren't axis-decomposable, individual KAN edge functions are not independently interpretable — the meaning lives in their composition, just as with MLP neurons.

---

## Three Research Directions Considered

### Direction 1: Tropical Geometry

ReLU networks compute tropical rational functions. The max/plus algebra underlying ReLU gives the piecewise affine structure a dual description as a tropical hypersurface — a polyhedral complex in dual space that encodes the partition globally and algebraically. The Newton polytope of the tropical polynomial gives a compact global description.

**Status:** Looking into this on the side. The mathematics is substantial and unfamiliar — requires learning algebraic geometry. This is a long-term investment, not an immediate tool. But it may be the most powerful approach eventually because it provides a single global object encoding both the subspace structure and the function, rather than requiring separate extraction steps.

**Key references to pursue:** Zhang et al., Alfarra et al. on tropical geometry of deep neural networks.

### Direction 2: Mesh Simplification

The polyhedral complex defined by a trained ReLU network is essentially a mesh in input space. Neighboring cells with nearly identical affine functions can be merged, like simplifying a triangulated surface by collapsing low-curvature edges. Scaffolding neurons correspond to unnecessary mesh refinement.

**Status:** Conceptually appealing and closest to existing expertise (implicit surface work). The open problems are: (a) which mesh simplification methods scale to high-dimensional ambient spaces, and (b) how to perform simplification when the full polyhedral complex cannot be enumerated explicitly (the number of polytopes grows exponentially with network size). The activation code framework (below) partially addresses (b) by allowing the partition to be sampled and probed on and near the data manifold without full enumeration.

### Direction 3: Active Subspace + Smooth Reparameterization

Use gradient-based active subspace methods on the trained network to identify the low-dimensional subspace where the function varies. Project onto that subspace. Fit the projected function with a smooth representation (B-splines, RBFs, or KANs).

**Status:** This is the direction with the most immediate traction. It directly separates what the MLP is good at (discovering the relevant subspace during training) from where it is representationally inefficient (storing the learned function as many piecewise linear/smooth segments when the underlying function is smooth). The transformer MLP compression experiment (see below) is a specific instance of this approach.

---

## Activation Codes and the Hamming Distance Framework

### The Key Tool (Exact for ReLU, Extendable to Soft Codes for GeLU/SiLU)

For any input x to a ReLU network, the vector of on/off states of all ReLU units gives a binary code identifying which polytope x occupies. This works in any dimension, no projection needed. For networks with smooth activations (GeLU, SiLU), an approximate version uses thresholded activation levels to produce soft/pseudo-binary codes (see "Smooth activations" in the Transformer Direction section).

- Points with identical codes → same polytope (same large region)
- Points with Hamming distance 1 → adjacent polytopes (share a boundary)
- Clusters of similar codes → large regions
- Rapid code changes between nearby points → dense boundary zone

This answers the two critical questions for compression:
1. How do you identify large partitions? → Cluster by activation code.
2. How do you find where boundaries are close together? → Look for regions of rapid code change (high Hamming distance between nearby points).

### Existing Work and Tools

- **GoL Toolbox** (Liu et al., 2023): Python toolbox for computing binary vectors, deriving bounding facets, enumerating neighbors. GitHub: https://github.com/cglrtrgy/GoL_Toolbox
- **Polytope Lens** (Alignment Forum, 2022): Used Hamming distance between spline codes as proximity metric. Showed polytope boundary density is higher between different-class activations. Defined features as groups of nearby polytopes implementing similar transformations.
- **Bosca et al. (2025)**: Used spectral graph theory (Fiedler partitioning) on the dual graph of the polyhedral decomposition. Found that weighting polytopes by size recovers decision boundaries. Connected to grokking — larger polytopes correspond to cleaner circuits.
- **Persistent homology approaches**: Hamming distance matrices fed into Ripser to compute topological invariants (Betti numbers) of the partition structure restricted to data.

### The 2D Slice Warning

The grokking paper's visualizations (showing boundary density accumulating near decision boundaries through training) are informative but potentially misleading. A 2D slice through a high-dimensional polyhedral complex can give a false picture — boundaries that appear close in projection may diverge in other dimensions. Any method that works by visualizing 2D slices is for analysis/intuition only, not for building losses. The activation code framework works in the ambient space without projections, which is essential.

---

## Geometric Loss Functions for Compression

### What Doesn't Work for Scaffold Removal: Pointwise Jacobian Matching as a Compression Objective

An initial idea was to include a Jacobian-matching term in the compression loss to preserve the local geometry during compression. This is counterproductive for the specific goal of removing scaffolding. If you match Jacobians pointwise, you're telling the compressed network to reproduce the same local linear behavior everywhere — including in the dense boundary regions where many small polytopes have rapidly varying Jacobians. This preserves the scaffolding instead of eliminating it. (Note: Jacobians are still useful as an *analysis* tool — e.g., measuring Jacobian discontinuity across boundaries to detect which boundaries are scaffolding vs. structural. The problem is specifically with using Jacobian matching as the compression *objective*.)

### What the Loss Should Preserve

The correct preservation target is the **coarse partition topology restricted to the data manifold:**

- **Function values on the data** — standard distillation loss handles this
- **Which data points are in the same large region** — points the original network assigns to the same partition cluster should remain together
- **Which data points are separated by boundaries** — points separated by dense boundary clusters should remain separated
- **No constraint on transition zone structure** — the compressed network is free to represent the boundary however it wants (smooth curve, single hyperplane, whatever) as long as the separation is maintained

### The Right Loss Structure

```
L = ||f_compressed(x) - f_original(x)||² + λ * TV(f_compressed)
```

Where the second term penalizes boundary count / total variation, encouraging the compressed network to use as few partition boundaries as possible while matching function values. The compressed network is free to replace a dense cluster of 50 small polytopes with a single smooth transition.

### Scaffolding Detection

A boundary is scaffolding if the affine functions on both sides are nearly identical (small Jacobian discontinuity across the ReLU transition). A boundary is structural if the Jacobian jump is large. This can be detected locally by perturbing inputs across boundaries and measuring the change in the network's Jacobian.

The topological/clustering approach (activation codes) is better suited than pointwise differential matching (Jacobian matching) for the compression objective, because the goal is preserving coarse structure while discarding fine structure — a topological property, not a differential one. Differential tools (Jacobians, gradients) remain essential for *analysis* — e.g., identifying which boundaries are scaffolding, estimating active subspaces, and probing local geometry.

---

## The Transformer Direction

### Why Switch to Transformers

The end goal is compressing billion-parameter models, not only ImageNet CNNs. The geometric approach needs to generalize beyond ReLU feedforward networks to transformer architectures (which use GeLU/SiLU activations and attention mechanisms) to reach its full practical potential. The ReLU feedforward work provides exact theory and clean experiments, but the research must target transformers to be relevant at scale.

### How the Framework Transfers

**MLP blocks in transformers:** These are the most direct analog. A transformer MLP with GeLU does: project up to 4d dimensions, apply GeLU independently to each dimension, project back down. This is already close to a KAN structure — univariate nonlinearities along learned directions. The scaffolding hypothesis applies: many up-projection directions may be near-parallel, with clusters of neurons collectively approximating smooth curves that single GeLUs cannot capture.

**Smooth activations (GeLU/SiLU):** The exact binary activation code framework doesn't apply — there's no sharp on/off state. But soft/thresholded activation codes work: neurons with pre-activation above ε are "on," below -ε are "off," in between are "transitional." The polytope picture becomes a fuzzy partition. This may actually be favorable for compression, since the architecture already smooths transitions rather than forcing piecewise linear approximation.

**Attention blocks:** These are geometrically different — data-dependent linear maps. The effective linear transformation changes based on context (other tokens). This means you're not partitioning a fixed input space. Analyzing attention requires fixing a context or averaging over contexts, which adds complexity. Tackle after the MLP block analysis is established — MLP blocks are the right starting point because they have fixed-function geometry analogous to feedforward networks.

**Residual stream:** All layers read from and write to the same ℝ^d. Representations live on a low-dimensional manifold within this space. The manifold structure is a potential compression target and may become important when scaling beyond single-block compression to whole-model reparameterization, but the immediate experiment focuses on individual MLP blocks.

### The Specific Transformer Experiment

**Observation:** A transformer MLP's W_up matrix has 3072 rows (in GPT-2), each a direction in ℝ^768. The scaffolding hypothesis predicts these cluster into far fewer effective directions. Each cluster of neurons with near-parallel input directions and spread biases is collectively computing a smooth univariate function that could be represented by a single learned spline.

**The replacement architecture:**
```
Original: x → W_up (768×3072) → GeLU → W_down (3072×768) → Δx  [~4.7M params]
Compressed: x → W_up_compressed (768×k) → [k learned splines] → W_down_compressed (k×768) → Δx
```

If k ≈ 200 (hypothesis), this is roughly 15× compression per MLP block.

**Critical check:** Joint input-output direction analysis. Neurons must cluster on BOTH the up-projection direction (what they read from the residual stream) and the down-projection direction (what they write back). If they cluster on input but scatter on output, they're doing different jobs despite reading from the same direction, and the simple "merge cluster into one spline" compression doesn't apply — a different compression strategy would be needed for those neurons.

**First experiment:** Compute cosine similarity between rows of W_up in trained GPT-2. If they cluster into significantly fewer than 3072 effective directions, the scaffolding hypothesis applies to transformer MLPs and the KAN replacement is worth pursuing. If directions are approximately uniformly distributed on the sphere, the scaffolding in transformer MLPs does not take the form of direction clustering, and the compression approach would need to be reconsidered — possibly targeting a different structural redundancy (e.g., low-rank structure, activation sparsity patterns, or redundancy in the output projection). Full plan written up separately (see experiment plan document).

---

## Dead Directions and Negative Results

### KANs as a Learning Architecture on High-Dimensional Non-Axis-Aligned Inputs
**Dead.** KANs apply independent univariate spline functions along each coordinate axis. When the relevant structure in the data lives in a subspace that is rotated relative to the coordinate axes — which is almost always the case in high-dimensional problems — KANs cannot discover or align with that subspace. They have no weight matrices and no mechanism for linear mixing. Tested empirically: fitting a 3D implicit surface directly in 3D worked fine; embedding the same surface in 100D and fitting with a KAN produced horribly slow training and poor results, because the relevant 3D subspace was rotated away from the coordinate axes. **KANs remain valuable for representing functions on low-dimensional, axis-aligned domains** — see the transformer MLP compression direction where KANs operate on a pre-extracted subspace.

### KANs as Drop-In Activation Function Replacements Inside MLPs
**Dead.** KANs are not "MLPs with spline activation functions." They are a fundamentally different architecture: MLP neurons compute σ(Σ w_i x_i) (linear mix then fixed nonlinearity), KAN nodes compute Σ φ_i(x_i) (learned univariate splines then sum, no weight matrices). You cannot swap GeLU/ReLU for a B-spline inside an MLP and call it a KAN — the architecture and parameterization are entirely different. **This does not preclude replacing MLP activation functions with B-splines as a reparameterization strategy** (which is a different idea — fitting splines to the piecewise linear/smooth response the MLP already learned — and remains a live direction).

### Pointwise Jacobian Matching as a Loss for Partition-Simplifying Compression
**Dead for the specific goal of removing scaffolding boundaries.** If the compression goal is to replace dense clusters of partition boundaries with cleaner smooth transitions (fewer boundaries, same coarse structure), then a loss that matches Jacobians pointwise works against that goal — it forces the compressed network to reproduce the rapidly varying local linear behavior in exactly the regions you want to simplify. **Jacobians remain useful for other purposes:** detecting scaffolding (small Jacobian discontinuity across a boundary = scaffolding), active subspace extraction (SVD of the Jacobian identifies locally important directions), and general function analysis. The point is: don't use Jacobian matching as the compression *objective* when the goal is partition simplification. The right objective is topological — preserve which data points are in the same coarse region and which are separated.

### MLP → KAN Distillation on High-Dimensional Inputs Without Prior Subspace Extraction
**Dead.** Training a KAN to directly match an MLP's outputs when the input space is high-dimensional and the relevant subspace is unknown fails for the same axis-alignment reason as KAN learning above. **MLP → KAN distillation is a live direction when preceded by subspace extraction** — first identify the relevant low-dimensional subspace from the trained MLP (via weight clustering, gradient analysis, or active subspace methods), project to that subspace, then fit a KAN on the projected coordinates. The MLP discovers the subspace; the KAN efficiently represents the function on it.

### 2D Slice Visualization as a Basis for Loss Functions or Compression Algorithms
**Dead for algorithm/loss design. Alive and valuable for building intuition.** 2D projections of high-dimensional polyhedral complexes can be misleading — boundaries that appear close in a slice may diverge in orthogonal dimensions, regions that look large may be thin in other directions. Never use 2D slice properties to construct losses or make compression decisions. Activation codes (binary patterns of neuron on/off states) provide projection-free, coordinate-free partition information in the full ambient space, and should be used instead for any algorithmic purpose.

---

## Alive Directions: Priority-Ordered

### Immediate: Transformer MLP Direction Clustering (Experiment 1)
Validate the scaffolding hypothesis for GPT-2 MLP blocks by checking whether up-projection directions cluster. One-afternoon experiment, high information value. Either confirms the path forward or forces a rethink.

### Near-term: KAN Replacement of Compressed MLP Blocks (Experiment 2)
Contingent on Experiment 1 success. Replace MLP blocks with projection → KAN → projection architecture. KANs are back in the picture here as the representation stage — they represent smooth functions on the discovered low-dimensional subspace.

### Near-term: Activation Code Framework for Analysis
Use the GoL Toolbox and related methods to compute activation patterns on trained networks, build adjacency graphs, and identify large regions vs. boundary zones. Start on small ReLU networks where the theory is exact, then extend to soft codes for GeLU networks.

### Medium-term: Topological Loss for Geometric Compression
Develop a loss function based on preserving coarse partition topology (via activation code clustering) rather than preserving local differential quantities. The loss should encourage the compressed network to maintain the large-region structure while freely simplifying transition zones.

### Medium-term: GNN for Structural Importance Prediction
Train a GNN on the computation graph of trained networks to predict which nodes/edges are scaffolding vs. structure. GNNs can capture non-local dependencies (a neuron that seems unimportant locally but is critical for partition structure elsewhere). Train on small networks where ground truth is available, test generalization to larger networks.

### Long-term: Tropical Geometry
Learn the mathematics. The tropical description of ReLU networks encodes the global geometry directly and algebraically, potentially providing a naturally compressible representation. This is the most theoretically powerful direction but requires substantial mathematical investment.

### Long-term: Scaling to Billion-Parameter Models
If the approach works at GPT-2 scale, extend to larger open-weight LLMs. The compression ratio may actually improve with model size (more redundancy per block in larger models). This is the ultimate test of practical value.

---

## Key Conceptual Connections

### The MLP as Discovery Tool, Splines as Representation
The central architectural insight: MLPs are good at *learning* (finding relevant subspaces via weight matrix optimization, fitting functions via gradient descent through piecewise linear approximation) but inefficient at *representing* the result (storing what was learned using many piecewise linear or piecewise smooth segments when the underlying function is smooth). Spline-based representations (including KANs) are efficient at *representing* smooth functions compactly but cannot discover relevant subspaces when those subspaces are not aligned with the input coordinates. The pipeline is: train MLP → extract the geometry it discovered (relevant subspace + function shape) → reparameterize using smooth representations.

### Smooth Functions from Piecewise Linear Approximations
The scaffolding hypothesis predicts that many of the piecewise linear segments in a trained ReLU network (or many of the clustered GeLU neurons in a transformer) are collectively approximating smooth functions. If this is true, the compression gain from reparameterization is fundamental: a B-spline with k knots represents a smooth curve far more efficiently than k piecewise linear segments representing the same curve. This isn't just pruning — it's a representational format change that unlocks a qualitatively different compression regime.

### The Etch-a-Sketch Analogy for KAN Limitations
A KAN can only warp along coordinate axes (like an Etch-a-Sketch can only draw horizontal and vertical). An MLP can move in any direction (weight vectors choose directions). When your problem has structure along directions that are rotated relative to the input coordinate axes (almost always the case in high dimensions), KANs must approximate those rotations through compositions of axis-aligned warpings, which is enormously inefficient. This is why KANs struggle on high-dimensional problems with non-axis-aligned structure, and why in this project they belong at the representation stage — operating on coordinates that have already been aligned by prior subspace extraction.

### Where KANs Belong in the Pipeline
KANs re-enter the picture specifically for transformer MLP compression. A transformer MLP already has the structure: linear projection to choose directions → univariate nonlinearity along each direction → linear projection back. This is almost a KAN. The compression move is: reduce the number of directions (cluster analysis), replace the fixed GeLU along each effective direction with a learned spline (KAN edge function). The KAN isn't discovering anything — it's efficiently encoding what was discovered.

---

## Open Questions

1. **Do transformer MLP directions actually cluster?** This is the immediate empirical question. Everything downstream depends on it.

2. **How does the scaffolding fraction scale with model size?** Larger models might have proportionally more scaffolding (predicting better compression ratios) or proportionally less (if the added parameters are doing genuinely different work).

3. **Can you formulate geometric losses that work without explicit geometry extraction?** The activation code approach works via sampling, but scaling to very large networks requires efficient probing strategies.

4. **Is there a unified framework connecting the polyhedral partition view (ReLU), the soft partition view (GeLU), and the tropical geometry view?** These feel like they should be aspects of the same theory.

5. **Can GNNs learn to predict structural importance in network computation graphs?** This would be a powerful meta-learning tool for compression but might be harder than the original problem.

6. **How does attention interact with MLP scaffolding?** If attention routes information through specific MLP neurons, the "which neurons are scaffolding" question becomes context-dependent, complicating the compression story.
