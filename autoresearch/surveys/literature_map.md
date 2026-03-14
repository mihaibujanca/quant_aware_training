# Geometric Compression — Literature Map

**Last updated:** 2026-03-14
**Problem:** Given a trained ReLU MLP, construct a smaller network computing the same function — analytically, not by distillation.

---

## How to read this document

Papers are rated:
- **CORE** — directly useful, actionable theory or algorithms
- **USEFUL** — relevant concepts, read before implementing
- **CONTEXT** — background / related problem, not directly applicable
- **DEAD END** — assessed and ruled out; don't revisit

---

## CORE

### MASO / Spline Theory (Balestriero & Baraniuk)

**Papers:**
- "A Spline Theory of Deep Learning" (Balestriero 2018)
- "Mad Max: Affine Spline Insights into Deep Learning" (Balestriero & Baraniuk 2020, ICML)
- Survey in cycle_001: `autoresearch/surveys/balestriero_spline_theory.md`

**Main idea:** ReLU networks are continuous piecewise affine (CPA) operators — Max Affine Spline Operators (MASOs). The network partitions input space into convex polytopes; within each polytope, the function is a fixed affine map. The partition is determined by which ReLU neurons are active (binary activation pattern). This is the theoretical foundation for everything else.

**What it gives us:**
1. Activation pattern = identity of linear region (hash it to count unique regions)
2. Effective affine map per region = product of masked weight matrices
3. SplineCam: exact 2D enumeration of the partition (BFS on hyperplane arrangement graph)
4. The Jacobian of the network is piecewise constant — each region has its own Jacobian
5. The "centroid" of a region (Jacobian averaged over inputs) summarizes what the region computes

**Gap:** Provides analysis tools but not a constructive synthesis route from partition → minimal network.

**Status:** Well-understood. Cycle 001 identified 6 implementable techniques. The activation pattern census (Technique 1) and affine map extraction (Technique 2) are the foundation for everything downstream.

---

### Local Complexity of Linear Regions (Patel & Montúfar, ICML 2025)

**arXiv:** 2412.18283
**User label:** "Foundational paper"

**Main idea:** Introduces *local complexity* — a measure of linear region *density* relative to the data distribution (not the total count of regions). Key result: local complexity upper-bounds the total variation of the network function over the data distribution. Networks that learn compressed feature representations have lower local complexity.

**Why it matters for us:** This is the missing bridge between "how many regions" and "how compressible." If a network has low local complexity on the data distribution:
- Its function varies slowly relative to the data
- Most of the partition is irrelevant (regions not visited by data)
- The function can be described by a smaller network that matches the partition only where data lives

**Actionable:** Compute local complexity on trained networks before attempting compression. Low local complexity = compression is possible. High local complexity = network is genuinely using its capacity.

**Limitation:** Primarily theoretical at this stage; no compression algorithm. The measure is an upper bound on variation, not a direct count of compressible structure.

---

### SplInterp: SAEs as Piecewise Affine Splines (Budd, Balestriero et al., 2025)

**arXiv:** 2505.11836
**User label:** "More Balestriero"

**Main idea:** Sparse Autoencoders (SAEs) are piecewise affine splines where the input partition forms a **K-th order power diagram** (weighted Voronoi diagram). TopK/JumpReLU SAEs encode data by mapping inputs to K active features; each partition cell Ω_S is linear. The geometry interpolates between k-means (piecewise constant) and local PCA (piecewise linear).

**Key theorem:** A TopK SAE with encoder W_enc, b_enc has activation regions forming a K-th order power diagram with centroids and weights derivable from encoder parameters.

**What it gives us:**
1. The **compressed representation of a function** has a natural geometric form: power diagram partitioning + local linear maps. This is what a compressed network "should" look like.
2. The k-means/PCA interpolation tells us the minimum structure needed: K cluster centroids + K local linear maps. A network with K regions needs ~K×(input_dim + output_dim) parameters.
3. **PAM-SGD**: alternating optimization (encoder via SGD, decoder via closed-form least squares) — could be adapted for fitting a compressed network to extracted affine maps without distillation.

**Gap:** SAE context — the target here is interpretability, not compression of the original network. The spline structure is of the SAE, not of the MLP being analyzed.

---

### Topological Signatures of Activation Patterns (Bosca et al., 2025)

**arXiv:** 2510.12700
**User label:** "Looks pretty promising"

**Main idea:** Build a *dual graph* of activation patterns: vertices = unique binary activation vectors (each vertex = one linear region visited by data), edges = pairs differing by one neuron flip (Hamming distance 1). Apply spectral analysis (weighted Laplacian) to this graph. The Fiedler vector (2nd eigenvector) partitions the graph in a way that correlates with the network's decision boundary.

**Results:**
- Two circles (2,6,6,2 net): weighted Fiedler → 0% misclassification (unweighted: 19%)
- Two moons (2,5,5,5,2 net): weighted Fiedler → 0% misclassification
- Training dynamics: Betti numbers (homology) correlate with training loss (r=0.82 detrended)

**What it gives us:**
1. A way to visualize and understand the partition structure without enumerating all regions
2. The Fiedler partition = natural coarsening of the activation pattern space into 2 clusters. With more eigenvectors, you get K clusters. **This is a geometric basis for region merging**: merge all patterns in the same spectral cluster.
3. The graph structure naturally handles the combinatorial explosion — you only work with regions actually visited by data, not all 2^n possibilities.

**Key limitation:** All experiments on 2D inputs, small networks. Laplacian computation scales as O(V²) in regions. The "polytope count grows polynomially with width, exponentially with depth" — acknowledged by authors. No compression experiments.

**Immediate next step:** Implement the dual graph + weighted Laplacian for our spirals/moons networks. The Fiedler partition gives us a free clustering of activation patterns. Merge patterns in the same Fiedler cluster and measure the approximation error.

---

### Centroid Affinity Hypothesis (Walker, Humayun, Balestriero, Baraniuk, 2025)

**OpenReview:** HxLGntmbf2
**User label:** "This one looks like it has good potential"

**Main idea:** Extends MASO theory. The *centroid* of a network (for a given layer) is a vector summarizing the network's Jacobians: a weighted average of Jacobian rows across data points. For continuous piecewise affine networks, centroids form **affine subspaces** that correspond to task-relevant features. This is an alternative to the linear representation hypothesis — features are not individual directions but affine subspaces defined by Jacobian centroids.

**Applications demonstrated:**
- Sparser feature dictionaries from DINO with better downstream performance
- Direct neuron identification in GPT2-Large circuits
- Better probes for detecting truthful vs false statements in Llama-3.1-8B

**What it gives us for compression:**
1. Centroids are a compact summary of what each layer computes. If two neurons have similar Jacobian centroids, they're computing similar functions → merge candidates.
2. The affine subspace structure means: the function lives in a low-dimensional subspace of the activation space, and the centroid identifies which subspace. Compression = projecting onto fewer affine subspaces.
3. **More principled than cosine similarity on weight rows**: centroids are defined with respect to the actual data distribution and the network's input-output behavior, not just the weight geometry.

**Gap:** The paper focuses on interpretability and circuits, not compression. No compression experiments or algorithms. The centroid computation requires a forward pass over data.

---

## USEFUL

### Auto-Compressing Networks (Dorovatas, Paraskevopoulos, Potamianos, NeurIPS 2025 oral)

**OpenReview:** eIDa6pd9iQ
**User label:** "This is trying to address our exact problem"

**Main idea:** Replace short residual connections with long feedforward connections from each layer to the output. During gradient descent, information naturally migrates to earlier layers, revealing redundancy in deeper ones. Post-training: prune the redundant layers.

**Results:** 30-80% architectural compression across ViT, MLP-Mixer, BERT. 18% reduction in catastrophic forgetting. Works without changing the loss function.

**Relevance assessment:** This is solving a related but different problem. ACNs require *training-time* architectural modification — you can't apply this to an already-trained network. It's also still SGD-based (just with a different architecture). The compression is made possible by training, not derived from the geometry of what the network learned.

**Why still useful:** ACNs provide empirical evidence that MLPs are genuinely over-parameterized for the functions they learn (30-80% of layers become redundant). This is consistent with our hypothesis. The mechanism (pushing info to earlier layers via long connections) is a training-time version of what we want to do analytically.

---

### Manifold Flattening and Reconstruction (Psenka, Pai, Raman, Sastry, Ma, JMLR 2024)

**JMLR:** 23-0615
**User label:** "Related manifold view work"

**Main idea:** FlatNet learns a pair of networks (encoder/decoder) that flatten an embedded submanifold to a linear space and reconstruct it. The flattening uses iterative geometric flows; the key property is theoretical guarantees on reconstruction fidelity from finite samples. Validated on synthetic high-dimensional manifolds and 2D image datasets.

**Connection to our problem:** If the data lives on a low-dimensional manifold, the network only needs to compute a piecewise affine function on that manifold — not on all of ℝⁿ. The number of linear regions needed is determined by the manifold dimension, not the ambient dimension. FlatNet's encoder is essentially doing what a compressed network would do: map the manifold to a compact linear representation.

**What it gives us:**
1. Theoretical grounding for why compression is possible: the function is low-complexity on the data manifold even if it looks high-complexity in ambient space.
2. The flattening approach is a constructive way to find the low-dimensional embedding — but requires training FlatNet, which is again SGD-based.

**Limitation:** Requires training a new network (encoder/decoder), not analytic extraction from a trained network.

---

### Curvature Tuning (Hu, Gamba, Balestriero, 2025)

**arXiv:** 2502.07783

**Main idea:** Inject a single hyperparameter into activation functions to modulate decision boundary curvature without weight updates. Viewed through the spline lens: the activation function parameterization controls the shape of the piecewise affine spline. One hyperparameter can project the network onto a smoother function space.

**Connection to our problem:** This shows that the piecewise affine structure of a ReLU network can be *globally modified* by adjusting activation function parameters. For compression: if we find the curvature parameter that best approximates the original function with a simpler partition, this is a form of analytic simplification. Limited to a 1D space of functions though.

**Verdict:** Weak connection. More relevant to fine-tuning/robustness than compression.

---

## CONTEXT (background, not directly applicable)

### Geometry of Grokking (Musat, 2025)

**arXiv:** 2511.01938
**User label:** "Potentially useful"

Explains grokking via norm minimization on the zero-loss manifold. Uses differential geometry and Morse theory. **Not relevant to compression** — about generalization dynamics during training, not about the structure of what the trained network computes.

---

### Flatness and Neural Collapse (Han, Adilova, Petzka, Kleesiek, Kamp, 2025)

**arXiv:** 2509.17738
**User label:** "Possibly good theory paper"

Relative flatness predicts generalization; neural collapse is not necessary for grokking. **Not relevant** — about loss landscape geometry, not functional structure.

---

### Into the Rabbit Hull / Minkowski Representation Hypothesis (Fel, Balestriero et al., 2025)

**arXiv:** 2510.08638

SAEs on DINOv2 show representations cluster around discrete archetypes via convex combinations. Minkowski geometry of representations. **Relevant to interpretability, not compression of MLPs.** The transformer attention context makes this distant from our problem.

---

## DEAD ENDS

### Tropical Geometry (Zhang 2018, Maragos 2023/2024)

**arXiv:** 1805.07091, 2306.15157, 2409.03945
**Cycle 002:** `autoresearch/results/cycle_002_summary.md`

The tropical polynomial representation has one monomial per linear region — exponentially large for deep networks. The only practical algorithm (TropNNC) does K-Means on weight rows using zonotope geometry as motivation, which is exactly the same heuristic as everything else. **Do not revisit tropical algebra as a compression framework.** The one useful takeaway (cluster similar neurons by cosine similarity) is already covered by the Centroid Affinity Hypothesis more rigorously.

---

### KAN Architectures

Dead as a compression target. Overfitting with L-BFGS; 30x slower than MLP; no advantage over distillation. See MEMORY.md.

---

## Open Threads (for next experiments)

### Thread 1: Region census + dual graph (immediate, empirical)
Implement activation pattern census on spirals/moons networks, build the dual graph, compute weighted Fiedler partition. This tests whether the partition is actually sparse (few regions visited) and gives us a free clustering for merging. Depends on: nothing. Blocks: region merging experiments.

Key question: **Is the Fiedler partition of the activation graph actually aligned with the decision boundary?** If yes, we have a principled basis for coarsening the partition.

### Thread 2: Local complexity measurement (foundational, empirical)
Implement local complexity from Patel & Montúfar 2412.18283. Compute it across layers of trained MLPs as a function of training progress. Is it low? Does it correlate with the compression potential measured by effective rank?

Key question: **Does local complexity decrease during training, and does it reach low values on our test problems?**

### Thread 3: Centroid-based neuron similarity (implementation)
Compute Jacobian centroids (average Jacobian row weighted by region volume) for each neuron. Use centroid distance (not weight cosine similarity) as the merging criterion. This is more principled than weight-space clustering.

Key question: **Do Jacobian centroids cluster more clearly than weight vectors? Does centroid-based merging give better accuracy/compression tradeoff?**

### Thread 4: Analytic compressed network construction (hardest, most important)
Given the extracted partition (activation patterns + affine maps from MASO), construct a minimal MLP that implements the same partition. The key theoretical question: what is the minimal number of neurons needed to implement a partition with K observed regions in ℝⁿ?

Relevant theory: O(K) neurons suffice for K pieces in ℝⁿ (from PWL representation theory). But the construction is not trivial — we need to find weight matrices that implement a given set of hyperplanes and affine maps.

**This is the core unsolved problem.** All threads above are analysis tools that tell us what the function is. This thread is synthesis: building the smaller network from scratch.

---

## What we're missing

1. **A constructive algorithm** that takes (partition, affine maps) → minimal MLP. The theory says O(K) neurons suffice; no practical algorithm exists for constructing the weights. This is the main gap.

2. **Empirical data on our specific networks.** We know spirals uses far fewer regions than the theoretical max (from the Balestriero survey), but we haven't measured it for our actual trained models.

3. **A clear connection between local complexity and compression ratio.** The theory says low complexity = compressible, but nobody has shown: "network with local complexity X can be compressed to Y parameters with Z% accuracy loss."
