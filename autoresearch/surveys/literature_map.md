# Geometric Compression — Literature Map

**Last updated:** 2026-03-18 (batch 12)
**Problem:** Given a trained ReLU MLP, construct a smaller network computing the same function — analytically, not by distillation.

---

## THE SYNTHESIS GAP (main open problem)

**What we know:**
1. A trained ReLU network defines a piecewise affine function f: ℝⁿ → ℝ with partition {P_i} and maps {A_i x + b_i}.
2. Rolnick & Kording (2020): the partition **uniquely determines** the network up to permutation and scaling — partition IS function.
3. Hanin & Rolnick (2019): the number of **actually observed** activation patterns is at most N^d (N=total neurons, d=input dim), independent of depth. Far below the theoretical 2^N.
4. Polytope Lens: regions within Hamming radius r of each other have **similar affine maps** — natural merging criterion.

**What we don't have:**
- A constructive algorithm to go from K observed (pattern, affine_map) pairs → minimal ReLU network computing the same function.
- A bound on how many neurons are needed to represent K piecewise affine regions.
- A tested merging procedure that produces a valid smaller network.

**The core algorithmic challenge:**
Two adjacent regions can be merged iff their affine maps agree: A_i = A_j (they're literally the same linear function). When does this happen? When all differing neurons (those with bit flip at the shared boundary) have zero effect on the output. This is testable: check if W_{out} × (W_{layer at boundary}) is zero for the differing neuron column.

**Strongest near-term lead (now confirmed):** Berzins (2306.07212) Section 5.4 demonstrates **78% parameter reduction with no accuracy loss** by:
1. Dead neuron removal (always-inactive → delete)
2. Always-active neuron collapse (always-linear → ≤D neurons per layer)
This is implementable directly from our white-box weight access.

**Next open question:** After removing stable neurons, how do you further compress the unstable (genuinely nonlinear) neurons? This is the hard part — no algorithm exists yet. The synthesis problem remains open for the unstable core.

**Papers to fetch next:** Balestriero et al. NeurIPS 2019 "The geometry of deep networks: Power diagram subdivision" — might bridge power diagrams (SplInterp) to general ReLU networks.

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

### SplineCam: Exact Partition Visualization (Humayun, Balestriero, Balakrishnan, Baraniuk, CVPR 2024)

**arXiv:** 2302.12828
**Code:** https://bit.ly/splinecam-git

**Main idea:** Provably exact method for computing the 2D partition of any CPWL network — the first exact method. BFS on the hyperplane arrangement graph, restricted to 2D slices. Computes partition statistics: region count, region volume, eccentricity, decision boundary.

**Key formula (Equations 3-4, direct use for us):**
For each region ω defined by activation pattern q_ω^ℓ (binary vector for layer ℓ), the affine map is:
```
A_ω^ℓ = ∏ diag(q_ω^i) W^i        (product of masked weight matrices)
b_ω^ℓ = diag(q_ω^ℓ) b^ℓ + sum of accumulated bias corrections
```
This is the **constructive formula** for extracting the affine map analytically from weights + activation patterns. This is what we need for the synthesis direction.

**Empirical findings directly relevant to compression:**
1. "Not all neurons actively partake in the regression task... while for the first layer all 10 hyperplanes intersect the input domain, for the last layer only 4 of the 10 neurons intersect." → **Direct visual evidence of the stable neuron pruning opportunity**
2. "Variation is considerably low between random 2D orientations" → a single 2D slice can characterize the partition structure. Supports the JL + SplineCam approach.
3. DA training → higher partition density near test data → better generalizability (partition density ↔ generalization)

**Limitation:** Only 2D slices. For D=3 (torus SDF) it's directly applicable. For higher-D, use random 2D projections.

**Partition statistics as compression quality metrics:**
- ARV (Average Region Volume), NR (Number of Regions): before vs after compression
- Ideally: same NR at training data locations, lower NR elsewhere

---

### Power Diagram Geometry of Deep Networks (Balestriero, Cosentino, Aazhang, Baraniuk, NeurIPS 2019)

**arXiv:** 1905.08443

**Main result:** Each MASO layer's partition corresponds to a **power diagram** (weighted Voronoi). Composing layers = progressively subdividing the power diagram. The full network's partition is an analytical expression of this progressive subdivision.

**Critical finding:** "The subdivision process constrains the affine maps on the (exponentially many) power diagram regions to greatly reduce their complexity." This means: adjacent regions don't have arbitrary affine maps — they're related by the subdivision structure (derived from the same product of weight matrices, just with different neuron masks). This is the theoretical basis for why many adjacent regions have similar affine maps → mergeable.

**What it gives us:**
1. The partition has a **structured geometric form** (power diagram), not arbitrary polytopes
2. The affine map on each region is determined by the power diagram parameters (centroids, weights) + layer weights
3. Compression = finding a coarser power diagram that approximates the same function on the data distribution
4. **Connects to SplInterp** (2025): SplInterp showed SAE partitions are K-th order power diagrams — Balestriero 2019 shows general ReLU nets produce power diagrams too. Same geometric structure, different architecture.

**Gap:** Provides analytical form but no algorithm for finding the minimal/coarsest power diagram consistent with a given function. The synthesis direction is: given a desired power diagram (fewer cells), derive the network weights.

---

### Deep Grokking: Local Complexity and Delayed Robustness (Humayun, Balestriero, Baraniuk, ICML 2024)

**arXiv:** 2402.15555

**Main finding:** Grokking (delayed generalization) is a **phase transition in the partition structure**:
- Pre-grokking: linear regions randomly distributed across input space
- Post-grokking: linear regions **migrate away from training data** toward the decision boundary
- Post-grokking: the mapping is locally **linear near training samples** (low local complexity there)

**Direct implication for compression:**
A well-trained, grokked network has the sparsest possible partition structure near its training data. The function is locally linear (constant affine map) over large neighborhoods of each training point. This means:
- Compression IS possible without accuracy loss on training data
- The minimal number of regions needed ≈ the number of distinct local linear behaviors on the training set
- After grokking, this number should be much smaller than the number of network neurons

**Compression strategy informed by this paper:**
1. Check if network has grokked (partition density at training data → low after grokking)
2. If grokked: high compression is achievable (function is "simple" near the data)
3. If not grokked: compression may reduce accuracy (function still uses all its capacity near data)

**Connection to Local Complexity (Patel & Montúfar 2025):** Same concept, different formalization. Humayun is empirical/phenomenological (first paper introducing the idea), Patel & Montúfar is the formal mathematical treatment.

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

**BUT: synthesis direction exists for SAEs.** Theorem 2.3 says: given a desired partition (K-th order power diagram), you can derive the SAE encoding parameters analytically. This is a working example of "desired geometry → network weights" for a specific architecture. The question is whether this extends to general MLPs.

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

**Reassessment (after reading full paper):** Harder constraint than expected — all experiments on ℝ² inputs explicitly. Cell complex homology computation becomes intractable for higher-D inputs (authors acknowledge exponential scaling). The Fiedler partition idea is sound and may still be useful for our 2D spirals/moons experiments, but doesn't scale to the compression problem in general. **Downgrade to USEFUL.**

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

### Equivalent and Approximate Transformations of Deep Neural Networks (Kumar, Serra, Ramalingam, 2019)

**arXiv:** 1905.11428

**Main idea:** Formalises the idea of transforming a trained ReLU network into an equivalent or approximate network with fewer units/layers — analytically, no retraining. Three transformation types:
1. **Exact (stable neuron removal):** ReLUs that are always-active or always-inactive for all valid inputs can be safely removed with zero loss. Always-inactive → delete; always-active → fold into next layer's affine map.
2. **Local exact (domain restriction):** For a smaller input domain, more neurons become stable → more can be removed + some layers collapsed.
3. **Global linear approximation:** For any feed-forward ReLU network there exists an equivalent 2-hidden-layer shallow network — but requires potentially exponential width to represent all activation patterns. **Key insight: only need to account for patterns that define actual linear regions, not all 2^N.**

**Results (MNIST):**
- L1-reg + adversarial training → dramatically fewer linear regions (more stable neurons → more compressible)
- This proves intentional training can enable effective lossless compression

**Directly relevant to Phase 1/2:**
- This is the theoretical paper behind Serra 2021 (same group). Proves equivalent transformations exist and gives the stable-neuron criterion.
- "Some layers collapsed" for local exact: if a layer becomes all-stable within a subdomain, its contribution becomes linear → collapse into adjacent layer. This is Berzins §5.4's always-active compression, formally grounded.
- The 2-layer global approximation result shows the depth lower bound for exact representation (log n₀+1 layers). Below this, exponential width needed → don't try to shallow-compress.

**Assessment:** CORE — foundational justification for Phase 1/2. Read before implementing stable neuron removal.

---

### The Geometry of ReLU Networks through the ReLU Transition Graph (Dhayalkar, 2025)

**arXiv:** 2505.11692

**Main idea:** Introduces the **ReLU Transition Graph (RTG)**: nodes = linear regions (unique activation patterns observed on data), edges = pairs of regions differing by exactly one neuron flip (Hamming distance 1). This is a direct formalisation of the Hamming-1 adjacency that underlies region merging.

**Key results:**
1. **Tight combinatorial bounds** on RTG size and diameter — precise counts, not just asymptotics
2. **RTG connectivity proof** — the graph of linear regions is always connected (you can always walk from one region to any other via Hamming-1 steps). This means the partition has no isolated components.
3. **VC-dimension interpreted as RTG graph-theoretic property** — the VC dimension corresponds to structural properties of the RTG
4. **Entropy and average degree of RTG → generalization error bound** — networks with sparser RTGs (fewer edges per region) generalise better
5. "Opens new avenues for compression, regularization, and complexity control rooted in RTG analysis"

**For compression:**
- The RTG diameter bound constrains how many merging steps are needed to collapse the partition
- Connectivity means any target coarser partition is reachable by sequential Hamming-1 merges
- Average degree = average number of mergeable neighbours per region → high-degree regions are "in the interior" of semantically similar clusters → merge candidates

**Assessment:** CORE — directly formalises the combinatorial structure of our merging problem. The RTG is precisely the graph we'd traverse in Phase 3.

---

### Unwrapping The Black Box of Deep ReLU Networks: Interpretability, Diagnostics, and Simplification (Sudjianto et al., 2020)

**arXiv:** 2011.04041

**Main idea:** Uses *local linear representation* — the same MASO framework (activation pattern → local affine map) — to build a practical toolkit for interpretability, diagnostics, and **simplification** of pre-trained ReLU networks. Each input maps to a Local Linear Model (LLM): W_eff(x) = product of active weight matrices along the forward path.

**Key contribution for compression — the merging strategy:**
- Clusters inputs by their LLM (if W_eff(x_1) ≈ W_eff(x_2), merge their regions)
- Proposes an "effective merging strategy" demonstrated to simplify networks while maintaining accuracy
- Validated on simulation, UCI benchmark datasets, and real credit risk data

**What it gives us:**
1. A working implementation of effective weight extraction: W_eff(x) = composition of masked layer weights along activation path. This IS SplineCam's Equation 3-4, implemented end-to-end.
2. A merging criterion: similarity of W_eff across inputs. Inputs mapping to similar W_eff can be served by a single merged region.
3. A practical simplification pipeline for pre-trained networks — closest thing in the literature to our Phase 3 target.

**Gap:** Paper is in the interpretability framing; compression results are not the main evaluation. No quantitative compression ratios reported. The merging strategy is described but not algorithmically specified in detail.

**Assessment:** CORE — must read for the merging strategy details. Full HTML paper worth fetching for algorithm section.

---

### On the Algorithmic Construction of Deep ReLU Networks (2025)

**arXiv:** 2506.19104
**Relevance:** CORE — the only paper we've found that explicitly constructs ReLU networks to implement known functions analytically

**Main idea:** Treats neural networks as algorithms, not trained models. Key contributions:
1. **Constructive synthesis:** Given a target function (e.g., a sorting algorithm), explicitly constructs a ReLU network implementing it exactly — not approximately, not by training.
2. **Recursion + parallelism:** Constructed networks are typically recursive (depth = recursion depth) and parallel (width = parallel branches). Deep networks are more efficient than shallow because recursion depth is bounded by network depth.
3. **Continuity constraint:** ReLU networks must be continuous — this is the binding constraint when constructing networks for discontinuous algorithms. The paper identifies this as the fundamental limitation of the ReLU representation.
4. **Complexity:** Constructed networks can have billions of parameters but are built analytically from the algorithm structure.

**Why this is CORE for our problem:**
- This is the most direct treatment of the synthesis direction: given a piecewise affine function (our compressed target), construct a minimal ReLU network implementing it.
- The recursive structure insight is directly applicable: after extracting the partition (activation patterns + affine maps), the construction of the compressed network can be formulated as a recursive problem on the partition's tree structure.
- Continuity constraint = the fundamental challenge of Phase 3 merging. Two adjacent regions can only be merged (their boundary hyperplane removed) if the affine maps agree on that boundary — exactly the continuity requirement.
- The "neural network as algorithm" framing inverts the usual problem: instead of training → analysis, it's specification → construction. Our compression pipeline needs exactly this: partition + maps → network.

**Gap:** Paper focuses on constructing networks for known mathematical functions (sorting, etc.), not for extracting a smaller network from a large trained one. The "trained large network → small network with same function" direction is not addressed directly. But the constructive tools developed here are the right primitives.

---

### Your Transformer is Secretly Linear (Razzhigaev et al., 2024)

**arXiv:** 2405.12250

**Main result:** Consecutive transformer layers produce representations that are almost perfectly related by a linear map: Procrustes linearity score ~0.99 for GPT-2, OPT, LLaMA. This means `X_{L+1} ≈ A·X_L` — the residual stream changes are near-linear between adjacent layers. Up to 25% of layers can be removed (replaced with identity) with distillation-based recovery, minimal degradation. Fine-tuning *increases* linearity (task-specific simplification).

**What it gives us:**
1. A cheap, per-layer linearity signal (Procrustes similarity) that directly predicts which layers are identity-like → compression candidates
2. Empirical evidence that the residual stream is doing less nonlinear work than the architecture allows, especially in fine-tuned models
3. A principled layer ranking for depth pruning that's complementary to our Jacobian variance approach (item 19): Procrustes measures whole-layer linearity (attention+FFN+residual), not within-FFN variation

**Gap:** Only tested identity removal and linear replacement (replace layer with skip connection or fitted linear map). Did not test affine cascade replacement (our compression move). The interplay between per-layer linearity and the layer's contribution to the overall function is not analyzed — a layer can be linear but still essential if it's a large-magnitude linear transformation.

**Assessment:** CORE — directly informs items 36 (Procrustes profiling), 38 (curvature trajectory), and the broader question of how much of transformer depth is functionally nonlinear.

---

### Emergence of a High-Dimensional Abstraction Phase in Neural Networks (Cheng et al., ICLR 2025)

**arXiv:** 2405.15471

**Main result:** Intrinsic dimension (ID) of representations peaks in middle layers across all tested LMs (OPT, LLaMA-3, Pythia, OLMo, Mistral). The peak layer is the locus of abstract linguistic processing: semantic probing accuracy also peaks there. Post-peak layers progressively reduce dimensionality — the representation stabilizes. Earlier/higher peak correlates with better model quality.

**What it gives us:**
1. The ID peak as a geometric marker dividing the network into "building" (pre-peak) and "refining" (post-peak) phases
2. Post-peak layers are compression candidates: the representation is already abstract, and later layers are doing progressively less new work
3. Methodology for our item 26 (ID trajectory): TwoNN estimator used here, consistent with our planned approach
4. Connection to items 36/43: if linearity increases and Betti numbers stabilize post-peak, we have three independent signals all pointing at the same compression boundary

**Gap:** Only measured ID, did not test whether post-peak layers can actually be removed. The "stabilization" claim is based on ID trajectory, not functional redundancy. A layer could maintain the same ID while performing a critical rotation/projection.

**Assessment:** CORE — provides the geometric framing for depth compression: identify the abstraction peak, compress post-peak layers. Directly informs items 26 (ID trajectory), 36 (linearity profiling), and 43 (topological truncation).

---

## USEFUL

### Shape of Learning: Anisotropy and Intrinsic Dimensions in Transformer Representations (Razzhigaev et al., EACL 2024)

**arXiv:** 2311.05928

**Main result:** Two-phase dynamics of intrinsic dimension (ID) during training: early training *expands* ID (model learns to use more dimensions), late training *compresses* ID (model finds lower-dimensional solutions). Bell-shaped anisotropy profile in decoder layers (highest in middle). Encoder and decoder have qualitatively different geometric profiles.

**What it gives us:**
1. Baseline methodology for our ID/anisotropy measurements (items 26, 38): their TwoNN + anisotropy pipeline is well-validated across model families
2. The expansion→compression training dynamics predict that longer training produces more compressible networks (lower ID at convergence) — connects to item 39 (grokking as compression catalyst)
3. The bell-shaped anisotropy profile gives a second geometric signal (alongside ID) for identifying compression-ready layers: high anisotropy = representation is concentrated in few directions = low effective rank = compressible

**Limitation:** Analysis only, no compression intervention. The connection between measured geometry and actual compressibility is hypothesized but not tested.

**Assessment:** USEFUL — provides methodology and training-dynamics context for our geometric profiling experiments.

---

### A Green AI Methodology: Persistent Homology for BERT Compression (PBCE, Applied Sciences 2025)

**Main result:** 47% parameter reduction for BERT Base using persistent homology (PH) to score neuron importance. Method: compute PH of per-neuron output distributions across a calibration set, use topological features (persistence, Betti numbers) as importance scores, prune neurons with lowest topological importance.

**What it gives us:**
1. Practical evidence that topology-based pruning works at transformer scale — validates the direction of items 42-45
2. A concrete neuron importance metric: PH-derived features outperform magnitude-based pruning at matched sparsity
3. The observation that topologically "boring" neurons (low persistence) are the prunable ones — aligns with our hypothesis that topological triviality indicates redundancy (item 43)

**Limitation:** Neuron-level pruning only, no partition-level or layer-level analysis. The PH is computed on individual neuron output distributions, not on the representation manifold. This is a much simpler use of topology than what items 42-45 propose.

**Assessment:** USEFUL — validates topology as a pruning signal at scale. The neuron-level PH importance scoring could be a quick baseline for our more ambitious topological compression experiments.

---

### Dimensional Collapse in Transformer Attention Outputs (2025)

**arXiv:** 2508.16929

**Main result:** Attention output matrices have intrinsic low-rank structure across models. Source: anisotropy of the output projection matrix W_O, not the attention patterns themselves. Robust across layers, datasets, and model families. The effective rank of attention outputs is substantially below head_dim.

**What it gives us:**
1. Extends our FFN-focused compression story to attention: if attention outputs are intrinsically low-rank, there's compression opportunity in W_O (item 40)
2. The source being W_O anisotropy (not attention patterns) means it's a weight-space property, not data-dependent — potentially exploitable analytically
3. Quantifies the gap between the nominal rank (head_dim) and effective rank across layers, providing a per-head compression budget

**Limitation:** Observational only, no compression algorithm or intervention tested. The low-rank structure might be compensated elsewhere in the residual stream.

**Assessment:** USEFUL — directly motivates item 40 (attention head geometry census) and quantifies the compression headroom in attention that we've been ignoring by focusing on FFN.

---

### Linear-Size Neural Network Representation of Piecewise Affine Functions in ℝ² (2025)

**arXiv:** 2503.13001
**Relevance:** USEFUL — constructive sanity aid for the native-2D reference lane, not the main high-D program

**Main idea:** In `R^2`, a continuous piecewise-affine function with `n` connected pieces can be represented by a sparse two-hidden-layer ReLU network whose size scales linearly with the number of connected pieces, even when the pieces are non-convex.

**Why it matters for us:**
- Gives a concrete synthesis endpoint for the `2D` diagnostic lane once we have extracted/coarsened a CPA atlas.
- Sharpens the distinction between `2D` constructive possibility and the real project target, which is high-dimensional observed input.
- Useful as a reference check for whether our extracted/coarsened 2D region structure is even in the right regime for compact ReLU resynthesis.

**Limitation:** This is explicitly a `2D` result. It should guide sanity checks and native-2D reconstruction experiments, but it does not change the main rule that a compression idea only counts as promising if it still works on high-dimensional embeddings such as `2->100` and `2->1000`.

---

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

### Cut Less, Fold More: Model Compression through the Lens of Projection Geometry (Saukh et al., ICLR 2026)

**arXiv:** 2602.18116

**Main idea:** Frames two dominant calibration-free compression approaches geometrically:
- **Structured pruning** = axis-aligned projection (set weights to zero)
- **Model folding** = low-rank projection via weight clustering (merge similar weight groups)

Both formalised as orthogonal operators. Proves: within rank-distance 1, folding yields **provably smaller parameter reconstruction error** than pruning. Under mild smoothness, also smaller functional perturbation.

**Results:** Evaluated on >1000 checkpoints (ResNet18, ViT-B/32, CLIP, LLaMA 60M/130M). Folding typically achieves higher post-compression accuracy, especially at moderate-high compression. No calibration data or retraining required.

**For our problem:** This is a working no-retraining compression method that achieves good results. Folding = neuron merging via weight clustering — closely related to our Phase 3 goal. The geometric framing (projections as operators) is a useful lens. Key difference from our approach: folding treats weights as vectors to cluster, not as components of a piecewise affine function. No activation pattern information used.

**Assessment:** USEFUL — confirms calibration-free compression works at scale. The folding = weight clustering idea is a practical Phase 3 fallback if exact activation-pattern merging proves too expensive.

---

### Unveiling the Training Dynamics of ReLU Networks through a Linear Lens (Ye, 2025)

**arXiv:** 2511.05628

**Main idea:** Recasts a multi-layer ReLU network as an equivalent single-layer linear model with **input-dependent effective weights**: W_eff(x) = product of weight matrices masked by the activation pattern of x. Tracks W_eff evolution during training. Key finding: as training progresses, W_eff for same-class inputs **converges** while W_eff for different classes **diverges** — class-specific decision boundaries emerge from this convergence.

**Connection to our problem:**
- W_eff(x) is exactly SplineCam's A_ω (the affine map for input x's region ω). This paper provides empirical evidence that W_eff is the right object to track.
- Convergence of W_eff within classes → regions belonging to the same class have similar affine maps → they're mergeable (the affine maps are approximately equal across the class).
- **Direct implication**: after training, partition the inputs by class label, compute W_eff within each class partition, check similarity. High similarity = high compression potential.

**Assessment:** USEFUL — confirms the effective weight framework. The within-class convergence finding is a new compression criterion: regions can be merged when their W_eff has converged to a common value.

---

### The Computational Complexity of Counting Linear Regions in ReLU Networks (NeurIPS 2025)

**arXiv:** 2505.16716

**Main result:** Counting the number of linear regions in a ReLU network is:
- **NP-hard and #P-hard** for networks with just one hidden layer
- **Strongly hard to approximate** for networks with two or more hidden layers
- **Polynomial-space solvable** for some common region definitions (PSPACE upper bound)

**For agenda item 3 (region census):** This rules out exact counting as a practical baseline for deep networks. The approach to use instead: enumerate activation patterns on the training data (not globally). Since we only care about regions visited by data, not all 2^N possible regions, we can count observed patterns — which is trivially polynomial in dataset size.

**Assessment:** USEFUL — directly resolves the algorithmic question for agenda item 3. Exact counting is intractable; pattern enumeration on data is the right approach.

---

### Global Minimizers of ℓp-Regularized Objectives Yield the Sparsest ReLU Neural Networks (2025)

**arXiv:** 2505.21791

**Main result:** For single-hidden-layer ReLU networks, a training objective based on minimizing ℓp quasinorms of weights (0 < p < 1) has **global minima that are exactly the sparsest interpolating networks** — i.e., networks with the fewest nonzero parameters/neurons. Recasts the combinatorial sparsest-network problem as a smooth optimization task (continuous, almost-everywhere differentiable).

**For compression:** This is "compression-aware training" — not post-hoc. The result tells us: if you train with ℓp regularization, you get the sparsest network that fits the data. The compression is baked in at training time. For our use case (post-hoc, no retraining), this is mostly theoretical motivation: it proves that maximally sparse networks that fit the data *exist* and are achievable. The question of finding the sparse network that is equivalent to an already-trained non-sparse one remains open.

**Limitation:** Proven for single-hidden-layer networks only. Extension to deep networks is future work.

**Assessment:** USEFUL — existence proof that sparsest-equivalent networks exist. Motivates the search for post-hoc sparsification. The ℓp quasinorm connection also explains why L1 regularization in Serra 2021 and Kumar et al. 1905.11428 increases stable neurons.

---

### The Geometric Structure of Fully-Connected ReLU Layers (Larsson, 2023)

**arXiv:** 2310.03482

**Main idea:** Formalises the geometry of a single ReLU layer. The layer induces a natural partition of input space into "sectors." Key result: **a ReLU layer = projection onto a polyhedral cone followed by an affine transformation**. This geometric view simplifies the preimage computation for decision boundaries.

**Results:**
- A feed-forward network with one hidden ReLU layer can only generate **at most d distinct decision boundaries** (modulo affine transformation), where d = input dimension
- Explicit characterisation of how adding more layers multiplies the number of reachable boundaries

**For compression:** The "at most d boundaries per layer" result is a compressibility bound. If a trained 1-hidden-layer network only uses ≤d of its neurons' boundaries, the rest are redundant. For deep networks, each additional layer multiplies the decision boundary count, but many multiplications may be trivial (near-identity layers → compressible).

**Assessment:** USEFUL — clean geometric characterisation, relevant for bounding what a single layer contributes.

---

### What Kinds of Functions do Deep Neural Networks Learn? Insights from Variational Spline Theory (Parhi & Nowak, SIAM 2022)

**arXiv:** 2105.03361

**Main idea:** Variational framework for deep ReLU networks. Proves a **representer theorem**: deep ReLU networks are solutions to regularized data fitting over a function space consisting of compositions of Banach spaces of 2nd-order bounded variation (Radon domain). The optimal solutions have:
- Skip connections
- **Low-rank weight matrices** (rank bounded by number of data points)
- Connection to variational splines

**For compression:**
1. The low-rank weight matrix result provides theoretical support for our empirical finding (rank₉₅ ≈ 2 at most layers). Minimum-norm training → low-rank weights → compressible.
2. The Radon-domain bounded variation space is the function class that ReLU networks are biased toward. Functions in this class are maximally "smooth" in a sparsity-promoting sense → fewer linear regions needed.
3. Skip connections are theoretically optimal → don't prune skip connections in residual networks.

**Assessment:** USEFUL — theoretical support for low-rank bias and function class of deep ReLU. Backs up our compression hypothesis from a functional analysis perspective.

---

### Sufficient Conditions for Stability of Minimum-Norm Interpolating Deep ReLU Networks (El Harzli et al., 2026)

**arXiv:** 2602.13910

**Main results:**
1. A deep ReLU network achieves good generalization stability if it contains a stable sub-network **followed by a low-rank weight layer**
2. Without the low-rank assumption on the subsequent layer, stability is not guaranteed even with a stable sub-network

**For our problem:** The low-rank layer requirement connects to the effective rank result from our prior experiments (rank₉₅ ≈ 2 at most layers). Networks trained with weight decay or gradient descent bias tend toward low-rank solutions. The "stable sub-network" concept (sub-network that doesn't change outputs under training point removal) is different from MASO stability (stable-0/1 neurons), but both point to identifying the "inert" part of the network.

**Assessment:** USEFUL (marginal) — generalization stability framework, not compression framework. The low-rank+stable-subnetwork result is suggestive but doesn't give a compression algorithm.

---

### ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization (Shopkhoev et al., NeurIPS 2025)

**OpenReview:** 334128876882158c871dd4bf17ccf8f6555a3cfd

**Main idea:** Training-free depth pruning for Transformers. Identifies contiguous blocks that can be replaced by a single linear transformation estimated from a small calibration dataset. The linear map is fused into adjacent remaining weights (no extra parameters). 25% depth pruning with ~90% performance retention on LLMs.

**Key technique:** Estimates a linear map L such that L @ x_in ≈ x_out for the removed blocks, using a small calibration set. This is the data-driven version of our always-active collapse: when a sequence of layers is "approximately linear" on the data distribution, replace it with a single affine map.

**For our problem:** Directly analogous to our Phase 2 (always-active collapse), but extended to blocks that are approximately (not exactly) linear. Calibration dataset acts as the domain over which linearity is checked — equivalent to checking stability on training data. The key difference: Transformers don't have the MASO geometric structure, so this is purely data-driven (not weight-analytic). For ReLU MLPs we can do this analytically.

**Assessment:** USEFUL — confirms that approximate linearization of network sub-blocks is practical and effective. The estimation approach (calibration + closed-form linear regression) is a practical fallback for Phase 2 when exact stability is hard to verify.

---

### LASER: The Truth Is In There — Improving Reasoning in Language Models with Layer-Selective Rank Reduction (Sharma, Ash, Misra, 2023)

**arXiv:** 2312.13558

**Main result:** Replacing certain weight matrices in pretrained transformers with their **rank-r SVD truncation** *improves* downstream task accuracy. No fine-tuning, no retraining — just truncate W ≈ U[:, :r]·diag(Σ[:r])·V[:r, :]^T and put it back. Tested on GPT-J, RoBERTa, LLaMA-2 across counterfactual reasoning benchmarks. Effect is layer-selective: gains concentrate on **MLP weight matrices in later layers** of the network, with the largest improvements when reducing to small fractions of the original rank (sometimes <10%).

**Their interpretation:** the high-order singular components store noisy/hallucinated information accumulated during training; truncation acts as a denoiser. Higher-rank components are *actively harmful* to held-out performance for some layers.

**Direct relevance to cycle 278/280 on this project:** cycle 280 reproduces this phenomenon on GPT-2 small block 1 c_proj — rank-2 truncation of a (3072, 768) matrix, plus a small B-factor retrain, beats float held-out PPL by 4.86. Cycle 278's eff_rank census provides the picking rule LASER lacked: cycle 280's rank choice (r=2) is predicted by `eff_rank(c_proj_1 OUTPUT) = 1.94` from the output covariance. **The runner has been doing 285 cycles of related work without referencing LASER**; this paper should be the framing reference for the cycle 270/280/282/283 line of results.

**What cycle 280 adds beyond LASER:**
1. Picks the rank from output covariance eff_rank rather than searching r empirically.
2. Adds a small retrain step on the right factor — pure SVD truncation alone gives +6.98% PPL; retrain pushes it to −9.0% (below float). The retrain step is what crosses below float.
3. Reports clean held-out PPL improvement on a disjoint split, not benchmark accuracy.
4. Finds the effect on an **early** layer (block 1), where LASER reported mostly later-layer gains.

**Assessment:** USEFUL — direct precedent for cycle 280. The "rank reduction can improve held-out performance" phenomenon has a name and a 2023 paper.

---

### ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models (Yuan et al., 2023)

**arXiv:** 2312.05821

**Main idea:** Activation-aware SVD compression. Instead of plain `W ≈ U Σ V^T`, scale each input column by its activation magnitude before SVD: `W' = W · diag(s)` where `s_j = ||X[:, j]||` over a small calibration set, then SVD `W'`, truncate, and rescale back: `W ≈ U_r Σ_r V_r^T · diag(1/s)`. This makes the SVD truncation minimize *activation-weighted* reconstruction error, not raw weight reconstruction error.

**Why it matters here:** plain SVD truncation (LASER, cycle 270, cycle 280) ignores which directions the data actually exercises. ASVD's activation weighting is **what cycle 278's output-covariance eff_rank computation already does implicitly** — `Y^T Y` over calib data is exactly the activation-weighted covariance of the output. So cycle 278's picking rule and ASVD's loss function point at the same quantity from opposite ends.

**Direct relevance:** if a follow-up wants to push cycle 280 below 49.03 PPL, the next try should be the **input**-side activation weighting (ASVD) on top of the **output**-side eff_rank picking (cycle 278) — they're orthogonal and combine cleanly.

**Assessment:** USEFUL — provides the activation-aware SVD recipe that cycle 280 should be using. The data-aware vs data-agnostic distinction is the key idea.

---

### SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression (Wang, Chen, Liu, et al., 2024)

**arXiv:** 2403.07378

**Main idea:** Combines (1) data whitening before SVD so the truncation minimizes reconstruction loss in the activation space (similar in spirit to ASVD but with a closed-form whitening derivation), and (2) **layer-wise closed-form weight update** after truncation to compensate for cumulative compression error across layers — "if you've already truncated layers 1..l−1, the optimal SVD for layer l should account for the perturbed inputs it now sees, not the original inputs." Posed as a sequential compensation problem with closed-form updates per layer.

**Why it matters here:** cycle 280's retrain step is a slow Adam-driven version of SVD-LLM's closed-form sequential update. The 4000 Adam steps on B + bias buy 8.6 PPL of recovery; SVD-LLM's closed-form update buys most of the same recovery in one shot. If you want to scale cycle 280's recipe to multiple blocks simultaneously without per-block Adam runs, SVD-LLM's sequential closed-form is the textbook approach.

**Direct relevance:** cycles 270, 280, 282, 283 are all **single-block** rank reductions retrained in isolation. The cycle 275 hybrid (rank-2 block 1 + rank-22 block 11) was REFUTED — composition fails. SVD-LLM's sequential compensation is exactly designed for this composition failure: it proves (under their assumptions) that *sequential* truncation + closed-form update *can* compose without the cycle 275 failure mode. The cycle 284 finding that homogeneous SVD-only hybrids compose non-destructively (52.88 PPL) is a partial replication of SVD-LLM's composability claim.

**Assessment:** USEFUL — the canonical reference for "low-rank weight decomposition + retrain on transformers." Provides the composition theory the runner needs to understand why cycle 275 failed and cycle 284 worked.

---

### Who Said Neural Networks Aren't Linear? — The Linearizer (Berman, Hallak, Shocher, 2025)

**arXiv:** 2510.08570

**Main idea:** Given a neural network f, finds non-standard vector spaces where f acts as a linear operator. Constructs a "Linearizer" architecture: f(x) = g_y⁻¹(A g_x(x)) where g_x, g_y are invertible neural networks and A is a linear operator. Uses algebraic "transport of structure" — the vector space operations on X and Y are defined through g_x and g_y. Composition of two Linearizers sharing a network = another Linearizer.

**Applications:** Single-step diffusion, idempotent generative models, modular style transfer.

**For our problem:** The Linearizer identifies the "hidden linear structure" of a nonlinear network. If the target function f is already approximately linear near the data, g_x and g_y can be trivial → f IS linear on the data → 1 region → maximally compressed. This is a different path to the same conclusion as our grokking compression argument. Not directly a compression algorithm but provides a theoretical lens.

**Assessment:** CONTEXT — elegant theoretical framework, different application domain. File for the "what does it mean for a ReLU network to be linear?" thread.

---

### Few Activation Patterns in Practice (Hanin & Rolnick, NeurIPS 2019)

**arXiv:** 1906.00904

**Main result:** The average number of distinct activation patterns in a ReLU network at initialization (and during training) is bounded by N^d, where N = total neurons, d = input dimension. This bound is **independent of depth** and empirically tight — even on memorization tasks.

**Why it matters for compression:**
- Theoretical max is 2^N (exponential in neurons). Actual patterns are N^d (polynomial in neurons, exponential in input dim).
- For our spirals network (2→32×12→1): N=384 neurons, d=2 inputs → bound = 384² ≈ 148k patterns. But we likely see far fewer in practice because training uses far less than the expressivity budget.
- Compression argument: the function only needs to be correct on the ~N^d regions that actually matter. The rest (exponentially many) are never visited and can be ignored.
- This directly bounds the complexity of the "synthesis target" — we need to represent at most N^d distinct regions, not 2^N.

**Caution:** The bound N^d scales **exponentially in input dimension d**. For high-D problems (e.g. CIFAR d=3072), this bound is vacuous. Only useful for low-D inputs or after dimensionality reduction.

---

### ReLU Stability and Exact Compression (Serra, Yu, Kumar, Ramalingam, NeurIPS 2021)

**arXiv:** 2102.07804
**Code:** https://github.com/yuxwind/ExactCompression

**Main idea:** A ReLU unit is *stable* on a given input domain if it always fires (positive pre-activation) or never fires (zero pre-activation) for all inputs in that domain. Stable units can be eliminated analytically without any retraining:
- Always-inactive (stable-0): remove entirely (contributes nothing)
- Always-active (stable-1): the ReLU is effectively linear → can fold into the affine part of the next layer

**Algorithm:** Instead of solving one MIP per neuron (prior art), solve **one MIP for all neurons jointly**, achieving 183x speedup over state-of-art on CIFAR-10.

**Results:**
- With L1 regularization (no accuracy drop): up to 56% parameter removal on CIFAR-10
- Works on 5×100 and 2×800 networks in minutes

**Critical for our approach:**
1. This is the cleanest form of "no-retraining compression" — provably exact on the given domain
2. The stability check needs the input domain to be bounded. For our networks: use the training data bounding box or convex hull.
3. L1 regularization during training increases the fraction of stable neurons → compression-aware training (but NOT our use case — we want post-hoc)
4. Without L1, typical networks have fewer stable neurons, but still some. The fraction depends on architecture and task complexity.

**Gap:** Requires solving a MIP per network, which has worst-case exponential complexity. In practice fast for small networks. For large networks, approximate methods (interval arithmetic, LP relaxation) are needed.

---

### Polyhedral Complex Extraction via Edge Subdivision (Berzins, 2023)

**arXiv:** 2306.07212
**Code:** https://github.com/arturs-berzins/relu_edge_subdivision

**Main idea:** Instead of subdividing regions (redundant: 2^(D-1) regions share each edge), subdivide **edges**. Sign-vectors (binary activation patterns at each vertex) encode all combinatorial structure. Linear interpolation between vertices with opposite signs finds new boundary vertices. GPU-accelerated tensor ops throughout.

**Performance:**
- O(|V|) optimal (with sorting: O(|V| log |V|)), hash-table improvement = linear
- 20x faster than SplineCam for D=2
- Seconds for millions of cells on consumer GPU (RTX 3090)
- D>8 impractical due to exponential growth of complex (hard limit)

**Pruning section (Section 5.4) — most relevant for us:**
1. **Dead neurons**: neuron whose folded hyperplane never intersects the input domain boundary → always-inactive → remove entirely
2. **Always-active neurons**: neuron whose whole shape is in the positive half-space of ReLU → contributes fixed affine function → any >D always-active neurons in one layer are linearly dependent → **compress to D neurons with adjusted outgoing weights**
3. **Demonstrated result**: 2,50,50,1 → 2,25,19,1 (2751 → 589 params, **78% reduction** with no accuracy loss!)

**Key insight (point 2):** If a layer has k neurons all always-active, their outputs are always on the linear region, so they just compute W_2 @ W_1 x = (W_2 W_1) x — a single linear map. This is equivalent to D neurons in the layer. No ReLU, no nonlinearity contributed.

**Connection to our problem:** This is the specific case of the synthesis gap that IS tractable:
- Identify stable neurons in each layer
- Remove dead neurons (zero params)
- Merge always-active neurons into the effective linear map (at most D survivors)
- The remaining "unstable" neurons define the piecewise linear structure
- Build the compressed network from the unstable neurons only

**Limitation:** Only applies to the bounded input domain defined by the extracted complex. For distribution-based compression (data-driven), use the training data convex hull as the domain.

**Note:** References "Balestriero et al. (2019) The geometry of deep networks: Power diagram subdivision" (NeurIPS 2019) — connects power diagrams to deep network geometry (pre-SplInterp). Should fetch.

---

### Bounding and Counting Linear Regions (Serra, Tjandraatmadja, Ramalingam, ICML 2018)

**arXiv:** 1711.02114

**Main results:**
1. Tighter upper/lower bounds for max number of linear regions in rectifier networks
2. First exact method for counting regions: formulated as mixed-integer linear programming (MIP)
3. Key condition: "a deep network can only have more regions than a shallow counterpart with the same number of neurons **if that number exceeds the input dimension**"

**For compression:**
- The condition "neurons > input_dim" is always true for our networks (2→32×12 has neurons >> 2). So deep architectures can represent more regions than shallow ones → depth is meaningful for our networks.
- The MIP approach for counting regions later evolved into the MIP approach for finding stable neurons (Serra 2021)
- A MNIST network (784-dim input, 22 total hidden neurons) produces 10^7 regions — computation takes tens of hours. **This is why approximate stability checking matters.**

**Assessment:** CONTEXT — foundational background for Serra 2021 (which supersedes this for compression purposes). Depth separation result confirms that depth reduction is expensive.

---

### Depth Separation for ReLU Networks (Arora et al., ICLR 2018)

**arXiv:** 1611.01491

**Main result:** Depth separation theorem — there exist functions representable by a depth-k² ReLU network (total size k³) that require at least ½k^(k+1) neurons if constrained to depth ≤ k. The lower bound uses zonotope geometry.

**Why it matters for compression:**
- Hard limit: you cannot in general compress a deep network to a shallow network without exponential blowup in width.
- **This bounds what "geometric compression" can achieve**: if the target function was learned specifically because of depth (depth efficiency), compression must preserve depth structure.
- For our use case: width reduction within the same depth is safe; depth reduction is exponentially costly in general.

**Practical implication:** Our compression strategy should target:
1. Pruning dead neurons (within-layer)
2. Merging neurons with nearly-identical activation patterns (within-layer)
3. NOT trying to reduce depth — this risks exponential representation loss

---

### Deep Learning as a Convex Paradigm: Minimizing Circuit Size with ResNets (Jacot, 2024)

**arXiv:** 2511.20888
**Relevance:** USEFUL — circuit complexity ↔ compression duality; HTMC norm characterises what gradient descent implicitly minimises

**Main idea:** Gradient descent on overparameterised ResNets minimises a functional norm (HTMC norm) that equals the **minimal binary circuit size** needed to ε-approximate the learned function. Three claims:
1. **Computational Occam's razor:** DNNs trained by SGD implicitly find the function with smallest circuit representation (among those fitting the data).
2. **Convexity in HTMC regime (γ > 2):** The set of functions ε-approximated by circuits of size ≤ c·ε^{-γ} is convex when γ > 2. This makes the optimisation landscape benign.
3. **ResNet = almost-minimal circuit:** The skip connections in ResNets correspond directly to circuit "shortcuts" — the residual block implements the increment, the skip adds it. ResNet norm ≈ minimal circuit size.

**Relevance to compression:**
- The HTMC norm gives a **principled complexity measure** for the learned function — not parameter count, not VC dimension, but circuit complexity. This is exactly the quantity a compression algorithm should target.
- If a trained network's function has HTMC complexity K (small circuit size), there exists a smaller network of roughly K neurons implementing the same function. The paper doesn't give the constructive algorithm but identifies the right objective.
- For our 3-phase algorithm: Phase 3 (merging unstable neurons) corresponds to finding the minimal circuit for the piecewise affine function restricted to the observed regions. HTMC norm is an upper bound on this.
- The convexity result (γ > 2) is relevant to whether compression-by-optimisation is tractable: if the target's HTMC complexity is in the convex regime, there's a polynomial-time algorithm to find it (in principle).

**Honest assessment:** This is a theory paper; the HTMC norm is defined formally but not yet practical. The constructive link between "ResNet of width W trained by SGD" and "minimal circuit" is approximate. Still, the circuit-complexity framing is the cleanest theoretical basis we've seen for why compression is possible — it says gradient descent already finds something close to minimal.

---

### A Generalized Spectral Framework to Explain Neural Scaling and Compression Dynamics (Zhang et al., 2024)

**arXiv:** 2511.07892
**Relevance:** USEFUL — spectral characterisation of compression potential; invariant relation between learning speed and compressibility

**Main idea:** Introduces a generalized spectral evolution function g(λ, t; β) that replaces the linear kernel form g(λt) = λt used in NTK/DMFT theory. The key contributions:
1. **Spectral-temporal elasticity:** Defines α(λ, t) = (∂ log g / ∂ log λ) · (∂ log g / ∂ log t)⁻¹ — a local measure of how much the network's spectral amplification changes with frequency vs. time.
2. **Invariant relation:** There is an invariant quantity relating the learning dynamics (how quickly a frequency component is learned) to its compressibility (how much it can be pruned post-training). Faster-learned components are less compressible; slower-learned components (high-frequency) are more compressible.
3. **Unifies scaling and compression:** The same spectral function explains both (a) neural scaling laws (test loss vs. data/compute) and (b) compression dynamics (accuracy vs. parameter count).

**Relevance to compression:**
- The invariant relation is practically useful: it says that for a trained network, the singular vectors of each weight matrix learned first (low singular values early in training) will also be most robust to removal. This gives a **training-informed pruning criterion** beyond simple singular value magnitude.
- For our setting (post-training, white-box access): after training a ReLU MLP, use the spectral dynamics to identify which activation directions are "late-learned" (high-frequency, slow convergence) and therefore most compressible.
- The compression curve g(λ, t; β) evaluated at t=∞ gives the final singular value spectrum; the β parameter controls how fast later components are learned — if β is large, the spectrum drops off fast → network is highly compressible.

**Honest assessment:** The framework is developed for smooth networks (kernel regime) and may not directly transfer to ReLU networks where the activation geometry changes the spectral structure fundamentally. The "invariant" holds under assumptions about the training trajectory that ReLU networks may not satisfy. Still, the empirical prediction — slower-learned singular vectors are more compressible — is testable on our networks and would be a useful additional compression criterion alongside the geometric Phase 1/2.

---

### Fast and Exact Enumeration of Deep Networks Partition Regions (2024)

**arXiv:** 2401.11188
**Relevance:** USEFUL — first exact algorithm for partition enumeration; directly enables region census experiments

**Main idea:** First parallel algorithm for exact enumeration of all linear regions of a piecewise affine deep network. Key findings:
1. **Exact vs. approximate:** Prior work (including SplineCam) only computed partitions on 2D/3D slices or estimated by random sampling. This gives the exact full-dimensional partition.
2. **Complexity scales linearly** with input dimension and region count — not exponentially.
3. **Key empirical finding:** Uniform random sampling is efficient for finding large-volume regions, but exponentially costly (in input dimension) for small regions. The distribution of region sizes is highly non-uniform — most of the partition's volume is in a few large regions, but many small regions exist near boundaries.

**Relevance to compression:**
- Directly implements the region census needed for agenda item 3. We can run this on our spiral/moons networks and get the exact region count and size distribution.
- The large-volume / small-volume split is crucial: large regions are stable (far from boundaries) and candidates for Phase 1/2 removal. Small regions are near boundaries and harder to compress.
- The linear scaling with input dimension makes this tractable for our networks (2D input spirals → trivially fast; 100D embedding → may be large but feasible).
- Running this tells us: how many distinct activation patterns are actually observed, how concentrated the volume is, and where small "corner" regions are.

---

### Training Dynamics of Deep Network Linear Regions (Balestriero group, 2023)

**arXiv:** 2310.12977
**Relevance:** USEFUL — explains where linear regions are at the end of training; directly informs compression strategy

**Main idea:** Studies how the partition of input space evolves during training for piecewise affine networks. Three key findings:
1. **Phases of local complexity:** Local complexity (density of linear regions around data points) follows a multi-phase curve: decreases after init, then rises, then descends.
2. **Region migration:** In the final descent phase, linear regions migrate **away from training/test data** toward the decision boundary. The network becomes nearly linear everywhere except at the decision boundary.
3. **Generalization correlation:** The different LC phases correlate with memorization vs. generalization regimes.

**Relevance to compression:**
- The migration finding is critical: after training, the complex partition structure is concentrated at the decision boundary. The vast majority of input space (where data lives) is nearly linear = just a few large affine regions.
- This directly justifies Phase 1/2 compression: most neurons are "dead" or "always-active" on the data distribution not because of pathology, but as a result of the training dynamics (regions migrate away from data).
- For our spirals experiments: by the time training converges, the bulk of the spiral interior has nearly-flat activation — only neurons near the decision boundary have complex patterns. Compression ratio ∝ boundary length / total volume.
- Measurement strategy: run region census at several training checkpoints to observe the migration, not just at final checkpoint.

---

### Gradient Descent Quantizes ReLU Network Features (Saxe group, 2018)

**arXiv:** 1803.08367
**Relevance:** USEFUL — SGD implicitly concentrates weight vectors to a small set of directions; compression can exploit this

**Main idea:** For feedforward ReLU networks trained with gradient descent (including plain GD) from small initialization:
1. **Quantization effect:** Weight vectors concentrate at a small number of directions determined by the input data distribution — finitely many directions, independent of network size.
2. **Data-determined directions:** The discrete set of weight directions is determined by the input data, not by random initialization or network width. Larger network → more weights pointing in the same directions.
3. **Analogy to linear interpolation:** The quantization of weight directions is analogous to linear interpolation — weights in over-parameterized networks end up as linear combinations of a small basis.

**Relevance to compression:**
- If SGD already concentrates weight vectors to K distinct directions for a dataset of N points (K << N and K << W), then a compressed network with K neurons per layer represents the same function.
- This is an implicit justification for why Phase 2 (always-active neuron collapse to ≤D directions per layer) works: training already pushes the neurons toward a D-dimensional subspace.
- The data-determined directions are exactly the basis for the affine maps A_ω in each region — if all weight vectors concentrate to K directions, then the effective rank of each A_ω is ≤ K.
- Practical implication: after training, compute the clustering of weight directions per layer. The number of distinct clusters = effective width needed for that layer.

**Honest assessment:** Published 2018, before modern understanding of feature learning. The "quantization" is an asymptotic statement for small init/LR and may not hold for modern training. But the direction is right: SGD doesn't explore the full weight space randomly — it finds structured solutions.

---

### Defining Neural Network Architecture through Polytope Structures of Dataset (2024)

**arXiv:** 2402.02407
**Relevance:** USEFUL — derives upper/lower bounds on required network width from dataset polytope structure; inverse algorithm recovers polytopes from trained network

**Main idea:** Connects the geometric complexity of a dataset (expressed as polytope structure) to the required neural network width.
1. **Width bounds:** Upper bound = number of polytope facets needed to separate classes. Lower bound = from simplicial complex structure of the boundary.
2. **Manifold generalization:** For datasets on manifolds, the required width scales with the geometric complexity of the manifold.
3. **Inverse algorithm:** Given a trained network, extract the polytope structure of the dataset it was trained on. Applied to MNIST/CIFAR10: these datasets can be "encapsulated" by ≤2 polytopes with few faces.

**Relevance to compression:**
- The width bound is a direct answer to "what is the minimal network width?" — it depends on the dataset geometry, not the training procedure.
- The inverse algorithm is exactly what we need for region census: run it on our spiral/moons networks to extract the polytope structure the network has learned to implement.
- "≤2 polytopes with few faces" for MNIST means the learned function is geometrically simple despite high-dimensional input — strong compression is possible.
- For spirals/moons (2D, clear decision boundary): the polytope structure should be recoverable exactly, giving us the minimal width analytically.

---

### Getting Away with More Network Pruning: From Sparsity to Geometry and Linear Regions (2023)

**arXiv:** 2301.07966
**Relevance:** USEFUL — directly links pruning sparsity to linear region count; derives per-layer sparsity bounds from partition geometry

**Main idea:** Studies how weight sparsification (pruning) affects the piecewise linear partition. Three contributions:
1. **Empirical finding:** Accuracy degradation from pruning tracks degradation in the number of observed linear regions — region count is the right intermediate variable between sparsity and accuracy.
2. **Theoretical bound:** Derives an upper bound on the maximum number of linear regions as a function of architecture *and* sparsity level. The bound decreases as sparsity increases (fewer nonzero weights → fewer activation combinations possible).
3. **Per-layer sparsity optimization:** Uses the region-count bound to derive the optimal per-layer sparsity distribution that maximizes region count at fixed overall sparsity. Outperforms uniform pruning at matched sparsity.

**Relevance to compression:**
- Confirms the region count ↔ accuracy link empirically: we can use region count as the primary optimization target for compression.
- The region-count bound as a function of architecture + sparsity can be inverted: given a target region count (sufficient for the task), what is the minimal weight count? This is a direct compression budget calculation.
- Per-layer sparsity optimization is directly applicable to our Phase 1/2 pipeline: instead of uniform pruning, use the region-count bound to allocate "budget" (neurons) per layer optimally.
- Connection to our existing results: the bound here is a coarser version of the geometry we're extracting (it uses architecture statistics, not activation patterns). Our approach is strictly better since we use the observed patterns, not worst-case counts.

---

### TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (Zandieh et al., Google Research, ICLR 2026)

**arXiv:** 2504.19874
**Companion papers:** QJL (2406.03482, AAAI 2025), PolarQuant (AISTATS 2026)

**Main idea:** Two-stage vector quantization achieving near-optimal distortion rates with zero data-dependent preprocessing:

**Stage 1 — PolarQuant (b−1 bits):** Random rotation Π (QR of Gaussian matrix) maps any vector x to y = Πx uniformly distributed on the hypersphere. In high-d, coordinates of y become near-independent with known distribution → optimal scalar quantizer (Lloyd-Max) applied independently per coordinate. Data-oblivious: the codebook depends only on the bit-width, not the data.

**Stage 2 — QJL on residual (1 bit):** Compute residual r = x − x̃_mse. Apply Quantized Johnson-Lindenstrauss: qjl = sign(S·r), where S is a random projection. Store 1 sign bit per coordinate + the residual norm γ = ‖r‖₂. Reconstruction: x̃ = x̃_mse + γ · (√(π/2)/d) · S⊤ · qjl. This eliminates bias in inner product estimation.

**Theoretical guarantees:**
- MSE distortion: D_mse ≤ (√(3π)/2) · 1/4^b — within factor ~2.7 of information-theoretic lower bound
- Inner product distortion: D_prod ≤ (√(3π)/2d) · ‖y‖² · 1/4^b — unbiased estimator
- Lower bounds proved via Yao's minimax + Shannon's source coding bound

**Key results:**
- KV cache: 3-bit quantization, 6× memory reduction, perfect downstream accuracy (Needle-in-Haystack recall 0.997 = same as full precision)
- 4-bit: up to 8× speedup over FP32 on H100
- Nearest neighbor search: outperforms Product Quantization and RabitQ in recall with near-zero indexing time
- Models: Gemma, Mistral, LLaMA-3.1-8B on LongBench, RULER, ZeroSCROLLS

**For our problem — the JL connection:**

1. **Random rotation as geometric normalizer.** PolarQuant's key insight: random rotation makes any vector distribution look uniform on the sphere → optimal scalar quantization becomes data-independent. For weight quantization: if we randomly rotate the weight matrix columns before quantizing, we could use a fixed codebook optimized for the sphere, eliminating per-layer calibration. The rotation is invertible (Π⊤), so we dequantize and rotate back.

2. **JL for error correction.** The QJL stage is a 1-bit error corrector for the residual. This is directly analogous to our correction framework: Stage 1 = quantize (introduce error), Stage 2 = correct the residual. The JL transform compresses the residual to 1 bit per dimension while preserving inner products. Question: could we use JL-compressed residuals as a lightweight alternative to our learned corrections?

3. **Unbiased inner product preservation.** TurboQuant guarantees E[⟨y, x̃⟩] = ⟨y, x⟩ — the quantized representation gives unbiased dot products. For weight quantization, this means the expected output of a quantized linear layer would match the float output exactly (in expectation). The variance of this estimator decreases as 1/4^b. At 4 bits, inner product distortion is 0.047/d — essentially zero for d > 100.

4. **Near-optimal distortion rate.** The 1/4^b scaling is a hard information-theoretic limit. Any b-bit quantizer must incur at least 1/4^b MSE distortion. This tells us the theoretical floor for any quantization scheme. Our topk+lstsq corrections are fighting against this floor — they can reduce effective distortion below what a b-bit quantizer alone achieves, but only by spending additional bits on the correction.

5. **Composing with our geometric analysis.** The pullback metric from Curved Inference (G = U⊤U) could be combined with TurboQuant's rotation: rotate weights into the semantic basis defined by G, then apply PolarQuant. Quantization error would be minimized in the semantically important directions first.

**Limitation:** Designed for KV cache (activation vectors seen once during inference), not for weight matrices (static, accessed repeatedly). Weight quantization has different tradeoffs: codebook/LUT overhead is amortized across all tokens, so more complex schemes are viable. But the theoretical framework (distortion bounds, JL residual correction) transfers.

**Assessment:** USEFUL — the JL residual correction idea is directly applicable. The theoretical distortion bounds set a floor we can reason against. Random rotation as a preprocessing step for quantization is a clean, implementable idea worth testing on weight matrices.

---

### SqueezeLLM: Dense-and-Sparse Quantization (Kim et al., UC Berkeley, ICML 2024)

**arXiv:** 2306.07629

**Main idea:** Post-training quantization of LLMs using two key innovations: (1) **sensitivity-based non-uniform quantization** — k-means clustering of weights weighted by diagonal Fisher information F_ii, so centroids are pulled toward values whose perturbation most impacts the final loss; (2) **dense-and-sparse decomposition** W = D + S — outliers (0.1-0.5% of weights) plus highly sensitive values extracted to sparse FP16 matrix, remainder quantized with contracted range.

**Key results:**
- LLaMA-7B 3-bit: 7.56 PPL on C4 (FP16 baseline 7.08), 2.2× inference speedup on A6000
- Outperforms GPTQ, AWQ, SpQR at 3-bit and 4-bit across LLaMA/OPT families
- Non-uniform quantization with LUT dequantization adds minimal latency overhead vs uniform
- Only 10 calibration samples needed (vs 100-200 for GPTQ/AWQ)
- Grouped GPTQ has severe latency penalty from permutation/non-coalesced access; SqueezeLLM avoids this

**For our problem:**

1. **Fisher-weighted k-means = semantically-weighted quantization error.** This is the same principle as the pullback metric G = U⊤U from Curved Inference, but applied at quantization time: minimize ∑(w_i − Q(w_i))² · F_ii instead of raw ∑(w_i − Q(w_i))². The Fisher diagonal measures "how much does the final output change if this weight moves" — precisely the sensitivity our correction framework targets.

2. **Dense-and-sparse decomposition parallels our topk+lstsq.** Their S matrix stores the values that would cause the most damage if quantized. Our topk correction targets the k directions of largest quantization error. Both identify and separately handle the "hard part" of quantization. The difference: SqueezeLLM preserves original values in FP16 (oracle), we learn corrections post-hoc.

3. **Non-uniform quantization insight.** LLM weight distributions are non-uniform → uniform binning wastes precision. Our geometric analysis of GELU channels (bump-dominated vs compression-important) confirms this: different channels have fundamentally different distributions. Per-channel non-uniform quantization guided by geometric channel type could combine both approaches.

4. **Memory-bound, not compute-bound.** Their analysis that single-batch generative inference is bottlenecked by memory bandwidth (not FLOPs) justifies compression approaches that add compute (LUT lookups, sparse matmul) in exchange for smaller memory footprint. Same economics apply to our correction approach: the correction network adds FLOPs but if it enables lower-bit quantization, net throughput improves.

**Limitation:** No geometric understanding of why outliers exist or which directions they align with. The Fisher diagonal is a scalar per-weight sensitivity — doesn't capture the directional structure our pullback/curvature analysis reveals.

**Assessment:** USEFUL — establishes that sensitivity-weighted non-uniform quantization works at scale. The Fisher-weighting principle directly validates our geometric approach (measure error impact, not raw error magnitude). Dense-and-sparse decomposition is a practical template for deploying corrections.

---

### Curved Inference: Concern-Sensitive Geometry in Large Language Model Residual Streams (Manson, 2025)

**arXiv:** 2507.21107

**Main result:** Defines a pullback metric G = U⊤U on the residual stream (U = unembedding matrix), giving a semantically meaningful inner product where distances correspond to output probability shifts. Under this metric, measures curvature (κ, via 3-point central differences) and salience (S = step distance) of the residual stream trajectory through layers. Key findings on Gemma-1b and LLaMA-3b: (1) salience and curvature are strongly anticorrelated (r = −0.89 in LLaMA), suggesting a "representational budget" — layers either move far or reorient sharply, not both. (2) Both metrics are sensitive to semantic perturbations (concern-shifted prompts), with statistically significant differences (p = 0.006 for curvature in LLaMA). (3) UMAP projections destroy curvature geometry (correlation < 0.47) — native-space analysis is essential.

**What it gives us:**
1. A principled semantic metric for measuring compression error: ||e||_G = ||Ue||₂ instead of ||e||_F. Error in directions that affect output probabilities is penalized more than error in semantically inert directions.
2. Layer-level curvature as a compression difficulty signal: high-curvature layers are performing nonlinear reorientation (risky to compress), low-curvature layers are doing near-linear transport (safer to compress).
3. The salience-curvature anticorrelation could guide compression budget allocation: layers that move far but straight (high S, low κ) may tolerate aggressive compression; layers that bend sharply (high κ) need fidelity.
4. G-weighted lstsq: minimize ||U(y - ŷ)||² instead of ||y - ŷ||² when fitting compressed output projections. Optimizes directly for semantic fidelity rather than Frobenius reconstruction.

**Limitation:** Validated on Gemma/LLaMA (26-28 layers), not on GPT-2 (12 layers — fewer curvature points, marginal statistical power). The connection to compression is hypothesized but not tested — no one has shown that curvature predicts compressibility. The metric is global (same G for all layers); a layer-local pullback (Jacobian of rest-of-network) would be more precise but much more expensive.

**Assessment:** USEFUL → potentially CORE. Provides the first principled geometric framework connecting residual stream geometry to compression decisions. Items 145-150 directly test this.

**Full survey:** `autoresearch/surveys/curved_inference_geometric_interpretability.md`

---

## CONTEXT (background, not directly applicable)

### Geometry of Grokking: Norm Minimization on the Zero-Loss Manifold (Musat, 2025)

**arXiv:** 2511.01938

**Main result:** Post-memorization training = gradient descent minimizing weight norm on the zero-loss manifold. Formally proved in the infinitesimal lr/weight-decay limit. Derives closed-form post-memorization dynamics for the first layer of a 2-layer network.

**For compression (updated assessment):** The norm-minimization result means post-grokking networks converge toward minimum-norm (= minimum-rank) solutions on the zero-loss manifold. This directly supports: (a) our empirical finding that rank₉₅ ≈ 2 at most layers, and (b) why compression should be maximally feasible post-grokking — the network has simplified its weights as much as possible while maintaining zero training loss. The zero-loss manifold framing also suggests: the set of networks equivalent on training data forms a manifold, and minimum-norm solutions are the most compressible points on this manifold.

**Assessment:** CONTEXT (upgraded) — not a compression algorithm, but provides strong theoretical basis for *why* well-trained networks are compressible. Read alongside Deep Grokking (2402.15555).

---

### Space Folding in ReLU Networks (Bréchet et al., 2025 / Manz et al., 2025)

**arXiv:** 2502.09954 (quantitative analysis), 2503.08502 (generalization + regularization)
**Relevance:** CONTEXT — analysis tool for the activation structure; not directly a compression algorithm

**Main idea (2502.09954 — Bréchet):** First quantitative analysis of the "space folding" phenomenon in ReLU networks. Measures how straight paths in input space become non-convex (folded) in activation (Hamming) space. Introduces a range-based measure of folding and proves equivalence between convexity in input space and in activation space.

**Main idea (2503.08502 — Manz):** Generalizes the folding measure to other activations via equivalence classes. Key finding: folding *increases* with depth when generalization error is low, and *decreases* when generalization degrades. Learned symmetries (e.g., reflection invariance) manifest as high folding values.

**Relevance to compression:**
- Space folding = same input region maps to the same activation pattern despite coming from different locations in input space. High folding = many inputs share activation patterns = fewer effective regions = more compressible.
- The correlation between folding and generalization means well-trained networks have high folding = are more compressible. Gives a measurable predictor of compression ratio before running the full region census.
- The symmetry connection: if the learned function is invariant under some transformation (reflection, rotation), the folding captures this and the compressed network only needs to implement one half.

**Honest assessment:** These papers develop analysis tools; they don't give a compression algorithm. Useful for measuring compression potential and understanding why it's high. Not yet actionable for Phase 3.

---

### Geometry-induced Regularization in Deep ReLU Neural Networks (2024)

**arXiv:** 2402.08269
**Relevance:** CONTEXT — geometric explanation of implicit regularization; connection between local dimension and region count

**Main idea:** The output space of a deep ReLU network partitions into regions of distinct local dimension (analogous to a stratified manifold). This stratification is invariant under ReLU symmetries (positive rescalings, permutations). Local dimension correlates with: (a) regularization strength, (b) flatness of minima, (c) saddle-to-saddle training dynamics. For shallow networks, local dimension ≈ count of linear regions.

**Relevance to compression:** Provides a unified geometric explanation for why trained networks find structured solutions. The stratified structure of the parameter space implies that training converges to low-dimensional strata — the same low-dimensionality that makes the function compressible. Context for understanding why Phase 1/2 work (low-dimensional parameter manifold = few effective neurons).

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

---

### Reverse-Engineering Deep ReLU Networks (Rolnick & Kording, ICML 2020)

**URL:** https://proceedings.mlr.press/v119/rolnick20a/rolnick20a.pdf
**User label:** "More ReLU theory stuff"

**Main idea:** Proves that you can reconstruct the full architecture, weights, and biases of a deep ReLU network by querying its outputs — without any prior knowledge of the network's structure. The key insight is that first-layer neuron boundaries are flat hyperplanes in input space; deeper neurons have "bent hyperplanes" that bend exactly where they intersect boundaries from earlier layers. By measuring these bends, you recover inter-layer weights.

**Algorithm (white-box version is simpler than the paper's black-box version):**
1. For each neuron z, its boundary Bz is an (n-1)-dim piecewise linear surface
2. First-layer neurons: Bz is a full hyperplane with equation W₁ᵢ·x + b₁ᵢ = 0 → weights and biases directly readable
3. Deeper neurons: Bz bends when it crosses Bz' for z' in an earlier layer; the amount of bending encodes the weight between z' and z
4. Boundary points found by binary search along line segments; hyperplane equations by local linear regression at boundary points

**Why it matters for our problem:** This is the closest thing to a constructive "function extraction" algorithm. Given a trained ReLU network (white-box: we have the weights), we can:
1. Enumerate boundary hyperplanes from weights directly (no queries needed)
2. Track bending structure to confirm layer assignment
3. Extract the affine map for each region from the masked weight product

The Rolnick & Kording algorithm is designed for black-box access (only output queries). For our white-box case, we already have all the machinery from MASO to do this analytically. But the paper's **theoretical result is important**: it confirms that the piecewise linear partition completely determines the network up to permutation and scaling isomorphisms. This is the theoretical basis for why partition-based compression is lossless — the partition IS the function.

**Key theoretical result:** Theorem 2 — given the set of all boundary points between linear regions, the full architecture and weights are recoverable (up to permutation and scaling). This means: two networks implementing the same piecewise linear function have the same partition up to these isomorphisms.

**What's new vs. MASO:** MASO gives analysis tools; Rolnick & Kording prove the converse: from partition → unique network (up to iso). This closes the loop for compression: if we can construct a network with the same partition, it computes the same function.

**Limitation:** Black-box algorithm requires O(n_in × total_neurons × log n₁) queries. For white-box, we bypass this entirely using weight matrices.

---

### Interpreting Neural Networks through the Polytope Lens (Black, Sharkey et al., Conjecture, 2023)

**URL:** https://www.lesswrong.com/posts/eDicGjD9yte6FLSie/interpreting-neural-networks-through-the-polytope-lens
**User label:** "another research thread relating to SplineCam"

**Main idea:** Proposes polytopes (linear regions defined by activation patterns / spline codes) as the fundamental unit of neural network computation. The key move: instead of interpreting individual neurons or directions, interpret the *polytope* — the set of inputs sharing the same binary activation pattern. Within a polytope, the network is a fixed affine map; between adjacent polytopes (differing by one neuron flip), the maps are similar.

**Empirical findings:**
1. **Monosemanticity:** Clustering spline codes by Hamming distance produces monosemantic groups — regions that "mean" the same thing. Individual neurons are polysemantic; polytope clusters are not.
2. **Semantic boundaries = polytope boundaries:** Inter-class activation pairs have statistically higher polytope boundary density (Hamming distance / Euclidean distance) than intra-class pairs. Effect increases at higher layers. The inter/intra difference is small but highly significant (p=3.8e-46).
3. **Spline code clustering:** Hamming distance in spline code space captures semantic similarity even without magnitude information.

**What it gives us for compression:**
1. **Hamming distance is the right merging metric.** Adjacent regions (Hamming distance 1) implement nearly the same affine map. This is the principled basis for region merging: merge regions within Hamming radius r.
2. **Spline code clustering = coarsening the partition.** Find clusters in Hamming space, assign one representative affine map per cluster. The accuracy loss is bounded by how different the affine maps are within each cluster.
3. **The "representational flow" framing:** Networks achieve representational convergence by routing semantically similar inputs to nearby polytopes. A compressed network with fewer polytopes just routes inputs to coarser regions — acceptable if the affine maps within each cluster are similar.
4. **Monosemantic regions are the natural compression target.** Rather than merging random adjacent regions, merge regions within monosemantic Hamming clusters. This preserves semantic structure.

**Gap:** All experiments on InceptionV1 and GPT2-small (large networks). No experiments on actual compression or accuracy impact of merging.

---

### The Geometry of Concepts: SAE Feature Structure (Li, Michaud, Baek, Engels, Tegmark, Entropy 2025)

**arXiv:** 2410.19750
**User label:** "potentially useful for metrics? Ways to probe networks and find structure?"

**Main idea:** Examines sparse autoencoder (SAE) feature geometry in LLMs (Gemma-2-2b) at three scales:
1. **Crystal (atomic):** Pairwise difference vectors between semantically related SAE features form approximate parallelograms/trapezoids — semantic transformations correspond to consistent directions (man→woman, country→capital). Revealed by projecting out "distractor" directions via LDA.
2. **Brain (meso):** Functionally co-occurring features also cluster geometrically; math/code features form a spatial "lobe." Quantified by mutual information between co-occurrence clusters and cosine-similarity clusters (AMI 954σ above null).
3. **Galaxy (macro):** Feature point cloud has power-law eigenvalue spectrum, steepest in middle layers (slope −0.47 at layer 12 vs −0.24 at early/late layers). Middle layers = information bottleneck.

**Relevance to compression:**
- The power-law eigenvalue finding is directly relevant: middle layers of trained networks have the most compressed (non-isotropic, low-dimensional) representations. This is empirical evidence that the function being computed is low-dimensional relative to the ambient activation space — the classic argument for compression.
- The co-occurrence + geometry alignment (features that fire together cluster together) provides an analysis tool: if you run similar analysis on MLP activations, you get a natural clustering of neurons for merging.
- LDA for removing "distractor" directions is useful: before computing region similarity, project out directions that vary due to irrelevant input features (e.g. word length, input scaling).

**Honest assessment for our problem:** Primarily about LLM interpretability via SAEs. The geometry analysis tools (power-law eigenvalue spectrum, co-occurrence clustering, LDA projection) are applicable to our MLPs but require adaptation. Not a compression paper.

---

### Johnson–Lindenstrauss Lemma + SplineCam Application

**URL:** https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma

**Statement:** For any ε ∈ (0,1) and N points in ℝⁿ, there exists a linear map f: ℝⁿ → ℝᵏ with k ≥ 8(ln N)/ε² such that all pairwise distances are preserved to within (1±ε) factor. The map is a random Gaussian projection (A ~ N(0,1)^{k×n}, scaled by 1/√k).

**Key parameters:** For N=1000 points, ε=0.5 (50% distortion): k ≥ 8·ln(1000)/0.25 ≈ 221 dimensions. For ε=0.1: k ≈ 5500. The bound is tight — you genuinely need O(log N / ε²) dimensions.

**The SplineCam problem:** SplineCam enumerates the exact piecewise linear partition in a 2D affine subspace (a plane cutting through input space). For 2D inputs it's exact. For high-D inputs (n >> 2), a single 2D cut reveals only the partition structure along that particular plane.

**The user's idea — JL-guided multi-projection SplineCam:**

The partition structure in ℝⁿ has an "effective dimension" determined by how many distinct activation patterns appear in the data. If K patterns are observed, the relevant structure lives in a subspace of dimension at most K (each pattern is a vertex in binary code space). JL says this structure can be approximately preserved in k = O(log K / ε²) dimensions.

This suggests a concrete algorithm:
1. Observe K distinct activation patterns from the data
2. Compute a JL random projection to ℝᵏ with k = O(log K / ε²) ≈ 10-50 dimensions for typical K
3. Run SplineCam in this k-dimensional projected space (still computationally intensive, but tractable for k=10-20)
4. The resulting partition in projected space approximates the important structure of the original partition

**What JL buys us:** It tells us that the "intrinsic dimensionality" of the partition structure is determined by K (number of observed patterns), not n (ambient input dimension). For our spirals/moons networks where K << 2^n, the relevant structure may live in a low-dimensional subspace.

**Practical numbers for our setting:**
- Spirals network (2→32×12→1): input dim = 2, so this is already low-D. JL irrelevant.
- Torus SDF (3D input): same.
- Higher-D test case (100D embedding, n=100): if K = 1000 observed patterns, JL says k ≈ 220 (ε=0.5) to see the structure. Still > 2D, so SplineCam still only gives a 2D slice.

**Revised assessment of the user's idea:** JL doesn't directly tell us we can get away with a few 2D projections. What it tells us is: the partition structure has effective dimension k = O(log K). For SplineCam, the right question becomes: how many random 2D projections cover this k-dimensional structure? Answer: O(k) projections of dimension 2 give O(k) independent "views" of a k-dimensional structure. With k = O(log K), that's O(log K) projections.

**Actionable:** For a network with K observed activation patterns, take O(log K) random 2D projections through the data distribution, run SplineCam on each, collect boundary densities across projections. This gives a statistically sound sample of the partition structure without full enumeration. The JL guarantee ensures these projections aren't redundant (they cover independent directions).

**Open question:** How does the partition look in a JL-projected space vs. the original? Is the boundary density in the projected space predictive of the partition's complexity in the full space? This is testable empirically.

---

### Deep Compression: Pruning, Trained Quantization, and Huffman Coding (Han, Mao, Dally, Stanford, ICLR 2016 Best Paper)

**arXiv:** 1510.00149

**Main idea:** Three-stage compression pipeline: (1) magnitude pruning → retrain → sparse representation (CSR with relative indices); (2) k-means weight sharing → retrain shared centroids via gradient aggregation; (3) Huffman coding on indices and centroids.

**Key results:**
- AlexNet: 240MB → 6.9MB (35×), no accuracy loss
- VGG-16: 552MB → 11.3MB (49×), no accuracy loss
- Stages are orthogonal: individually each degrades below 8% model size, combined they reach 3% without loss
- Linear initialization for k-means centroids outperforms density-based (preserves large-magnitude weights)
- Conv layers need 8 bits, FC layers only 5 bits
- 3-7× energy savings from fitting compressed model in SRAM cache

**For our problem:** This is the foundational "pruning + quantization + encoding" pipeline that all modern methods build on. Two relevant details:

1. **k-means weight sharing with gradient-aggregated retraining** is the precursor to SqueezeLLM's sensitivity-weighted k-means. Deep Compression uses uniform k-means; SqueezeLLM adds Fisher weighting. Both do non-uniform quantization via clustering — same family as model folding (Wang et al., 2502.10216).

2. **Linear centroid initialization > density-based.** This is because weight distributions have long tails — density-based puts too many centroids near zero, under-representing the important large weights. Same principle as SqueezeLLM's outlier extraction and our observation that bump-dominated channels (small weights, nonlinear) need different treatment from compression-important channels (large weights, linear).

**Limitation:** Pre-LLM era. Results on AlexNet/VGG; no attention layers. Retraining required between stages. But the conceptual framework (prune → quantize → encode, each stage orthogonal) remains the template.

**Assessment:** CONTEXT — foundational reference. The k-means weight sharing + retraining idea is the ancestor of most modern non-uniform quantization.

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
