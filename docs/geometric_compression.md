# Geometric Neural Network Compression Using KANs - OBSOLETE
Note: KAN failed for a very clear reason: it's crap at high dimensional data. Did loads of experiments, in no experiment was KAN able to beat an MLP of the same size.


## What This Subproject Is About

Neural networks trained via gradient descent are massively overparameterized relative to the functions they learn. The optimization process (SGD, Adam, etc.) *requires* this redundancy — it's how gradient-based methods navigate the loss landscape. But the resulting trained function is typically much simpler than the network that represents it. A ReLU network, for example, builds piecewise linear approximations of smooth curves using many small linear segments, each requiring its own parameters. The network is the shipyard; the function is the ship.

**The thesis:** given a trained network and the dataset it was trained on, we can extract the geometric structure of the learned function and re-encode it in a much smaller network — not by pruning or distilling via gradient descent, but by *reading out the geometry* and constructing a compressed representation analytically or semi-analytically.

This is different from distillation. Distillation trains a small network against the big network's outputs using SGD, which means the small network inherits the same overparameterization problems. We want to bypass SGD for the compression step entirely, or reduce it to a brief cleanup phase.

**The long-term goal** is a general-purpose "compression algorithm" (possibly a meta-network / hypernetwork) that takes any trained network + dataset and produces a compressed network. This project is the first experimental step toward that goal.

## Why Implicit Representations on Meshes

We use implicit neural representations (INRs) fitted to 3D mesh data as the testbed. This is NOT a project about NeRFs, 3D reconstruction, or computer graphics. We are using this domain because:

1. **The geometry is literal.** The function maps 3D coordinates to properties (occupancy, SDF, color). Geometric structure in the function corresponds to actual spatial geometry, making it easy to visualize and debug.
2. **The input is low-dimensional (3D).** This lets us validate the approach thoroughly before moving to higher dimensions. We can densely sample, compute Jacobians cheaply, and visualize everything.
3. **Results are immediately visual.** Compression quality is obvious — you can see if the compressed network preserves surfaces, edges, and smooth regions.
4. **It's simple to set up.** No volumetric rendering pipeline, no camera models, no image formation. Just: sample points on/near a mesh → train a network to predict occupancy/SDF/color → compress that network.
5. **It will generalize.** The compression method we build here must NOT depend on the input being 3D. Everything we build should work identically if the input were 100-dimensional — we'd just lose the ability to visualize.

**Critical constraint: every design decision must be made with high-dimensional generalization in mind.** If a technique only works because we can densely grid-sample 3D space, it's not useful. The techniques must work on-distribution (using only data points we actually have) and must not assume spatial structure in the input domain.

## This Experiment: Jacobian-Matching Compression with KAN Target

### Core Idea

The local differential structure of a trained network — its output, Jacobian (first derivative), and Hessian (second derivative) at data points — captures the geometry of the learned function on the data manifold. We use this information to:

1. **Decide where complexity lives:** High Hessian magnitude = high curvature = the function is doing something nonlinear here = needs more capacity. Low Hessian = locally linear = trivially compressible.
2. **Construct a compressed network analytically:** Using KAN (Kolmogorov-Arnold Networks) as the target architecture, because KAN edges use learnable spline functions that can natively represent smooth curves — unlike ReLU networks which must approximate them with many linear segments.
3. **Match the geometry, not just the outputs:** The loss for evaluating compression quality includes Jacobian matching, which forces the compressed network to have the same local linear behavior as the original. This is strictly more informative than output-only matching.

### Why KANs

In a ReLU network, 50 neurons might collaborate to approximate a smooth curve via piecewise linear segments. In a KAN, a single spline edge with a few knots can represent the same curve directly. The mapping from "geometric complexity profile" to "KAN architecture" is natural:

- More knots where curvature is high
- Fewer knots where the function is smooth
- Width determined by effective local dimensionality

KANs are the compressed target. ReLU networks (or any architecture) are the source. The compression reads the geometry from the source and constructs the target.

## Detailed Steps

### Phase 0: Infrastructure (do this first)

**0.1 — Mesh sampling pipeline**

- Take meshes from a standard dataset (ShapeNet, Thingi10K, or even simple procedural meshes — spheres, tori, Stanford bunny)
- Implement point sampling: uniform sampling in a bounding volume, with labels for inside/outside (for occupancy) or signed distance values (for SDF)
- Also sample points ON the surface with normals (for surface-focused analysis)
- Target: ~100K–500K points per mesh, with higher density near surfaces
- **Store these as simple (position, value) pairs.** The mesh is only used to generate the dataset. After this step, we never touch the mesh again — we only work with the point cloud and a network.

**0.2 — Train source networks (ReLU MLPs)**

- Architecture: standard MLP with ReLU activations. Start with something moderately overparameterized — e.g., 4-6 hidden layers, 256 units each. This gives ~200K-500K parameters for a function that should be representable with far fewer.
- Task: predict SDF (signed distance function) from 3D coordinates. SDF is better than binary occupancy because it's a continuous function with meaningful gradients everywhere, not just near the surface.
- Train to convergence with standard Adam optimizer. Nothing fancy — we want a typical overparameterized trained network.
- Train on multiple meshes (at least 10-20 different shapes) to ensure the compression method isn't overfit to one function.
- **Deliverable:** a collection of trained ReLU MLPs, each representing the SDF of a different mesh.

**0.3 — Jacobian/Hessian computation infrastructure**

- Implement efficient batched computation of:
  - Network output f(x) at data points
  - Jacobian ∂f/∂x (3×1 vector for SDF, since output is scalar and input is 3D)
  - Hessian ∂²f/∂x² (3×3 matrix) or at minimum the Hessian trace (Laplacian)
- Use PyTorch autograd (torch.func.jacrev / jacfwd, or torch.autograd.functional.hessian). For Hessians, Hutchinson's trace estimator is fine if exact computation is too slow.
- This must be implemented as a generic utility that takes ANY network and ANY set of input points — not hardcoded to a specific architecture.
- **Test:** verify Jacobians numerically (finite differences) on a few points. This is critical — wrong Jacobians will silently corrupt everything downstream.

**0.4 — KAN implementation or integration**

- Use an existing KAN library (e.g., pykan, efficient-kan) or implement a minimal version.
- Key requirement: we need control over knot placement per edge. Most KAN implementations use uniform knots — we need adaptive/non-uniform knot placement.
- If no library supports this well, implement a simple B-spline KAN layer where knot vectors are parameters we can set directly (not just learn via SGD).
- **Test:** verify a small KAN can fit a known smooth function (e.g., sin(x)) with few knots, and that adding knots in the right places improves accuracy on sharp features.

### Phase 1: Geometric Profiling of Trained Networks

**1.1 — Compute geometric profiles**

For each trained source network and its training data:

- Evaluate f(x), J(x) = ∇f(x), and H(x) = ∇²f(x) at all data points
- Compute scalar curvature measure at each point: e.g., ‖H(x)‖_F (Frobenius norm of Hessian) or the Laplacian |tr(H(x))|
- Compute the effective rank of the Jacobian at each point (trivial for scalar SDF — it's always rank 1 or 0 — but implement it generally for when we move to vector-valued outputs)

**1.2 — Build a complexity map**

- Partition the data points into regions by curvature: low, medium, high
- Visualize this as a heatmap on the mesh surface (for debugging/understanding only — the method must not depend on visualization)
- Expected result: low curvature in flat regions and far from the surface, high curvature near edges, corners, and thin features

**1.3 — Cluster by local affine behavior**

- For each data point, the network is locally an affine function: f(x) ≈ f(x₀) + J(x₀)(x - x₀)
- Cluster data points by Jacobian similarity (e.g., k-means on the Jacobian vectors, or DBSCAN)
- This identifies "macro-regions" where the network is doing essentially the same linear thing
- Count the number of distinct macro-regions — this is a lower bound on the complexity of the function and tells you the minimum capacity the compressed network needs

This phase is analytical — no training, no optimization. It produces a geometric description of the trained function.

### Phase 2: KAN Construction from Geometric Profile

**2.1 — Architecture determination**

From the geometric profile:

- Number of macro-regions → minimum network width
- Maximum curvature → number of knots needed in the highest-complexity regions
- Distribution of curvature → overall knot budget and allocation

Design a KAN architecture. Start simple: same depth as source or shallower, with width determined by the geometric profile. The goal is to have significantly fewer total parameters than the source.

**2.2 — Knot placement**

This is the key novel step. For each spline edge in the KAN:

- The spline's domain corresponds to some subspace of the input
- Place more knots where the geometric profile indicates high curvature along that subspace
- Place fewer (or no) knots where the profile is locally linear

For the initial prototype, a simpler approach: use the curvature map to create a non-uniform sampling density, and place knots according to that density. This doesn't require knowing which KAN edge corresponds to which input subspace — it's a global allocation strategy.

**2.3 — Coefficient fitting (NOT backprop)**

Given the KAN architecture with placed knots, solve for the spline coefficients using least-squares fitting:

- Minimize: ‖f_KAN(X) - f_source(X)‖² + λ_J ‖J_KAN(X) - J_source(X)‖² + λ_H ‖H_KAN(X) - H_source(X)‖²
- Where X is the dataset, f_source is the trained network, f_KAN is the compressed KAN
- The Jacobian and Hessian terms are the geometric matching loss
- If this is linear in the spline coefficients (it is for B-spline KANs with fixed knots), solve it as a linear system. If not, use a direct solver (L-BFGS, not SGD/Adam).

**Important: the point of avoiding SGD is not religious. It's that SGD introduces its own overparameterization dynamics. If a brief SGD fine-tuning step at the end improves results, that's fine — but the initial construction should be analytic.**

### Phase 3: Evaluation

**3.1 — Baselines**

Compare the geometric compression against:

1. **Naive distillation:** train a small KAN (same architecture as our compressed one) from scratch against the source network's outputs using Adam. Same number of parameters. This is the key baseline — if geometric construction beats distillation, the geometry is doing real work.
2. **Small ReLU trained from scratch:** train a small ReLU MLP (same parameter count as compressed KAN) directly on the SDF data. This tests whether KANs are inherently better for this or whether the geometric construction is the key.
3. **Small KAN trained from scratch on data:** train a small KAN on the original SDF data (not distilled). This tests whether having the source network matters at all.
4. **Pruning baseline:** magnitude-prune the source network to the same parameter count. This tests whether our approach beats the simplest structural compression.

**3.2 — Metrics**

For each method, measure:

- **Reconstruction quality:** held-out pointwise error (MSE as default baseline; loss function may evolve)
- **Surface quality (3D diagnostic only):** Chamfer distance and Hausdorff distance between the original mesh and the mesh extracted (via marching cubes) from each network's SDF. These are useful sanity checks when a ground-truth mesh is available, but they require dense grid evaluation + marching cubes and **do not generalize beyond 3D**. Never use them as loss functions — they are evaluation-only diagnostics.
- **Compression ratio:** parameters in source / parameters in compressed
- **Construction time:** wall-clock time for the compression procedure (excluding source training)
- **Jacobian fidelity:** L2 error of Jacobians on held-out points (do the gradients match, not just the values?)

**3.3 — Ablations**

- Jacobian matching ON vs OFF in the fitting loss (does it help?)
- Hessian matching ON vs OFF (does second-order info help beyond first-order?)
- Adaptive knot placement vs uniform knots (does the geometric profile help KAN construction?)
- Vary compression ratio: 2x, 5x, 10x, 20x. Where does each method break?

**3.4 — Visualization**

- For each compressed network, extract and render the mesh (marching cubes on the SDF)
- Show the curvature/complexity heatmap alongside compression error heatmap — do errors concentrate where we predicted complexity?
- Show knot density maps vs curvature maps — are we allocating capacity correctly?

### Phase 4: Analysis and Next Steps

Based on results:

- If geometric construction significantly beats distillation: the local differential structure is sufficient geometric information for compression. Next step: scale to higher-dimensional inputs, try on image classifiers.
- If it matches distillation but doesn't beat it: the Jacobian/Hessian captures the same info that SGD discovers anyway. The geometric profiling is redundant. Pivot to Direction 2 (meta-learning) or Direction 3 (spectral matching).
- If it loses to distillation: the analytic fitting is worse than iterative optimization even with geometric initialization. Consider using the geometric construction as initialization for brief SGD fine-tuning.

## What we need to add as tools:

- A KAN library with non-uniform knot support, or a custom implementation
- trimesh or similar for mesh I/O and point sampling

## What To Avoid

**Do not treat this as a NeRF / 3D reconstruction project.** We are not optimizing for beautiful renders. We are not comparing against Instant-NGP or Gaussian Splatting. We are not using positional encoding or hash grids. The 3D setting is a testbed for a general compression method. Every function we write should work on any MLP mapping R^n → R^m.

**Do not hardcode 3D assumptions.** The Jacobian computation, curvature analysis, knot placement, and KAN fitting must work for any input/output dimensionality. The only 3D-specific code should be the mesh sampling (Phase 0.1) and the marching cubes visualization (Phase 3.4).

**Do not skip the baselines.** The entire point of this experiment is to determine whether geometric information helps compression beyond what naive distillation achieves. Without rigorous baselines, we learn nothing.

**Do not over-optimize the source network.** We want a typical, somewhat overparameterized trained network. Don't use tricks (positional encoding, SIREN activations, etc.) that would make the source already efficient. The point is to compress a bloated standard network.

**Do not add complexity to the source architecture to make it easier to compress.** The source should be a vanilla ReLU MLP. If we can only compress specially-designed networks, the method isn't useful.

**Do not spend time on hyperparameter tuning for the source networks.** They just need to fit the SDF reasonably well. The research effort goes into the compression pipeline.

**Do not use SGD/Adam to optimize the compressed KAN as the primary method.** Analytic construction first, optional fine-tuning second. If the method only works with extensive fine-tuning, it's not geometric compression — it's distillation with better initialization.

## File Structure
Make sure to reuse existing components in the quant-aware-training project and structure the contributions appropriately. The overall project is about using geometry to understand and improve neural networks.

## Definition of Success

**Minimum:** geometric compression achieves comparable quality to distillation at the same parameter count, with faster construction time.

**Target:** geometric compression achieves the same quality as distillation at 2-5x fewer parameters, or significantly better quality at the same parameter count.

**Stretch:** geometric compression matches or exceeds the quality of a small network trained from scratch on the original data, meaning the source network + geometric extraction provides a genuine advantage over just retraining.

Any of these outcomes produces a publishable result. Even negative results (geometric info doesn't help) are informative for the field.
