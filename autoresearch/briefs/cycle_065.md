# Cycle 065: Topological Truncation — Betti-Guided Depth Reduction

**Item**: 43 (P2, MOONSHOT)

**Hypothesis**: Hidden layer representations progress through distinct topological phases (class separation, manifold untangling) that stabilize before the final layer. A "topological triviality point" T marks where Betti numbers (β₀, β₁) freeze. Layers after T merely refine metrics, not topology. Replacing all post-T layers with least-squares affine map preserves accuracy.

**Method**:
1. Compute persistent homology (Ripser) on activations at each layer
2. Track β₀ (components), β₁ (loops) — identify stabilization point T
3. Fit affine map (layer T → output) via least squares
4. Measure accuracy loss vs. depth reduction

**Scope**:
- Toy (primary): spirals 2D/100D, moons 2D (known topology)
- Stretch: GPT-2 FFN with PCA-projected activations (30D)

**Success**: T ≤ L−2 AND post-T affine replacement ≤5% accuracy loss (toy) / ≤15% ppl (GPT-2)

**Fail**: Betti unstable across depth OR >30% accuracy loss from replacement

**Inconclusive**: Stabilization observed but refinement still task-critical
