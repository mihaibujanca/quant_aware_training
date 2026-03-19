# Item
27. **Persistent homology of activation manifold through depth — toy ReLU**

# Context
Study topological simplification through depth in a trained spirals ReLU network. Hypothesis: early layers entangle classes (high Betti numbers β₀, β₁), late layers untangle them (low Betti → clean separation). Measure whether topological drop correlates with compression ease.

# Dependencies
**Depends on item 20** (CONFIRMED, cycle 010): single-affine collapse quality in toy ReLU. Item 20 computed per-layer residual r (L1=0.43, L2–L4≤0.06), showing deep layers are near-linear. Item 27 will ask if topology aligns: layers with largest Betti drop should have lowest r.

Item 28 (GPT-2 homology) blocks on item 27 outcome.

# Key Files
- `/Users/mihai/Projects/quant_aware_training/aleph/relu_geometric_compression.py` — network architecture, spirals config, inference
- `/Users/mihai/Projects/quant_aware_training/autoresearch/results/cycle_010_single_affine.json` — baseline r values for correlation
- `cycle_019_entropy_cascade.json` — prior entropy signal (conditional H cascade ≠ topology, but reference for similar analysis pattern)

# Constraints
- **Dependency met**: item 20 complete (r values available in cycle_010 artifacts)
- **ripser unavailable** — need `pip install ripser` or `scikit-tda` in venv
- **Config**: spirals 2→32×4→1, 2000 train, scale=2.0, lr=0.001 (per cycle 010, 019, 020 convention)
- **Confirm criterion**: β complexity decrease ≥80% of layers AND Spearman(Betti_drop, r) ≥0 with r values from cycle_010
- **Refute criterion**: Betti numbers flat (no trend) or same for trained vs random init

# Handoff
Collect hidden-layer activations (x_L1, x_L2, x_L3, x_L4) on probe set (test 500 points). Run ripser (requires pip install). Compute Betti₀, Betti₁ persistence diagrams at each layer. Plot Betti complexity vs depth. Correlate Betti drop with cycle_010 r values via Spearman. Compare trained vs random-init network topologies to ground signal. Output to `cycle_047_phom_toy.json` + `cycle_047_phom_toy.png`.

# Notes
- **Literature ref**: Ripser via scikit-tda is standard persistent homology tool (used in topology ML papers). Bottleneck/Wasserstein distance available for diagram comparison.
- **Prior signal**: entropy cascade (item 37, cycle 019) found sharp drop after L1; expect similar pattern in Betti if topology and entropy are aligned.
- **Post-task**: if item 27 confirmed, item 28 (GPT-2 homology) becomes eligible.
