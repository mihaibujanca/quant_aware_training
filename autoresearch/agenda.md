# Research Agenda — Transformer-First Geometric Compression

This queue supersedes the old toy-model-first plan. The goal is to answer the right transformer FFN compression questions, in the right order, with distillation as a control and not the mainline.

## NEXT EXPERIMENTS

### P0



### P0 — New geometric directions (cycle 006+, added 2026-03-18)



22. **Effective number of affine behaviors via Jacobian k-means — toy ReLU** *(REFUTED — cycle 012)*
   - A ReLU network with K observed activation patterns has exactly K distinct affine maps A_ω. In practice, many A_ω are similar and can be served by a single representative. For a trained spirals ReLU network, extract all distinct activation patterns on probe data, compute their A_ω matrices using the MASO formula A_ω = W_L·diag(q^L)·…·W_1·diag(q^1), then run k-means on the flattened A_ω vectors. Find the minimum K* such that cluster-centroid A_ω matrices reproduce network outputs with ≤ 1% accuracy drop. K* / N_patterns is the functional redundancy ratio: how much of the activation-pattern diversity is semantically irrelevant. Plot K* vs. accuracy for a range of K to get a compression-accuracy curve.
   - Type: empirical
   - Confirmed if: K* / N_patterns ≤ 0.2 for the spirals network (i.e., fewer than 20% of observed patterns are functionally distinct), indicating high redundancy exploitable for compression.
   - Refuted if: K* / N_patterns > 0.7, indicating that most activation patterns compute meaningfully different functions.
   - Inconclusive if: accuracy drops sharply as soon as any merging is applied (K* ≈ N_patterns) even though many A_ω look geometrically similar, suggesting that functional similarity does not survive k-means clustering.
   - Depends on: none
   - **Outcome: REFUTED** — K*/N_patterns = 1.00 (375/375). Accuracy at K=64 (17% of patterns) = 74.9%, never recovers to within 1% of float (98.3%) until K=N_patterns. Affine maps are genuinely diverse (std 42–83 per component); k-means clustering cannot find redundancy. Implies experiment 23 (GPT-2 Jacobian k-means) is unmotivated per agenda dependency note. RTG spectral coarsening (exp 24) may still work as it respects local adjacency structure.
   - Cycle: 012
   - Commit: see below
   - Artifacts: `autoresearch/results/cycle_012_kmeans_toy.json`, `autoresearch/results/cycle_012_kmeans_curve.png`, `autoresearch/results/cycle_012_summary.md`

23. **Effective number of affine behaviors via Jacobian k-means — GPT-2 FFN**
   - Apply the same clustering idea from experiment 22 to GPT-2 FFN blocks. Instead of MASO affine maps (unavailable for GELU networks), use per-token Jacobians J(x) ∈ ℝ^{768×768} computed via `torch.func.jacrev`. Run k-means on the flattened Jacobians for each block. Measure the reconstruction error of block outputs when each token's output is replaced by J̄_cluster(x − x̄_cluster) + f(x̄_cluster) (first-order Taylor approximation using the cluster centroid Jacobian). Find minimum K* such that reconstruction error < 5% of block output norm. Compare K* across blocks and correlate with single-block swap damage from cycle 005.
   - Type: empirical
   - Confirmed if: for a majority of GPT-2 blocks, K* < 0.1 × N_probe AND K* correlates with compression difficulty (Spearman ≥ 0.4).
   - Refuted if: K* > 0.5 × N_probe for most blocks, indicating that every probe token activates a genuinely distinct functional regime.
   - Inconclusive if: K* is low but reconstruction error does not translate to perplexity improvement when the approximation is used for inference.
   - Depends on: 9, 22. Skip if experiment 22 is refuted (functional redundancy does not exist in toy networks, making the LLM analog unmotivated).


27. **Persistent homology of activation manifold through depth — toy ReLU**
   - For a trained spirals ReLU network (2D input, embedded into 100D), collect the hidden-layer representations at each layer. Apply the ripser persistent homology algorithm to the resulting point clouds, computing Betti numbers β_0 (connected components) and β_1 (loops/holes) at each layer. A well-trained classifier should progressively "untangle" the two spiral classes through depth: early layers should show high topological complexity (many β_1 loops = classes still entangled), while late layers should approach trivial topology (β_0 = 2 isolated clusters, β_1 = 0 = classes cleanly separated). Measure whether topological simplification (decrease in total Betti complexity) correlates with layer-wise compression ease from experiment 20.
   - Type: empirical
   - Confirmed if: total Betti complexity decreases monotonically through at least 80% of layers AND layers with the largest Betti drop have the lowest single-affine residual r from experiment 20.
   - Refuted if: Betti complexity does not decrease or is the same for trained and randomly-initialized networks.
   - Inconclusive if: topology simplifies overall but the relationship to compression ease is not consistent.
   - Depends on: 20

28. **Persistent homology of activation manifold through depth — GPT-2**
   - Apply the same analysis from experiment 27 to GPT-2 hidden states: collect token representations at each block output, apply PCA to 30D, run ripser. The question is whether the progressive topological simplification seen in toy networks also appears in a trained language model, and whether it can serve as a compression predictor. If layers with high Betti drop are also the easy-to-compress layers from cycle 005, persistent homology becomes a geometry-driven compression signal that requires no compression attempt.
   - Type: empirical
   - Confirmed if: Betti complexity decreases monotonically through at least 60% of GPT-2 blocks AND Spearman correlation between (per-block Betti drop) and (1 − compression difficulty) from cycle 005 is ≥ 0.35.
   - Refuted if: Betti numbers are approximately constant across all GPT-2 blocks after PCA, indicating that topology is not a useful signal in the residual stream.
   - Inconclusive if: topology simplifies in toy but GPT-2 shows noise or no trend, suggesting the signal does not survive the transformer architecture's residual connections.
   - Depends on: 9, 27. Skip if experiment 27 is refuted (topology does not simplify through depth even in toy networks).

29. **Local complexity as a compression difficulty predictor — toy ReLU**
   - Implement the local complexity measure from Patel & Montúfar 2025: for each probe input x, count the number of distinct activation patterns observed in a ball of radius ε around x (using probe samples). Average over x. This is the empirical local complexity. A network with high local complexity has many linear regions packed near the data — it is using its nonlinearity heavily and will be hard to compress. For the spirals toy network, compute local complexity at each layer and correlate it with the layer-wise single-affine residual r from experiment 20. Also compare local complexity across training checkpoints (early, mid, late training) to test the Humayun/Patel prediction that local complexity decreases during grokking.
   - Type: empirical
   - Confirmed if: local complexity correlates with single-affine residual r (Spearman ≥ 0.5) AND local complexity decreases by ≥ 30% between early and late training checkpoints.
   - Refuted if: local complexity shows no correlation with r and no consistent trend across training.
   - Inconclusive if: the correlation with r holds but the training-dynamics prediction fails, or vice versa.
   - Depends on: 20

30. **Local complexity proxy as a compression difficulty predictor — GPT-2**
   - Apply the local complexity idea from experiment 29 to GPT-2 FFN blocks. Since GELU makes exact activation-pattern counting undefined, use a proxy: for each probe token x, compute the standard deviation of block Jacobians J(x') for x' in the k nearest neighbors of x in the input representation space. High std → the block is highly nonlinear near x → high local complexity. Low std → the block is locally smooth → easy to approximate. Aggregate this proxy over the full probe set per block and correlate with single-block swap damage from cycle 005.
   - Type: empirical
   - Confirmed if: Spearman correlation between local complexity proxy and compression damage ≥ 0.5, making this a cheap compression-difficulty oracle.
   - Refuted if: correlation < 0.2, indicating that local nonlinearity does not predict block-level compression difficulty.
   - Inconclusive if: the toy result from experiment 29 is strong but the GPT-2 proxy gives weak correlation, suggesting the Jacobian-variation proxy is too indirect.
   - Depends on: 9, 29. Skip if experiment 29 is refuted (local complexity is not a useful signal even in toy networks).

31. **CKA representational similarity matrix across layers (GPT-2)**
   - Compute centered kernel alignment (CKA) between the token representation matrices at all pairs of (layer_i, layer_j) in GPT-2. CKA is a scale-invariant similarity measure between two sets of representations: CKA ≈ 1 means the two layers are computing essentially the same geometry (up to rotation/scale), CKA ≈ 0 means they are unrelated. Plot the full 13×13 CKA heatmap. Identify "representational plateaus" — groups of consecutive layers with high mutual CKA (≥ 0.8) — as candidates for block-level depth pruning. Correlate the CKA drop between layer i and layer i+1 with the single-block swap damage for block i from cycle 005: blocks that contribute large CKA drops are doing new work and should be harder to remove.
   - Type: empirical
   - Confirmed if: at least one contiguous group of ≥ 3 GPT-2 layers has pairwise CKA ≥ 0.80 AND the (CKA drop to next layer) × (compression difficulty) Spearman correlation is ≥ 0.4.
   - Refuted if: CKA is uniformly low (< 0.5) for all non-adjacent layer pairs, indicating every layer computes a distinct representation.
   - Inconclusive if: CKA plateaus exist but removing one layer from a plateau still causes > 20% relative perplexity increase, indicating that representational similarity does not imply functional redundancy.
   - Depends on: 9

32. **GELU → 3-piece PWL approximation fidelity — toy GELU MLP** *(INCONCLUSIVE — cycle 016)*

33. **GELU → 3-piece PWL approximation fidelity — GPT-2**
   - Apply the same PWL substitution from experiment 32 to GPT-2: replace all GELU activations with the 3-piece PWL function, no retraining, measure perplexity on the probe validation set. If perplexity degrades < 5% relative, then GPT-2's FFN blocks are approximately piecewise-affine and all the partition-based theory (activation patterns, RTG, SplineCam) applies as an approximation. This would justify using ReLU-derived compression signals on GELU networks. Also measure block-by-block contribution: which blocks' perplexity degrades most when only that block's GELU is replaced, to identify which blocks are most sensitive to smooth activation curvature.
   - Type: empirical
   - Confirmed if: full GELU → 3-piece PWL substitution causes < 5% relative perplexity increase on the validation set.
   - Refuted if: perplexity increases > 20%, indicating that GELU's smooth transition zone is functionally critical at the LLM scale.
   - Inconclusive if: perplexity increase is moderate (5–20%) and is concentrated in specific blocks rather than spread evenly.
   - Depends on: 32. Skip if experiment 32 is refuted (the PWL approximation is fundamentally inadequate even for toy GELU networks).

34. **Region volume Pareto: do large regions carry most task accuracy? — toy ReLU** *(INCONCLUSIVE — cycle 017)*
   - For a trained spirals ReLU network (2D input → 100D embedding → 4 layers × 64 hidden), enumerate all observed activation patterns on the full probe set. Define region "volume" as the fraction of probe inputs that fall in each region. Sort regions by volume in descending order. Compute the cumulative accuracy recovery as a function of regions kept (the "Pareto compression curve"): start with only the largest region, classify all test points using that region's A_ω, add the next largest region and allow each test point to use its own region's A_ω if it falls there, otherwise fall back to the nearest large region. If the Pareto curve is steep (top 20% of regions by volume covers ≥ 80% of correctly classified points), compression can be achieved by keeping only large regions and routing small-region inputs to the nearest large region.
   - Type: empirical
   - Confirmed if: the top 20% of regions by volume covers ≥ 70% of correctly classified test points.
   - Refuted if: regions have an approximately uniform volume distribution (Gini coefficient < 0.3), indicating no Pareto structure.
   - Inconclusive if: Pareto holds in 2D but disappears when the same analysis is run in the 100D embedding, suggesting the structure is input-space-specific.
   - Depends on: none
   - **Outcome: INCONCLUSIVE** — Gini=0.408 confirms a skewed (non-uniform) volume distribution, but the top 20% of regions (75 of 375) covers only 48.2% of correctly-classified test points, falling short of the 70% threshold. The region-count distribution is Pareto-skewed, but coverage is spread across many medium-sized regions rather than concentrated in the top 20%. Accuracy with A_ω fallback at 20% kept = 86.7% vs float 98.3%. Item 35 (GPT-2 analog) depends on 34 not being refuted — it can proceed since Pareto structure is present, just weaker than expected.
   - Cycle: 017
   - Artifacts: `autoresearch/results/cycle_017_pareto.json`, `autoresearch/results/cycle_017_pareto.png`

35. **Region volume Pareto — GPT-2 FFN analog**
   - Apply the Pareto analysis from experiment 34 to GPT-2 FFN blocks. Since exact activation patterns are unavailable for GELU, use k-means clusters on probe-token Jacobians as a proxy for "regions." Sort clusters by token count (proxy for volume). For each block, compute the cumulative reconstruction error of block outputs as clusters are added in descending size order. If a Pareto structure exists (top 20% of clusters covers 80% of total output variance), it suggests that a few dominant "computational modes" drive most of the block's behavior — and keeping only those modes gives a strong compression baseline. Compare the Pareto coefficient to compression damage from cycle 005 to test whether Pareto-skewed blocks are the easy ones.
   - Type: empirical
   - Confirmed if: top 20% of clusters by size covers ≥ 70% of output variance in a majority of blocks AND Gini coefficient of cluster-size distribution correlates with (1 − compression difficulty) at Spearman ≥ 0.35.
   - Refuted if: cluster size distribution is approximately uniform across all blocks (Gini < 0.2), indicating no dominant computational modes.
   - Inconclusive if: Pareto holds in toy but the LLM cluster distribution is flat, suggesting the Jacobian k-means proxy is not capturing the same structure as true activation regions.
   - Depends on: 9, 22, 34. Skip if experiment 34 is refuted (Pareto structure does not appear even in toy ReLU networks).


### P0 — Novel directions: structured probing + topology (items 36–45, added 2026-03-18)


37. **Activation pattern entropy cascade — toy ReLU** *(CONFIRMED — cycle 019)*
   - For each layer in a trained spirals network, compute Shannon entropy H(layer) of the activation pattern distribution over data. High H = many equally-used patterns = complex = hard to compress. Low H = few dominant patterns = compressible. Also compute conditional entropy H(pattern_L | pattern_{L-1}) — how much NEW partition information each layer adds. If conditional entropy is near zero for some layer, that layer is partitioning redundantly relative to the previous layer → safe to remove. Plot the entropy cascade: H(L1), H(L2|L1), H(L3|L1,L2), ... This is an information-theoretic characterization of partition redundancy, different from affine-map clustering (item 22) or volume analysis (item 34).
   - Type: empirical
   - Confirmed if: conditional entropy drops ≥50% for at least one layer AND low-conditional-entropy layers also have low single-affine residual (item 20 correlation).
   - Refuted if: conditional entropy is uniformly high across all layers (each layer adds substantial new partition structure).
   - Inconclusive if: conditional entropy varies but does not correlate with any compression signal from items 20/22.
   - Depends on: none
   - **Outcome: CONFIRMED** — Conditional entropy drops sharply after layer 1: cH(L2|L1)=1.64 bits (ratio 0.26), cH(L3|L1,L2)=0.26 bits (ratio 0.040), cH(L4|L1-3)=0.27 bits (ratio 0.043). Layers 2-4 all have single-affine residual r < 0.013, far below layer 1's r=0.168. The entropy cascade reveals that layer 1 establishes the bulk of the partition structure; later layers refine minimally. Spearman(cH_ratio, r)=0.40 (underpowered at n=4 but direction consistent).
   - Cycle: 019
   - Artifacts: `autoresearch/results/cycle_019_entropy_cascade.json`, `autoresearch/results/cycle_019_entropy_cascade.png`, `autoresearch/results/cycle_019_summary.md`

38. **Manifold curvature trajectory — GPT-2** *(REFUTED — cycle 020)*
   - At each layer, estimate local manifold curvature of the representation space. Method: for each probe token, fit local PCA in its k-NN neighborhood (k=50 in activation space). Curvature proxy = fraction of variance NOT explained by top-d principal components (d = estimated local intrinsic dimension). Average over tokens to get per-layer curvature. The curvature trajectory should reveal: (a) where the model builds curvature (early layers creating nonlinear features), (b) where it flattens (late layers approaching linearity). This is the geometric dual of the Procrustes test (item 36): Procrustes measures global linearity, curvature measures local manifold geometry. Correlate curvature with single-block swap damage (cycle 005).
   - Type: empirical
   - Confirmed if: curvature decreases monotonically in the final third of layers AND Spearman correlation between curvature and compression difficulty ≥ 0.4.
   - Refuted if: curvature has no clear trend or anti-correlates with compression difficulty.
   - Inconclusive if: curvature decreases but correlation with compression difficulty is weak (0.2–0.4).
   - Depends on: 9
   - **Outcome: REFUTED** — Curvature is nearly uniform across all 13 layers (range 0.0451–0.0471, <5% unexplained variance everywhere), with no monotone trend in the final third. Spearman rho = −0.42 (p=0.17, non-significant), indicating if anything a slight anti-correlation. The k-NN local PCA curvature proxy is insensitive to GPT-2 residual-stream geometry: LayerNorm at each block normalises representations before they are read by subsequent layers, collapsing curvature differences that would otherwise be visible. The signal is too compressed and homogeneous to serve as a compression predictor.
   - Cycle: 020
   - Artifacts: `autoresearch/results/cycle_020_curvature.json`, `autoresearch/results/cycle_020_curvature_trajectory.png`, `autoresearch/results/cycle_020_curvature_vs_damage.png`

39. **Grokking as compression catalyst — toy ReLU** *(REFUTED — cycle 021)*
   - Train spirals network for 10× normal epochs (well past convergence into overfitting/grokking regime). At regular checkpoints: (a) count distinct activation patterns on held-out data, (b) compute pattern entropy (item 37), (c) measure single-affine residual per layer (item 20 method). Look for a phase transition where partition complexity drops sharply — this is grokking manifesting as geometric simplification. If found: run the geometric compression pipeline (from `relu_geometric_compression.py`) AT the grokking point vs BEFORE — quantify how much easier compression becomes post-grokking. This bridges "Deep Grokking" theory (already in lit map) with actionable compression timing.
   - Type: empirical
   - **Outcome: REFUTED** — Classic grokking (late test accuracy rise) IS detected on spirals: train saturates at 92% by epoch 2000, test accuracy jumps from 0.747 → 0.805+ around epoch 17000. However, the geometry signal is opposite to the hypothesis: pattern count INCREASES through grokking (369 → 564+, rising to 723 at epoch 29000 with no drop). Pattern complexity is NOT reduced by grokking on spirals — instead the network develops a richer partition. Compression quality improves slightly (+5.7% accuracy at 4× compression) but this tracks the teacher's improved accuracy, not geometric simplification. Pattern drop from peak = 4.3% (need 40%), compression gain = 5.7% (need 10%). Verdict: REFUTED on both criteria. The "partition collapses during grokking" hypothesis is wrong for continuous 2D tasks; it may only apply to discrete/algebraic tasks (modular arithmetic) where grokking enables compact algorithmic solutions.
   - Cycle: 021
   - Artifacts: `experiments/grokking_compression_catalyst.py`, `autoresearch/results/cycle_021_grokking.json`, `autoresearch/results/cycle_021_grokking_trajectory.png`

40. **Attention head geometry census — GPT-2** *(INCONCLUSIVE — cycle 022)*
   - Extend beyond FFN-only analysis. For each attention head in GPT-2 (12 layers × 12 heads = 144 heads): (a) effective rank of attention matrix mean over probe sequences (nuclear norm / operator norm), (b) head output Jacobian variation (std of head output norms across tokens), (c) pairwise CKA between heads within the same layer to measure redundancy. Rank heads by a combined "prunability score" = low_rank × low_variation × high_redundancy. Test: zero-out the top-K most prunable heads, measure perplexity cost. This answers: is there geometric compression opportunity in attention, not just FFN?
   - Type: empirical + intervention
   - Confirmed if: ≥20% of heads have effective rank < 0.3 × head_dim AND zeroing the 10 most prunable heads causes ≤5% perplexity increase.
   - Refuted if: all heads have high effective rank and zeroing any 5 heads causes >15% perplexity increase.
   - Inconclusive if: effective rank is low but zeroing heads causes high perplexity increase, indicating low-rank heads are still functionally critical.
   - Depends on: 9
   - **Outcome: INCONCLUSIVE** — 89.6% of heads (129/144) have effective rank < 19.2 (0.3×64) — strong geometric low-rank signal across all layers. Eff_rank concentrates 1–7 in deeper layers (5–11), up to 104 in early layers. But zeroing top-10 most prunable heads causes +14.6% PPL increase (criterion: ≤5%). Only K=5 is tolerable at +2.7%. Low-rank geometry does not imply prunability: heads are individually redundant but collectively load-bearing.
   - Cycle: 022
   - Artifacts: `experiments/attn_head_census.py`, `autoresearch/results/cycle_022_head_census.json`, `autoresearch/results/cycle_022_head_census.png`

41. **Constructive partition synthesis — toy ReLU (2D input) [MOONSHOT]**
   - THE open problem in the literature. Given a trained spirals network (2→32→32→1): (a) Extract all observed activation patterns + affine maps via MASO formula (A_ω = ∏ diag(q_ω^i) W^i). (b) Cluster into K macro-regions via k-means on affine maps (leverage item 22). (c) For each macro-region, compute target affine map (support-weighted average of constituent maps). (d) **SYNTHESIS**: solve for weights W'_1, ..., W'_L of a smaller network (2→K'→1 or 2→K'→K'→1) such that for each macro-region k with pattern q_k, the product ∏ diag(q'_k^i) W'^i ≈ target A_k. This is a system of linear equations in the unknown weights, one set per macro-region. Boundary continuity adds constraints: if macro-regions k1, k2 are adjacent, their affine maps must agree on the shared boundary hyperplane. Method: Solve via least-squares with boundary penalty. Start with K=8 macro-regions, target architecture 2→8→1.
   - Type: implementation + empirical
   - Confirmed if: synthesized network achieves ≥85% of original accuracy with ≤50% parameters.
   - Refuted if: the constrained system has no feasible solution for any K ≤ 32, or synthesized network accuracy is below 60%.
   - Inconclusive if: synthesis works for simple datasets (moons) but fails on spirals, indicating the approach doesn't scale to moderate partition complexity.
   - Depends on: 22

42. **Topology-preserving distillation — toy ReLU [MOONSHOT]**
   - Standard distillation matches outputs. This experiment adds a TOPOLOGICAL loss: the student's representation manifold must have the same persistent homology as the teacher's at each layer. Architecture: teacher 2→32→32→32→32→1 (deep), student 2→16→16→1 (shallow+narrow). Training: L_output = MSE(student(x), teacher(x)), L_topo = Σ_layers Wasserstein₁(PD_teacher(layer_k), PD_student(layer_j)) where layer_j is the student layer aligned to teacher layer_k via optimal layer matching, Total = L_output + λ · L_topo. Use Ripser for PH computation (Betti₀ and Betti₁), differentiable through the Wasserstein distance on persistence diagrams (use TopologyLayer / torch-tda or finite-difference approximation). Compare with: (a) output-only distillation, (b) Jacobian-matching distillation, (c) CKA-matching distillation.
   - Type: implementation + empirical
   - Confirmed if: topo-distilled student achieves ≥5pp higher test accuracy than output-only distilled student at matched architecture AND the topology loss converges (PD distance decreases during training).
   - Refuted if: topology loss is too noisy to optimize (PD distance oscillates or diverges) OR topo-distilled student performs ≤ output-only student.
   - Inconclusive if: topology loss converges but accuracy advantage is < 3pp, suggesting the signal is real but weak.
   - Depends on: none

43. **Topological truncation: Betti-guided depth reduction [MOONSHOT]**
   - Compute persistent homology (via Ripser) on hidden representations at each layer. Track Betti numbers (β₀ = connected components, β₁ = loops/holes) through depth. Hypothesis: there exists a "topological triviality point" T — the first layer where Betti numbers stabilize at their final values. Before T, the network performs topological operations (separating classes, untangling manifolds). After T, it's just refining linear separators within an already-correct topology. Test: replace all layers after T with a single affine map (fit via least squares on layer T activations → final output). This is "topologically-justified depth pruning." Toy (primary): spirals (known topology — two interleaved spirals → β₀ should go 1→2, β₁ should decrease). Moons (simpler topology — β₀ goes 1→2). Run on 2D and 100D-embedded variants. GPT-2 (stretch): PH on PCA-projected activations (30D) per layer.
   - Type: empirical + intervention
   - Confirmed if: triviality point T exists at layer ≤ L−2 AND post-T affine replacement loses ≤5% accuracy (toy) or ≤15% perplexity (GPT-2).
   - Refuted if: Betti numbers never stabilize (continuously changing through all layers) or post-T replacement causes >30% accuracy loss.
   - Inconclusive if: Betti numbers stabilize but the affine replacement still degrades accuracy substantially, suggesting sub-topological refinement is still task-relevant.
   - Depends on: none

44. **Topological autoencoder for partition space — toy ReLU [MOONSHOT]**
   - Build an autoencoder that learns a compressed encoding of the activation pattern space: Input: per-sample activation pattern (binary vector, one bit per neuron across all layers — e.g., 128-dim for 4×32 network). Encoder: MLP mapping to low-dimensional latent z ∈ ℝ^K (K << total neurons). Decoder: from z, predict the affine map A_ω (flattened weight matrix of the effective linear map for that pattern). Loss = MSE(A_ω_predicted, A_ω_true) + λ · PH_reg. The PH regularization term: compute persistent homology of the latent space neighborhoods (using k-NN graph on z) and match it to the persistent homology of the activation pattern graph (the RTG — Hamming-1 adjacency graph from Bosca et al.). If the autoencoder achieves good reconstruction with K << number_of_patterns, it proves: (1) the partition space has low intrinsic dimensionality, (2) the decoder provides a direct "K codes → affine map" lookup — a compressed functional representation. Sweep K = {2, 4, 8, 16}. Train on spirals network (~50-100 patterns).
   - Type: implementation + empirical
   - Confirmed if: autoencoder with K ≤ 8 achieves affine map reconstruction R² ≥ 0.9 AND the latent PH matches the RTG PH (Wasserstein distance < threshold).
   - Refuted if: reconstruction requires K ≥ number_of_patterns/2 (no compression), or PH regularization destabilizes training.
   - Inconclusive if: reconstruction works but PH regularization has no measurable effect (topology matching is trivially satisfied or trivially ignored).
   - Depends on: 22, 24

45. **Weight-space diffusion on topological level sets — toy ReLU [MOONSHOT]**
   - Starting from trained weights θ₀, explore the weight space to find maximally compressed parameters that preserve the representation topology. Method: (a) Compute baseline persistent diagram PD₀ of representations at each layer (on training data). (b) Run projected Langevin dynamics: θ_{t+1} = θ_t − ε·∇||θ||₁ + √(2ε)·noise (drift toward minimum L1 norm = sparsity). (c) After each step, compute PD_t. If Wasserstein₁(PD_t, PD₀) > δ (topology changed), reject the step. (d) Run for T steps. The trajectory traces the "topological level set" — all weights producing the same representational topology. (e) Select the point θ* with minimum ||θ||₁ on the trajectory. If the topological level set is large (θ* has much lower norm than θ₀), the network is over-parameterized for its topology. Then: prune θ* (zero out small weights) and measure accuracy. Compare with: (a) standard magnitude pruning of θ₀, (b) L1-regularized retraining.
   - Type: implementation + empirical
   - Confirmed if: topological diffusion achieves ≥30% weight norm reduction while preserving PD within δ AND the pruned θ* achieves ≥5pp better accuracy than standard magnitude pruning at matched sparsity.
   - Refuted if: topological level set is tiny (any perturbation changes PD) or the minimum-norm point within the level set has essentially the same norm as θ₀ (no topological redundancy).
   - Inconclusive if: norm reduction is achieved but pruned accuracy is not better than standard magnitude pruning, suggesting topology preservation is too coarse a constraint.
   - Depends on: none

### P1

14. **Rank-only baseline versus geometry versus distillation**
   - Add a matched-budget SVD / low-rank FFN baseline and compare it directly to the current geometric method and distillation on the same single-block swaps and budgets. This asks whether the current "geometric" method is actually better than a plain matrix-factorization baseline.
   - Type: empirical
   - Confirmed if: the geometric method beats the rank-only baseline on full-model metrics in more than `50%` of tested block-budget pairs at equal FFN parameter count.
   - Refuted if: rank-only beats the geometric method in more than `80%` of tested block-budget pairs.
   - Inconclusive if: wins trade back and forth by block or budget with no consistent edge.
   - Depends on: 10

15. **Centroid and Jacobian affinity as merge signals**
   - Evaluate whether data-conditioned similarity measures from the centroid-affinity / local-map literature predict safe compression better than raw weight cosine. For each block, compare weight similarity, centroid similarity, Jacobian-spectrum similarity, and any combined score against actual swap damage under controlled budgeted compressions.
   - Type: empirical
   - Confirmed if: a centroid/Jacobian-based score improves Spearman correlation with swap error by at least `0.15` over raw weight cosine on a majority of tested blocks.
   - Refuted if: raw weight cosine is just as good or better across the board.
   - Inconclusive if: the better signal changes by block depth or corpus.
   - Depends on: 10, 11

16. **Exact prune plus linear collapse baseline**
   - Implement the strongest exact baseline from the literature: stable-channel deletion and always-active / near-linear collapse before any heuristic merging. Compare exact-prune-only, exact-prune-plus-least-squares-refit, and the current geometric method on single-block swaps. This is the cleanest test of how much "free" no-SGD compression exists.
   - Type: implementation
   - Confirmed if: exact or near-exact pruning alone yields at least `1.1x` FFN parameter reduction with `<=1%` relative perplexity increase on the tested single-block swaps.
   - Refuted if: exact or near-exact pruning yields negligible savings or immediately breaks block behavior.
   - Inconclusive if: savings exist only under loose thresholding that is hard to justify.
   - Depends on: 11

17. **Local-chart analytic compression inside one FFN block**
   - Replace the current global top-k analytic compression with a local-chart construction: cluster FFN inputs on the probe distribution, derive a small set of local charts, and fit the compressed block analytically or semi-analytically per chart. Test this on representative easy, medium, and hard blocks from item 10.
   - Type: implementation
   - Confirmed if: local-chart compression improves full-model single-block metrics by at least `20%` over the global analytic method at equal budget on at least `3` representative blocks.
   - Refuted if: local-chart compression fails to improve over the global analytic baseline or is less stable.
   - Inconclusive if: it helps one class of blocks but not the others.
   - Depends on: 10, 13, 15

18. **Learned compressor generalization on held-out blocks**
   - Train the descriptor-autoencoder / hypernetwork compressor on a subset of blocks and evaluate on held-out blocks from the same model family. If exact nanochat checkpoints at multiple depths are available, also test cross-checkpoint generalization. The point is to determine whether the learned track discovers reusable structure or merely memorizes block identities.
   - Type: empirical
   - Confirmed if: on held-out blocks, the learned compressor beats the current geometric baseline in a majority of cases and stays within `25%` of its in-distribution performance.
   - Refuted if: held-out performance collapses relative to both the geometric baseline and in-distribution performance.
   - Inconclusive if: generalization works only within narrow depth bands or only within one checkpoint.
   - Depends on: 10

## PAST EXPERIMENTS

### Cycle 022

40. **Attention head geometry census — GPT-2** — see item 40 above (moved inline for full context). **INCONCLUSIVE**: 89.6% heads low-rank (eff_rank < 19.2), K=5 tolerable (+2.7% PPL), K=10 fails (+14.6% PPL). Geometric low-rank ≠ functional prunability at scale.
   - Artifacts: `experiments/attn_head_census.py`, `autoresearch/results/cycle_022_head_census.json`, `autoresearch/results/cycle_022_head_census.png`

### Cycle 021

39. **Grokking as compression catalyst — toy ReLU** — see item 39 above (moved inline above for full context). **REFUTED**: pattern count increases during grokking on spirals; compression gain 5.7% < 10% threshold.
   - Artifacts: `experiments/grokking_compression_catalyst.py`, `autoresearch/results/cycle_021_grokking.json`, `autoresearch/results/cycle_021_grokking_trajectory.png`

### Cycle 018

36. **Procrustes linearity profile + identity-layer pruning — GPT-2**
   - Outcome: **INCONCLUSIVE** — Linearity scores confirmed (7/12 layers ≥0.98, layers 3–10 all highly linear, consistent with "Secretly Linear" paper). Spearman correlation with cycle 005 damage = 0.557 (PASS). But identity replacement of top-2 most linear blocks (3, 4) causes 33.6% PPL increase (FAIL; criterion ≤10%). Individually: block 3 alone = 10.2%, block 4 = 8.5%, block 6 cheapest at 3.9%. Compounding makes simultaneous removal far worse than either alone. Linearity ≠ removability for joint removal — though single blocks near threshold. Block 0 catastrophic (2314%); block 11 severe (595%). Layers 1–2 have lower linearity (0.846, 0.852) consistent with their early representational role.
   - Cycle: 018
   - Artifacts: `experiments/procrustes_linearity_gpt2.py`, `autoresearch/results/cycle_018_linearity.json`, `autoresearch/results/cycle_018_linearity.png`

### Cycle 017

34. **Region volume Pareto: do large regions carry most task accuracy? — toy ReLU**
   - Outcome: **INCONCLUSIVE** — Gini=0.408 (Pareto structure present, not refuted), but top 20% of regions (75/375) covers only 48.2% of float-correct test points, below the 70% confirmation threshold. Volume is skewed but spread across many medium regions. Accuracy at 20% kept (with Hamming fallback) = 86.7% vs float 98.3%. Item 35 (GPT-2 analog) may still proceed since Pareto structure is present (dependency condition is "not refuted").
   - Cycle: 017
   - Artifacts: `experiments/region_volume_pareto.py`, `autoresearch/results/cycle_017_pareto.json`, `autoresearch/results/cycle_017_pareto.png`

### Cycle 016

32. **GELU → 3-piece PWL approximation fidelity — toy GELU MLP**
   - Outcome: **INCONCLUSIVE** — K=10 pieces recovers nearly float accuracy (drop=2.59pp, below 3% threshold), but K=3 and K=5 drop ~35–39pp (far above 3% confirmed threshold). GELU curvature is task-critical at low piece count; the model is piecewise-affine at fine enough granularity (≈10 pieces) but not at the coarse 3–5 piece scale where ReLU theory directly applies. 17% of activations fall outside the [-3, 3] domain. Float model test accuracy = 86.75% (spirals, noise=0.5, 3 turns).
   - Cycle: 016
   - Artifacts: `experiments/gelu_pwl_approximation.py`, `autoresearch/results/cycle_016_gelu_pwl_approximation.png`

### Cycle 015

26. **Intrinsic dimension trajectory through depth (GPT-2)**
   - Outcome: **INCONCLUSIVE** — ID increases monotonically (1.6 → 7.5, not decreases) through depth. Final-third mean ID (7.32) is 62% higher than first-third (4.52), opposite of the ≥30% decrease criterion. Spearman correlation between |ΔID| and cycle-005 damage = -0.16 (p=0.62), well below the ρ≥0.4 threshold. The token-representation manifold becomes progressively higher-dimensional through GPT-2 depth; later layers do not simplify structure — they elaborate it.
   - Cycle: 015
   - Artifacts: `experiments/intrinsic_dim_trajectory.py`, `autoresearch/results/cycle_015_intrinsic_dim.json`, `autoresearch/results/cycle_015_intrinsic_dim.png`

### Cycle 014

25. **RTG spectral coarsening — GPT-2 FFN analog**
   - Outcome: **REFUTED** — Spectral clustering of k-NN Jacobian graph provides no advantage over k-means on raw Jacobians. Maximum relative improvement across all blocks and K values was 1.3% (block 2, K=2), far below the 15% threshold. Both methods show monotonically decreasing error with K, but at parity. Graphs were connected (1 component) in all 4 blocks tested. Result aligns with cycle 013 toy result — spectral coarsening adds no value when the Jacobian space is genuinely high-dimensional and non-clustered.
   - Cycle: 014
   - Artifacts: `experiments/rtg_spectral_gpt2.py`, `autoresearch/results/cycle_014_rtg_spectral_gpt2.json`, `autoresearch/results/cycle_014_rtg_spectral_gpt2.png`, `autoresearch/results/cycle_014_summary.md`

### Cycle 012

22. **Effective number of affine behaviors via Jacobian k-means — toy ReLU**
   - Outcome: **REFUTED** — K*/N_patterns = 1.00 (375/375). Even at K=64 (17% of patterns), accuracy = 74.9% vs float 98.3%. MASO affine maps are genuinely diverse (component ranges spanning ±175 to ±575); k-means cannot find functional redundancy. Experiment 23 (GPT-2 Jacobian k-means) is skipped per dependency rule. RTG spectral coarsening (exp 24) remains open.
   - Cycle: 012
   - Artifacts: `autoresearch/results/cycle_012_kmeans_toy.json`, `autoresearch/results/cycle_012_kmeans_curve.png`, `autoresearch/results/cycle_012_summary.md`

### Cycle 005

10. **Single-block swap localization curve**
   - Type: empirical
   - Outcome: **confirmed (proxy)** -- on the latest `gpt2` + Shakespeare sweep, `22/24` latest `>=4x` geometric swaps met the agenda criterion; only layer `0` failed
   - Cycle: 005
   - Commit: NONE (recovered from existing sweep artifacts; no replay commit)
   - Artifact: `autoresearch/results/cycle_005_summary.md`
   - Key finding: Single-block FFN compression is sharply depth-sensitive rather than uniformly broken; most middle and late blocks survive `8x-16x` analytic compression with `<=10%` relative perplexity increase, while layer `0` is a clear hard failure case.

### Cycle 001

1. **Survey: Balestriero's piecewise linear / spline theory of deep learning**
   - Type: literature
   - Outcome: **confirmed** — 6 concrete, implementable techniques identified (≥3 required)
   - Cycle: 001
   - Commit: (see cycle_001_manifest.txt)
   - Artifact: `autoresearch/surveys/balestriero_spline_theory.md`
   - Key finding: Activation pattern census (Technique 1) and affine map extraction (Technique 2) are directly actionable from MASO theory; SplineCam (Technique 3) provides exact 2D enumeration. Gap: analysis-to-synthesis route for minimal network construction remains open.

### Cycle 003

7. **Survey: Non-distillation model compression methods (2023-2026)**
   - Type: literature
   - Outcome: **confirmed** — found ≥2 papers with working no-SGD compression methods
   - Cycle: 003
   - Commit: 66021d0
   - Artifact: `autoresearch/surveys/literature_map.md`
   - Key findings:
     - **Serra et al. NeurIPS 2021 (2102.07804)**: ReLU stability criterion — MIP-based dead/always-active neuron detection → 56% connection removal, 183x faster than prior art. No SGD required.
     - **Berzins 2023 (2306.07212)**: GPU-accelerated partition extraction + §5.4 pruning demo → 78% parameter reduction (2,50,50,1 → 2,25,19,1) via dead + always-active removal. Constructive, no SGD.
     - **3-Phase algorithm** identified: Phase 1 (dead neuron removal) + Phase 2 (always-active collapse to ≤D neurons/layer) = solvable analytically. Phase 3 (merging genuinely unstable neurons) = open problem.
     - Depth reduction ruled out: Arora et al. (1611.01491) proves exponential neuron cost for depth reduction → target width compression only.

### Cycle 002

2. **Survey: Tropical geometry and neural networks**
   - Type: literature
   - Outcome: **inconclusive** — tropical polynomial form is exponentially large (intractable for deep nets); practical TropNNC algorithm uses zonotope clustering but never builds a compact tropical form
   - Cycle: 002
   - Commit: TBD
   - Artifact: `autoresearch/results/cycle_002_summary.md`
   - Key finding: Tropical geometry = dead end for direct compression. One salvageable idea: neurons-as-zonotope-generators motivates clustering-based pruning (overlaps with MASO region merging). Do NOT pursue tropical algebra further.

### Cycle 004

9. **Benchmark lock-in and asset audit**
   - Type: analysis
   - Outcome: **confirmed** — exact-open nanochat benchmark and explicit `gpt2` proxy benchmark are both now documented, with concrete path conventions and acquisition procedures
   - Cycle: 004
   - Commit: e25470e
   - Artifact: `autoresearch/results/cycle_004_summary.md`
   - Key finding: the exact-open target is `karpathy/nanochat` base checkpoints plus `climbmix-400b-shuffle`, but those assets are not present locally; the only benchmark runnable immediately on this machine is the cached `gpt2` proxy path, which must remain clearly labeled proxy-only.

### Cycle 006

11. **Stable and near-linear channel census in pretrained FFNs**
   - Type: empirical
   - Outcome: **refuted (proxy)** — 0/12 proxy gpt2 FFN blocks reached ≥5% stable-or-near-linear channels; 10/12 stayed below 1%
   - Cycle: 006
   - Commit: NONE (recovered from hidden runtime artifacts)
   - Artifact: `autoresearch/results/cycle_006_summary.md`
   - Key finding: Exact stable-channel pruning (dead + always-active) is insufficient for GELU transformer FFNs — stable channels account for < 1% of hidden width in most blocks on the proxy benchmark.

### Cycle 007

12. **Progressive multi-block accumulation curve**
   - Type: empirical
   - Outcome: **confirmed (proxy)** — easiest-first outperforms hardest-first by 27.4% at k=6 and 36.2% at k=9, exceeding the ≥25% threshold. Advantage grows with block count.
   - Cycle: 007
   - Commit: 980dec5
   - Artifact: `autoresearch/results/cycle_007_summary.md`, `autoresearch/results/cycle_007_accumulation.json`
   - Key finding: Ordering matters enormously for multi-block accumulation. Hardest-first is catastrophically worse. Single-block damage ranking from cycle 005 correctly predicts safe accumulation order. Sweet spot is k=4-5 easiest blocks (ppl < 150, < 25% degradation). All-block 16x compression fails regardless of ordering (ppl=403).

### Cycle 008

13. **Probe-corpus robustness of geometry signals**
   - Type: empirical
   - Outcome: **confirmed (proxy)** — 12/12 blocks pass both criteria; channel score Spearman ≥ 0.95, selected-set Jaccard ≥ 0.82, Jacobian cosine ≥ 0.997 across disjoint Shakespeare probe halves
   - Cycle: 008
   - Commit: 63fb41d
   - Artifact: `autoresearch/results/cycle_008_summary.md`, `autoresearch/results/cycle_008_robustness.json`
   - Key finding: Compression geometry signals are highly robust to probe-corpus splits. Channel rankings, Jacobian spectra, and selected hidden-unit sets are nearly identical across disjoint probe halves. This validates the single-probe approach used in all prior cycles.

### Cycle 009

19. **Jacobian variance atlas: which FFN blocks are near-linear on probe data?**
   - Type: empirical
   - Outcome: **refuted (proxy)** — 0/12 blocks have L ≥ 0.85; all blocks have L < 0.6 (max L = 0.38). Spearman ρ = −0.14, p = 0.66.
   - Cycle: 009
   - Commit: NONE
   - Artifact: `autoresearch/results/cycle_009_summary.md`, `autoresearch/results/cycle_009_jacobian_atlas.json`
   - Key finding: No GPT-2 FFN block is near-linear on the probe distribution. Most blocks have negative L (mean Jacobian deviation exceeds mean norm), indicating substantial per-token functional diversity. Single-affine collapse is not viable for any block. Multi-region approaches (items 22–25) are the necessary path.

### Cycle 011

21. **Single-affine collapse quality — GPT-2 FFN blocks**
   - Type: empirical
   - Outcome: **refuted (proxy)** — 0/12 blocks have r_held < 0.15 (best: block 2 r_held=0.178). Swap of best block causes +33.7% PPL increase (> 20% threshold). Best-two simultaneous swap causes +344% PPL increase.
   - Cycle: 011
   - Commit: c4ea941
   - Artifact: `autoresearch/results/cycle_011_summary.md`, `autoresearch/results/cycle_011_single_affine_gpt2.json`, `autoresearch/results/cycle_011_single_affine_gpt2.png`
   - Key finding: No GPT-2 FFN block is approximable by a single global affine map on held-out tokens. Block 2 comes closest (r_fit=0.018) but fails to generalize (r_held=0.178), indicating the affine fit memorizes rather than captures structure. Even the best block causes 33.7% PPL increase when swapped, confirming that GELU nonlinearity is functionally critical. This is consistent with cycle 009 (Jacobian variance too high for single-affine). Multi-region approaches (items 22–25) are required.

### Cycle 010

20. **Single-affine collapse quality — toy ReLU**
   - Type: empirical
   - Outcome: **confirmed** — 3/4 hidden layers have r < 0.15 on held-out data. r values: L1=0.430, L2=0.050, L3=0.033, L4=0.024. r decreases monotonically with depth.
   - Cycle: 010
   - Commit: 3d07fcd
   - Artifact: `autoresearch/results/cycle_010_summary.md`, `autoresearch/results/cycle_010_single_affine.json`
   - Key finding: Deep layers of a trained ReLU MLP (layers 2–4) are near-linear on the data distribution (r < 0.06). Only the first hidden layer performs genuinely nonlinear computation (r=0.43). Dead/always-active neuron fraction correlates negatively with r: layers with more stable neurons are more affinely collapsible. Train accuracy was 99.45%. The single-affine collapse is viable for compression of deep layers.

### Cycle 012

22. **Effective number of affine behaviors via Jacobian k-means — toy ReLU**
   - Type: empirical
   - Outcome: **refuted** — K*/N_patterns = 1.0 (> 0.7 threshold). 375 patterns, accuracy flat at ~75% for K≤64. k-means on raw MASO affine maps finds no functional redundancy.
   - Cycle: 012
   - Commit: 66273d3
   - Artifact: `autoresearch/results/cycle_012_summary.md`, `autoresearch/results/cycle_012_kmeans_toy.json`
   - Key finding: k-means clustering on raw MASO affine maps cannot find functional redundancy. 375 observed activation patterns all compute meaningfully different linear functions under Euclidean distance. Accuracy never recovers within 1pp of float until K=N_patterns. Items 23, 35, 41 blocked (depend on 22 confirmed).

### Cycle 013

24. **RTG spectral coarsening: Fiedler-based region merging — toy ReLU**
   - Type: implementation + empirical
   - Outcome: **inconclusive** — spectral beats random by up to +9.4pp at K=8,16 (signal exists), but accuracy plateaus at 55-60% vs float 95.5%. Hamming-1 RTG is highly fragmented (79 components, 330 nodes).
   - Cycle: 013
   - Commit: 202daf9
   - Artifact: `autoresearch/results/cycle_013_rtg_spectral.json`, `experiments/rtg_spectral_toy.py`
   - Key finding: RTG spectral signal is real (beats random significantly) but Hamming-1 graph is too sparse for useful compression — 79 disconnected components means spectral clustering must average across unrelated regions. Item 25 (GPT-2 analog with k-NN Jacobian graph) avoids this sparsity issue and remains eligible.
