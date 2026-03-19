# Research Agenda — Transformer-First Geometric Compression

This queue supersedes the old toy-model-first plan. The goal is to answer the right transformer FFN compression questions, in the right order, with distillation as a control and not the mainline.

## NEXT EXPERIMENTS

### P0

62. **Domain robustness: compression quality on non-Shakespeare text — GPT-2 FFN**
   - All experiments so far use Shakespeare as the probe and eval corpus. This tests whether the geometric compression generalizes to different text domains. Run k=4@6× with correction on 3 corpora: (a) Shakespeare (baseline), (b) WikiText-2 validation, (c) OpenWebText sample. Use Shakespeare-fit compressed blocks on all 3 corpora (zero-shot transfer), then re-fit channel selection on each corpus separately. If zero-shot transfer works, the compression is data-robust; if re-fitting is needed, the method is corpus-specific. 3 seeds, 200 eval seqs per corpus.
   - Type: empirical
   - Confirmed if: zero-shot transfer PPL cost is within 3pp of corpus-specific fit on ≥2 of 3 corpora.
   - Refuted if: zero-shot transfer causes >20% PPL increase on any non-Shakespeare corpus while corpus-specific fit stays <10%.
   - Inconclusive if: one corpus transfers well and the other does not.
   - Depends on: 037, 039

63. **Practical compression recipe: consolidated findings summary — GPT-2 FFN**
   - Synthesize all confirmed results (cycles 005–041) into a single executable compression recipe with exact hyperparameters, block ordering, correction placement, and expected quality. Document: (a) which blocks to compress and at what ratio, (b) channel selection method + lstsq refit procedure, (c) correction fitting procedure and placement, (d) expected PPL cost envelope across seeds, (e) failure modes (block 0, block-count wall). This is a documentation + validation experiment: re-run the full recommended recipe end-to-end once to confirm the documented numbers match. Output: a standalone script + results JSON.
   - Type: implementation + validation
   - Confirmed if: end-to-end recipe reproduces k=4@6× with correction at ≤10% PPL, matching cycle 037/039 numbers within 1pp.
   - Refuted if: reproduced numbers differ by >3pp from documented, indicating hidden dependencies or stale artifacts.
   - Inconclusive if: numbers match but the recipe requires corpus-specific tuning not captured in the documentation.
   - Depends on: 037, 039, 040, 041

### P1

55. **Lstsq refit quality degradation analysis across compression ratios — GPT-2 FFN**
   - The geometric method uses least-squares refit of W_down after channel selection. At high compression (6x+), the refit becomes increasingly underdetermined (fewer channels = fewer columns in the design matrix). Measure the condition number of the refit system and held-out reconstruction MSE at ratios {2x, 4x, 6x, 8x} on blocks 7, 5, 4. If the refit condition number explodes at a specific ratio, that explains any compression ceiling and motivates regularized refit (ridge/truncated SVD) as a next step. This is an analysis experiment that explains mechanism, not just measures outcome.
   - Type: analysis
   - Confirmed if: condition number increases >10x between 4x and 8x AND held-out MSE / train MSE ratio exceeds 2.0 at 8x for a majority of blocks (overfitting in the refit).
   - Refuted if: condition number stays below 1e4 across all ratios and held-out MSE tracks train MSE closely, indicating the refit is well-conditioned and the ceiling is from information loss, not fitting artifacts.
   - Inconclusive if: condition number rises but held-out MSE remains close to train MSE, suggesting the system is ill-conditioned but regularization would not help.
   - Depends on: 54

50. **Centroid-merge quality as a function of probe corpus size — GPT-2 FFN**
   - Cycle 027 established centroid similarity (ρ=0.85) as the merge signal on a fixed probe corpus. It is unknown whether this signal is stable under small probe sets (e.g., 64 vs 512 vs 2048 tokens), which determines minimum data requirements for deployment. Run single-block swap with centroid-guided merge on one representative block across probe sizes {64, 128, 256, 512, 1024, 2048} and measure PPL. Also measure channel selection stability (Jaccard index of selected channels across runs with different random seeds at the same probe size).
   - Type: empirical
   - Confirmed if: Jaccard ≥ 0.85 and PPL within 2% of the 2048-token reference at probe size ≤ 256 tokens.
   - Refuted if: Jaccard < 0.6 or PPL degrades > 5% relative at probe size 256, indicating the signal needs large data.
   - Inconclusive if: Jaccard is stable but PPL variance across seeds is high (> 3%), suggesting instability from a different source.
   - Depends on: 26, 27

51. **Per-block learned compressor with block-conditioned bottleneck — GPT-2 FFN**
   - Cycle 030 showed a shared compressor underfits (INCONCLUSIVE): held-out blocks got geometric-level PPL but the shared compressor didn't beat per-block geometric. Rather than a fully shared compressor, train one lightweight compressor per block (same architecture, separate weights, 500 distill steps each) and measure: (a) whether per-block distillation beats the no-training geometric baseline, and (b) at what step budget the gap closes. This tests whether the cycle 030 failure was due to cross-block sharing vs. insufficient capacity.
   - Type: empirical
   - Confirmed if: per-block learned compressor beats geometric baseline by ≥1 PPL point for ≥8 of 12 blocks.
   - Refuted if: per-block compressor does not beat geometric baseline for any block after 500 distill steps.
   - Inconclusive if: per-block beats geometric on some blocks but with high variance across blocks, making no systematic conclusion possible.
   - Depends on: 14, 18

23. **Effective number of affine behaviors via Jacobian k-means — GPT-2 FFN**
   - ⚠ **BLOCKED**: Depends on item 22 (REFUTED). Item 22 found no functional redundancy in toy networks, making the GPT-2 Jacobian k-means analog unmotivated per agenda dependency rule.
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

30. **Local complexity proxy as a compression difficulty predictor — GPT-2**
   - Apply the local complexity idea from experiment 29 to GPT-2 FFN blocks. Since GELU makes exact activation-pattern counting undefined, use a proxy: for each probe token x, compute the standard deviation of block Jacobians J(x') for x' in the k nearest neighbors of x in the input representation space. High std → the block is highly nonlinear near x → high local complexity. Low std → the block is locally smooth → easy to approximate. Aggregate this proxy over the full probe set per block and correlate with single-block swap damage from cycle 005.
   - Type: empirical
   - Confirmed if: Spearman correlation between local complexity proxy and compression damage ≥ 0.5, making this a cheap compression-difficulty oracle.
   - Refuted if: correlation < 0.2, indicating that local nonlinearity does not predict block-level compression difficulty.
   - Inconclusive if: the toy result from experiment 29 is strong but the GPT-2 proxy gives weak correlation, suggesting the Jacobian-variation proxy is too indirect.
   - Depends on: 9, 29. Skip if experiment 29 is refuted (local complexity is not a useful signal even in toy networks).

35. **Region volume Pareto — GPT-2 FFN analog**
   - Apply the Pareto analysis from experiment 34 to GPT-2 FFN blocks. Since exact activation patterns are unavailable for GELU, use k-means clusters on probe-token Jacobians as a proxy for "regions." Sort clusters by token count (proxy for volume). For each block, compute the cumulative reconstruction error of block outputs as clusters are added in descending size order. If a Pareto structure exists (top 20% of clusters covers 80% of total output variance), it suggests that a few dominant "computational modes" drive most of the block's behavior — and keeping only those modes gives a strong compression baseline. Compare the Pareto coefficient to compression damage from cycle 005 to test whether Pareto-skewed blocks are the easy ones.
   - Type: empirical
   - Confirmed if: top 20% of clusters by size covers ≥ 70% of output variance in a majority of blocks AND Gini coefficient of cluster-size distribution correlates with (1 − compression difficulty) at Spearman ≥ 0.35.
   - Refuted if: cluster size distribution is approximately uniform across all blocks (Gini < 0.2), indicating no dominant computational modes.
   - Inconclusive if: Pareto holds in toy but the LLM cluster distribution is flat, suggesting the Jacobian k-means proxy is not capturing the same structure as true activation regions.
   - Depends on: 9, 22, 34. Skip if experiment 34 is refuted (Pareto structure does not appear even in toy ReLU networks).


### P0 — Novel directions: structured probing + topology (items 36–45, added 2026-03-18)


### P2 — Moonshots

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


## PAST EXPERIMENTS

### Cycle 032

47. **Dead channel census — GPT-2 FFN v2 (GELU-correct)** *(REFUTED — cycle 032)* ⚠ **FLAWED: MP1 (threshold above trivial baseline)**
   - Pre-GELU dead (T=-3): 0.00%. Near-linear (T=+3): 0.00%. Pruning 1% (by contribution): +1.14% PPL > 1% threshold. Max safe prune = 0%. Contribution-based metric: 93% of channels below 0.05% threshold — but this is an artifact of near-uniform distribution (equal share = 0.033%); not meaningful "dead" channels. Confirms and closes cycle 028 finding with correct GELU-aware definitions.
   - ⚠ **Post-hoc review**: the 0.05% contribution threshold exceeds the equal-share baseline (1/3072 = 0.033%), so nearly all channels are trivially flagged. A design review would have caught this via R2 (measurement soundness) and required a threshold well below equal-share.
   - Outcome: REFUTED
   - Commit: b83ddd0
   - Artifacts: autoresearch/results/cycle_032_prune_collapse.json, autoresearch/results/cycle_032_prune_collapse.png

### Cycle 031

46. **Grokking as compression catalyst — modular arithmetic (v2)** *(REFUTED — cycle 031)* ⚠ **FLAWED: MP5 (insufficient model capacity)**
   - Task: (a+b) mod 97. Grokking detected (memorized epoch 1000, grokked epoch 17500, delay 16500 epochs). BUT: distilled student acc = 0.002 both pre and post grokking (essentially chance). Compression gain = +0.001, below 2% threshold. Pattern count stays at 9409 (= P², all examples unique) throughout — no geometric compaction signal. Post-grokking algebraic structure cannot be distilled from training data alone into a 4× smaller network.
   - Outcome: REFUTED — grokking occurred but compression gain <2% (0.001 actual)
   - ⚠ **Post-hoc review**: student hidden=32 is insufficient capacity for mod-97 (97 residue classes). A design review would have caught this via R3 (capacity adequacy) and required either a larger student or a smaller prime.
   - Commit: c0cb0dd
   - Artifacts: autoresearch/results/cycle_031_grokking.json, autoresearch/results/cycle_031_grokking_trajectory.png

### Cycle 028

16. **Exact prune plus linear collapse baseline** *(REFUTED — cycle 028)* ⚠ **FLAWED METHODOLOGY**
   - Implement the strongest exact baseline from the literature: stable-channel deletion and always-active / near-linear collapse before any heuristic merging. Compare exact-prune-only, exact-prune-plus-least-squares-refit, and the current geometric method on single-block swaps. This is the cleanest test of how much "free" no-SGD compression exists.
   - Type: implementation
   - Confirmed if: exact or near-exact pruning alone yields at least `1.1x` FFN parameter reduction with `<=1%` relative perplexity increase on the tested single-block swaps.
   - Refuted if: exact or near-exact pruning yields negligible savings or immediately breaks block behavior.
   - Inconclusive if: savings exist only under loose thresholding that is hard to justify.
   - Depends on: 11
   - **Outcome: REFUTED** — GPT-2 FFN has effectively zero dead or near-linear channels at threshold 1e-5/5%. Mean dead=0.00%, mean saved (dead+near-linear collapse)=0.008% across 12 blocks. Only 3 blocks had any classifiable channels at all (blocks 8, 9, 11: 1 channel each at 0.033%). Param reduction=1.000x. Prune-only and prune+collapse PPL unchanged (+0.00%). Prune+collapse+lstsq refit improves PPL by 0.21% (127.44→127.18) but that is a full lstsq refit on all live channels (3072/block), not structural pruning. GELU activations in GPT-2 use all channels meaningfully — no free structural compression exists at any practical threshold.
   - ⚠ **Post-hoc review: flawed experiment.** Dead channel definition used `max|post-GELU act| < 1e-5`, but GELU is never truly zero: gelu(-3)≈-0.004, gelu(-5)≈-0.00002 — both well above 1e-5. Threshold is calibrated for ReLU, not GELU. Correct definition: check *pre*-GELU values (max(h_j) < -T for T∈{2,3,4}) to find channels stuck in GELU's saturated tail. Also missing: output-contribution-based metric (||W_down[:,j]|| × rms(act_j)) and smooth pruning curve. **Superseded by item 47.**
   - Cycle: 028
   - Artifacts: `experiments/exact_prune_linear_collapse_gpt2.py`, `autoresearch/results/cycle_028_prune_collapse.json`, `autoresearch/results/cycle_028_prune_collapse.png`

### Cycle 027

15. **Centroid and Jacobian affinity as merge signals** *(CONFIRMED — centroid; REFUTED — Jacobian; cycle 027)*
   - Evaluate whether data-conditioned similarity measures from the centroid-affinity / local-map literature predict safe compression better than raw weight cosine. For each block, compare weight similarity, centroid similarity, Jacobian-spectrum similarity, and any combined score against actual swap damage under controlled budgeted compressions.
   - Type: empirical
   - Confirmed if: a centroid/Jacobian-based score improves Spearman correlation with swap error by at least `0.15` over raw weight cosine on a majority of tested blocks.
   - Refuted if: raw weight cosine is just as good or better across the board.
   - Inconclusive if: the better signal changes by block depth or corpus.
   - Depends on: 10, 11
   - **Outcome: CONFIRMED (centroid) / REFUTED (Jacobian)**
     - Weight cosine: mean ρ = +0.041 (effectively zero — no predictive power)
     - Centroid similarity: mean ρ = +0.853, Δ = +0.812 vs weight cosine, criterion met in 12/12 blocks
     - Jacobian-channel similarity: mean ρ = −0.037, Δ = −0.078 — slightly anti-predictive
     - Centroid (activation time-series cosine) is a dominant merge predictor across all depths
     - Jacobian signal (mean weighted projection) fails — likely because it collapses the token-level variation that centroid preserves
   - Cycle: 027
   - Artifacts: `autoresearch/results/cycle_027_affinity_signals.json`, `autoresearch/results/cycle_027_affinity_signals.png`

### Cycle 026

14. **Rank-only baseline versus geometry versus distillation**
   - Add a matched-budget SVD / low-rank FFN baseline and compare it directly to the current geometric method and distillation on the same single-block swaps and budgets. This asks whether the current "geometric" method is actually better than a plain matrix-factorization baseline.
   - Type: empirical
   - Confirmed if: the geometric method beats the rank-only baseline on full-model metrics in more than `50%` of tested block-budget pairs at equal FFN parameter count.
   - Refuted if: rank-only beats the geometric method in more than `80%` of tested block-budget pairs.
   - Inconclusive if: wins trade back and forth by block or budget with no consistent edge.
   - Depends on: 10
   - **Outcome: CONFIRMED** (cycle 026) — Two SVD baselines tested: (1) SVD on composed map W_down@W_up: catastrophically bad (PPL 3K–350M) because GELU does not commute with change-of-basis; (2) SVD-norm: select top-k channels by W_up row-norm + lstsq refit (proper rank baseline). Geometric beats svd_norm in 11/12 (92%) block-budget pairs. Baseline PPL 122.4; geometric +0.8–5.8%; svd_norm +2.2–7.2%; distillation (200 steps) +0.1–10.1%. Only exception: block 11 at 16× (deep layer, aggressive compression, svd_norm=129.0 vs geo=129.5). Key insight: data-driven channel scoring (activity × output-norm) outperforms weight-magnitude-based selection at all depths and budgets.

### Cycle 025

33. **GELU → 3-piece PWL approximation fidelity — GPT-2** *(REFUTED — cycle 025)*
   - Apply the same PWL substitution from experiment 32 to GPT-2: replace all GELU activations with the 3-piece PWL function, no retraining, measure perplexity on the probe validation set. If perplexity degrades < 5% relative, then GPT-2's FFN blocks are approximately piecewise-affine and all the partition-based theory (activation patterns, RTG, SplineCam) applies as an approximation. This would justify using ReLU-derived compression signals on GELU networks. Also measure block-by-block contribution: which blocks' perplexity degrades most when only that block's GELU is replaced, to identify which blocks are most sensitive to smooth activation curvature.
   - Type: empirical
   - Confirmed if: full GELU → 3-piece PWL substitution causes < 5% relative perplexity increase on the validation set.
   - Refuted if: perplexity increases > 20%, indicating that GELU's smooth transition zone is functionally critical at the LLM scale.
   - Inconclusive if: perplexity increase is moderate (5–20%) and is concentrated in specific blocks rather than spread evenly.
   - Depends on: 32. Skip if experiment 32 is refuted (the PWL approximation is fundamentally inadequate even for toy GELU networks).
   - **Outcome: REFUTED** — Baseline PPL=125.7. 3-piece: 6997 (+5466%), 5-piece: 3221 (+2462%), 10-piece: 198 (+57.8%), 20-piece: 127 (+1.1%). Even single-block 3-piece replacement causes catastrophic damage (block 0: +1239%; best block 9: +135%). GELU curvature is functionally critical at every granularity in GPT-2; 3-piece PWL is completely inadequate. Partition-based theory (SplineCam, RTG, MASO) does NOT apply to GELU networks even as a first-order approximation at this scale. 20-piece PWL is the minimum for near-lossless approximation.
   - Cycle: 025
   - Artifacts: `experiments/gelu_pwl_gpt2.py`, `autoresearch/results/cycle_025_gelu_pwl_gpt2.json`, `autoresearch/results/cycle_025_gelu_pwl_gpt2.png`

### Cycle 024

31. **CKA representational similarity matrix across layers (GPT-2)** *(INCONCLUSIVE — cycle 024)*
   - Compute centered kernel alignment (CKA) between the token representation matrices at all pairs of (layer_i, layer_j) in GPT-2. CKA is a scale-invariant similarity measure between two sets of representations: CKA ≈ 1 means the two layers are computing essentially the same geometry (up to rotation/scale), CKA ≈ 0 means they are unrelated. Plot the full 13×13 CKA heatmap. Identify "representational plateaus" — groups of consecutive layers with high mutual CKA (≥ 0.8) — as candidates for block-level depth pruning. Correlate the CKA drop between layer i and layer i+1 with the single-block swap damage for block i from cycle 005: blocks that contribute large CKA drops are doing new work and should be harder to remove.
   - Type: empirical
   - Confirmed if: at least one contiguous group of ≥ 3 GPT-2 layers has pairwise CKA ≥ 0.80 AND the (CKA drop to next layer) × (compression difficulty) Spearman correlation is ≥ 0.4.
   - Refuted if: CKA is uniformly low (< 0.5) for all non-adjacent layer pairs, indicating every layer computes a distinct representation.
   - Inconclusive if: CKA plateaus exist but removing one layer from a plateau still causes > 20% relative perplexity increase, indicating that representational similarity does not imply functional redundancy.
   - Depends on: 9
   - **Outcome: INCONCLUSIVE** — Massive 10-block plateau found (blocks 1–10, pairwise CKA ≥ 0.991), satisfying plateau criterion. But Spearman(CKA_drop, damage) = 0.32 (p=0.31), below ρ≥0.40 threshold. Root cause: CKA drops within the plateau are near-zero (0.0002–0.004) for all middle transitions, even though damage varies (0.12%–3.07%). Residual connections make the Gram matrix geometry nearly constant despite meaningful functional differences per block. CKA is a poor per-block compression predictor in GPT-2: representational similarity does not imply functional redundancy. The plateau exists but cannot identify which blocks are safe to remove.
   - Cycle: 024
   - Artifacts: `autoresearch/results/cycle_024_cka_matrix.json`, `autoresearch/results/cycle_024_cka_heatmap.png`, `autoresearch/results/cycle_024_summary.md`

### Cycle 023

29. **Local complexity as a compression difficulty predictor — toy ReLU**
   - Implement the local complexity measure from Patel & Montúfar 2025: for each probe input x, count the number of distinct activation patterns observed in a ball of radius ε around x (using probe samples). Average over x. This is the empirical local complexity. A network with high local complexity has many linear regions packed near the data — it is using its nonlinearity heavily and will be hard to compress. For the spirals toy network, compute local complexity at each layer and correlate it with the layer-wise single-affine residual r from experiment 20. Also compare local complexity across training checkpoints (early, mid, late training) to test the Humayun/Patel prediction that local complexity decreases during grokking.
   - Type: empirical
   - Confirmed if: local complexity correlates with single-affine residual r (Spearman ≥ 0.5) AND local complexity decreases by ≥ 30% between early and late training checkpoints.
   - Refuted if: local complexity shows no correlation with r and no consistent trend across training.
   - Inconclusive if: the correlation with r holds but the training-dynamics prediction fails, or vice versa.
   - Depends on: 20
   - **Outcome: INCONCLUSIVE** — Spearman ρ=0.80 (p=0.20, n=4) between local complexity and r_held PASSES the ≥0.5 threshold. Layer ordering by LC matches compression difficulty (L2 highest LC, L1 next, consistent with large r drop from L1→L2 in cycle 010). However, local complexity does NOT decrease during training: early=15.08, mid=14.32, late=15.91 (−5.5% change vs required −30%). The Humayun/Patel grokking prediction fails — spirals may not exhibit the same training-phase collapse as reported. Note: ε=0.39 yielded avg 72 neighbors (2× intended ~30); this doesn't change the qualitative finding. Experiment 30 (GPT-2 proxy) can proceed since the correlation signal holds.
   - Cycle: 023
   - Artifacts: `autoresearch/results/cycle_023_local_complexity.json`, `autoresearch/results/cycle_023_local_complexity.png`

### Cycle 022

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

### Cycle 021

39. **Grokking as compression catalyst — toy ReLU** *(REFUTED — cycle 021)* ⚠ **FLAWED METHODOLOGY**
   - Train spirals network for 10× normal epochs (well past convergence into overfitting/grokking regime). At regular checkpoints: (a) count distinct activation patterns on held-out data, (b) compute pattern entropy (item 37), (c) measure single-affine residual per layer (item 20 method). Look for a phase transition where partition complexity drops sharply — this is grokking manifesting as geometric simplification. If found: run the geometric compression pipeline (from `relu_geometric_compression.py`) AT the grokking point vs BEFORE — quantify how much easier compression becomes post-grokking. This bridges "Deep Grokking" theory (already in lit map) with actionable compression timing.
   - Type: empirical
   - **Outcome: REFUTED** — Classic grokking (late test accuracy rise) IS detected on spirals: train saturates at 92% by epoch 2000, test accuracy jumps from 0.747 → 0.805+ around epoch 17000. However, the geometry signal is opposite to the hypothesis: pattern count INCREASES through grokking (369 → 564+, rising to 723 at epoch 29000 with no drop). Pattern complexity is NOT reduced by grokking on spirals — instead the network develops a richer partition. Compression quality improves slightly (+5.7% accuracy at 4× compression) but this tracks the teacher's improved accuracy, not geometric simplification. Pattern drop from peak = 4.3% (need 40%), compression gain = 5.7% (need 10%). Verdict: REFUTED on both criteria. The "partition collapses during grokking" hypothesis is wrong for continuous 2D tasks; it may only apply to discrete/algebraic tasks (modular arithmetic) where grokking enables compact algorithmic solutions.
   - ⚠ **Post-hoc review: flawed experiment.** (1) Spirals don't grok — continuous classification tasks lack the memorization→generalization phase transition. The "grokking" detected is just gradual fitting. (2) Raw pattern count is the wrong metric — grokking is about partition *rearrangement* (regions align with task structure), not pattern count *reduction*. Should track region-label purity. (3) Pattern increase is expected and correct for a network refining spiral boundaries on 2000 test points. **Superseded by item 46.**
   - Cycle: 021
   - Artifacts: `experiments/grokking_compression_catalyst.py`, `autoresearch/results/cycle_021_grokking.json`, `autoresearch/results/cycle_021_grokking_trajectory.png`

### Cycle 020

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

### Cycle 019

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

### Cycle 017

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

### Cycle 016

32. **GELU → 3-piece PWL approximation fidelity — toy GELU MLP** *(INCONCLUSIVE — cycle 016)*

### Cycle 012

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


### Cycle 030

18. **Learned compressor generalization on held-out blocks** — **INCONCLUSIVE** *(cycle 030)*
   - Shared low-rank bottleneck (768→384→768) trained on blocks 0-7, evaluated on blocks 8-11 with bias-only adaptation (mean-shift). Beats geometric on only 1/4 held-out blocks (block 10 only, where shared PPL=117.4 vs geometric=122.1). Generalization ratio = 0.76 (held-out PPL / in-distribution train PPL), within the 1.25 threshold. But the train-block PPL itself is poor (mean=163.4 vs baseline=122.4): the shared U,V underfit because each block computes a qualitatively different function. The compressor memorizes a rough average rather than reusable structure. Per-block distillation achieves 122-132 PPL (near baseline) with the same parameter budget, confirming the bottleneck is task-specific. Geometric baseline wins on 3/4 blocks without any training.
   - Artifacts: `experiments/learned_compressor_generalization.py`, `autoresearch/results/cycle_030_generalization.json`

### Cycle 029

17. **Local-chart analytic compression inside one FFN block** — **REFUTED**: k-means clustering of FFN inputs into K local charts (K=4, K=8) at matched budget (384 total hidden channels) consistently hurts vs global geometric. Block MSE increases on all blocks (+41% for K=4 on block 2, +53% on block 11). Full-model PPL worse on all 3 blocks for both K values. Splitting the channel budget per chart causes each chart to be too thin; chart-local channel selection cannot overcome the loss from reduced per-chart capacity. The global top-k + lstsq refit remains the stronger approach because GELU FFN outputs are approximately a global low-dimensional function of the input, not a collection of locally separable functions.
   - Artifacts: `experiments/local_chart_ffn_compression.py`, `autoresearch/results/cycle_029_local_chart.json`

### Cycle 025

### Cycle 022

### Cycle 021

### Cycle 018

36. **Procrustes linearity profile + identity-layer pruning — GPT-2**
   - Outcome: **INCONCLUSIVE** — Linearity scores confirmed (7/12 layers ≥0.98, layers 3–10 all highly linear, consistent with "Secretly Linear" paper). Spearman correlation with cycle 005 damage = 0.557 (PASS). But identity replacement of top-2 most linear blocks (3, 4) causes 33.6% PPL increase (FAIL; criterion ≤10%). Individually: block 3 alone = 10.2%, block 4 = 8.5%, block 6 cheapest at 3.9%. Compounding makes simultaneous removal far worse than either alone. Linearity ≠ removability for joint removal — though single blocks near threshold. Block 0 catastrophic (2314%); block 11 severe (595%). Layers 1–2 have lower linearity (0.846, 0.852) consistent with their early representational role.
   - Cycle: 018
   - Artifacts: `experiments/procrustes_linearity_gpt2.py`, `autoresearch/results/cycle_018_linearity.json`, `autoresearch/results/cycle_018_linearity.png`

### Cycle 017

### Cycle 016

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

11. **Stable and near-linear channel census in pretrained FFNs** ⚠ **FLAWED: MP2 (detection method mismatched to GELU)**
   - Type: empirical
   - Outcome: **refuted (proxy)** — 0/12 proxy gpt2 FFN blocks reached ≥5% stable-or-near-linear channels; 10/12 stayed below 1%
   - Cycle: 006
   - Commit: NONE (recovered from hidden runtime artifacts)
   - Artifact: `autoresearch/results/cycle_006_summary.md`
   - Key finding: Exact stable-channel pruning (dead + always-active) is insufficient for GELU transformer FFNs — stable channels account for < 1% of hidden width in most blocks on the proxy benchmark.
   - ⚠ **Post-hoc review**: same GELU detection flaw as cycle 028 — stability thresholds calibrated for ReLU zero-output, but GELU never truly zeros. Pre-GELU input values should be used for dead channel detection. Superseded by cycle 032 (item 47).

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

21. **Single-affine collapse quality — GPT-2 FFN blocks** ⚠ **FLAWED: MP3 (underdetermined fitting)**
   - Type: empirical
   - Outcome: **refuted (proxy)** — 0/12 blocks have r_held < 0.15 (best: block 2 r_held=0.178). Swap of best block causes +33.7% PPL increase (> 20% threshold). Best-two simultaneous swap causes +344% PPL increase.
   - Cycle: 011
   - Commit: c4ea941
   - Artifact: `autoresearch/results/cycle_011_summary.md`, `autoresearch/results/cycle_011_single_affine_gpt2.json`, `autoresearch/results/cycle_011_single_affine_gpt2.png`
   - Key finding: No GPT-2 FFN block is approximable by a single global affine map on held-out tokens. Block 2 comes closest (r_fit=0.018) but fails to generalize (r_held=0.178), indicating the affine fit memorizes rather than captures structure. Even the best block causes 33.7% PPL increase when swapped, confirming that GELU nonlinearity is functionally critical. This is consistent with cycle 009 (Jacobian variance too high for single-affine). Multi-region approaches (items 22–25) are required.
   - ⚠ **Post-hoc review**: 512 probe tokens fitting 769 parameters (768 weights + 1 bias) is underdetermined. The r_fit=0.018 → r_held=0.178 gap confirms overfitting. A design review would have caught this via R3 and required n_samples >= 1538.

### Cycle 010

20. **Single-affine collapse quality — toy ReLU**
   - Type: empirical
   - Outcome: **confirmed** — 3/4 hidden layers have r < 0.15 on held-out data. r values: L1=0.430, L2=0.050, L3=0.033, L4=0.024. r decreases monotonically with depth.
   - Cycle: 010
   - Commit: 3d07fcd
   - Artifact: `autoresearch/results/cycle_010_summary.md`, `autoresearch/results/cycle_010_single_affine.json`
   - Key finding: Deep layers of a trained ReLU MLP (layers 2–4) are near-linear on the data distribution (r < 0.06). Only the first hidden layer performs genuinely nonlinear computation (r=0.43). Dead/always-active neuron fraction correlates negatively with r: layers with more stable neurons are more affinely collapsible. Train accuracy was 99.45%. The single-affine collapse is viable for compression of deep layers.

### Cycle 012

### Cycle 013

24. **RTG spectral coarsening: Fiedler-based region merging — toy ReLU**
   - Type: implementation + empirical
   - Outcome: **inconclusive** — spectral beats random by up to +9.4pp at K=8,16 (signal exists), but accuracy plateaus at 55-60% vs float 95.5%. Hamming-1 RTG is highly fragmented (79 components, 330 nodes).
   - Cycle: 013
   - Commit: 202daf9
   - Artifact: `autoresearch/results/cycle_013_rtg_spectral.json`, `experiments/rtg_spectral_toy.py`
   - Key finding: RTG spectral signal is real (beats random significantly) but Hamming-1 graph is too sparse for useful compression — 79 disconnected components means spectral clustering must average across unrelated regions. Item 25 (GPT-2 analog with k-NN Jacobian graph) avoids this sparsity issue and remains eligible.

### Cycle 033

48. **Easiest-first multi-block accumulation up to 6 blocks — GPT-2 FFN** *(CONFIRMED — cycle 033)*
   - Re-ran the easiest-first multi-block accumulation (cycle 007) with the improved geometric method (cycles 026-027): topk channel selection + centroid similarity merge signal at 4× compression. Blocks [7,5,4,1,3,6] accumulated in easiest-first order. k=1: +0.56%, k=2: +2.60%, k=3: +6.38%, k=4: +8.84%, k=5: +17.46%, k=6: +25.99%. Degradation is sub-additive through k=4, then becomes super-additive (k=4→k=5 jump is +8.6pp). The improved method significantly outperforms cycle 007 (which had k=4 at +9.89% with the old method).
   - Outcome: CONFIRMED — k=4 PPL=130.96 (+8.84%) is within the ≤10% threshold
   - Cycle: 033
   - Commit: NONE (recovered from incomplete prior session)
   - Artifacts: `autoresearch/results/cycle_033_accumulation.json`, `autoresearch/results/cycle_033_accumulation.png`, `experiments/multiblock_accumulation_6block.py`

### Cycle 034

49. **Block-specific compression ratio sweep — GPT-2 FFN** *(INCONCLUSIVE — cycle 034, audit-downgraded)*
   - Swept compression ratios {2×, 3×, 4×, 5×, 6×} independently on each of 6 easiest blocks [7,5,4,1,3,6]. All 6 blocks tolerate 6× compression within 5% PPL threshold (29/30 points). Block 5 at 2× is the only failure (+5.41%). PPL is non-monotonic with ratio (higher compression sometimes gives sub-baseline PPL). Reconstruction MSE increases monotonically. Per-block max safe ratios: {7:6, 5:6, 4:6, 1:6, 3:6, 6:6}. Audit downgraded to INCONCLUSIVE: single seed + 50 eval sequences insufficient to confirm sub-baseline PPL values are real vs noise. The directional finding (high compression tolerance) is likely correct but needs multi-seed confirmation.
   - Outcome: INCONCLUSIVE (audit-downgraded from CONFIRMED)
   - Cycle: 034
   - Commit: b8dad28
   - Artifacts: `autoresearch/results/cycle_034_ratio_sweep.json`, `autoresearch/results/cycle_034_ratio_sweep.png`, `experiments/block_ratio_sweep_gpt2.py`

### Cycle 035

52. **Multi-seed confirmation of single-block compression tolerance at 6x — GPT-2 FFN** *(INCONCLUSIVE — cycle 035)*
   - 3 seeds (42, 123, 7), 200 eval sequences each, blocks [7,5,4,1,3,6] at 6× with topk + random null baseline. Baselines differ across seeds (123.4, 125.9, 125.0) confirming shuffle_seed works. All 6/6 blocks within 5% threshold (mean rel PPL: block 7 +0.51%, block 5 -0.20%, block 4 -0.13%, block 1 -1.00%, block 3 -1.55%, block 6 -0.57%). Cross-seed std low (0.18-0.83%). Topk beats random on all 6 blocks. BUT: 5/6 blocks show sub-baseline mean PPL — the non-monotonicity from cycle 034 is REAL, not noise. The lstsq refit at 6× acts as regularization, removing channels that hurt generalization. Verdict: INCONCLUSIVE because confirmation criterion requires no sub-baseline mean PPL.
   - Outcome: INCONCLUSIVE — 6/6 within 5% but 5/6 sub-baseline (regularization effect confirmed real)
   - Cycle: 035
   - Artifacts: `autoresearch/results/cycle_035_multiseed_6x.json`, `autoresearch/results/cycle_035_multiseed_6x.png`, `experiments/multiseed_6x_confirmation_gpt2.py`

### Cycle 036

53. **Heterogeneous-ratio multi-block accumulation — GPT-2 FFN** *(INCONCLUSIVE — cycle 036)*
   - 6× per-block accumulation on blocks [7,5,4,1,3,6]. k=1: -0.16%, k=2: +2.04%, k=3: +6.40%, k=4: +9.59%, k=5: +17.40%, k=6: +26.57%. Baseline 120.32. Compared to cycle 033 (uniform 4×): k=4 at +9.59% vs +8.84% — only +0.75pp worse despite 50% more per-block compression (6× vs 4×). The compounding degradation is dominated by block count, not per-block ratio. Curves nearly overlap at k=3 and diverge slightly at k=4+. Average compression at k=4 is 6.0× (vs criterion >4×).
   - Outcome: INCONCLUSIVE — k=4 at +9.59% is in the 8-12% range; better compression but unclear win over uniform 4×
   - Cycle: 036
   - Artifacts: `autoresearch/results/cycle_036_accumulation_6x.json`, `autoresearch/results/cycle_036_accumulation_6x.png`, `experiments/multiblock_accumulation_6x.py`

### Cycle 037

56. **Total FFN parameter reduction budget at <10% PPL — GPT-2 FFN** *(REFUTED — cycle 037)*
   - k=4@6× (blocks 7,5,4,1): mean +9.77% PPL (±0.26%, 2 seeds), 27.8% total FFN param reduction. k=4 budget is solid. k=5 heterogeneous: tested blocks 3 and 6 as 5th at 2×/3×/4×. ALL k=5 configs exceed +15% (best: block 6 at 4× = +16.99%). The block count wall is hard — reducing the 5th block's ratio to 2× still gives +20%. Per-block ratio doesn't matter; the compounding through the residual stream is the binding constraint.
   - Outcome: REFUTED — all k=5 configs exceed 15% PPL threshold (best +16.99%)
   - Positive finding: k=4@6× achieves 27.8% FFN param reduction at <10% PPL
   - Cycle: 037
   - Artifacts: `autoresearch/results/cycle_037_param_budget.json`, `autoresearch/results/cycle_037_param_budget.png`, `experiments/ffn_param_budget_hetero_k5.py`

### Cycle 038

57. **Block ordering optimization for 5-block stacking — GPT-2 FFN** *(REFUTED — cycle 038)*
   - All C(6,5)=6 five-block combos from the 6 easiest blocks at 6×. Best: drop block 4 (+16.12%). Worst: drop block 1 (+20.88%). Spread 4.76pp — combo matters but not enough. All combos >15%. Insertion order is a no-op (pipeline compresses all blocks independently from float activations). The k=5 wall is robust across all block selections.
   - Outcome: REFUTED — all 5-block configs exceed 15% regardless of combo
   - Cycle: 038
   - Artifacts: `autoresearch/results/cycle_038_block_ordering.json`, `autoresearch/results/cycle_038_block_ordering.png`, `experiments/ffn_block_ordering_k5.py`

### Cycle 039

58. **Inter-block error correction via residual-stream lstsq adjustment — GPT-2 FFN** *(CONFIRMED — cycle 039)*
   - Affine correction C fit via lstsq on probe data at the residual stream after the last compressed block (block 8 input). k=5 (blocks 7,5,1,3,6 at 6×): +16.12% → +12.53%, recovery 3.6pp (≥3pp threshold). k=4 also benefits: +9.59% → +8.35%, recovery 1.23pp (no overcorrection). Correction MSE reduction: 20.8%. Correction is high-rank (rank-95 = 392/768) — the error is distributed, not low-rank, but a significant affine component exists. Zero extra inference parameters (correction applied via hook).
   - Outcome: CONFIRMED — k=5 correction recovers 3.6pp (≥3pp criterion met), k=4 not worsened
   - Cycle: 039
   - Artifacts: `autoresearch/results/cycle_039_residual_correction.json`, `autoresearch/results/cycle_039_residual_correction.png`, `experiments/ffn_residual_correction_k5.py`

### Cycle 040

54. **Compression ceiling scan: 8x and 10x single-block — GPT-2 FFN** *(CONFIRMED — cycle 040)*
   - All 3 blocks (7, 5, 4) tolerate 8× AND 10× compression with mean PPL within 5% (most sub-baseline). 3 seeds, 200 eval seqs. Results: block 7: 6×=+0.51%, 8×=+0.12%, 10×=+0.05%. Block 5: 6×=-0.20%, 8×=-0.49%, 10×=-0.59%. Block 4: 6×=-0.13%, 8×=-0.56%, 10×=-0.54%. Cross-seed std 0.17-0.59% (low). 6× cross-check matches cycle 035 exactly. The ceiling has NOT been found — even 10× compression is essentially free for single-block swaps. The lstsq refit regularization effect strengthens with compression.
   - Outcome: CONFIRMED — 3/3 blocks tolerate 8× at ≤10% (all also tolerate 10×)
   - Cycle: 040
   - Commit: pending
   - Artifacts: `autoresearch/results/cycle_040_ceiling_scan.json`, `autoresearch/results/cycle_040_ceiling_scan.png`, `experiments/compression_ceiling_scan_gpt2.py`

### Cycle 041

59. **Generalization probe: single-block 4× geometric compression on GPT-2-medium — FFN** *(CONFIRMED — cycle 041)*
   - GPT-2-medium (24 layers, d=1024, FFN=4096). 6 blocks (0,4,8,12,16,20) at 4× with 3 seeds, 200 eval seqs. Baseline PPL: 94-97. Block 0: +29.74% (catastrophic, same as GPT-2-small). Blocks 4,8,12,16,20: all within +1% PPL (block 8 actually sub-baseline at -0.16%). 5/6 blocks pass ≤10% threshold. The method generalizes cleanly from GPT-2-small to GPT-2-medium with identical block-0-hard, rest-easy pattern.
   - Outcome: CONFIRMED — 5/6 blocks pass (≥4 required)
   - Cycle: 041
   - Artifacts: `autoresearch/results/cycle_041_medium_generalization.json`, `autoresearch/results/cycle_041_medium_generalization.png`, `experiments/gpt2_medium_generalization_probe.py`

### Cycle 042

60. **Multi-block stacking at 10× per-block — GPT-2 FFN** *(CONFIRMED — cycle 042)*
   - k=4 blocks [7,5,4,1] at 10× compression, 3 seeds, 200 eval seqs. Uncorrected: +10.62% ±0.82%. With residual correction at block 8: **+8.45% ±0.66%**, recovery 2.17pp. This essentially matches k=4@6× corrected (+8.35% from cycle 039). Per-block ratio (6× vs 10×) makes no difference once correction is applied. Total FFN param reduction at 10×: ~30%.
   - Outcome: CONFIRMED — k=4@10x corrected mean +8.45% (≤12% threshold)
   - Cycle: 042
   - Artifacts: `autoresearch/results/cycle_042_multiblock_10x.json`, `autoresearch/results/cycle_042_multiblock_10x.png`, `experiments/multiblock_10x_stacking_gpt2.py`

### Cycle 043

61. **Multi-point residual correction at every compressed block boundary — GPT-2 FFN** *(INCONCLUSIVE — cycle 043)*
   - Fitted N sequential affine corrections (one per compressed block in layer order). k=4: multi-point (+7.68%) is WORSE than single-point (+6.22%) by 1.45pp — intermediate corrections interfere (block 4 MSE increases 41% after its correction). k=5: multi-point (+12.08%) barely improves over single-point (+12.31%) by 0.23pp — within noise. Per-correction diagnostic: first correction (block 1) reduces MSE by 40.8%, but subsequent corrections are destructive or neutral. Cascading lstsq corrections are not composable.
   - Outcome: INCONCLUSIVE — k=5 multi-point at +12.08% touches but doesn't clearly beat 12% threshold; k=4 multi-point is worse than single-point
   - Cycle: 043
   - Artifacts: `autoresearch/results/cycle_043_multipoint_correction.json`, `autoresearch/results/cycle_043_multipoint_correction.png`, `experiments/ffn_multipoint_correction.py`
