# Research Agenda — Transformer-First Geometric Compression

This queue supersedes the old toy-model-first plan. The goal is to answer the right transformer FFN compression questions, in the right order, with distillation as a control and not the mainline.

## NEXT EXPERIMENTS

### P0
11. **Stable and near-linear channel census in pretrained FFNs**
   - Implement the exact or near-exact pruning census implied by Kumar/Serra/Berzins for transformer FFNs: dead channels, always-positive channels, always-negative channels, and near-linear channels under the actual probe distribution. Report fractions per block and test whether these fractions are robust enough to justify an exact-pruning baseline.
   - Type: empirical
   - Confirmed if: stable or near-linear channels account for at least `5%` of hidden width in a majority of tested FFN blocks.
   - Refuted if: stable or near-linear channels stay below `1%` in more than `80%` of tested blocks.
   - Inconclusive if: the fraction is nontrivial but highly dependent on the chosen probe corpus.
   - Depends on: 9


13. **Probe-corpus robustness of geometry signals**
   - Compare the FFN geometry atlas across disjoint probe subsets and, if available, across a proxy corpus vs a more teacher-like corpus. Measure robustness of stable-channel masks, channel rankings, Jacobian spectra, and selected hidden-unit sets. The aim is to determine whether the compression signal is genuinely teacher-specific or just probe-noise.
   - Type: empirical
   - Confirmed if: for a majority of tested blocks, score rank correlation is `>=0.8` and selected-set overlap is `>=50%` across disjoint probes.
   - Refuted if: selected-set overlap drops below `20%` and score correlations stay below `0.5` for most blocks.
   - Inconclusive if: early and late blocks behave very differently.
   - Depends on: 9, 11

### P0 — New geometric directions (cycle 006+, added 2026-03-18)

19. **Jacobian variance atlas: which FFN blocks are near-linear on probe data?**
   - The existing `FfnLayerGeometry.jacobian_spectrum` stores the *mean* singular value spectrum across probe points. This experiment extends that by computing the *variance* of singular values across probe tokens — the coefficient of variation (std/mean) per singular value and per block. A block where J(x) barely changes across inputs has a near-constant Jacobian: it behaves as a single linear operator on the data distribution, which means it can be collapsed to one affine map without needing to model multiple linear regions. This is cheaper than any region-merging approach and can be evaluated analytically. Compute linearity index L = 1 − mean_x(||J(x) − J̄||_F / ||J̄||_F) per block. Correlate L with the single-block swap damage from cycle 005 to test whether near-linear blocks are also the easy-to-compress ones.
   - Type: empirical
   - Confirmed if: at least 4 of 24 GPT-2 blocks have L ≥ 0.85 AND Spearman correlation between L and (1 − relative perplexity increase from cycle 005) is ≥ 0.4.
   - Refuted if: all blocks have L < 0.6, indicating that every block exhibits substantial Jacobian variation and single-affine collapse is not viable for any block.
   - Inconclusive if: linearity index correlates with depth but not with compression damage, suggesting L measures something other than compressibility.
   - Depends on: 9

20. **Single-affine collapse quality — toy ReLU**
   - For a trained ReLU MLP on spirals (2D input, 4 layers, 64 hidden), fit a single affine map (A, b) to each hidden layer via least squares on probe data: A, b = argmin_A ||h_ℓ(x) − A·h_{ℓ−1}(x) − b||_F. Measure the normalized residual r = ||h_ℓ − Ah_{ℓ−1} − b||_F / ||h_ℓ − mean(h_ℓ)||_F on held-out data. This tells us how well a single affine map approximates what each layer computes on the training distribution. If r is small for some layers, those layers are functionally linear on the data and can be collapsed analytically. This is the simplest possible compression move: replace a nonlinear layer with a matrix multiplication.
   - Type: empirical
   - Confirmed if: at least one hidden layer achieves r < 0.15 on held-out probe data, indicating near-linear behavior.
   - Refuted if: all layers have r > 0.5, indicating that single-affine collapse is uniformly poor and the function uses its nonlinearity everywhere.
   - Inconclusive if: r is moderate (0.15–0.5) and does not correlate with layer depth or neuron stability statistics.
   - Depends on: none

21. **Single-affine collapse quality — GPT-2 FFN blocks**
   - Apply the same single-affine fit from experiment 20 to each GPT-2 FFN block: fit A, b from probe tokens, measure normalized residual r on held-out tokens. The question is whether any FFN block is functionally linear on the GPT-2 probe corpus (Shakespeare text), which would mean it can be replaced by a single matrix multiplication with negligible perplexity cost. If yes, this provides a zero-shot depth-compression baseline that does not require distillation or retraining. Additionally test the action: for blocks with the lowest r, actually swap in the affine approximation and measure resulting perplexity.
   - Type: empirical
   - Confirmed if: at least 2 GPT-2 blocks achieve r < 0.15 AND swapping in the single-affine approximation causes ≤ 5% relative perplexity increase.
   - Refuted if: all blocks have r > 0.5 or the swap always causes > 20% perplexity increase regardless of r.
   - Inconclusive if: r is low but the perplexity increase is high, suggesting the affine approximation captures training-corpus behavior but not the function that matters for generation.
   - Depends on: 9, 20. Skip if experiment 20 is refuted (single-affine collapse is uniformly poor on toy networks, so there is no reason to expect it to work on a harder architecture).

22. **Effective number of affine behaviors via Jacobian k-means — toy ReLU**
   - A ReLU network with K observed activation patterns has exactly K distinct affine maps A_ω. In practice, many A_ω are similar and can be served by a single representative. For a trained spirals ReLU network, extract all distinct activation patterns on probe data, compute their A_ω matrices using the MASO formula A_ω = W_L·diag(q^L)·…·W_1·diag(q^1), then run k-means on the flattened A_ω vectors. Find the minimum K* such that cluster-centroid A_ω matrices reproduce network outputs with ≤ 1% accuracy drop. K* / N_patterns is the functional redundancy ratio: how much of the activation-pattern diversity is semantically irrelevant. Plot K* vs. accuracy for a range of K to get a compression-accuracy curve.
   - Type: empirical
   - Confirmed if: K* / N_patterns ≤ 0.2 for the spirals network (i.e., fewer than 20% of observed patterns are functionally distinct), indicating high redundancy exploitable for compression.
   - Refuted if: K* / N_patterns > 0.7, indicating that most activation patterns compute meaningfully different functions.
   - Inconclusive if: accuracy drops sharply as soon as any merging is applied (K* ≈ N_patterns) even though many A_ω look geometrically similar, suggesting that functional similarity does not survive k-means clustering.
   - Depends on: none

23. **Effective number of affine behaviors via Jacobian k-means — GPT-2 FFN**
   - Apply the same clustering idea from experiment 22 to GPT-2 FFN blocks. Instead of MASO affine maps (unavailable for GELU networks), use per-token Jacobians J(x) ∈ ℝ^{768×768} computed via `torch.func.jacrev`. Run k-means on the flattened Jacobians for each block. Measure the reconstruction error of block outputs when each token's output is replaced by J̄_cluster(x − x̄_cluster) + f(x̄_cluster) (first-order Taylor approximation using the cluster centroid Jacobian). Find minimum K* such that reconstruction error < 5% of block output norm. Compare K* across blocks and correlate with single-block swap damage from cycle 005.
   - Type: empirical
   - Confirmed if: for a majority of GPT-2 blocks, K* < 0.1 × N_probe AND K* correlates with compression difficulty (Spearman ≥ 0.4).
   - Refuted if: K* > 0.5 × N_probe for most blocks, indicating that every probe token activates a genuinely distinct functional regime.
   - Inconclusive if: K* is low but reconstruction error does not translate to perplexity improvement when the approximation is used for inference.
   - Depends on: 9, 22. Skip if experiment 22 is refuted (functional redundancy does not exist in toy networks, making the LLM analog unmotivated).

24. **RTG spectral coarsening: Fiedler-based region merging — toy ReLU**
   - For a trained spirals ReLU network (2D input), enumerate all observed activation patterns on probe data and build the Hamming-1 adjacency graph (RTG): nodes = unique binary activation vectors, edges = pairs differing by one neuron flip. Compute the weighted graph Laplacian, its Fiedler vector (second eigenvector), and a K-way spectral clustering for K = 2, 4, 8, 16. For each K, merge all activation patterns in the same spectral cluster: compute the cluster's affine map as the weighted average A_ω of its members (weighted by region support size), then evaluate network output accuracy using these K affine maps. This is the first empirical test of whether the RTG spectral structure (Bosca 2025, Dhayalkar 2025) gives a principled merging criterion, and whether it outperforms random or size-based merging at the same K.
   - Type: implementation + empirical
   - Confirmed if: RTG spectral coarsening achieves ≥ 3× region reduction with < 5% accuracy drop AND outperforms random region merging by ≥ 5 percentage points in accuracy at the same K.
   - Refuted if: spectral coarsening does not outperform random merging, suggesting the RTG eigenvectors carry no useful merging signal.
   - Inconclusive if: spectral coarsening outperforms random on spirals but the benefit disappears for the 100D embedded version, indicating the advantage is geometry-dependent.
   - Depends on: none

25. **RTG spectral coarsening — GPT-2 FFN analog**
   - Build a k-NN graph on per-token Jacobians for each GPT-2 FFN block (k=10, distance = Frobenius norm between J(x_i) and J(x_j)). This is the smooth-activation analog of the Hamming graph from experiment 24: instead of Hamming-1 adjacency between binary patterns, we use geometric proximity in Jacobian space. Compute the Laplacian of this graph, extract K eigenvectors, cluster tokens into K groups. For each cluster, fit a per-cluster affine approximation and measure the output reconstruction error relative to the full block output. Compare this to k-means on raw Jacobians (experiment 23) — the hypothesis is that the graph-Laplacian clustering is a better merging criterion because it respects local neighborhood structure.
   - Type: empirical
   - Confirmed if: spectral coarsening reduces reconstruction error by ≥ 15% relative to k-means at matched K for a majority of blocks.
   - Refuted if: spectral coarsening performs at parity with or worse than k-means on raw Jacobians.
   - Inconclusive if: the benefit appears only at small K (K ≤ 4) and vanishes as K grows.
   - Depends on: 9, 24. Skip if experiment 24 is refuted (RTG spectral merging does not outperform random merging even on toy networks).

26. **Intrinsic dimension trajectory through depth (GPT-2)**
   - At each of GPT-2's 13 block output checkpoints, collect token representation vectors from the probe corpus and compute their intrinsic dimensionality using the TwoNN estimator (Facco et al. 2017). TwoNN estimates the manifold dimension from the ratio of 1st- and 2nd-nearest-neighbor distances and requires no hyperparameters. A decreasing intrinsic dimension trajectory means later blocks are projecting the data onto a simpler manifold — those blocks are doing less work and are candidates for compression. Compare the intrinsic dimension profile to the single-block swap damage curve from cycle 005: if high-damage blocks coincide with layers where intrinsic dimension is still changing rapidly, that confirms the manifold contraction is task-relevant.
   - Type: empirical
   - Confirmed if: intrinsic dimension decreases by ≥ 30% over the final third of layers AND Spearman correlation between (per-block intrinsic dimension change) and (compression difficulty from cycle 005) is ≥ 0.4.
   - Refuted if: intrinsic dimension is approximately constant across all layers (< 10% variation), indicating that all layers operate on the same-complexity manifold.
   - Inconclusive if: intrinsic dimension varies but the trend is non-monotonic with no correlation to compression difficulty.
   - Depends on: 9

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

32. **GELU → 3-piece PWL approximation fidelity — toy GELU MLP**
   - Train a GELU MLP on spirals (2D input → 100D embedding → 4 layers × 64 hidden → output) to convergence. Replace each GELU with a 3-piece piecewise-linear approximation: g̃(x) = 0 for x < −3, a linear ramp from 0 to 0 over [−3, 3] matching GELU at endpoints and midpoint, x for x > 3. No retraining. Re-evaluate accuracy. This tests whether the smooth transition zone of GELU carries any task-relevant information, or whether the network can be treated as approximately piecewise-affine (making MASO/SplineCam/RTG theory applicable). Also measure the accuracy drop as a function of the number of PWL pieces (1, 3, 5, 10) to find the minimum resolution at which ReLU theory kicks in.
   - Type: empirical
   - Confirmed if: 3-piece PWL substitution causes < 3% accuracy drop and 5-piece causes < 1%, indicating that GELU networks are approximately piecewise-affine at low piece count.
   - Refuted if: even 10-piece PWL causes > 5% accuracy drop, indicating that smooth curvature is task-critical and the piecewise-affine approximation is fundamentally flawed.
   - Inconclusive if: accuracy recovers only with a large number of pieces (> 10), suggesting GELU networks are piecewise-affine but at much finer granularity than ReLU.
   - Depends on: none

33. **GELU → 3-piece PWL approximation fidelity — GPT-2**
   - Apply the same PWL substitution from experiment 32 to GPT-2: replace all GELU activations with the 3-piece PWL function, no retraining, measure perplexity on the probe validation set. If perplexity degrades < 5% relative, then GPT-2's FFN blocks are approximately piecewise-affine and all the partition-based theory (activation patterns, RTG, SplineCam) applies as an approximation. This would justify using ReLU-derived compression signals on GELU networks. Also measure block-by-block contribution: which blocks' perplexity degrades most when only that block's GELU is replaced, to identify which blocks are most sensitive to smooth activation curvature.
   - Type: empirical
   - Confirmed if: full GELU → 3-piece PWL substitution causes < 5% relative perplexity increase on the validation set.
   - Refuted if: perplexity increases > 20%, indicating that GELU's smooth transition zone is functionally critical at the LLM scale.
   - Inconclusive if: perplexity increase is moderate (5–20%) and is concentrated in specific blocks rather than spread evenly.
   - Depends on: 32. Skip if experiment 32 is refuted (the PWL approximation is fundamentally inadequate even for toy GELU networks).

34. **Region volume Pareto: do large regions carry most task accuracy? — toy ReLU**
   - For a trained spirals ReLU network (2D input → 100D embedding → 4 layers × 64 hidden), enumerate all observed activation patterns on the full probe set. Define region "volume" as the fraction of probe inputs that fall in each region. Sort regions by volume in descending order. Compute the cumulative accuracy recovery as a function of regions kept (the "Pareto compression curve"): start with only the largest region, classify all test points using that region's A_ω, add the next largest region and allow each test point to use its own region's A_ω if it falls there, otherwise fall back to the nearest large region. If the Pareto curve is steep (top 20% of regions by volume covers ≥ 80% of correctly classified points), compression can be achieved by keeping only large regions and routing small-region inputs to the nearest large region.
   - Type: empirical
   - Confirmed if: the top 20% of regions by volume covers ≥ 70% of correctly classified test points.
   - Refuted if: regions have an approximately uniform volume distribution (Gini coefficient < 0.3), indicating no Pareto structure.
   - Inconclusive if: Pareto holds in 2D but disappears when the same analysis is run in the 100D embedding, suggesting the structure is input-space-specific.
   - Depends on: none

35. **Region volume Pareto — GPT-2 FFN analog**
   - Apply the Pareto analysis from experiment 34 to GPT-2 FFN blocks. Since exact activation patterns are unavailable for GELU, use k-means clusters on probe-token Jacobians as a proxy for "regions." Sort clusters by token count (proxy for volume). For each block, compute the cumulative reconstruction error of block outputs as clusters are added in descending size order. If a Pareto structure exists (top 20% of clusters covers 80% of total output variance), it suggests that a few dominant "computational modes" drive most of the block's behavior — and keeping only those modes gives a strong compression baseline. Compare the Pareto coefficient to compression damage from cycle 005 to test whether Pareto-skewed blocks are the easy ones.
   - Type: empirical
   - Confirmed if: top 20% of clusters by size covers ≥ 70% of output variance in a majority of blocks AND Gini coefficient of cluster-size distribution correlates with (1 − compression difficulty) at Spearman ≥ 0.35.
   - Refuted if: cluster size distribution is approximately uniform across all blocks (Gini < 0.2), indicating no dominant computational modes.
   - Inconclusive if: Pareto holds in toy but the LLM cluster distribution is flat, suggesting the Jacobian k-means proxy is not capturing the same structure as true activation regions.
   - Depends on: 9, 22, 34. Skip if experiment 34 is refuted (Pareto structure does not appear even in toy ReLU networks).


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

### Cycle 007

12. **Progressive multi-block accumulation curve**
   - Type: empirical
   - Outcome: **confirmed (proxy)** — easiest-first outperforms hardest-first by 27.4% at k=6 and 36.2% at k=9, exceeding the ≥25% threshold. Advantage grows with block count.
   - Cycle: 007
   - Commit: TBD
   - Artifact: `autoresearch/results/cycle_007_summary.md`, `autoresearch/results/cycle_007_accumulation.json`
   - Key finding: Ordering matters enormously for multi-block accumulation. Hardest-first is catastrophically worse. Single-block damage ranking from cycle 005 correctly predicts safe accumulation order. Sweet spot is k=4-5 easiest blocks (ppl < 150, < 25% degradation). All-block 16x compression fails regardless of ordering (ppl=403).
