# Cycle 134 Plan: Data-Conditioned Local Affine Map Extraction (Item 135)

## Task
Compute per-token Jacobians of GPT-2 FFN blocks {4, 7} on 128 OWT tokens. Measure pairwise cosine similarity to find distinct local affine behaviors.

## Hypothesis
If GELU FFNs act as piecewise-affine maps with few distinct regions (Walker et al. 2025), then most token Jacobians should cluster into <=50 groups with high intra-group cosine similarity (>0.95).

## Pre-flight Checks
- P1 (duplicate): item 23 was similar but OBSOLETE (used k-means, not cosine+spectral). No prior cycle ran this.
- P2 (deps): none.
- P3 (scope): 128 tokens x 2 blocks = 256 Jacobians. ~5-10 min CPU. Single-cycle.
- P4 (program.md): no modifications needed.
- P5 (null baseline): if Jacobians were random 768x768 matrices, pairwise cosine of flattened vectors would be ~0 (high-dimensional). Any significant clustering is meaningful.
- P6 (memory): each Jacobian is 768x768 float32 = 2.25 MB. 128 Jacobians = ~288 MB per block. Pairwise cosine similarity matrix: 128x128 = tiny. No memory issue.

## Steps
1. Load GPT-2 (124M) from HuggingFace. Load 128 tokens from OWT (first sequence, truncated).
2. For each block in {4, 7}:
   a. Define `ffn_fn(x) -> y` that runs the block's FFN (ln_2 -> c_fc -> GELU -> c_proj) on a single token embedding x (768,).
   b. Compute Jacobian J_i = torch.func.jacrev(ffn_fn)(x_i) for each token i. Shape: (768, 768).
   c. Flatten each J_i to (768*768,). Normalize to unit norm.
   d. Compute pairwise cosine similarity matrix S (128x128).
   e. Report: mean, median, min, max of off-diagonal S entries.
   f. Spectral analysis: eigendecompose S, find spectral gap (ratio of consecutive eigenvalues). Effective number of clusters = number of eigenvalues above 1/N threshold.
   g. Threshold clustering: for tau in {0.90, 0.95, 0.99}, greedily assign tokens to groups (token joins the first group whose centroid has cos > tau, else starts a new group). Report group count and coverage.
3. For each pair (i, j) with S[i,j] > 0.95:
   a. Compute affine map: y_approx = J_i @ x_j + (FFN(x_i) - J_i @ x_i). The bias term b_i = FFN(x_i) - J_i @ x_i.
   b. Compute relative error: ||y_approx - FFN(x_j)|| / ||FFN(x_j)||.
   c. Report: mean, median, max relative error across all high-similarity pairs.
4. Save results JSON and similarity heatmap.
5. Apply verdict criteria:
   - CONFIRMED: <=50 groups at tau=0.95 cover >=90% of tokens in BOTH blocks.
   - REFUTED: >200 groups at tau=0.95 for 90% coverage in either block.
   - INCONCLUSIVE: 50-200 groups, or groups exist but interchangeability error >5%.

## Validation
- Sanity: Jacobian diagonal should be non-zero (GELU is nonlinear). If all-zero, hook is wrong.
- Sanity: S[i,i] = 1.0 (self-similarity). If not, normalization is broken.
- Sanity: affine approximation error for (i, i) should be ~0 (first-order Taylor at the expansion point).
- If Jacobian computation takes >20 min total, abort and report timing only.

## Criteria (verbatim from agenda)
- **Confirmed** if: <=50 distinct affine groups cover >=90% of tokens in >=2/4 blocks AND centroid-affine compression matches topk+lstsq single-block PPL within 5%.
- **Note**: centroid-affine compression test is DEFERRED (scope reduction). Cycle can be at most INCONCLUSIVE-positive (groups exist) or REFUTED (groups don't exist). CONFIRMED requires follow-up cycle with compression test.

## Artifacts
- `experiments/affine_map_extraction_cycle134.py`
- `autoresearch/results/cycle_134_affine_maps.json`
- `autoresearch/results/cycle_134_similarity_heatmap.png`

## Fallbacks
- If `torch.func.jacrev` is too slow (>2s per Jacobian): use finite differences with h=1e-4 on 50 random input dimensions instead of full 768. Report partial Jacobian similarity as a proxy.
- If pairwise cosine similarity is uninformative (all values in [0.4, 0.6]): try cosine on the TOP-10 singular vectors of each Jacobian instead of the full flattened matrix.
