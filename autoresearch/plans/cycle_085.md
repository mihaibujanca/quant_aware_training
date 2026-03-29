# Task
Item 97: GELU activation pattern diversity — GPT-2 FFN. For all 12 blocks, compute GELU'(z) on 2048 mixed-corpus tokens (3072 channels) → soft activation patterns. Binarize at 0.5, count unique patterns, Hamming distances. K-means on continuous GELU' vectors, k in {8,16,32,64,128,256}. Find K* = min k with centroid reconstruction error < 5% of block output norm. Uses consistent affine linearization: d_c = GELU'(z_c), not k-means centroid of GELU'.

# Hypothesis
- **Confirmed** if: K* ≤ 64 for ≥6/12 blocks (FFN has manageable piecewise-linear complexity).
- **Refuted** if: K* > 256 for majority of blocks (GELU patterns too continuous/diverse to cluster).
- **Inconclusive** if: K* varies wildly across blocks.

# Pre-flight Results
- P1: Not duplicate. Cycle 083 (item 94) did per-channel near-linearity census; this is per-token pattern analysis. Cycle 084 (item 95) was rejected (affine inconsistency) and targeted block 7 only. This covers all 12 blocks with corrected methodology.
- P2: No dependencies.
- P3: Concrete — single script, clear metric, well-defined sweep.
- P4: No program.md changes.
- P5: Null baseline — N(0,1) inputs to GELU: GELU'(z) ≈ Φ(1.702z) ≈ 0.5 at z=0. Binarized patterns at threshold 0.5 have ~50% active bits. For 3072 independent Bernoulli(0.5) bits, expected unique count = 2048 (all distinct w.p.1, since 2^3072 >> 2048). K-means on random 3072-dim vectors in [0,1]: within-cluster variance scales as k^{-2/d} ≈ k^0 for d=3072 (curse of dimensionality; random clustering barely improves). Report k=1 reconstruction error as null.
- P6: GPT-2 uses tanh-approx GELU. All GELU/GELU' computations use `approximate='tanh'`.
- P7: 2048 tokens, k-means k_max=256. Ratio n/k = 8 at k=256 — marginal. Use 1536 train / 512 holdout split. At k=256, 1536/256=6 tokens per cluster — report cluster sizes and flag degenerate clusters (<3 members).

# Steps
Script: `experiments/gelu_pattern_diversity.py`. Adapt infrastructure from `experiments/gelu_nearlinearity_census_v2.py`.

1. **Setup.** Load GPT-2. Build 2048-token probe: 512 tokens each from shakespeare, wikitext2, OWT, C4 (4× ProbeCorpusSpec with probe_train_sequences=4, seq_len=128). Split: first 1536 tokens = train, last 512 = holdout.
2. **Collect pre-GELU z.** Hook `model.transformer.h[i].mlp.c_fc` for all 12 blocks. Forward pass in batches of 4. Collect z (n_tokens, 3072) per block. Also store ground-truth FFN outputs y = block.mlp(hidden_states) per block.
3. **Compute GELU' patterns.** For each block: z.requires_grad_(True), g = F.gelu(z, approximate='tanh'), d = autograd.grad(g.sum(), z)[0]. Sanity: assert d.min() >= -0.02, d.max() <= 1.02.
4. **Binary pattern analysis.** Binarize d at 0.5. Count unique patterns per block. Compute pairwise Hamming distances (subsample 512 pairs for speed). Report: n_unique, mean/std/min/max Hamming distance per block.
5. **K-means sweep.** For each block, for k in {8, 16, 32, 64, 128, 256}: sklearn KMeans on D_train (1536×3072), n_init=5, max_iter=300. Record cluster assignments, inertia. Assign holdout tokens to nearest centroid. Compute per-cluster: z_c = mean(z_train[cluster]), d_c = GELU'(z_c) via autograd (NOT k-means centroid of d). Diagnostic: report ||d_c_recomputed - d_c_kmeans_centroid||₁ / 3072 per cluster (Jensen gap).
6. **Centroid reconstruction.** For each holdout token x in cluster c: y_approx = W_out @ [GELU(z_c) + diag(d_c) @ (z - z_c)] + b_proj. Error = ||y_approx - y_true|| / ||y_true||. Report mean and p95 error per k per block.
7. **Find K*.** Per block: K* = min k with mean relative error < 5%. If k=256 still > 5%, report K* = >256.
8. **Save results.** JSON to `autoresearch/results/cycle_085_gelu_pattern_diversity.json`. Plot to `autoresearch/results/cycle_085_gelu_pattern_diversity.png`: (a) K* per block bar chart, (b) reconstruction error vs k curves for all 12 blocks.
9. **Bookkeeping.** Update agenda item 97. Commit: `[autoresearch] cycle 085: GELU activation pattern diversity — GPT-2`.

# Validation
- GELU'(0) = 0.5, GELU'(3) ≈ 1.0, GELU'(-3) ≈ -0.012. Assert before proceeding.
- k=1 reconstruction error = null baseline. Must be > 5% for experiment to be informative.
- Unique binary patterns: must be > null expectation (~2048) is impossible since n=2048, so expect ~2048. If n_unique << 1000, something is structured.
- Jensen gap diagnostic: if mean ||d_c_recomputed - d_c_kmeans|| > 0.05 for majority of clusters, flag in results.
- Cluster size check: flag any cluster with <3 members at k=256.

# Artifacts
- `experiments/gelu_pattern_diversity.py`
- `autoresearch/results/cycle_085_gelu_pattern_diversity.json`
- `autoresearch/results/cycle_085_gelu_pattern_diversity.png`
- Agenda update: item 97 outcome + cycle tag.
- Commit: `[autoresearch] cycle 085: GELU activation pattern diversity — GPT-2`

# Fallbacks
- KMeans slow at k=256 × 12 blocks → use MiniBatchKMeans(batch_size=256).
- k=256 still > 5% error → report K* = >256, note refuted, still report binary pattern diversity.
- OOM on 12 blocks × 2048 tokens → process blocks sequentially, discard z after each block.
- Any data source missing → fall back to 3 sources × 683 tokens each (2049 tokens, trim to 2048).
