# Cycle 101 — Item 106: GELU output clustering (corrected) — GPT-2 FFN

## Task

Cluster tokens by GELU'(z) similarity (3072-dim k-means) and measure whether tokens in the same cluster produce similar **actual FFN outputs**. This corrects cycle 085 (item 97), which used Taylor linearization around centroids — a methodology that trivially fails for smooth nonlinearities at large displacements.

## Hypothesis

**Confirmed if:** K* <= 64 for >=6/12 blocks (tokens in same GELU' cluster DO produce similar outputs).
**Refuted if:** K* > 256 for majority of blocks (even with correct reconstruction, outputs are too diverse).
**Inconclusive if:** K* varies wildly across blocks.

Where K* = min k such that mean within-cluster reconstruction error < 5% of output norm.

## Pre-flight Results

**P1 (duplicate):** Not a duplicate. Cycle 085/item 97 tested this question with WRONG methodology (Taylor linearization: `y ≈ gelu(z_c) + gelu'(z_c)*(z - z_c)`). Item 106 explicitly supersedes it with correct per-token GELU reconstruction. Different methodology = different experiment.

**P2 (dependencies):** None. Supersedes item 97.

**P3 (one cycle):** Yes. Single analysis script: capture activations, cluster, measure errors. No training loop.

**P4 (program/prompt changes):** No.

**P5 (null baseline):** k=1 baseline (global mean output) gives the random-clustering score. Cycle 085 k=1 null error was 0.69–1.93 per block. Under the corrected methodology (cluster-mean actual outputs), the k=1 error should still be large but potentially different. Must compute explicitly.

**P6 (GELU appropriateness):** Yes. Clustering on GELU'(z) groups tokens by derivative similarity — tokens with similar gate profiles. The key question is whether derivative similarity implies output similarity. Method applies actual GELU per-token, no linearization.

**P7 (statistical power):** 2048 tokens for k-means on 3072-dim with k up to 256: at k=256, only ~8 tokens/cluster on average (2048/256). This is borderline. Use 4096 tokens (8 sequences × 4 sources × 128 length) to get ~16 tokens/cluster at k=256. Train/holdout split: 3072/1024.

## Steps

1. **Copy and modify** `experiments/gelu_pattern_diversity.py` → `experiments/gelu_output_clustering_corrected.py`. Key changes:
   - Increase to N_TOKENS=4096 (8 sequences per source instead of 4).
   - Train/holdout split: N_TRAIN=3072, N_HOLDOUT=1024.
   - **Cluster on GELU'(z)** (not z) via k-means. The derivative vector captures gate state.
   - **CORRECT reconstruction**: for each cluster c, compute `y_c = mean(W_out @ gelu(z_i) + b_out)` for training tokens i in cluster c. Reconstruction error per holdout token: `||W_out @ gelu(z_j) + b_out - y_c|| / ||W_out @ gelu(z_j) + b_out||` where c = assigned cluster of token j.
   - Remove all Taylor linearization code.
   - Remove binary pattern analysis (already done in cycle 085, not needed here).
   - Add within-cluster vs between-cluster output variance ratio.

2. **K-means sweep** over k ∈ {8, 16, 32, 64, 128, 256}, all 12 blocks.

3. **For each (block, k)** compute and store:
   - Mean within-cluster reconstruction error (fraction of output norm).
   - 95th percentile within-cluster reconstruction error.
   - Within-cluster output variance: `mean_c[var(y_i for i in c)]`.
   - Between-cluster output variance: `var(y_c across clusters)`.
   - Variance ratio: between / (within + eps).
   - Number of degenerate clusters (train count < 3).

4. **Determine K*** per block: min k where mean reconstruction error < 0.05.

5. **Plot**: heatmap of mean reconstruction error, blocks (rows) × k (columns). Save to `autoresearch/results/cycle_101_gelu_output_clustering.png`.

6. **Save JSON** results to `autoresearch/results/cycle_101_gelu_output_clustering.json`.

7. **Verdict**: apply criteria. Update agenda item 106 with outcome. Append to experiments.tsv.

## Validation

- k=1 null baseline error must exceed 5% for all blocks (sanity check that experiment is informative).
- Reconstruction errors must decrease monotonically with k (more clusters = better fit). If not, flag as anomalous.
- Verify actual GELU is applied: add assertion that `gelu(z_i) @ W_out + b_out` matches hooked FFN output within float32 tolerance (~1e-5). If GPT-2 MLP includes dropout or layer norm, the hook captures the full MLP output which may differ — compare shapes and verify.
- Cross-check: for k=256 under the old (wrong) linearized method, cycle 085 got error 0.69–1.93. Under correct method, k=256 error should be strictly lower (cluster-mean of actual outputs is a better approximation than linearized reconstruction). If not, something is wrong.

## Artifacts

- `experiments/gelu_output_clustering_corrected.py` — experiment script
- `autoresearch/results/cycle_101_gelu_output_clustering.json` — full results
- `autoresearch/results/cycle_101_gelu_output_clustering.png` — heatmap

## Fallbacks

- If 4096 tokens OOM on GPU: reduce to 2048, note reduced statistical power at k=256.
- If k-means on 3072-dim GELU'(z) is slow: use MiniBatchKMeans (already used in cycle 085 code).
- If FFN hook output doesn't match `gelu(z) @ W_out + b_out` (due to dropout/residual): inspect GPT-2 MLP structure. The MLP forward is `c_proj(gelu(c_fc(x)))`, so the MLP output IS `gelu(z) @ W_out + b_out` (Conv1D). Verify `c_proj` is the only post-GELU op in MLP.
- If all blocks have K* > 256: report refuted, but also report the actual error curve shape (does it plateau or keep decreasing?). This determines whether more clusters would eventually work or if the approach is fundamentally limited.
