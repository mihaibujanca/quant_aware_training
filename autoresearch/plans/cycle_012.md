# Task
Item 22: Effective number of affine behaviors via Jacobian k-means — toy ReLU.
Train a small spirals ReLU MLP, extract all observed activation patterns and their exact MASO affine maps, cluster with k-means, find minimum K* preserving accuracy within 1%.

# Hypothesis
- **Confirmed if**: K* / N_patterns <= 0.2 for the spirals network (i.e., fewer than 20% of observed patterns are functionally distinct), indicating high redundancy exploitable for compression.
- **Refuted if**: K* / N_patterns > 0.7, indicating that most activation patterns compute meaningfully different functions.
- **Inconclusive if**: accuracy drops sharply as soon as any merging is applied (K* ~ N_patterns) even though many A_w look geometrically similar, suggesting that functional similarity does not survive k-means clustering.

# Pre-flight Results
- P1 PASS: No prior experiment clusters full-network affine maps. Cycle 010 tested single global affine per layer (different question).
- P2 PASS: No dependencies (agenda says "Depends on: none").
- P3 PASS: Concrete pipeline — extract patterns, compute A_w, k-means sweep, measure accuracy.
- P4 PASS: No changes to program.md or prompts needed.

# Steps
1. Create `experiments/jacobian_kmeans_toy.py` with:
   - Dataset: spirals (n=2000, 3 turns, scale=2.0, noise=0.0, seed=42), split 50/50 train/test.
   - Architecture: `aleph.models.MLP` with dims `[2, 32, 32, 1]` (2 hidden layers, manageable pattern count).
   - Train: Adam lr=0.001, BCE loss, up to 500 epochs, stop if train acc > 99%.
   - Verify test accuracy >= 90% before proceeding (undertrained model = meaningless analysis).

2. Extract activation patterns on test set (n=1000):
   - Forward pass collecting per-layer ReLU masks via `aleph.relu_geometric_compression._forward_with_masks`.
   - Pack into activation codes via `_pack_activation_codes`. Record N_patterns = number of unique codes.
   - For each unique pattern, call `_effective_affine_from_mask(model, mask_layers)` -> (A_w, b_w).
     Each A_w is shape (1, 2), b_w is scalar. Flatten to vector [A_w | b_w] of length 3.

3. K-means sweep:
   - K values: `[2, 4, 8, 12, 16, 24, 32, 48, 64]` capped at N_patterns.
   - For each K, fit `sklearn.cluster.KMeans(n_clusters=K)` on the (N_patterns, 3) matrix.
   - Assign each test point to its pattern's cluster. Replace output with centroid affine map:
     `y_approx_i = A_centroid @ x_i + b_centroid` for the cluster that point i's pattern belongs to.
   - Compute accuracy = fraction of test points where `sign(y_approx) == sign(y_true)` matches original network.
   - Find K* = smallest K where accuracy drop vs float <= 1 percentage point.

4. Save results to `autoresearch/results/cycle_012_kmeans_toy.json`:
   ```
   {N_patterns, K_star, ratio: K_star/N_patterns, float_acc, per_K_accuracy: {K: acc}, arch, verdict}
   ```

5. Save plot `autoresearch/results/cycle_012_kmeans_curve.png`:
   - X-axis: K (log scale), Y-axis: test accuracy. Horizontal line at float accuracy and at float-1%.

6. Update `autoresearch/agenda.md`: mark item 22 with verdict and one-line result.

7. Write `autoresearch/results/cycle_012_summary.md` with setup, table of K vs accuracy, K*, ratio, verdict.

8. Commit all artifacts.

# Validation
- Train accuracy >= 99%, test accuracy >= 90% before analysis begins.
- N_patterns is in plausible range (expect 50-500 for 1000 probes through 2×32 hidden).
- K=N_patterns accuracy equals float accuracy exactly (sanity check: no clustering error at full K).
- K* and ratio directly determine Confirmed/Refuted/Inconclusive per hypothesis.

# Artifacts
- `experiments/jacobian_kmeans_toy.py`
- `autoresearch/results/cycle_012_kmeans_toy.json`
- `autoresearch/results/cycle_012_kmeans_curve.png`
- `autoresearch/results/cycle_012_summary.md`

# Fallbacks
- If N_patterns > 1000 with 2×32 arch: subsample to 500 most-populated patterns.
- If k-means is unstable (high variance across seeds): run 5 seeds, report median K*.
- If accuracy never recovers within 1%: report Inconclusive, include elbow-point K as soft K*.
