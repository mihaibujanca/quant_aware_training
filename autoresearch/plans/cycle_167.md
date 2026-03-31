# Cycle 167 Plan: Sign-Pattern Census (Item 177)

## Task
Count unique binary sign patterns (z > 0) of GPT-2 FFN pre-GELU activations on real data. Cluster with k-means, find K* = min k for <5% within-cluster FFN output variance.

## Hypothesis
With ~91% channels locked negative (cycle 165), sign patterns are highly redundant. K* <= 64 means the FFN's piecewise-linear structure is tractably partitioned by sign patterns.

## Pre-flights
- P1 (duplicate): no prior cycle tested sign-pattern clustering on GPT-2 FFN.
- P2 (deps): none. Standalone analysis.
- P3 (scope): 12 blocks x 4096 tokens x 4 k-values = light compute. Single cycle.
- P4 (program.md): no modifications needed.

## Steps
1. Load GPT-2, tokenize 4096 OWT tokens (from `data/openwebtext_sample.txt`).
2. For each of 12 blocks, hook `mlp.c_fc` output (pre-GELU z).
3. Forward pass, collect z per block. Binary patterns: `sign = (z > 0).bool()`.
4. Per block:
   a. Count unique sign patterns (pack bits, use `torch.unique`).
   b. Sample 10000 pairs, compute Hamming distances, report mean/std/histogram.
   c. Also hook `mlp.c_proj` output (FFN output y) for variance computation.
   d. For k in {16, 32, 64, 128, 256}: run k-means on float sign vectors.
      Metric: within-cluster FFN output variance / total FFN output variance.
   e. K* = min k where ratio < 0.05.
5. Save JSON results, plot Hamming histogram + K* per block bar chart.
6. Verdict: K* <= 64 in >=6/12 blocks → CONFIRMED. K* > 256 → REFUTED.

## Validation
- Baseline PPL sanity: must be 20-25 (GPT-2 on OWT).
- Unique pattern count << 4096 per block (if ~91% channels are locked).
- Hamming distances should cluster around ~2 * p_flip * 3072 where p_flip ~ 9%.

## K-means implementation
Use `sklearn.cluster.MiniBatchKMeans` on boolean patterns cast to float32. 3072 dims, 4096 samples — tractable. Within-cluster variance: for each cluster, compute var(y) for tokens assigned to that cluster, average weighted by cluster size. Total variance: var(y) over all tokens. Ratio = weighted_within / total.

## Artifacts
- `experiments/sign_pattern_census_cycle167.py`
- `autoresearch/results/cycle_167_sign_census.json`
- `autoresearch/results/cycle_167_sign_census.png`

## Fallbacks
- If k-means on 3072-dim bools is slow, use PCA to 50 dims first (sign patterns are ~277-dim effective).
- If MPS runs out of memory on 4096 x 3072 float32, reduce to 2048 tokens.
