# Task
230. **W_down column geometry census — anti-podal pairs, clusters, cross-block alignment**
For each of the 12 GPT-2 small blocks, treat `mlp.c_proj.weight` (Conv1D, shape `(3072, 768)`) as a stack of 3072 columns of W_down in R^768. Measure: pairwise cosine structure, anti-podal pair count, top-20 singular values, k=32 KMeans cluster quality, and cross-block principal-angle alignment between block l's W_down writing subspace and block l+1's W_up reading subspace. Weights-only, no activations, no PPL eval.

# Hypothesis
- **Confirmed**: "≥20% of columns in blocks 1-5 have a near-anti-podal twin (cos < −0.9), AND column clusters are geometrically distinct (mean within-cluster cos > 0.5, between-cluster cos < 0.2 at k=32)."
- **Refuted**: "<5% anti-podal pairs in any block, AND columns are near-uniformly distributed on the sphere (no cluster structure)."
- **Inconclusive**: some structure but no clean story.

# Pre-flight Results
- P1 (duplicate): NO. Items 188-214 are activation-based; cycle 195 measured Jacobian effective rank (not columns); cycle 201 did SVD of block 11 input/output (not W_down columns). This is the first W_down column weight-geometry census in the lineage.
- P2 (dependency): NONE.
- P3 (≤30 min): YES. Per block: 3072×768 mat → 3072×3072 cosine (~36 MB fp32), SVD of 768×3072 (<1s CPU), KMeans k=32 with sklearn n_init=5 (<5s). 12 blocks × ~10 s = ~2 min total.
- P4 (program/prompts): NO edits needed.
- Conv1D convention verified from brief: `c_proj.weight` shape `(3072, 768)`, `c_fc.weight` shape `(768, 3072)`. The column c of W_down as a vector in R^768 = `c_proj.weight[c, :]` (row slice).

# Steps
1. Create `experiments/wdown_column_census_cycle216.py`.
2. Imports: `torch`, `json`, `time`, `numpy as np`, `from transformers import GPT2LMHeadModel`, `from sklearn.cluster import KMeans`, `from pathlib import Path`.
3. Load `GPT2LMHeadModel.from_pretrained("gpt2")`, `model.eval()`, CPU is fine.
4. Assertion: for block 0, `assert model.transformer.h[0].mlp.c_proj.weight.shape == (3072, 768)` and `assert model.transformer.h[0].mlp.c_fc.weight.shape == (768, 3072)`.
5. Per block `l` in `range(12)`, compute and store:
   a. `W_down_cols = model.transformer.h[l].mlp.c_proj.weight.detach().float()` → shape `(3072, 768)`. Each row = one column of W_down in R^768.
   b. `cols_norm = W_down_cols / W_down_cols.norm(dim=1, keepdim=True).clamp_min(1e-12)` → shape `(3072, 768)`.
   c. `cos_mat = cols_norm @ cols_norm.T` → shape `(3072, 3072)`.
   d. Upper-triangle mask `iu = torch.triu_indices(3072, 3072, offset=1)`; `cos_upper = cos_mat[iu[0], iu[1]]`.
   e. For each threshold τ in `[-0.95, -0.90, -0.85, -0.80]`: `n_pairs_τ = int((cos_upper < τ).sum())`, `pct_columns_with_twin_τ = 100.0 * 2 * n_pairs_τ / 3072` (capped at 100.0).
   f. Cosine histogram: 50 bins in [-1, 1], store `bin_edges` + `counts`. Also `cos_upper.min()`, `mean()`, `std()`.
   g. SVD: `U, S, Vh = torch.linalg.svd(W_down_cols, full_matrices=False)`; record `S[:20].tolist()` and full `S.shape[0]` for reference. Compute `rank_95 = int((S.cumsum(0) / S.sum() < 0.95).sum() + 1)`.
   h. KMeans: `km = KMeans(n_clusters=32, n_init=5, random_state=0).fit(cols_norm.cpu().numpy())`. Labels `lab` shape `(3072,)`. Centroids `C = torch.from_numpy(km.cluster_centers_)`; normalize centroids.
      - Mean within-cluster cos: for each cluster, compute mean pairwise cos of its members (skip singleton clusters); aggregate weighted by cluster size. Implementation: `within = []; for k in range(32): mask = lab == k; pts = cols_norm[mask]; if pts.shape[0] < 2: continue; cm = pts @ pts.T; iu_k = torch.triu_indices(pts.shape[0], pts.shape[0], offset=1); within.append(cm[iu_k[0], iu_k[1]].mean().item()); sizes.append(pts.shape[0])`. Report weighted mean.
      - Mean between-cluster centroid cos: normalize `C`; `C_cos = C @ C.T`; upper-triangle mean.
      - Cluster size distribution: `sizes` list (length 32).
   i. Block l's "writing subspace" V_l in R^768: top-20 right singular vectors of `c_proj.weight` → columns of `Vh[:20, :].T` (shape `(768, 20)`). Note: `torch.linalg.svd` of `(3072, 768)` returns `Vh` shape `(768, 768)`; take `V_l = Vh[:20, :].T` (shape `(768, 20)`), columns are the top-20 right singular vectors.
6. Cross-block alignment (for `l in range(11)`):
   a. Get `c_fc_next = model.transformer.h[l+1].mlp.c_fc.weight.detach().float()` → shape `(768, 3072)`.
   b. `U_next, S_next, Vh_next = torch.linalg.svd(c_fc_next, full_matrices=False)`. The reading subspace in R^768 is span of `U_next[:, :20]` (shape `(768, 20)`). Columns are top-20 left singular vectors of `c_fc` (= column space of `c_fc`).
   c. `M = V_l.T @ U_next[:, :20]` → shape `(20, 20)`. Principal-angle cosines = `torch.linalg.svdvals(M)` → length 20. Record as list. Also record `mean_cos`, `top5_mean_cos`.
7. Random baseline (run once, not per block):
   a. For each `seed in [0, 1, 2]`: `g = torch.Generator().manual_seed(seed); X = torch.randn(3072, 768, generator=g); X = X / X.norm(dim=1, keepdim=True)`; `cos_mat_rand = X @ X.T`; `cos_upper_rand = cos_mat_rand[iu[0], iu[1]]`.
   b. For each τ: count `(cos_upper_rand < τ).sum()`. Record per-seed and mean±std and max-negative `cos_upper_rand.min()`.
   c. Also: KMeans k=32 within/between cos on `X` (single seed=0) for cluster-baseline comparison.
8. Verdict computation in code after all blocks done:
   - `confirmed = all(pct_columns_with_twin_-0.9[l] >= 20.0 for l in [1,2,3,4,5]) and all(within_mean[l] > 0.5 and between_mean[l] < 0.2 for l in [1,2,3,4,5])`
   - `refuted = all(pct_columns_with_twin_-0.9[l] < 5.0 for l in range(12)) and all(within_mean[l] < 0.2 for l in range(12))`
   - `verdict = "CONFIRMED" if confirmed else ("REFUTED" if refuted else "INCONCLUSIVE")`
9. Write `autoresearch/results/cycle_216_wdown_geometry.json`:
   ```
   {"cycle":216,"item":230,"verdict":...,"config":{"model":"gpt2","n_blocks":12,"n_cols":3072,"d":768,"kmeans_k":32,"kmeans_n_init":5,"thresholds":[-0.95,-0.9,-0.85,-0.8]},
    "per_block":[{"block":l,"pct_twin":{"-0.95":...,"-0.9":...,"-0.85":...,"-0.8":...},"n_pairs":{...},"cos_stats":{"min":...,"mean":...,"std":...},"hist":{"edges":[...],"counts":[...]},"svd_top20":[...],"rank_95":...,"within_cluster_cos":...,"between_centroid_cos":...,"cluster_sizes":[...],"cross_block_cos":[...] or null,"cross_block_mean":...,"cross_block_top5_mean":...,"claim":"one sentence"} for l in 0..11],
    "random_baseline":{"seeds":[0,1,2],"n_pairs":{"-0.95":[...],"-0.9":[...],"-0.85":[...],"-0.8":[...]},"mean_pairs_-0.9":...,"std_pairs_-0.9":...,"min_cos_per_seed":[...],"kmeans_within":...,"kmeans_between":...},
    "aggregate":{"blocks_meeting_20pct_twin":[...],"blocks_meeting_within_gt_0.5":[...],"blocks_meeting_between_lt_0.2":[...],"twin_fold_over_random":[...]}}
   ```
10. Per-block one-sentence claim generated in code: `f"Block {l}: {pct_twin_-0.9:.1f}% of columns have an anti-podal twin (cos<-0.9); {rank_95} singular values capture 95% of W_down energy; KMeans k=32 within/between = {within:.2f}/{between:.2f}."`
11. Write `autoresearch/results/cycle_216_summary.md`:
    - 1-2 paragraphs: model/config, overall verdict, headline finding (e.g. "X of 12 blocks meet the anti-podal threshold; random baseline = Y pairs per 3072; fold = Zx"), pointer to JSON.
    - Per-block table: `| block | %twin@-0.9 | within | between | rank_95 | xblock_top5 |`.
    - One-sentence claim per block.
    - Explicit verdict statement and which criterion decided it.

# Validation
- Shape asserts pass (step 4).
- Per-block `cos_upper.shape == (3072*3071//2,)`.
- SVD: `S.shape == (768,)`; top SV strictly positive and decreasing.
- KMeans labels in `[0, 32)`.
- Cross-block cosines ∈ [0, 1] (singular values of sub-matrix), length 20 per pair, 11 pairs total.
- Random baseline `n_pairs_-0.9` should be near-zero (expect <5 per seed in R^768 with 3072 points); if ≫0 the Gaussian normalization is wrong.
- Total wall time <10 min.

# Artifacts
- `experiments/wdown_column_census_cycle216.py`
- `autoresearch/results/cycle_216_wdown_geometry.json`
- `autoresearch/results/cycle_216_summary.md`
- `autoresearch/results/cycle_216_manifest.txt` (written by execute stage)

# Fallbacks
- If `sklearn` unavailable: pure-torch KMeans — init centroids via random 32 columns, 20 Lloyd iterations. Deterministic seed=0.
- If `torch.linalg.svd` of `(3072, 768)` hits numerical issues: fall back to `torch.svd_lowrank(W_down_cols, q=30)` for top-20 SVs only.
- If cross-block alignment produces `nan` (rank-deficient submatrix): log and set `cross_block_cos=null`, `cross_block_mean=null` for that pair; do not block verdict.
- If wall-time exceeds 20 min (unexpected): drop KMeans `n_init` from 5 to 1 and note under `config.reduced_n_init`.

# Update bookkeeping
- Move item 230 from NEXT/P0 to PAST EXPERIMENTS under new `### Cycle 216` section in `autoresearch/agenda.md` with verdict + key metrics + artifact list.
- Append row to `autoresearch/logs/experiments.tsv`.
- Commit: `[autoresearch] cycle 216: W_down column geometry census`.
- Refresh `autoresearch/STATUS.md`.
