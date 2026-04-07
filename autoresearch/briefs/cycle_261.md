# Item
272 (P1): W_down column geometry vs per-channel sign frequency σ_c — the rescue-note experiment.
For each block, compute σ_c = E_x[1{z_c > 0}], split channels into σ ∈ [0,0.1] / [0.45,0.55] / [0.9,1.0], compute PCA spectrum of each subspace, principal angles between subspaces, special check at block 2 for rank-1 positive subspace (cycle 174 finding).

# Context
- The original rescue note (2026-04-02) said "directions come from W_down columns, not activations". This is the experiment that finally tests it.
- Cycle 174: positive-regime token-dependent signal at block 2 is rank-1.
- Cycle 216: anti-podal hypothesis on W_down columns refuted.
- This experiment partitions columns by sign frequency, not anti-podal cosine. New angle.

# Dependencies
None.

# Key Files
- HF gpt2 weights.
- Calib data: 200 wikitext-2 val × 128, seed 42.

# Constraints
- For each block l ∈ {0..11}:
  - Single forward pass with c_fc input hook → preact z(x) shape (B, T, 3072).
  - σ_c = (preact > 0).float().mean(dim=(0,1)) shape (3072,) — per-channel sign frequency.
  - Split channels into 3 groups: σ ∈ [0, 0.1] (mostly_neg), [0.45, 0.55] (balanced), [0.9, 1.0] (mostly_pos).
  - W_down = c_proj.weight (3072, 768). Each row is a column of W_down.
  - For each group: extract those rows, normalize, compute PCA spectrum (SVD), report top-20 singular values, rank_95.
  - Block 2 special: rank of `{w_c : σ_c ≥ 0.9}` subspace; report top-1 SV energy fraction.
  - Principal angles between mostly_pos and mostly_neg subspaces (top-20 SVs each → svdvals(V_pos.T @ V_neg)).
  - Null distribution: shuffle σ_c → channel labels 100 times, compute principal-angle distribution under random partitions.
- Confirmed if block 2 mostly_pos top-1 SV ≥ 0.7 AND median principal angle (pos vs neg) decreasing monotonically from b1 to b11 by ≥ 30° AND each angle outside random null ±2σ.
- Refuted if all blocks within ±2σ of null.
- Inconclusive otherwise.
- Trainable params: 0.

# Handoff
Plan agent: design carefully. The principal angles need to be in degrees. New script `experiments/wdown_sign_geometry_cycle261.py`. Output JSON + summary with the per-block stats.
