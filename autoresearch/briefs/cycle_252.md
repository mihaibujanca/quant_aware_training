# Item
262 (P1): Block 5 geometric deep-dive — what makes block 5 unique?
For each block in {1,2,3,4,5,6,7,8}, compute per-channel std/kurtosis distributions, effective rank, topk coverage, c_proj row cosine matrix, and Jacobian linearization residual on calibration data.

# Context
- Cycle 245 (CONFIRMED): dropping block 5 → 53.826 (closes the 6-block gap).
- Cycle 250 (REFUTED): dropping block 5 + adding block 3 → 60.023 (~same as cycle 221). **Cycle 245's success was a COUNT transition (6→5), NOT block-5-specific**.
- Cycle 250 audit explicitly downgraded item 262's urgency.
- However, the analysis is still cheap and may reveal whether block 5 has any geometric outlier signal. Even if not, the result will close this direction cleanly.

# Dependencies
None.

# Key Files
- experiments/adam_wout_correction_cycle206.py — for the data loader.
- experiments/jacobian_rank_census_cycle195.py if it exists.

# Constraints
- Calib: all_ids[:200], SEQ_LEN=128.
- For each block in {1, 2, 3, 4, 5, 6, 7, 8}:
  - Extract c_fc activation (post-GELU pre-c_proj input), shape (n_tokens, 3072).
  - (a) Per-channel std distribution
  - (b) Per-channel kurtosis distribution
  - (c) Effective rank (SVD of (n_tokens, 3072), entropy of normalized squared singular values)
  - (d) topk-128 and topk-512 coverage (sum of top-K std-weighted norms / total)
  - (e) c_proj row cosine similarity matrix (3072 rows in R^768)
  - (f) Jacobian linearization residual (skip if hard; this is the most expensive)
- Run cap ~15 min.
- Confirmed if block 5 is ≥1.5σ outlier on ≥2 metrics in the harder direction.
- Refuted if block 5 is within 1σ of {1,2,4,7,8} on ALL 6 metrics.
- Inconclusive if outlier on exactly 1 metric.
- After cycle 250, the prior is that block 5 is NOT special — the test is mostly to confirm this.

# Handoff
Plan agent: design via reusing data loader. New script `experiments/block5_geometric_cycle252.py`. Skip metric (f) Jacobian residual if it would blow the time budget. Focus on (a)-(e). Output JSON + summary markdown.
