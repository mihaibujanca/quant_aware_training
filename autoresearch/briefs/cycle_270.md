# Item
280 (P0): Targeted block 11 low-rank c_proj compression — does the rank-22 SVD subspace pay for itself?
For r ∈ {5, 22, 50, 100}: replace c_proj_11.weight with rank-r SVD approximation, measure PPL. For r=22: also retrain W_out (the 22×768 parameter) via Adam.

# Context
- Cycle 267 (CONFIRMED): block 11 c_proj output eff_rank = 22.49.
- Cycle 173: block 11 single-block PPL +4.02%.
- Cycle 264: block 11 is 11/12 in κ disruption (output-channel structure, not trajectory).

# Dependencies
None.

# Key Files
- experiments/block11_cproj_decomp_cycle267.py — has the SVD computation pattern.
- experiments/adam_wout_correction_cycle206.py — for the retrain harness.

# Constraints
- Load HF gpt2.
- Compute SVD of c_proj_11.weight (3072, 768 in HF Conv1D layout — but the W_out application is x @ weight + bias; weight is "in_features × out_features").
- For each r ∈ {5, 22, 50, 100}:
  - Replace c_proj_11.weight with U[:, :r] @ diag(Σ[:r]) @ Vh[:r, :] (the rank-r approximation).
  - bias unchanged.
  - Eval PPL on eval set (`all_ids[200:300]`, SEQ_LEN=128).
- For r=22 specifically: rerun with Adam W_out retrain.
  - The trainable W_out becomes the rank-r factor structure: e.g. parameterize c_proj_11.weight = A @ B where A is (3072, 22), B is (22, 768). 22*3072 + 22*768 = 67584 + 16896 = 84,480 trainable. Or simpler: keep c_proj_11.weight frozen at rank-22 SVD, train ONLY W_out_block11 (768) — wait, that doesn't make sense, c_proj_11.weight IS the W_out for that block.
  - Brief says "rank-22 W_out treated as the trainable parameter (22 × 768 + 768 = 17,664-param trainable)". So: parameterize c_proj_11 as P = U_22 @ S_22 @ B where B is the trainable (22, 768). Initialize B = Vh[:22, :]. Train.
- 3 seeds for the retrain.
- N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore. Disjoint splits.
- Confirmed if r=22 no-retrain ≤ 55 OR r=22 retrain ≤ 54.5.
- Refuted if r=22 no-retrain ≥ 60 AND retrain ≥ 58.
- Inconclusive otherwise.

# Handoff
Plan agent: design the script. Note that c_proj_11 in HF has weight shape (3072, 768) — Conv1D applies as input @ weight + bias. The SVD of that is U (3072, k) Σ (k,) Vh (k, 768). Rank-r approx: U[:,:r] @ diag(Σ[:r]) @ Vh[:r, :] → still shape (3072, 768).
For retraining, the trainable param is the 22-dim W_out factor — fix U_22 @ Σ_22 (one (3072, 22) matrix) and train Vh[:22, :] (22, 768). Initialize from SVD.
New script `experiments/block11_lowrank_cycle270.py`. Output JSON + summary.
