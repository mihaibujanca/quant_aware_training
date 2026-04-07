# Item
283 (P1): Heterogeneous K guided by cycle 267 eff_rank — block 11 at K=22, blocks 1/4/7 at K=512.
Test BLOCKS=[1,4,7,11] with K_PER_BLOCK={1:512, 4:512, 7:512, 11:22}. Plus control uniform K=22.

# Context
- Cycle 267: block 11 c_proj output eff_rank = 22.49.
- Cycle 270: block 11 rank-22 retrain → 54.469 (single-block patch).
- Cycle 266: 2-block [1,4] K=512 → 43.85 (best Pareto so far).
- Cycle 247: hetero K (cycle 243 ranking) at 6 blocks was 64.45 (no benefit).

# Dependencies
- Cycle 267 SVD result.
- Cycle 247 hetero K infrastructure.

# Key Files
- experiments/hetero_k_6block_cycle247.py — heterogeneous K base.
- experiments/four_block_sweep_cycle255.py — 4-block joint Adam.

# Constraints
- BLOCKS=[1, 4, 7, 11], K_PER_BLOCK={1:512, 4:512, 7:512, 11:22}
- W_out+b_out trainable, 1.2M params total.
- Adam LR=1e-4, 4000 steps, 3 seeds, best-state restore. Disjoint splits.
- Control: BLOCKS=[1,4,7,11] uniform K=22, 3 seeds.
- 6 fits total (3 hetero + 3 control).
- Confirmed if hetero best_mean ≤ 44.5 AND control ≥ 60.
- Refuted if hetero ≥ 47.
- Inconclusive otherwise.

# Handoff
Plan agent: design via extending cycle 247's K_PER_BLOCK to support per-block K. New script `experiments/hetero_k_eff_rank_cycle273.py`.
