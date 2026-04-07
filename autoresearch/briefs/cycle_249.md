# Item
259 (P0): Heterogeneous K complement — K=512 on {7,8}, K=128 on {1,2,4,5}.
Tests cycle 247's residual-MSE-driven hypothesis: blocks 7,8 had HIGHER per-block MSE under K=128 than blocks 4,5. Upgrade THOSE blocks to K=512 instead.

# Context
- Cycle 221: K=512 6-block → 59.515.
- Cycle 231: K=128 6-block → 67.562.
- Cycle 247 (INCONCLUSIVE): K=512 on {4,5}+K=128 elsewhere → 64.453, only 38.6% gap recovery. Per-block MSE seed-42: blocks 7=0.581, 8=0.818 vs blocks 4=0.162, 5=0.224.
- Hypothesis: blocks 7,8 are the actual K=128 binding constraint.

# Dependencies
None.

# Key Files
- experiments/hetero_k_6block_cycle247.py — direct template, swap K_PER_BLOCK.

# Constraints
- BLOCKS=[1,2,4,5,7,8]
- K_PER_BLOCK={1: 128, 2: 128, 4: 128, 5: 128, 7: 512, 8: 512}
- W_out+b_out trainable, W_fc frozen.
- Adam LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore.
- Disjoint calib/eval. 3 seeds.
- Trainable: 4×(128×768 + 768) + 2×(512×768 + 768) ≈ 1.18M (same as cycle 247).
- Confirmed if best_mean ≤ 61.0 (or significantly better than cycle 247's 64.453).
- Refuted if best_mean ≥ 65.0.
- Inconclusive if 61-65.

# Handoff
Plan agent: trivial K_PER_BLOCK swap on cycle 247 template. New script `experiments/hetero_k_complement_cycle249.py`. Output JSON + summary.
