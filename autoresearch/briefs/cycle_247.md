# Item
257 (P1): Heterogeneous K — K=512 on binding blocks 4,5; K=128 on 1,2,7,8.
Test whether the K=128 penalty (cycle 231: 67.6) concentrates on cycle 243's mid-block contributors.

# Context
- Cycle 221: K=512 6-block → 59.515 (quality upper bound).
- Cycle 231: K=128 6-block → 67.562 (cost lower bound).
- Cycle 243: blocks 4 and 5 are biggest per-block contributors (+4.90, +5.10) in the joint cascade.
- Cycle 244 (just-completed): in fresh-fit, the cost tracks the deepest block, NOT mid blocks.
- Cycle 245 (just-completed): dropping block 5 entirely closes the entire 6-block gap (best 53.826).
- This cycle: keep block 5 in the compression set but at K=512 (more capacity); drop K=128 only on the easier blocks.

# Dependencies
None.

# Key Files
- experiments/adam_6block_earlystop_cycle221.py — base.
- experiments/adam_6block_k128_cycle231.py — K=128 6-block.

# Constraints
- BLOCKS=[1,2,4,5,7,8]
- Per-block K: {1: 128, 2: 128, 4: 512, 5: 512, 7: 128, 8: 128}
- W_out+b_out trainable, W_fc frozen.
- Adam LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore. 3 seeds.
- Disjoint calib/eval (`all_ids[:200]`/`[200:300]`, SEQ_LEN=128).
- Trainable: 4×(128×768 + 768) + 2×(512×768 + 768) = 4×98,432 + 2×393,984 = 393,728 + 787,968 = 1,181,696 ≈ 1.18M.
- Confirmed if best_mean ≤ 61.0.
- Refuted if best_mean ≥ 65.0.
- Inconclusive if 61-65.

# Handoff
Plan agent: design via combining cycles 221+231 with per-block K. New script `experiments/hetero_k_6block_cycle247.py`. Output JSON + summary.
