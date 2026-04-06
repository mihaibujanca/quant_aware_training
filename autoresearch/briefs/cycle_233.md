# Item
243 (P1): Alternative 6-block selections — is BLOCKS=[1,2,4,5,7,8] representative?
Sweep 3 alternative 6-block subsets: (a) [1,2,3,4,5,6] contiguous early, (b) [3,4,5,6,7,8] contiguous middle, (c) [1,3,5,7,8,10] spread. K=512, joint Adam, 4000 steps. SEED=42 first; rerun winner with 3 seeds if it beats 59.0.

# Context
- Cycle 221 (item 226, INCONCLUSIVE): 6-block joint Adam at [1,2,4,5,7,8] → 59.515 ± 0.094.
- Cycle 229 (item 239, REFUTED): sequential at [1,2,4,5,7,8] → 60.071 ± 0.086.
- Cycle 231 (item 241, REFUTED): K=128 6-block at [1,2,4,5,7,8] → 67.562. K is not the lever.
- Cycle 232 (item 242, REFUTED): sequential+polish 3 blocks → 46.911. Composition is not the lever.
- Open question: is the 59.5 ceiling structural to "any 6 blocks" or specific to [1,2,4,5,7,8]?

# Dependencies
- 226 / cycle 221 (done).

# Key Files
- experiments/adam_6block_earlystop_cycle221.py — direct template, just swap BLOCKS.

# Constraints
- K=512, N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100.
- Phase 1: 1 seed (42) for each of 3 alternative subsets. Compare best PPL.
- Phase 2: if any beats 59.0, rerun with 3 seeds.
- Exclude blocks 0 (cycle 198, irreplaceable) and 11 (cycle 201/202, output decompressor).
- Confirmed if any subset best ≤ 58.0.
- Refuted if all 3 alternatives best ∈ [59.0, 62.0].
- Inconclusive if spread > 2 PPL with no clear winner.

# Handoff
Plan agent: design via cycle 221 template with 3 BLOCKS sweeps. New script `experiments/adam_6block_subset_sweep_cycle233.py`. Output JSON + summary.
