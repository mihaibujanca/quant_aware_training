# Item
242 (P0): Sequential init + joint polish hybrid at 3 blocks — break the 46.5 floor.
Run cycle 215 sequential to convergence (2000 steps per block, best-state restore), then initialize a JOINT Adam pass on all 3 W_out+b_out for 1000 more steps with LR=5e-5.

# Context
- Cycle 215 (item 225, INCONCLUSIVE): sequential per-block Adam at 3 blocks → 47.253 ± 0.047. Cycle 215 audit recommended hybrid: "sequential + lightweight joint polish".
- Cycle 206 (item 217, CONFIRMED): joint Adam at 3 blocks 1000 steps → 48.111. Cycle 210 step curve: 46.556 best at 4000 steps.
- Open question: does sequential warm-start give joint Adam a better basin?

# Dependencies
- 217 / cycle 206 (done) — joint baseline.
- 225 / cycle 215 (done) — sequential.

# Key Files
- experiments/adam_sequential_cycle215.py — sequential template.
- experiments/adam_wout_correction_cycle206.py — joint training reference.

# Constraints
- BLOCKS=[1,4,7], K=512.
- Phase 1: sequential per-block Adam (block_idx ∈ [1,4,7]) — 2000 steps per block, EVAL_EVERY=100, best-state restore per block. Same as cycle 215.
- Phase 2: after sequential, set all 3 blocks' W_out+b_out trainable simultaneously, fresh Adam with LR=5e-5 (lower to avoid wiping the sequential init), 1000 joint steps with EVAL_EVERY=100, best-state restore.
- Final eval after restore.
- 3 seeds: 42, 137, 2024.
- Confirmed if hybrid best_mean ≤ 45.8 (breaks joint 46.556 by ≥0.75).
- Refuted if best_mean ≥ 46.5.
- Inconclusive if 45.8 < best_mean < 46.5.

# Handoff
Plan agent: design via combining cycle 215 (phase 1) and cycle 206 (phase 2). New script `experiments/sequential_polish_hybrid_cycle232.py`. Output JSON + summary, comparing against cycle 206 (46.556), cycle 210 best (46.556), cycle 213 W_out+W_fc (46.424), cycle 215 sequential (47.253).
