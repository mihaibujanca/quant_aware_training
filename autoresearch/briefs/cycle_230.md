# Item
240 (P0): K ∈ {8, 4, 2, 1} Adam compression cliff sweep — find the absolute floor.
Extend the cycle 225 curve into the sub-16 regime. BLOCKS=[1,4,7], K ∈ {8, 4, 2, 1} Adam W_out+b_out, 4000 steps, 3 seeds, best-state restore.

# Context
- Curve so far: K=512 (46.556) → K=256 (47.509) → K=128 (48.989) → K=64 (50.781) → K=32 (53.038) → K=16 (55.133, first to cross float=53.895). Roughly +1-2 PPL per halving.
- This cycle pushes below K=16 to K∈{8,4,2,1}.

# Dependencies
- Item 235 / cycle 225 (done) — most recent template.

# Key Files
- experiments/adam_lower_k_cliff_cycle225.py — direct template, just change K_VALUES.

# Constraints
- BLOCKS=[1,4,7]
- K_VALUES=[8, 4, 2, 1]
- N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Same calib/eval split as cycles 206/223/225
- Trainable per block at K=1: 1×768+768=1536. K=2: 2304. K=4: 3840. K=8: 6912.
- Confirmed if: K=4 best_mean ≤ 59.28 (within +10% of float). Floor is K≤4.
- Refuted if: K=8 best_mean ≥ 61.98 (≥+15% of float). Cliff is steeper than predicted.
- Inconclusive if: K=8 lands between +2.3% and +15%.

# Handoff
Plan agent: trivial extension of cycle 225 script. Output JSON + summary with the extended K curve {512, 256, 128, 64, 32, 16, 8, 4, 2, 1}.
