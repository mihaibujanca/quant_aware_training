# Item
235 (P0): K=64/32/16 Adam compression cliff sweep.
Push the cycle 223 efficiency curve until it breaks. BLOCKS=[1,4,7], K ∈ {64, 32, 16}, Adam W_out+b_out, 3 seeds, 4000 steps, eval every 100, best-state restore. Same calib/eval split as cycle 223.

# Context
- Cycle 223 (CONFIRMED): K=128 → 48.989 ± 0.033, K=256 → 47.509 ± 0.023, K=512 → 46.556. Curve is monotone but very flat — 4× compression of W_out (512→128) costs only +2.43 PPL. ALL three K beat float (53.895).
- Channel capacity is NOT the bottleneck for low-K compression (cycle 223 audit, interesting=true).
- This item finds where the cliff actually is.

# Dependencies
None.

# Key Files
- experiments/adam_lower_k_cycle223.py — direct template. Just change K_VALUES.
- experiments/adam_wout_correction_cycle206.py — base init.

# Constraints
- BLOCKS=[1,4,7]
- K_VALUES=[64, 32, 16]
- Same N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100
- 3 seeds (42, 137, 2024)
- Best-state snapshot+restore per seed
- Confirmed if: at least one K ≤ 64 still beats float (best_mean ≤ 53.0)
- Refuted if: all three K give best_mean ≥ 60.0
- Inconclusive if: best_mean lies in 53-60 PPL for the smallest viable K (i.e., the cliff sits inside this range)

# Handoff
Plan agent: design via reusing cycle 223 script. New script `experiments/adam_lower_k_cliff_cycle225.py`. Output JSON + summary. Should be a quick variant of cycle 223. Verdict against the cliff criteria.
