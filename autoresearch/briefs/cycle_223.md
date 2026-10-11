# Item
228 (P1): Lower K with Adam correction — compression efficiency curve.
Test K=256 (25%) and K=128 (12.5%) with Adam W_out correction for BLOCKS=[1,4,7]. Compare to K=512 (50%) Adam baseline.

# Context
- Cycle 206 (item 217, CONFIRMED): K=512 Adam W_out correction → 48.111 PPL after 1000 steps; cycle 210 best 46.556 at step 4000.
- Cycle 215: sequential per-block Adam K=512 → 47.253 ± 0.047.
- All Adam experiments used K=512 = 50% of full FFN width 1024 wait, K=512 is 1/6 of 3072 (16.7%) — actually K=3072 is the full intermediate dim. K=512 ≈ 16.7% of intermediate.
- Wait — let me check. cycle 206 says K=512 and "1.18M trainable params (W_out/b_out of 3 compressed MLPs)". W_out shape is (K, 768). K=512: 512×768 + 768 = 393984+768 = 394752 per block × 3 = 1.18M. ✓
- K=256 → 197376 params per block, 592K total.
- K=128 → 98688 params per block, 296K total.
- Confirmed if K=256 + Adam achieves PPL ≤52.0 (+3.5% vs float 53.895).
- Refuted if K=256 + Adam PPL >58.0.
- Inconclusive if 52-58.

# Dependencies
None.

# Key Files
- experiments/adam_wout_correction_cycle206.py — K=512 Adam baseline. Reuse loader, init, training loop.
- experiments/adam_step_curve_cycle210.py — eval-every-100-steps + best-state machinery.

# Constraints
- BLOCKS=[1,4,7], same as canonical baseline.
- Test K ∈ {128, 256}. Optionally include K=512 for comparison (already known: best 46.556).
- Same N_CALIB=200, SEQ_LEN=128, LR (1e-4 likely), N_STEPS=4000 (with EVAL_EVERY=100 + best-state restore).
- 3 seeds: 42, 137, 2024.
- Verdict criterion is K=256-specific.

# Handoff
Plan agent: design via reusing cycle 206/210 patterns. New script `experiments/adam_lower_k_cycle223.py`. The script must:
1. For K in {128, 256} (and optionally 512 for sanity reproduction):
   - For each seed:
     - Build topk+lstsq compressed model with that K.
     - Compute init PPL.
     - Adam W_out + b_out training, 4000 steps, EVAL_EVERY=100, best-state snapshot/restore.
     - Record best_ppl, best_step, init_ppl, final_ppl_at_4000, history.
   - Aggregate.
2. Output JSON with all K values and verdict against the K=256 criterion.

Tight scope. ≤30 min wall time.
