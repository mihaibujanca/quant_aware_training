# Item
226 (P0): Adam correction for 6 blocks with proper early stopping — confirm INCONCLUSIVE.
BLOCKS=[1,2,4,5,7,8] K=512. Eval every 100 steps with proper early stopping; settle the cycle 211 INCONCLUSIVE.

# Context
- Cycle 211 (item 221, INCONCLUSIVE): 6-block joint Adam W_out correction. Best PPL=59.680 at step 1500 (criterion ≤60.0 met) but rapid overfitting: step 2000=61.141, step 4000=72.250. Used fixed 4000-step budget without per-eval-step early stopping.
- Cycle 215 (item 225, INCONCLUSIVE): 3-block sequential per-block Adam → 47.253 ± 0.047, 0.7 PPL above joint baseline 46.556.
- Cycle 211 ran the 6-block trajectory but didn't formally early-stop. This cycle reruns with eval cadence and tracks the best step.

# Dependencies
None.

# Key Files
- experiments/adam_6block_cycle211.py — original 6-block run. Reuse loader, optimizer, eval pattern.
- experiments/adam_step_curve_cycle210.py — has the eval-every-100-steps machinery.
- experiments/adam_wout_correction_cycle206.py — 3-block joint baseline.

# Constraints
- BLOCKS=[1,2,4,5,7,8], K=512.
- N_CALIB=200, N_EVAL=100 (or whatever cycle 211 used — match), SEQ_LEN=128.
- EVAL_EVERY=100 steps.
- Total steps: 4000 (matching cycle 211 budget) — DO NOT cut short, we want to see full trajectory.
- Snapshot best (W_out, b_out) state across all 6 trainable blocks when eval PPL improves.
- 3 seeds: 42, 137, 2024.
- Restore best state at end; report best_ppl_mean ± std AND final_ppl (last step) for comparison.
- Confirmed if: best_ppl_mean ≤ 58.0.
- Refuted if: best_ppl_mean > 62.0.
- Inconclusive if: 58 < best_ppl_mean ≤ 62.

# Handoff
Plan agent: design via reusing cycle 211 + cycle 210 patterns. New script `experiments/adam_6block_earlystop_cycle221.py`. The script must:
1. Build the 6-block compressed model with topk+lstsq init.
2. Set up Adam on W_out/b_out of the 6 blocks (frozen W_fc).
3. Train for 4000 steps; eval every 100 steps; track best per-seed.
4. Aggregate across 3 seeds.
5. Output JSON + summary.
6. Compare against the cycle 211 best PPL=59.680.
