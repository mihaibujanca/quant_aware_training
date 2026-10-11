# Item
263 (P2): gelu_noneg + μ targeted to compressed blocks only.
Patch ONLY blocks [1,4,7] with relu(gelu(z))+μ_l, leave the other 9 blocks on raw GELU. Then K=512 topk+lstsq+Adam on those 3 blocks against the patched targets.

# Context
- Cycle 210: K=512 [1,4,7] raw-GELU → 46.556.
- Cycle 246 (REFUTED): all-12 patched + K=128 [1,4,7] → 57.811. The all-12 patch hurt the K=128 lstsq init.
- Cycle 246 audit: try the inverse — patch ONLY the compressed blocks.
- Hypothesis: targeted patch sidesteps the "all-12 hurts raw blocks" failure.

# Dependencies
- Cycle 246's GeluNonegCorrected class.

# Key Files
- experiments/gelu_noneg_lora_k128_cycle246.py — has GeluNonegCorrected and the patching pattern.
- experiments/adam_wout_correction_cycle206.py — K=512 topk+lstsq+Adam template for [1,4,7].

# Constraints
- BLOCKS=[1,4,7], K=512 (NOT K=128).
- Patch ONLY blocks 1, 4, 7 with GeluNonegCorrected(μ_l). μ_l precomputed on calib for blocks {1,4,7} only.
- Leave the OTHER 9 blocks on raw GELU.
- After patching, run K=512 topk+lstsq init on [1,4,7] (against the patched activations).
- Joint Adam W_out+b_out, LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore.
- 3 seeds, disjoint calib/eval.
- Confirmed if best_mean ≤ 46.0.
- Refuted if best_mean ≥ 47.0.
- Inconclusive if 46-47.

# Handoff
Plan agent: design the experiment. New script `experiments/gelu_noneg_targeted_cycle253.py`. Output JSON + summary.
