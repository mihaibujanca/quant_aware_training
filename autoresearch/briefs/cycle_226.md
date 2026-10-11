# Item
236 (P0): KD distillation baseline rerun with corrected loss normalization.
Cycle 224 was downgraded to INCONCLUSIVE because `F.kl_div(..., reduction='batchmean')` on `(B=1, T=128, V=50257)` divides only by B=1, returning the SUM over T·V ≈ 6.4M entries while CE averages over tokens. Effective ~10× KD weighting. Rerun with reshape (B*T, V) before kl_div (or reduction='mean'); add a short α-sweep at low budget; then long-budget run with the best α.

# Context
- Cycle 213 (CE pure, W_out+W_fc): best 46.424.
- Cycle 224 (broken KD+CE, 50/50): best 60.938 (barely moved off init 61.746).
- This cycle: same surface, fixed KD normalization.

# Dependencies
None.

# Key Files
- experiments/distillation_baseline_cycle224.py — broken version. Read to understand the bug.
- experiments/adam_full_correction_cycle213.py — pure CE baseline (W_out+W_fc).

# Constraints
- BLOCKS=[1,4,7], K=512, FullCompressedMLP (W_out+b_out+W_fc+b_fc trainable).
- KD loss: reshape student/teacher logits to (B*T, V), then `F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * τ²` divides by B*T (per-token).
- α-sweep at 2000 steps × 1 seed: α ∈ {0.0 (pure CE), 0.5, 1.0 (pure KD)}, find the best.
- Then full run: 4000 steps × 3 seeds at the best α.
- LR=1e-4, τ=2.0.
- Confirmed if best_mean ≤ 44.0 (better than cycle 213's 46.424).
- Refuted if best_mean ≥ 47.0.
- Inconclusive if 44 < best_mean < 47.

# Handoff
Plan agent: design exact code for the corrected KD loss + α-sweep + full run. New script `experiments/distillation_corrected_cycle226.py`. Output JSON + summary with the α-sweep table and the best-α full run results.
