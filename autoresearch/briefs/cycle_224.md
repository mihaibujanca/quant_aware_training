# Item
229 (P2): Distillation baseline for 3-block compression — matched budget.
KD from float GPT-2 to compressed GPT-2 with BLOCKS=[1,4,7] K=512. Train W_out + W_fc for 10k steps with KD loss.

# Context
- Cycle 206 (CONFIRMED): Adam W_out only → 48.111 PPL after 1000 steps.
- Cycle 210 (best 46.556): K=512 W_out only at 4000 steps.
- Cycle 213 (INCONCLUSIVE): W_out + W_fc Adam (CE loss) → 46.424 (≈ W_out-only).
- Cycle 223 (CONFIRMED): K=128/256 also clear float with Adam W_out alone.
- This item: KD loss instead of CE — does soft-target supervision give a meaningful edge over CE?

# Dependencies
None. Standalone.

# Key Files
- experiments/adam_full_correction_cycle213.py — has W_out + W_fc training pattern.
- experiments/adam_wout_correction_cycle206.py — has the BLOCKS=[1,4,7] K=512 init.

# Constraints
- BLOCKS=[1,4,7], K=512.
- Train W_out + W_fc.
- KD loss: KL divergence between student logits (compressed model) and teacher logits (float GPT-2). Use temperature τ (e.g., τ=2). Add small CE term as a regularizer (weight 0.5 KD + 0.5 CE) or pure KD — plan should specify.
- 10k steps (10× cycle 206/213's step counts). EVAL_EVERY=500. Best-state snapshot/restore.
- 3 seeds: 42, 137, 2024.
- Use SAME N_CALIB=200, SEQ_LEN=128 calibration set; eval on the standard eval set.
- Confirmed if KD ≤ 44.0 (meaningfully better than 46.556).
- Refuted if KD ≥ 47.0.
- Inconclusive if 44-47.

# Handoff
Plan agent: design the KD loss carefully. Forward pass:
- teacher_logits = float_model(input_ids) (no_grad)
- student_logits = compressed_model(input_ids)
- kd_loss = KL(softmax(student_logits/τ), softmax(teacher_logits/τ)) * τ²
- optionally + λ * CE(student_logits, input_ids[1:])
Use 10000 steps with eval every 500. Track best per-seed.
Compare best PPL mean to thresholds 44.0/47.0.
Compare against cycle 213 W_out+W_fc best 46.424 (CE loss, 1500 steps).

Wall time: 10000 steps × 3 seeds × forward through TWO models (float teacher + compressed student) — 30-60s per step amortized — could be tight. Plan should include the timing budget and a fallback (5000 steps if needed).
