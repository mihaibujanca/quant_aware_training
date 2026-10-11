# Cycle 224 Design Review — Item 229

**Item**: 229 (P2) — Distillation baseline for 3-block compression — matched budget
**Verdict**: APPROVE

## Summary

Plan trains W_out + W_fc + biases of compressed blocks [1,4,7] (K=512) using a
mixed loss (0.5 KD + 0.5 CE), KL divergence with τ=2 against the float GPT-2
teacher. 10000 steps × 3 seeds, eval every 500, best-state snapshot/restore.
Compares against cycle 213 W_out+W_fc CE-only baseline of 46.424.

## Pitfalls Check (MP1–MP8)

- **MP1** (threshold above trivial baseline): N/A — threshold (44.0) is *below*
  the CE baseline (46.42), not above.
- **MP2** (activation regime mismatch): N/A — KL on softmax logits is in the
  right regime.
- **MP3** (underdetermined fit): trainable params ≈ 3 blocks × (W_fc + W_out +
  biases) ≈ 3 × (768·512 + 512·768 + 512 + 768) ≈ 2.4M. Calib set is
  200 × 128 = 25600 tokens × 50257 logit targets via KD soft labels — far
  exceeds parameter count. OK.
- **MP4** (domain mismatch): same task as cycle 206/210/213, direct comparison.
- **MP5** (insufficient capacity): cycle 213 already shows W_out+W_fc trains
  to 46.4 — capacity exists.
- **MP6** (verdict misclassification): metric (PPL) varies meaningfully in the
  44–47 range; INCONCLUSIVE band is explicit.
- **MP7** (post-LayerNorm): N/A — not measuring geometry.
- **MP8** (rank vs dimension): N/A — not measuring rank.

## Concerns

1. **Wall time**: 10000 steps × 3 seeds × dual forward (teacher + student) on
   MPS is tight. Plan has documented 5000-step fallback (cycle 213's best was
   at step 1500, so 5000 is ample headroom). Acceptable.
2. **Threshold 44.0 is aggressive**: the CE ceiling at the same parameter
   budget is 46.42 (cycle 213). Asking KD to break it by ≥2.4 PPL on a near-
   saturated optimum is a strong demand. Soft targets sometimes outperform hard
   CE, but rarely by 2+ PPL on a 200-sample calib set. Most likely outcome:
   INCONCLUSIVE. Plan acknowledges this in the Fallbacks section. Acceptable —
   this is exactly the question being asked.
3. **Mixed 0.5/0.5 KD+CE vs pure KD**: pure KD would be a cleaner test of
   "soft targets help"; the mixed version is reasonable as it hedges against
   KD divergence. Plan has a pure-KD fallback if mixed diverges. Acceptable.
4. **Best-state restore round-trip check**: explicit (`abs(restored - best) <
   1e-4`). Good guard against state-dict bugs.

## Comparison Set

- cycle 206 (W_out only, CE, 1000 steps): 48.111
- cycle 210 (W_out only, CE, 4000 steps): 46.556
- cycle 213 (W_out + W_fc, CE, ?): 46.424
- cycle 224 (W_out + W_fc, KD+CE, 10000): TBD

The cycle 213→224 delta isolates the KD contribution at matched param budget.
Step count differs (cycle 213 best at 1500 vs cycle 224 budget 10000), but
plan tracks `best_step` per seed so saturation is observable.

## Decision

APPROVE. The experiment is well-specified, has appropriate fallbacks, the
verdict bands are calibrated to the existing CE ceiling, and the most likely
outcome (INCONCLUSIVE in 44–47) is itself an informative result. Proceed.
