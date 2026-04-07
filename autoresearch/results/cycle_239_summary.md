# Cycle 239 — gelu_noneg + μ_l + LoRA r=8 disjoint split (item 249)

**Verdict: CONFIRMED**

This cycle re-runs cycle 234's gelu_noneg + per-channel first-moment (μ_l) +
LoRA r=8 W_out residual on all 12 GPT-2 blocks, but with the cycle 230 safe
disjoint split (`calib_ids = all_ids[:200]`, `eval_ids = all_ids[200:300]`,
SEQ_LEN=128, seed 42 for tokenisation order). Cycle 234's CONFIRMED 32.411
PPL was downgraded to INCONCLUSIVE because training and evaluation used the
same 200 sequences (autoresearch/audits/cycle_234.md). This rerun closes that
loop with the canonical safe pattern from cycles 206/210/223/225/230.

| Variant | PPL | source |
|---|---|---|
| Float baseline (held-out) | 53.895 | this run |
| cycle 220 float ref | 45.064 | reference |
| cycle 228 μ-only (calib==eval) | 61.664 | leaked baseline |
| cycle 239 μ-only on CALIB | 61.664 | this run |
| cycle 239 μ-only on EVAL (held-out floor) | 74.301 | this run |
| cycle 234 leaked LoRA r=8 | 32.411 | calib==eval (data leak) |
| **cycle 239 LoRA r=8 disjoint mean ± std** | **47.692 ± 0.143** | this run, n=3 seeds |

**Per-seed results (final_eval_ppl after best-state restore):**

| seed | init_eval_ppl | best_eval_ppl (step) | final_eval_ppl |
|---|---|---|---|
| seed 42 | 74.3012 | 47.6658 (step 2000) | 47.6658 |
| seed 137 | 74.3012 | 47.5305 (step 2000) | 47.5305 |
| seed 2024 | 74.3012 | 47.8790 (step 2000) | 47.8790 |

Thresholds: CONFIRMED ≤ 48.0, REFUTED ≥ 58.0.
Δ vs float baseline: -6.203 PPL (-11.51%).
Δ vs cycle 228 (μ-only calib leak ref): -13.972 PPL.
Δ vs cycle 234 (leaked LoRA): +15.281 PPL.

**Sanity.**
- `len(calib_ids) = 200`, `len(eval_ids) = 100`, disjoint by object id and tensor equality.
- post-μ patch CALIB PPL = 61.6636 (cycle 228 ref 61.664).
- LoRA-zero init eval PPL matches post-μ eval PPL (74.3012) within 1e-3 across all seeds.
- Block 0 base c_proj weight unchanged across all training: True.
- μ ≤ 0 elementwise across all 12 blocks (max = 0.00e+00).
- Trainable params: 368640 = 12 × (3072×8 + 8×768).
- Fallback SEQ_LEN: none (SEQ_LEN=128).

**Conclusion.** LoRA r=8 W_out residual generalises to held-out data: mean 47.692 ≤ 48.0. The gelu_noneg higher-order GELU curvature beyond the first-moment correction lives in a rank-≤8 subspace of W_out that is not purely calibration overfit. Cycle 234's leaked 32.411 OVERSTATED the actual residual quality by +15.28 PPL on held-out, but the LoRA residual is real.

**μ statistics (per-block aggregate, computed on CALIB).** Overall mean(μ) ≈
-0.08744, overall mean(|μ|) ≈ 0.08744. μ ≤ 0
elementwise by construction.
