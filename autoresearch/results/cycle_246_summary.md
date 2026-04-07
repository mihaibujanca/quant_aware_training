# Cycle 246 — gelu_noneg + μ + LoRA r=8 + K=128 W_out on [1,4,7] (item 256)

**Verdict: REFUTED**

Combines cycle 239's gelu_noneg + per-channel μ_l + LoRA r=8 on every block's
`mlp.c_proj` recipe (47.692 ± 0.143 held-out, all 12 blocks) with cycle 223's
K=128 topk+lstsq W_out channel compression on BLOCKS=[1,4,7] (48.989 alone).
Tests whether activation fix and channel compression compose (≤ 47.0) or trade
(≥ 49.5).

**Install order.** (1) μ precomputation on CALIB; (2) patch 12 × mlp.act with
`GeluNonegCorrected(μ_l)`; (3) collect (ln_out, mlp_out) on the patched model
and fit K=128 topk+lstsq against the patched target on [1,4,7], replacing
those mlp modules with `CompressedMLPWithLoRA` (built-in LoRA r=8 residual);
(4) wrap the 9 float blocks' `mlp.c_proj` in `LoraConv1D` (r=8);
(5) joint Adam LR=0.0001, N_STEPS=2000, EVAL_EVERY=200, best-state
restore. Trainable surface = LoRA A/B across all 12 blocks ∪ W_out + b_out on
the 3 compressed blocks.

| Variant | PPL | source |
|---|---|---|
| Float baseline (held-out) | 53.895 | this run |
| cycle 220 float ref | 45.064 | reference |
| μ-only CALIB | 61.664 | this run |
| μ-only EVAL (held-out) | 74.301 | this run |
| μ + K=128 [1,4,7] EVAL (LoRA=0 init) | 106.742 | this run |
| cycle 223 K=128 [1,4,7] W_out-only | 48.989 | no μ, no LoRA |
| cycle 239 gelu_noneg+μ+LoRA r=8 (all 12) | 47.692 | no K compression |
| **cycle 246 combined best_mean ± std** | **57.811 ± 0.277** | this run, n=3 seeds |
| cycle 246 combined final_mean ± std | 57.811 ± 0.277 | after full 2000 steps |

**Per-seed results (final_eval_ppl after best-state restore):**

| seed | init_eval_ppl | best_eval_ppl (step) | final_eval_ppl |
|---|---|---|---|
| seed 42 | 106.7418 | 57.8992 (step 2000) | 57.8992 |
| seed 137 | 106.7418 | 57.4368 (step 2000) | 57.4368 |
| seed 2024 | 106.7418 | 58.0970 (step 2000) | 58.0970 |

Thresholds: CONFIRMED ≤ 47.0, REFUTED ≥ 49.5.
Δ vs float baseline: +3.916 PPL (+7.27%).
Δ vs cycle 223 K=128 (48.989): +8.822 PPL.
Δ vs cycle 239 LoRA-only (47.692): +10.119 PPL.

**Trainable budget.** 595,200 params
 = 9 × (3072×8 + 8×768) [float-block LoRA]
 + 3 × (128×8 + 8×768) [compressed-block LoRA]
 + 3 × (768×128 + 768) [compressed W_out + b_out].

**Sanity.**
- `len(calib_ids) = 200`, `len(eval_ids) = 100`, disjoint by object id and tensor equality.
- μ ≤ 0 elementwise across all 12 blocks (max = 0.00e+00).
- LoRA-zero init eval PPL matches post-compress eval PPL (106.7418) within 1e-3 across all seeds.
- Fallback SEQ_LEN: none (SEQ_LEN=128).
