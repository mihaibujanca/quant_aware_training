# Cycle 253 — gelu_noneg + μ targeted to compressed blocks ONLY (item 263)

**Verdict: INCONCLUSIVE**

Tests the complement of cycle 246 (which patched all 12 blocks with gelu_noneg
+ μ_l and failed on K=128 [1,4,7] compression, 57.811). This experiment patches
ONLY the 3 compressed blocks [1, 4, 7] with `GeluNonegCorrected(μ_l)`, leaves
the other 9 blocks on raw GELU, and runs the standard K=512 topk+lstsq + Adam
W_out+b_out recipe against the patched targets. Isolates whether the activation
surgery is a per-block target-side improvement or a whole-model change.

**Install order.** (1) μ precomputation on CALIB for blocks [1, 4, 7] ONLY
(3 entries in mu_dict, not 12); (2) patch ONLY those 3 mlp.act modules with
`GeluNonegCorrected(μ_l)`, leave the other 9 blocks on raw GELU; (3) collect
(ln_out, mlp_out) on the patched model and fit K=512 topk+lstsq against the
patched target on [1,4,7], replacing those mlp modules with `CompressedMLP`
(W_fc_sub, b_fc_sub, mu_sub, W_out, b_out); (4) freeze all; unfreeze only
W_out + b_out on the 3 compressed blocks; (5) joint Adam LR=0.0001,
N_STEPS=4000, EVAL_EVERY=100, best-state restore.

| Variant | PPL | source |
|---|---|---|
| Float baseline (held-out) | 53.895 | this run |
| cycle 220 float ref | 45.064 | reference |
| μ-only CALIB (targeted, 3 blocks) | 47.774 | this run |
| μ-only EVAL (targeted, 3 blocks) | 57.311 | this run |
| μ + K=512 [1,4,7] EVAL (lstsq init) | 63.732 | this run |
| cycle 210 K=512 [1,4,7] raw-GELU | 46.556 | direct baseline |
| cycle 246 all-12 patched K=128 | 57.811 | what NOT to do |
| cycle 239 gelu_noneg+μ+LoRA (all 12, no compression) | 47.692 | no K compression |
| **cycle 253 targeted best_mean ± std** | **46.510 ± 0.038** | this run, n=3 seeds |
| cycle 253 final_mean ± std | 46.510 ± 0.038 | after full 4000 steps |

**Per-seed results (final_eval_ppl after best-state restore):**

| seed | init_eval_ppl | best_eval_ppl (step) | final_eval_ppl |
|---|---|---|---|
| seed 42 | 63.7318 | 46.4718 (step 4000) | 46.4718 |
| seed 137 | 63.7318 | 46.4953 (step 4000) | 46.4953 |
| seed 2024 | 63.7318 | 46.5618 (step 3800) | 46.5618 |

Thresholds: CONFIRMED ≤ 46.0, REFUTED ≥ 47.0.
Δ vs float baseline: -7.385 PPL (-13.70%).
Δ vs cycle 210 K=512 raw (46.556): -0.046 PPL.
Δ vs cycle 246 all-12 (57.811): -11.301 PPL.
Δ vs cycle 239 LoRA (47.692): -1.182 PPL.

**Trainable budget.** 1,181,952 params = 3 × (768×512 + 768) [W_out + b_out on 3 blocks].

**Sanity.**
- `len(calib_ids) = 200`, `len(eval_ids) = 100`, disjoint by object id and tensor equality.
- `mu_dict` has exactly 3 entries (for blocks [1, 4, 7]), not 12.
- μ ≤ 0 elementwise across the 3 targeted blocks (max = -2.84e-11).
- Phase 2 sanity: compressed blocks [1, 4, 7] patched to `GeluNonegCorrected`; other 9 blocks still raw GELU.
- `init_eval_ppl == post_compress_eval_ppl` across all seeds (within 1e-3).
- `|final_eval_ppl - best_eval_ppl| < 1e-3` per seed (best-state restore round-trip).
- Fallback SEQ_LEN: none (SEQ_LEN=128).
