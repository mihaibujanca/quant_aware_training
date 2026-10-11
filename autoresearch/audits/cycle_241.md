# Cycle 241 Audit — Item 251

**Status: PASS**

**Item 251 verbatim:** Confirmed if best_mean ≤ 58.0; Refuted if best_mean ∈ [59.0, 60.5]; else Inconclusive.

**Claimed:** REFUTED, best_mean=59.533 ± 0.104. Per-seed 59.385 / 59.611 / 59.602.

## Phase 1 — verdict check
59.533 ∈ [59.0, 60.5] → REFUTED. Threshold gate applied correctly in script
(lines 274–279). Verdict unambiguous and lands within 1 std of cycle 221's
59.515 ± 0.094 (Δ = +0.018 PPL).

## Phase 2 — methodology check
- BLOCKS=[1,2,4,5,7,8], K=512, LR=1e-4, 4000 steps, EVAL_EVERY=100, 3 seeds
  42/137/2024, disjoint calib=all_ids[:200] / eval=all_ids[200:300], SEQ_LEN=128.
- Trainable surface = W_out + b_out + FULL c_fc bias (3072,) per block; W_fc
  matrix remains a register_buffer (frozen). Per-block count
  512·768 + 768 + 3072 = 397,056; × 6 = **2,382,336** (asserted, line 174).
- Forward indexes `b_fc_full[topk_idx]` so only K=512 bias entries reach
  the output; the 2560 unselected entries receive zero gradient but count
  toward the trainable budget as specified by the agenda's 2.38M envelope.
- Init PPL = 86.686 with zero seed variance — matches cycle 221 exactly
  because the forward is identical at init (same topk indices, same
  b_fc[topk] = cycle 221's post-topk buffer).
- Per-seed best_ppl 59.385 / 59.611 / 59.602 at steps 1300 / 1200 / 1600
  (seed-for-seed identical best steps to cycle 221).
- Best-state restore round-trip clean for all 3 seeds
  (|restored − best| < 1e-3, assertion line 234).
- Final@4000 mean 71.948 → strong overfit phase past step 1500, same
  trajectory shape as cycle 221.

## Phase 3 — interpretation
Adding the full pre-topk c_fc bias on top of W_out-only joint Adam is a
perfect noop at 6 blocks: delta is +0.018 PPL (≪ 1 σ). This mirrors
cycle 213 (3-block W_fc bias 46.424 vs W_out-only 46.556, Δ = −0.13):
the c_fc bias direction is closed at both 3 and 6 blocks. Combined with
cycle 240 (K=1 6-block 96.886, REFUTED) the 6-block 59.5 floor is rigid
to trainable-parameter-count expansion via c_fc bias, but collapses under
channel capacity reduction — the binding constraint at 6 blocks is K,
not parameter count. The W_fc matrix itself is untested (would add
2.36M more params and violates the agenda's 2.38M budget).
