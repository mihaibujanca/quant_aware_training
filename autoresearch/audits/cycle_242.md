# Cycle 242 audit — item 252

**Status: PASS. Verdict: CONFIRMED. Interesting: true.**

## Item 252 criterion
- CONFIRMED if either new config Δ ≥ +2.0 PPL vs c210 [1,4,7]=46.556
- REFUTED if both within ±0.8
- INCONCLUSIVE otherwise

## Phase 1 — verdict arithmetic
- A=[1,2,3]: mean 47.135 ± 0.093 → Δ +0.579 (within ±0.8 band)
- B=[8,9,10]: mean 55.192 ± 0.040 → Δ +8.636 (≥ 2.0)
- max(Δ_A, Δ_B) = +8.636 ≥ 2.0 → **CONFIRMED**. Arithmetic checks out.

## Phase 2 — quality
- 3 seeds (42/137/2024); per-seed PPL tight: A std 0.093, B std 0.040.
- Identical recipe across configs (K=512, joint Adam W_out+b_out, LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore, calib=all_ids[:200], eval=all_ids[200:300], SEQ_LEN=128, trainable=1,181,952). Only BLOCKS differs → fair contrast.
- Float held-out 53.895. A beats float by 6.76 PPL (consistent with cycles 206/210/223/230/239 early-block family). B lands +1.30 above float → late blocks resist correction.
- B's 3 seeds all peak at step 1400 then plateau-then-climb to ~62.25 by step 4000 (overfit signature seen in cycles 221/241). Convergence is genuine, not budget-bounded.

## Phase 3 — interpretation
At matched count=3, depth placement alone gives an 8.06 PPL gap (47.135 vs 55.192). Combined with c210 [1,4,7]=46.556 and c233 [3..8]=67.826, this rules out the cycle 241 hypothesis that block-count arithmetic drives cascade penalty. New null hypothesis (already in manifest next): cascade floor is set by late-block compression cost, not block count. Strong, well-controlled, structurally informative — interesting=true.
