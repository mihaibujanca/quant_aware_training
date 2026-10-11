# Cycle 240 Audit — Item 250

**Status: PASS**

**Item 250 verbatim:** Confirmed if best_mean ∈ [58, 62]; refuted if ≥70; else inconclusive.

**Claimed:** REFUTED, best_mean=96.886 ± 0.071 (≥70). Init=203.246.

## Phase 1 — verdict check
96.886 ≥ 70 → REFUTED by a margin of 26.9 PPL. Threshold applied correctly in script
(lines 236–241). Verdict unambiguous.

## Phase 2 — methodology check
- Code is a direct K-swap of cycle 221: BLOCKS=[1,2,4,5,7,8], joint Adam W_out+b_out,
  LR=1e-4, 4000 steps, EVAL_EVERY=100, 3 seeds (42/137/2024), disjoint calib/eval
  splits (first 200 / next 100), SEQ_LEN=128.
- Trainable params asserted == 9216 (6 × (768 + 768)); assertion lives at line 147.
- Per-seed best_ppl: 96.943 / 96.786 / 96.929 — std 0.071 is tiny, result is seed-robust.
- Restore round-trip clean for all 3 seeds (|restored − best| < 1e-3), assertion line 195.
- init_std=0 confirms lstsq fit is deterministic (as expected — seed only affects Adam).
- All 3 seeds hit best at step 4000 (still descending), so the 96.9 number is a
  budget-bounded upper floor, not a converged minimum — but it is already ~35 PPL
  above the REFUTED threshold so more steps cannot flip the verdict.

## Phase 3 — interpretation
Init PPL at K=1 6-block is 203.246 vs K=512 6-block ~86 (cycle 221). Channel capacity
binds hard when starting from topk+lstsq at 6 blocks: a single c_fc channel per block
cannot span the 768-dim residual subspace the MLP normally writes, and 9216 W_out+b_out
params cannot recover the missing rank across 6 stacked blocks. Cycle 230's K=1 3-block
result (59.245) does not extrapolate — at 3 blocks the uncorrected blocks provide
mid-cascade slack, at 6 blocks the K=1 projection compounds multiplicatively.
