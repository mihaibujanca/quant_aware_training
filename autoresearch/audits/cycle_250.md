# Cycle 250 audit — item 260

**Status: PASS** — claimed verdict REFUTED stands.

## Item 260 criteria (verbatim)
- Confirmed if `best_mean ≤ 55.0`
- Refuted   if `best_mean ≥ 58.0`
- Inconclusive if 55-58

## Phase 1 — verdict arithmetic
JSON aggregate `best_mean = 60.02321965` (reported 60.023 ± 0.020).
60.023 ≥ 58.0 → **REFUTED** ✓.

Deltas cross-check against sources:
- Δ vs cycle 221 (59.515): +0.5082 ✓ (matches exec_result +0.508)
- Δ vs cycle 245 (53.826): +6.1972 ✓ (matches +6.197)
- Δ vs float (53.895):     +6.1283 ✓ (matches +6.128)

Per-seed bests [60.0492, 60.0199, 60.0006] at steps [1500, 1600, 1600] — seeds identical in agenda, exec_result, and JSON. Init PPL variance 8e-6 (lstsq deterministic up to MPS float noise).

## Phase 2 — result quality
- 3 seeds; best_std = 0.020 (tightest seed band at this scale; cycle 221 had 0.094, cycle 245 had 0.032).
- Same best-step cluster (1500-1600) as cycle 221 — same overfit-then-degrade trajectory. Final@4000 = 70.965 (10.94 PPL above best). Consistent with prior 6-block training dynamics.
- Best-state restore round-trip clean (|restored-best|<1e-3 asserted in code and passed for all seeds).
- Trainable params 2,363,904 match 6×(512×768 + 768).
- Code review: K=512 topk via `act.std(0) * W_proj.norm(dim=1)`, joint lstsq with bias column, W_out/b_out unfrozen on BLOCKS only, rest of model frozen — clean.

## Phase 3 — reinterpretation (IMPORTANT)
**Cycle 245's "block 5 is special" narrative is contradicted by cycle 250.**
- Cycle 245 [1,2,4,7,8] → 53.826
- Cycle 250 [1,2,3,4,7,8] → 60.023 (block 5 dropped, block 3 substituted)
- Cycle 221 [1,2,4,5,7,8] → 59.515

Substituting block 3 (cycle 244's "cheap" block in the [1,2,3] window) for block 5 moves the cascade by only +0.5 PPL — the 6-block cascade cost is ~invariant to which specific block from the early-mid range fills the sixth slot. Cycle 245's 5.7 PPL gain was a **count transition (6→5)**, not a block-5 removal benefit. Cycles 244 and 250 together suggest each compressed block in the early-mid range adds ~5-6 PPL of cascade penalty regardless of identity, bounding K=512 6-block compression to a ~59-60 PPL band. Cycle 243's per-block attribution under fixed joint fit does not predict cascade tolerance under rejoint fit. Item 262 (block-5 geometric deep-dive) is now LESS urgent; the open question shifts to characterising the cascade-count constant.

**Interesting: true** — this overturns the prior cycle's interpretation and reframes the selection-vs-count question.
