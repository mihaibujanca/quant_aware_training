# Cycle 231 Audit — Item 241 (P0)

**Claim:** REFUTED. K=128 6-block Adam best_mean=67.562 ± 0.061 (Δ=+8.047 vs cycle 221 K=512 ceiling 59.515).
**Audit verdict:** PASS

## Phase 1 — Threshold check
- Criteria: Confirmed ≤61.0, Refuted ≥65.0, else Inconclusive.
- best_mean = 67.562 ≥ 65.0 → REFUTED. Correct.

## Phase 2 — Quality
- Config (`experiments/adam_6block_k128_cycle231.py`): BLOCKS=[1,2,4,5,7,8], K=128, LR=1e-4, 4000 steps, EVAL_EVERY=100, seeds [42,137,2024]. Mirror of cycle 221 with only K changed (512→128). Float PPL 53.895.
- Per-seed best: 67.630 / 67.483 / 67.574; std=0.061; best_step_mean=3833 ± 170 (all three seeds bottom out near end of budget). Init PPL 115.056 deterministic across seeds (lstsq fit, no RNG). Restore-best sanity (`assert |restored−best|<1e-3`) passes; trajectories monotone; no collapses.
- Init PPL is +28.37 worse than cycle 221 K=512 init (86.686). Adam descent (~47.5 PPL) is large but the K=128 6-block surface bottoms out 8.05 PPL above the K=512 ceiling — well above the 65.0 refutation gate.

## Phase 3 — Cross-cycle context
- 3-block K=128 penalty (cycle 223 vs cycle 206 K=512): +2.43 PPL.
- 6-block K=128 penalty (this cycle vs cycle 221 K=512): +8.05 PPL.
- The K=128 penalty triples 3→6 blocks; cycle 230's "no compression cliff in [1,512] at 3 blocks" does not generalise. Triangulates with cycle 229 (sequential 6-block 60.071) the structural ~59.5 PPL floor for 6-block W_out+b_out correction.

## Issues
None. Code, criteria, and verdict all consistent.
