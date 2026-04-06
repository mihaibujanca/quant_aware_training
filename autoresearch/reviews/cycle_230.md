# Cycle 230 Design Review — Item 240

**Verdict: APPROVE**

Trivial extension of cycle 225 (which was itself a trivial extension of cycle 223). Same script template (`experiments/adam_lower_k_cliff_cycle225.py`), same machinery (topk+lstsq init → Adam W_out+b_out, 4000 steps, best-state restore, 3 seeds), same calib/eval split as cycles 206/223/225. Only `K_VALUES`, verdict thresholds, and bookkeeping change.

## Strengths

- K_VALUES ∈ {8,4,2,1} fully disjoint from prior cycles (210: K=512; 223: K=128/256; 225: K=16/32/64). No overlap, extends the efficiency curve monotonically into sub-16 regime.
- Verdict criteria are crisp and numeric (CONFIRMED: K=4 ≤ 59.28; REFUTED: K=8 ≥ 61.98; INCONCLUSIVE otherwise), anchored to the float baseline (53.895) and the observed K=16 value (55.133 ≈ +2.3%).
- Trainable params scale predictably (K=1 → 1,536/block, ~4.6K total across 3 blocks) — a clean stress test.
- Validation block covers init_ppl monotonicity, param count, restore round-trip, float baseline, and verdict-label consistency.
- REFERENCE_CURVE extended with all prior K values for a single extended efficiency table.
- Fallbacks (drop K=1, then K=2) preserve verdict computability since only K=8 and K=4 are needed.
- Floor diagnostic section (|best - init| < 1.0 = no learning; best > 2× float = catastrophic) is useful for interpreting "cliff" structure vs "plateau."

## Risks

- None of substance. At K=1 the per-block correction is a rank-1 outer product in the output direction plus a bias; if that's insufficient even Adam can't do anything, which is exactly what the floor diagnostic is designed to detect.
- Minor: reference_curve tuple formatting must match the template's existing structure — verbatim copy should handle this.

## Recommendation

Approve as-is. Execute.
