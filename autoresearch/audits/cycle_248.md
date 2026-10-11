# Cycle 248 Audit — Item 258

**Status: PASS** (interesting=true)

## Claim
Item 258: K=128 + W_out + W_fc trainable at 6 blocks. Confirmed if best_mean ≤ 61.0; refuted if ≥ 66.0; inconclusive in between. Reported REFUTED at best_mean=67.609 ± 0.047.

## Phase 1 — Verdict logic
67.609 ≥ 66.0 → REFUTED. Correct per item 258 criteria.

## Phase 2 — Evidence quality
- 3 seeds {42, 137, 2024}, per-seed bests 67.671 / 67.596 / 67.559, std=0.047 — tight.
- Init PPL=115.056 with zero seed variance and exact match to cycle 231 K=128 init (lstsq deterministic).
- Float baseline 53.895; cycle 231 K=128 W_out-only baseline 67.562; Δ = +0.047 (within seed noise).
- Cycle 221 K=512 ceiling 59.515; Δ = +8.094.
- W_fc relative Frobenius drift mean = 7.85e-2 at best (per-block 0.049–0.098). W_fc *did* train ~8% off init, so the noop is structural, not "did not train".
- Best-state restore round-trip clean for all 3 seeds (|restored − best| < 1e-3).
- final@4000 ≈ best (no overfit); LR=5e-5 monotone descent at ~0.003 PPL/100 steps by step 3000 — no plausible path below 66 in any reasonable budget.
- Wall ~133.4s/seed, ~400s total. Trainable 1,185,024 (sliced (768, K) W_fc per cycle 213 FullCompressedMLP, the only budget-coherent K=128 interpretation; the agenda's "14.77M" formula was wrong and is correctly flagged in the experiment docstring).

## Phase 3 — Interesting
Combined with cycle 241 (W_fc bias-only at K=512, also noop), W_fc surface enlargement is now refuted at *both* K endpoints. The 8.09 PPL K=128→K=512 gap is a K-bandwidth (information bottleneck) phenomenon, not a trainable-surface-size phenomenon. This closes a hypothesis cleanly via two-endpoint triangulation.
