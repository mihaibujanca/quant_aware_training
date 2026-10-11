# Audit — Cycle 254 / Item 264

**Status: PASS** — verdict REFUTED is correct.

## Item criterion (verbatim)
≤56 confirmed; ≥58.5 refuted; else inconclusive.

## Verdict logic
- best_mean = 59.018 ± 0.041 (3 seeds {42, 137, 2024})
- 59.018 ≥ 58.5 → **REFUTED** (Phase 1 ✓)

## Phase 2 — sanity
- Per-seed best: [58.996, 58.982, 59.076] — std 0.041, tight cluster
- Per-seed best step: [4800, 4400, 5200] — all plateau within step 4400-5200
- Final@8000 mean = 60.517 ± 0.081 — clear plateau-then-climb (+1.50 PPL past best)
- Init PPL = 88.341 (zero seed variance — lstsq deterministic, identical to cycle 251)
- Best-state restore round-trip clean for all 3 seeds (|restored - best| < 1e-3)
- Δ vs cycle 251 K=128 4k 59.119 = −0.101 (within seed noise; doubling budget bought essentially nothing)
- Δ vs float 53.895 = +5.123

## Phase 3 — interpretation
The plateau-then-climb pattern at step 4400-5200 directly falsifies cycle 251's
linear extrapolation that "Adam still descending" would reach ~57.1 at step 8000.
The descent decelerates rapidly past step 4000 and converts to overfitting before
step 5500. The 5-block K=128 floor is structurally bounded near 59 PPL — not an
optimisation-budget artifact. To beat 59 PPL at this regime, a different recipe
is needed (K=256, LR schedule, W_fc trainable, or LoRA on c_proj), not more steps.

interesting=true (resolves cycle 251 inconclusive; closes "more budget" lever for
the 5-block K=128 recipe space).
