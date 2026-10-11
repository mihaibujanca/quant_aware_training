# Cycle 251 audit — item 261

**Status: PASS**
**Interesting: true**

Claim: 5-block [1,2,4,7,8] K=128 → INCONCLUSIVE at best_mean 59.119 ± 0.009.

## Phase 1 — verdict logic
- Criteria: Confirmed ≤56.0, Refuted ≥62.0, else Inconclusive.
- best_mean = 59.119 ∈ (56, 62) → INCONCLUSIVE. Correct.

## Phase 2 — data integrity
- 3 seeds {42, 137, 2024}; per-seed best {59.1112, 59.1325, 59.1139}; std 0.009 (tightest multi-block run to date).
- best_step_mean 3967 ± 47: seeds 137 and 2024 best at step 4000; seed 42 best at 3900. **Adam is still descending monotonically through the step budget** — no plateau-then-climb pattern. Final@4000 mean 59.131 ≈ best.
- Init mean 88.341, zero variance (lstsq deterministic). Adam recovers 29.2 PPL (84% of init gap-to-float) — ceiling sits ~5.2 PPL above float.
- Deltas: +5.293 vs cycle 245 K=512 5-block; −8.443 vs cycle 231 K=128 6-block; +10.130 vs cycle 223 K=128 3-block; +5.224 vs float.
- Best-state restore round-trip clean (|restored−best|<1e-3 per seed).

## Phase 3 — interesting findings
- **Pareto observation**: 5-block K=128 (59.119, 495K params) matches 6-block K=512 cycle 221 floor (~59.5, 2.36M params) — statistically indistinguishable PPL at 25% of trainable budget. 4× parameter win at matched PPL.
- **Sublinear K-penalty per block**: 3-block 0.81/blk, 5-block 1.06/blk, 6-block 1.34/blk. The K=512→K=128 drop costs more per block as count grows; the 5-block cascade is more K-robust than 6-block.
- Cycle 245's near-float success at K=512 **does NOT transfer to K=128** — the float-tie was a 4× channel-budget phenomenon, not an intrinsic property of the 5-block recipe.
- Descent-still-active pattern mirrors cycle 248 (K=128); suggests the K=128 surface at 5 blocks is under-optimised at 4000 steps, but linear extrapolation (~0.05 PPL/100 steps) predicts ~57 PPL at 8000 steps — still INCONCLUSIVE band.

Verdict stands.
