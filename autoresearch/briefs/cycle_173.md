# Cycle 173 — Brief

**Item:** 184. Unified per-block regime ablation table — GPT-2 FFN [INTERPRETABILITY]

**Goal:** Reconcile contradictory claims from prior cycles using ONE metric (single-block PPL) across 6 conditions on all 12 blocks independently.

**Prior results to reconcile:**
- Cycle 102: GELU→ReLU swap — block 0 catastrophic (615 PPL), blocks 1-11 mild (+8-56%)
- Cycle 109: Negative regime is 87-98% bias energy (b_leak dominates token-dependent s_neg)
- Cycle 164: Positive regime = 99.9% of L2 norm; killing positive is catastrophic

**Conditions (per block, all others float):**
- C0: float baseline
- C1: kill negative (→ReLU, set output=0 where z<0)
- C2: kill positive (set output=0 where z>0)
- C3: replace negative with token-mean E[y_neg]
- C4: replace positive with token-mean E[y_pos]
- C5: replace both with means (constant block output)

**Confirmed if:** C3 ≈ C0 (within +2% PPL) in ≥8/12 blocks.
**Refuted if:** C3 >> C0 (>+10% PPL) in ≥8/12 blocks.

**Budget:** 2048 OWT tokens, 72 forward passes, ~10 min.
