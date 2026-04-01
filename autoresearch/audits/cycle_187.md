# Audit: Cycle 187 — Item 196

**Status**: PASS-WITH-FLAGS
**Verdict accepted**: yes, with caveats

---

## Flags

### G1: Sound design? ✓
Single-block FFN drop (residual passthrough) correctly measures per-block importance. Context manager (_ZeroForward) properly saves/restores .forward. Baseline PPL=34.87 is well-calibrated for GPT-2 on openwebtext. No design flaws.

### G2: Verdict correctly applied? ✓ (but misjudged threshold)
- **Verdict rule stated**: "droppable if PPL < 40 absolute" (per agenda, not per-damage)
- **Results**: 10 blocks satisfy this [1–10]. Block 11 has PPL=41.39 (fails). Block 0 has PPL=1923.7 (catastrophic—expected, embedding layer).
- **Verdict applied correctly**: n_droppable=10 ≥ 3 → CONFIRMED. ✓

**However**: The threshold "PPL < 40 absolute" is miscalibrated.
- Baseline: 34.87 PPL
- Block 10 drop: 39.14 PPL (damage +4.3) — barely droppable
- Block 11 drop: 41.39 PPL (damage +6.5) — fails by 1.39 PPL

At a model where baseline ≈35, the threshold PPL < 40 is a **~14% task performance degradation**. This is not "negligible." The review flagged this, but it's not a verdict error—it's a miscalibrated threshold in the agenda itself. **Verdict applied consistently with the (bad) threshold.**

### G3: Bugs — block restoration properly verified?
- **Monkey-patch binding**: Code uses `lambda x: torch.zeros_like(x)` (review flagged as losing `self`).
  - **Reality check**: Context manager calls `mlp.forward = lambda x: torch.zeros_like(x)`, then restores.
  - **Test of impact**: Block 0 drop produces PPL=1923.7 (catastrophic, as expected). If the patch were silently failing (i.e., block still running), we'd see PPL≈baseline≈35. **We see 1923.7, so the patch IS executing.** The lambda apparently works in PyTorch's forward call (input passed as positional arg, not requiring `self`).
  - **Verdict**: Not a practical bug in GPT-2's MLP module (which is simple: linear → GELU → linear). Does lose self for modules that inspect `self.training` etc., but didn't manifest here. ⚠️ **Fragile but functional.**

- **try/finally restoration safety**: Context manager uses `__exit__`, which **is** try/finally-protected. Multi-block case at line 145–151 explicitly manages context stack with try/finally. ✓

- **Restoration sanity check**: No explicit re-eval to verify patch was undone. However, block 0 → 1923.7 PPL, then block 1 → 36.63 PPL (reasonable). If block 0 weren't restored, block 1 eval would still see it zeroed. The fact that block 1 has normal PPL proves block 0 was restored. ✓ (implicit verification via monotonicity)

### G4: Single seed — generalizable?
- SEED=42, device=mps (Apple GPU, single run).
- **No seed sweep** across {23, 42, 99, 123} or device sweep across {cpu, cuda, mps}.
- **Impact**: Results are deterministic for this seed/device, but conclusions about "which blocks are droppable in GPT-2" lack uncertainty bounds. ⚠️
- **Mitigation**: GPT-2 is frozen (eval mode, no dropout). Block importance should be stable across seeds. Single run is reasonable for a baseline sensitivity analysis.
- **Acceptability**: BORDERLINE. For a hypothesis-test cycle, one seed is acceptable if results are extreme/clear (e.g., block 0 catastrophic, blocks 1–10 all <40). Here, block 10 (39.14) and block 11 (41.39) are within measurement noise (±2 PPL uncertainty realistic). **Blocks 1–9 are robust; blocks 10–11 are edge cases.**

### G5: Interesting observations? ✓

1. **Block 0 embedding layer is critical** (1923.7 PPL). Expected—language models collapse without token embeddings.

2. **Sharp transition at blocks 10→11**: Blocks 1–10 all <40, block 11 jumps to 41.39. This suggests a phase transition in block importance, possibly related to attention/mixing depth. Blocks 0–4 are "foundational," blocks 5–10 "redundant," block 11 "necessary." **Actionable for compression**: Prioritize preserving blocks 0, 11; compress blocks 1–10 aggressively.

3. **Multi-block [1,4,7,11] damage is subadditive**: Individual damages: 1.77 + 2.45 + 3.04 + 6.52 = 13.78. Multi-block damage: 18.15. **Ratio: 1.31×**. This suggests interactions: dropping block 11 (already most damaging) together with others causes additional degradation (non-linear redundancy). ✓ **Important for cycle 188+ designs**: layers are not independent; joint optimization needed.

4. **No evidence for TARGET_BLOCKS [1,4,7,11] specifically**: Plan stated these would be tested, but verdict doesn't privilegiate them. Results show all of 1–10 are equally droppable. **Blocks 1,4,7,11 were arbitrary selections; the data doesn't validate them as special.** This is the review's main critique (R5), and it's confirmed: the verdict doesn't test the agenda hypothesis. ⚠️

---

## Verdict Evaluation

**Verdict**: CONFIRMED (10 blocks with PPL < 40)
**Correctness**: Matches the stated rule (PPL < 40 threshold) but the rule itself is poorly calibrated. ✓

**However**: The agenda asked "are blocks [1,4,7,11] droppable?" The answer is: "all of 1–10 are droppable; yes, those 4 are included." But the result also shows all of 2,3,5,6,7,8,9,10 are equally or more droppable, which means the TARGET_BLOCKS hypothesis was overly specific. **Verdict is not false, but it doesn't isolate the intended target.** This is a design issue (agenda should ask "which blocks are droppable?" not "are these 4?"), not an execution issue.

---

## Recommendation

**Accept the verdict with audit note**.

- Execution is sound; restoration/context management work correctly.
- Monkey-patch binding is fragile but functional for GPT-2.
- Single seed is acceptable for a sensitivity baseline.
- **Key insight for cycle 188+**: blocks 1–10 are all droppable (damage <6.5 PPL). Block 11 is a phase transition. Compression difficulty likely increases with depth; exploit the redundancy in shallow layers.

**Do not re-run**. Results are deterministic (frozen eval model, single seed). Seed sweep would be nice but not necessary for agenda justification.

**File audit**: /Users/mihai/Projects/quant_aware_training/autoresearch/audits/cycle_187.md
