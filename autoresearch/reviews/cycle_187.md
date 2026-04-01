# Cycle 187 Design Review: Layer-drop baseline — GPT-2 FFN skip blocks [1,4,7,11]

## Review Summary
Design tests per-block FFN importance via single-block zero-pass ablation. **Sound hypothesis, but critical implementation and verdict bugs block execution.**

---

## R1: Design Soundness ✓
Single-block drop correctly measures FFN contribution to language modeling loss. Architecture is clean: save reference → monkey-patch → eval → restore.

---

## R2: Verdict Criteria — MISCALIBRATED ✗

**Issues:**
1. **Threshold mismatch**: Plan states GPT-2 baseline "typically 30-50 PPL" but verdict hardcodes "damage < 40" as droppable. This is inconsistent:
   - At baseline 45 PPL, damage < 40 → PPL < 85 (89% increase, not modest)
   - At baseline 30 PPL, damage < 40 → PPL < 70 (133% increase, critical)
   - No calibration shown.

2. **Verdict logic doesn't test the hypothesis**: Verdict returns "confirmed" if "≥3 blocks have damage < 40". But plan never states the hypothesis being tested. Confirmed *what*? That blocks 1,4,7,11 are droppable? The verdict doesn't check TARGET_BLOCKS at all—it just counts any 3 blocks.
   - Current logic would return "confirmed" even if blocks 0,2,3 are droppable and blocks 1,4,7,11 are all critical.
   - **Fix**: Filter verdict to check if TARGET_BLOCKS are among the top-3 least-damaging blocks.

3. **No literature reference**: Why "damage < 40"? Why "≥3 blocks"? These thresholds need justification or prior art (e.g., lottery ticket magnitude, quantization error bounds).

---

## R3: Implementation — CRITICAL BUGS ✗

### Bug 1: Monkey-patch binding error
```python
original_forward = model.transformer.h[block_id].mlp.forward
model.transformer.h[block_id].mlp.forward = lambda x: torch.zeros_like(x)
```
**Problem**: `lambda x: ...` loses `self`. MLPs access `self.training`, `self.device`, and other module state. This will crash or silently corrupt computation.

**Fix**: Use `types.MethodType`:
```python
import types
original = model.transformer.h[block_id].mlp.forward
model.transformer.h[block_id].mlp.forward = types.MethodType(
    lambda self, x: torch.zeros_like(x),
    model.transformer.h[block_id].mlp
)
```
Or use a proper wrapper class.

### Bug 2: No restoration safety
If eval crashes mid-run, block is left zero-patched and subsequent evals are corrupted. No try/finally to guarantee restoration.

**Fix**: Wrap in try/finally:
```python
try:
    model.transformer.h[block_id].mlp.forward = zero_fn
    ppl = eval_ppl(model, dataloader)
finally:
    model.transformer.h[block_id].mlp.forward = original_forward
```

### Bug 3: Loss aggregation ambiguity
Plan specifies `BATCH_SIZE=4` but doesn't show PPL calculation:
- Is loss per-token (sum over tokens in batch, then divide by count)?
- Is loss per-batch (cross-entropy on logits × lengths)?
- Does batch aggregation use mean or sum?

**Critical**: PPL = exp(mean_loss_over_all_tokens), not exp(mean_loss_over_batches). Clarify in pseudocode.

---

## R4: Edge Cases ✗

1. **Missing `torch.no_grad()` context**: Plan mentions in notes but doesn't enforce at top level. Should wrap all eval in `with torch.no_grad():`.

2. **Device handling**: Plan assumes CUDA or CPU silently. No check for MPS (Apple GPU). Will error on Mac without explicit handling.

3. **File existence**: No check that `openwebtext_sample.txt` exists or has ≥200×128 tokens. If truncated, results are biased.

4. **No restoration verification**: After unpatching block, no sanity check that `.forward` is restored. If patch silently fails, next block tests corrupted model.

---

## R5: Verdict Label Mismatch ✗

**Problem**: Verdict returns "confirmed/refuted/inconclusive" but the plan never states a binary hypothesis.

- Is the hypothesis "blocks 1,4,7,11 are droppable"? (Then test only those 4, not all 12.)
- Is the hypothesis "≥3 blocks are droppable in GPT-2"? (Then justify why 3.)
- Is the hypothesis "FFN blocks are redundant"? (Then calibrate thresholds to quantitative redundancy metrics, e.g., effective rank, gradient flow.)

**Current implementation**: Counts any 3 blocks with low damage, regardless of which blocks they are. This decouples verdict from the stated objective (evaluating TARGET_BLOCKS [1,4,7,11]).

**Fix**: Redefine verdict to check if TARGET_BLOCKS are among the top-3 least-damaging blocks:
```python
target_set = {1, 4, 7, 11}
least_damaging = sorted(damage.items(), key=lambda x: x[1])[:3]
least_damaging_ids = {b for b, d in least_damaging}

if target_set.issubset(least_damaging_ids):
    return "confirmed"
elif target_set.isdisjoint(least_damaging_ids):
    return "refuted"
else:
    return "inconclusive"
```

---

## Summary Table

| Criterion | Status | Severity |
|-----------|--------|----------|
| R1: Soundness | ✓ Pass | — |
| R2: Verdict calibration | ✗ Fail | CRITICAL |
| R3: Implementation | ✗ Fail | CRITICAL |
| R4: Edge cases | ✗ Fail | HIGH |
| R5: Verdict label match | ✗ Fail | CRITICAL |

---

## Recommendation: REVISE

**Must fix before execution:**
1. Fix monkey-patch binding (use `types.MethodType` or wrapper)
2. Add try/finally for restoration safety
3. Clarify loss aggregation (per-token PPL calculation)
4. Redefine verdict to check TARGET_BLOCKS membership
5. Justify damage thresholds (or calibrate to baseline ± 2σ std)

**Should fix:**
6. Add restoration sanity check (re-eval one block after unpatch)
7. Add `torch.no_grad()` context at top level
8. Handle MPS device explicitly
9. Verify data file exists and has sufficient tokens

---

## Notes for Implementer
- This is a **sensitivity analysis**, not a compression test. Verdict should reflect which blocks are least important, not whether compression is feasible.
- TARGET_BLOCKS [1,4,7,11] need justification in the brief. Why these blocks? Were they selected based on prior layer-drop literature?
- Consider adding per-layer statistics (mean damage, std, quantiles) to contextualize individual block results.
