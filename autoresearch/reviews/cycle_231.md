# Cycle 231 Design Review — Item 241

**Verdict: APPROVE**

Trivial K-swap on cycle 221's validated protocol. Only `K=512 → K=128` changes; BLOCKS, LR, N_STEPS, EVAL_EVERY, SEEDS, restore-best logic all mirror cycle 221 exactly.

## Rationale

- **Well-motivated contrast**: cycle 223 showed K=128 beats float at 3 blocks; cycle 221 showed K=512 ceilings at 59.5 at 6 blocks. Testing whether K=128 survives the 3→6 cascade jump is a clean one-variable question.
- **Protocol reuse**: template script `experiments/adam_6block_earlystop_cycle221.py` is already validated. Copying and swapping a single constant minimizes implementation risk.
- **Param accounting correct**: 768×128 + 768 = 99,072/block × 6 = 594,432 ≈ 0.59M, half of K=512's ~2.36M. Checks out.
- **Verdict thresholds reasonable**: ≤61.0 Confirmed (within ~1.5 PPL of K=512 ceiling at 4× compression = clear win), ≥65.0 Refuted (collapse at higher cascade depth), else Inconclusive.
- **Wall-time**: K=128 strictly faster than K=512 in CompressedMLP forward; 2h budget safe.
- **Init guard**: abort-if-init>120 catches lstsq bugs before wasting Adam steps.

## Concerns

None material. Standard K-sweep continuation.

## Notes

- Ensure `cycle` and `item` fields in JSON flip to 231/241 (listed in step 1; flagged here so execute stage doesn't miss it).
- Restore-best assertion tolerance `1e-3` preserved from cycle 221 — keep.
- Delta vs 59.515 cycle 221 ceiling must appear in summary markdown.
