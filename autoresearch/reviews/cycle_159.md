# Review — Cycle 159 / Item 169: Correction-first compression (Round 2)

## Verdict: APPROVE

## Round 1 issue resolution

1. **Transpose bug (high)**: FIXED. Step 2 now extracts `C_b = LoRA_B.weight.T @ LoRA_A.weight.T` and validates orientation on a probe batch with cosine > 0.99 assertion, transposing and re-checking on failure.
2. **OOD correction application (high)**: FIXED. Step 3 adds per-block cosine similarity between compressed-model and float-model residuals. Cosine < 0.5 skips Phase B first-order and goes directly to iterative fallback.
3. **lstsq residual gate (medium)**: FIXED. Step 4 and Validation section make the 2x residual ratio a blocking gate producing INCONCLUSIVE, not REFUTED. Hypothesis section also includes this path.
4. **Redundant phases (low)**: FIXED. Phase A and Phase B are now distinct — A is baseline compress+correct+extract, B is correction-first compression.

## R1: Premise validity

Hypothesis is clear with well-defined thresholds: confirmed <57, refuted >62, inconclusive 57-62 or lstsq gate. The correction-first reordering has theoretical motivation (preserve the subspace corrections can't reach). First-order approximation risk is mitigated by the OOD gate and iterative fallback. PASS.

## R2: Measurement soundness

Baselines documented (70.7 / 59.0 / 51.7). PPL is absolute. lstsq residual gate is blocking. Probe-batch orientation check prevents silent errors.

Minor note: Step 4 computes `Y_target = Y_float - C_b @ (residual_in_float + Y_float)`. The correction input includes `Y_float` (the FFN output), which creates a mild circular dependency — the correction was trained on FFN input (residual stream), not residual + FFN output. If the correction architecture actually wraps the block output, this is fine; if it wraps the FFN input only, the formula should be `Y_target = Y_float - C_b @ residual_in_float`. This is protected by the lstsq residual gate — if the formula is wrong, the residual will blow up and trigger INCONCLUSIVE. Not blocking.

## R3: Capacity and statistical adequacy

3 seeds x 200 eval sequences. Matches established protocol. PASS.

## R4: Method-architecture alignment

GPT-2 FFN blocks [7,5,4,1] at 6x, LoRA r=8. Consistent with prior cycles. PASS.

## R5: Known pitfall scan

- MP1 (threshold above trivial): N/A — PPL is absolute.
- MP2 (activation regime mismatch): N/A.
- MP3 (underdetermined fitting): PASS — k=4 at 6x oversampling.
- MP4 (phenomenon-domain mismatch): N/A — same architecture/task.
- MP5 (insufficient capacity): PASS — float baseline works, same correction rank.
- MP6 (verdict misclassification): PASS — lstsq gate and OOD gate produce INCONCLUSIVE when method fails, not REFUTED.
- MP7 (post-LayerNorm): N/A.
- MP8 (representation vs functional rank): N/A.
