# Design Review — Cycle 191 (Item 201): Mean Residual Error Correction

## Correctness

**R1 — Mean correction computation**: `mean_correction[l] = (r_float[l] - r_swapped[l]).mean(dim=0)`.
This is the first moment of the error distribution over tokens. Shape: (768,). Computed from
calibration data (seed=1337), evaluated on analysis data (seed=42). **CORRECT**.

**R2 — Hook output handling**: GPT-2 block returns `(hidden_states, present_key_value, ...)`.
Hook must add correction to `out[0]` and reconstruct the tuple. Plan handles this correctly.
**CORRECT**.

**R3 — Correction application**: The mean correction is added to the FULL residual stream
output of each block (after residual connection). This is equivalent to adding a constant
bias to the residual stream at each depth. Not the same as cycle 182 (which added bias to
c_proj before residual addition). **CORRECT — distinct from cycle 182**.

**R4 — Config matches cycle 189**: Same seeds, SEQ_LEN=128, N_SEQS_CALIB=200, N_SEQS_ANALYSIS=100.
Uncorrected PPL should match cycle 189's 44.842. **CORRECT**.

## Potential Issues

**P1 — Mean error may be small**: If the error distribution is symmetric (zero mean), the
correction will be near-zero and provide no benefit. The directional coherence finding (cycle 189,
cos=0.716) does not necessarily mean the mean is large — it means CONSECUTIVE layer errors
are aligned, not that the mean is large relative to the variance.

The correct predictor of benefit: ||mean_error[l]||₂ / ||e_l||_F × sqrt(N_tokens).
If this ratio is small (say <5%), the correction will be small. The experiment captures this
via `fraction_of_frobenius`.

**P2 — Order of operations**: Bias vectors in step 1 are computed from the float model on calib
data (pre-residual-collection). Then residual collection in step 2 uses the swapped model (after
bias correction). The mean_correction computed in step 2 is then the RESIDUAL error after
bias correction — not the raw error. This is correct: we want to correct the remaining error,
not the full error (which the bias already partially corrects). **CORRECT**.

## Verdict Criteria Check

Reference: cycle 189 uncorrected = 44.842. Corrected baseline = ~35.9.
- CONFIRMED threshold: corrected < 38 (saves > 6.8 PPL from 44.842, ~15%)
- REFUTED: corrected within 1 PPL of 44.842 (correction has < 2% effect)

These seem reasonable. Even a small PPL improvement would be informative.

## Flags

None blocking.

**REVIEW RESULT: APPROVED.**
