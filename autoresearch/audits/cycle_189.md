# Audit — Cycle 189 (Item 199)

**Verdict:** CONFIRMED
**Mean cos(e_l, e_{l-1}):** 0.716 (threshold: >0.3 in ≥8/11 — achieved 11/11)

## Result Quality

Layer cosines: 0.373, 0.525, 0.669, 0.743, 0.760, 0.778, 0.802, 0.817, 0.832, 0.834, 0.744
Frobenius norms: 685, 974, 1337, 1675, 1901, 2228, 2604, 3136, 3782, 4694, 6054, 11472
Rank_90: 371, 468, 480, 503, 530, 545, 553, 550, 542, 534, 518, 320

The result is clean and unambiguous. No NaN or Inf values. All metrics computed for all 12 layers.

## Methodological Notes

1. **Baseline PPL=35.886**: Higher than the ~22-24 typically seen because SEQ_LEN=128 provides
   less context than the 1024 used in cycles 173/181/188. This is expected and does not affect
   the directional analysis (both models evaluated identically).

2. **Full-swap PPL=44.842 (+25%)**: Lower delta than cycle 181 (which used 1024-token sequences
   and saw ~88% degradation). Short sequences have less cascade opportunity — consistent with
   the cascade barrier being a sequential-depth phenomenon.

3. **Device bug (fixed)**: `mean_delta.cpu() @ c_proj.weight.mps()` → added `.cpu()` to weight
   retrieval. Minor fix, does not affect results.

4. **Block 11 anomaly**: ||e||_F jumps from 6054 to 11472 (largest single-block step), and
   rank_90 drops 518→320. This repeats the pattern from cycle 184. The last block amplifies
   errors most severely and the error becomes less diffuse (rank_90 drops).

## Reconciliation with Prior Cycles

- **Cycle 184** found rank_90=300-434 (diffuse), concluded low-rank correction not viable. This
  cycle confirms high rank_90 (320-553), so the diffuse content finding stands. However, cycle 189
  adds the directional finding: even though errors are high-rank, they point in consistent
  directions. These findings are COMPATIBLE, not contradictory.
- **Implication**: The errors cannot be corrected by low-rank projection (cycle 184 correct), but
  they could in principle be corrected by tracking the error direction and applying a directional
  correction. Whether this is achievable without training is an open question.

## Items to Address Next

Item 200 (GELU-neg+ReLU-pos hybrid): The directional compounding implies the error source is
structural, not random. The hybrid test checks whether preserving GELU in the negative regime
(which cycle 188/C3 tested differently) changes the directionality.
