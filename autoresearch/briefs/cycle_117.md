# Brief: Cycle 117 — Cascade Corrections at k=5 and k=8 Stacking (Item 133)

## Context
Cycle 115 showed LoRA-style cascade corrections (r=8, 49K params) recover 74% of k=4 stacking
degradation (PPL 80.7 → 59.1, baseline 51.7). Cycle 116 confirmed this is structural, not
just extra optimization budget. Item 133 asks: does this scale to k=5 and k=8?

## Prior data
- **k=5 uncorrected** (cycle 106): +22-24% PPL after 500-step polish (mean over seeds/configs).
  Best config: [7,5,4,1,6] slightly better than [7,5,1,3,6].
- **k=8 uncorrected** (cycle 089): PPL explodes to +914% (c≈0.41 compounding).
- **Correction saturation at k=4**: r=4 (72.6%) ≈ r=8 (74.3%) ≈ r=16 (74.1%).

## Key risk
At k=8, uncorrected PPL is ~524. The correction must recover a much larger absolute gap.
Rank-8 may be insufficient — the error subspace likely grows with more compressed blocks.
Cycle 116 audit: extended polish shows U-shape (best at step 1000, overfits by 2500).
Corrections at 2000 steps may benefit from implicit regularization of the UV parameterization.

## Criteria
- **Confirmed**: ≥30% recovery at k=5 AND ≥20% at k=8 with r≤16
- **Refuted**: <10% recovery at k=5
- **Inconclusive**: works at k=5 but fails at k=8, or needs r>16
