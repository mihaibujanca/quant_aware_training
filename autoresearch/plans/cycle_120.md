# Cycle 120 Plan — Item 137: Cascade corrections at k=8 stacking

## Block set (REVISED per review A1)
- Agenda item 137 specifies blocks `[1,2,3,4,5,7,9,11]`.
- Review correctly notes this is **not** a superset of the k=5 set `[7,5,4,1,6]` (block 6 missing),
  breaking the k→k+3 scaling comparison. Block 11 is also unnecessarily hard.
- **Adopted set: [1,2,3,4,5,6,7,9]** = k=5 set {7,5,4,1,6} ∪ next 3 easiest {3,2,9}.
  Deviation from agenda documented; scientific rationale: superset requirement for valid comparison.

## Seeds and verdicts
- Single seed (42), per agenda. Flagged but accepted as exploratory scope.
- Verdicts use agenda thresholds (20/10/10-20), not brief thresholds.

## Changes to script

1. **Copy** `experiments/cascade_correction_k5_k8_gpt2.py` → `experiments/cascade_correction_k8_cycle120.py`.
2. Set `SEEDS = [42]`, `FFN_BLOCKS_K8 = [1, 2, 3, 4, 5, 6, 7, 9]`.
3. **Remove** all k=5 logic. This cycle is k=8 only.
4. Keep r=8 primary, r=16 fallback. Fallback trigger: r=8 recovery < 10%.
5. Output: `autoresearch/results/cycle_120_cascade_correction_k8.json`.

## Expected params
- r=8:  8 blocks × 2 × 768 × 8  =  98,304
- r=16: 8 blocks × 2 × 768 × 16 = 196,608

## Execution
```bash
cd /Users/mihai/Projects/quant_aware_training
python experiments/cascade_correction_k8_cycle120.py 2>&1 | tee autoresearch/logs/cycle_120_run.log
```

## Verdicts (from agenda item 137)
- **CONFIRMED**: recovery ≥ 20% with r ≤ 16.
- **REFUTED**: recovery < 10% at r=16.
- **INCONCLUSIVE**: 10-20% recovery, or r=16 needed but r=8 fails.

## Sanity checks
- P1: Baseline PPL ∈ [25, 65] (abort otherwise).
- P2: k=8 uncorrected PPL > baseline (degradation is positive).
- P3: Correction params match expected counts above.
- P4: Loss curve decreasing (first > last).
- P5: Compare k=8 degradation to k=5 (cycle 119: ~70 PPL). k=8 should be larger.
- P6: Recovery % is sensible (not >100%, not negative).
- P7: Runtime ~20 min for 1 seed (flag if >40 min).

## Advisory notes
- A3 (from review): 8 simultaneous corrections may interact. If training diverges,
  reduce lr to 5e-5 before declaring failure.

## Limitation
Single seed — not sufficient for statistical claims. If recovery is ambiguous (10-30%),
recommend follow-up with 3 seeds.
