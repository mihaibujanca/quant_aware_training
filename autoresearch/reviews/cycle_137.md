# Cycle 137 Review — AE Param-Matched Comparison (Item 151)

Item 151. Reviewer: design-review agent.

## R1: Premise validity — OK

Cycle 131's INCONCLUSIVE verdict explicitly flagged the 2.5x param gap. Isolating
param count vs architecture is the correct next step. Testing at 6x (the standard
recipe ratio) instead of 8x is also right — results at the working compression point
are what matter.

## R2: Measurement soundness — OK

Three seeds, pre/post polish, 200 eval sequences. The polish-survival test is
well-motivated by cycle 133 (polish erases pre-polish differences). Param-count
printing at init catches silent mismatches. Float baseline gate at 57.6 prevents
measuring on a broken pipeline.

## R3: Capacity and statistical adequacy — OK

3 seeds x 4 blocks x 3 methods x 2 phases = 72 PPL measurements. The >=3%
threshold on >=2/4 blocks is reasonable for 3-seed statistics. No single-seed
screening needed since the full matrix is tractable.

## R4: Method-architecture alignment — MINOR NOTE

**d_bn=64 dead neuron risk.** A 3072->64 encoder with ReLU may kill >50% of the 64
units, leaving effective capacity well below 64. The plan's fallback (d_bn=96) is
appropriate but: log the fraction of dead neurons (mean activation = 0 across eval
batch) for each block. If >30% are dead, the comparison is unfair (effective AE
capacity << topk capacity). Report this metric alongside PPL.

**Polish semantics.** For topk+lstsq, "polish" fine-tunes the 393K compressed weight
matrix. For AE, "polish" fine-tunes all AE params (encoder+decoder). These are the
same param sets that were already trained (AE) or derived (topk), so polish is
apples-to-apples in terms of what's being optimized. Correct.

## R5: Known pitfall scan — OK

- Cycle 133 showed polish erases pre-polish gaps: plan explicitly tests this.
- AE training at 1000 steps with 2000-step fallback is reasonable.
- No MP issues (no underdetermined system, no capacity mismatch at the experiment level).

## Summary

No blocking issues. One advisory:
1. Log dead neuron fraction for d_bn=64 AE. If >30% dead, the param-matched
   comparison is confounded by effective capacity loss, not just architecture.

**Verdict: ACCEPT.**
