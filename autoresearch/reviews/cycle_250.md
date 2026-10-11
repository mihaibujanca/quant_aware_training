# Cycle 250 Design Review — APPROVED

## Item
260 (P0): 6-block [1,2,3,4,7,8] — drop 5, add 3.

## Decision
APPROVED.

## Rationale
- Trivial one-line swap of `BLOCKS` on the cycle 221 template; no new code paths, no new infrastructure.
- Hyperparameters identical to cycles 221 and 245, so results are directly comparable (59.515 baseline vs 53.826 5-block).
- Decision thresholds (≤55.0 confirmed, ≥58.0 refuted) are cleanly separated and operational.
- The experiment distinguishes two concrete hypotheses: (a) block 5 is uniquely hard, vs (b) any 6-block config near 60 PPL is a count phenomenon. Either outcome is informative.
- Dependencies: none. Compute budget: same as cycle 221 (3 seeds × 4000 steps).

## Checks
- BLOCKS list matches brief: [1,2,3,4,7,8]. OK.
- Seeds [42,137,2024] match cycles 221/245. OK.
- Best-state restore preserved. OK.
- New script path does not overwrite prior cycle experiments. OK.

## Risks / Notes
- If block 3 is itself problematic, the result may be inconclusive; still informative but doesn't close the question.
- No scalar-narrative risk: only per-seed PPLs are reported.

## Outcome
Approved to execute.
