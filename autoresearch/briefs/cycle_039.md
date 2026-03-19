# Cycle 039 Brief — Item 58

## Task
**Inter-block error correction via residual-stream lstsq adjustment — GPT-2 FFN**

Cycles 036–038 hit the k=5 wall: all five-block configs exceed +15% PPL regardless of per-block ratio, combo selection, or ordering. Root cause: compounding error propagates through residual stream, not addressable by single-block tuning alone.

Item 58 proposes a post-hoc correction: after compressing k blocks in sequence, fit a single global affine correction C (via least-squares on probe data) at the residual stream AFTER the last compressed block, minimizing ||h_float - h_compressed|| at that layer. This correction is zero-parameter at inference (folds into the next layer's weights).

## Hypothesis
Inter-block error is mostly a mean shift + low-rank distortion in the residual stream. A single insertion point correction can recover several percentage points of PPL.

## Success Criteria
- **Confirmed if**: correction reduces the k=5 PPL gap by ≥3pp (from +17.46% toward +14% or better).
- **Refuted if**: correction reduces gap by <1pp (error is distributed, not centralized).
- **Inconclusive if**: correction helps k=5 but worsens k=4 (overcorrection signal).

## Dependency
- Item 53: DONE (cycle 036, heterogeneous multi-block at 6×)

## Key Context
- Baseline PPL: 120.32 (cycle 038)
- k=4 reference: +9.77% (blocks 7,5,4,1 at 6×)
- k=5 target (best): +16.12% (blocks 7,5,1,3,6, drop 4)
- Gap to close: 6.35pp (from +16.12% to ~+10% threshold)
