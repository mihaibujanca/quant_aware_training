# Cycle 333 Review: LASER on tiny GPT activation variants (item 332)

## Checklist

| Check | Status | Notes |
|---|---|---|
| Spectrum preflight before retrain | PASS | Learned from cycles 327-329; gate at top-5 ≥ 10% |
| nn.Linear vs Conv1D handled | PASS | Weight shape (256,1024), forward via F.linear; explicitly different from cycle 280's HF Conv1D |
| Full-rank sanity check mentioned | WARN | Plan mentions it in Risks but doesn't make it a mandatory step. Should be Step 4.0: verify rank-256 no-retrain = baseline ±0.01 PPL for one variant |
| Multiple seeds | PASS | 3 seeds {42, 137, 2024} for retrain; seed-42 for base checkpoint |
| Baseline from correct source | PASS | Each variant's own uncompressed PPL, not GPT-2's |
| Verdict logic matches agenda | PASS | CONFIRMED/REFUTED/INCONCLUSIVE criteria match item 332 |
| Time budget realistic | PASS | 15 min for 36 retrain runs at 500 steps on MPS — conservative |
| No bias handled | PASS | Tiny model c_proj has no bias; LowRankCProj omits bias parameter |

## Issues

### WARN: Add full-rank sanity as mandatory step
The transposition between nn.Linear and Conv1D is the #1 bug risk. Add a mandatory sanity check: for one variant, construct LowRankCProj at rank=256 (full rank), verify PPL matches baseline within 0.01. If it doesn't, there's a transposition bug. This should abort the entire experiment, not just be a "risk note."

**Fix**: Add to Step 4 before the rank sweep: "Step 4.0: Full-rank sanity. For GELU seed-42, construct LowRankCProj with r=256. Eval PPL. Assert matches baseline within 0.05 PPL. If fails, debug transposition before proceeding."

### INFO: Target block selection
Plan uses highest top-5 energy fraction (from spectrum preflight) as the target block. This is correct — LASER works best where the spectrum is concentrated. But if all blocks have similar spectrum, default to b0 (first block) for consistency with GPT-2 b1 being early.

### INFO: 500 vs 1000 steps
Plan notes 500 steps may be short. For a tiny model with PPL ~185-200, the loss landscape is steeper and Adam may need fewer steps. 500 is fine as a first pass; if results are borderline, the verdict should note "could benefit from extended budget" but not change the outcome.

## Decision
APPROVE (with the full-rank sanity check added as mandatory Step 4.0)
