# Cycle 324 Brief — Item 337

## Item
Re-classify cycle 319 (curvature residual) verdict under simplified rank-only criterion.
- Original verdict: INCONCLUSIVE (rank_95 ≤ 50 at b4/b5/b12, but magnitude bar missed)
- New criterion: rank_95 ≤ 50 alone (no magnitude requirement)
- Task: extract per-block rank_95 from existing JSON, count blocks passing, annotate

## Context
Cycle 319 audit identified that geometric concentration ≠ magnitude amplification. The strict verdict tree mixed two conditions that shouldn't be AND'ed. Audit recommends rank-only criterion as more appropriate for this phenomenon.

## Dependencies
- Input: `autoresearch/results/cycle_319_pgolf_curvature_residual.json` (exists)
- No new measurements needed; re-analysis only

## Key Files
- `autoresearch/results/cycle_319_pgolf_curvature_residual.json` — source data (profile → per-block rank_95)
- `autoresearch/audits/cycle_319.md` line 28 — audit recommendation paragraph
- Output: `autoresearch/results/cycle_319_reclassification.md` (new)

## Constraints
- Expected rank_95 values: b4=18, b5=33, b12=21 (all ≤ 50); others ≥ 127
- Write 3-section output: strict verdict, rank-only verdict, justification
- ~2 minutes

## Handoff
Extract (lrelu², relu²) rank_95 profile from JSON. Count blocks with rank_95 ≤ 50 and rank_95 > 100 separately. If exactly 3 blocks pass the threshold (b4, b5, b12) and 10+ others are above, confirm the reclassification logic and write the annotation.
