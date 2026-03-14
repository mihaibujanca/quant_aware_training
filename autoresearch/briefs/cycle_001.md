# Cycle 001: Balestriero Piecewise Linear / Spline Theory Survey

## Item
1. Literature survey: Balestriero's piecewise linear / spline theory of deep learning
   - Read "A Spline Theory of Deep Learning" (2018), "Mad Max" (2020), "From deep to shallow" (if findable)
   - Focus on: (a) ReLU networks create affine spline partitions, (b) linear region enumeration, (c) compression via partition analysis

## Context
- Active project: geometric compression (investigating function-level structure for parameter reduction)
- Previous work (quantization/correction) completed; transitioning to representation learning
- No prior literature survey on this topic completed yet (PAST EXPERIMENTS empty)
- Repo has no existing code for region enumeration, partition extraction, or tropical geometry (items 2-8 depend on this survey)

## Dependencies
- None (independent literature task)

## Key Files
- `autoresearch/agenda.md` — full agenda with 8 items, P0-P2 priority
- Target papers: Balestriero 2018, Balestriero & Baraniuk 2020, + "From deep to shallow" if locatable
- Future code will land in `experiments/region_census.py` (item 3), `aleph/models.py` (existing models for testing)

## Constraints
- Confirmed if: ≥3 concrete, implementable techniques for analyzing/compressing via piecewise linear structure
- Refuted if: papers purely theoretical with no computational pathway
- Inconclusive if: techniques exist but applicability to model compression unclear

## Handoff
Read the 3 papers, extract: (1) how ReLU networks partition input space, (2) methods to enumerate / count / analyze linear regions, (3) any compression or representation reduction techniques. Assess applicability to item 3 (region census) and item 8 (minimal network construction). Produce a 2-3 page technical summary with implementability verdict.
