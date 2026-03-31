# Cycle 170 — Item 180: Literature scan: FFN compression 2025-2026

## Context
163 cycles completed. We have exhausted many directions: topk+lstsq+corrections is best (PPL ~59 at 6x FFN compression on GPT-2), topk near-optimal for channel selection, joint training exhausted, composition barrier unresolved. Before burning more cycles on incremental variants, scan recent literature for methods we may have missed.

## Test
Search arxiv and 2025 proceedings for: (a) structured pruning of FFN/MLP in transformers (post-2024), (b) calibration-only compression (SliceGPT, ShortGPT, LaCo, FLAP), (c) activation-aware structured pruning (Wanda-structured, OWL), (d) any method at <15% PPL degradation with FFN-only compression at comparable ratios. Annotated table of >=5 methods with gap analysis.

## Criteria
- **CONFIRMED**: >=2 untried methods at comparable or better results.
- **REFUTED**: all found methods already tried or clearly worse.
- **INCONCLUSIVE**: methods target different scales/settings.

## Dependencies
None. ~30 min research.
