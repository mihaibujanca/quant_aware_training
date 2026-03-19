# Cycle 053 Brief — Per-Block Corpus Sensitivity Isolation

## Item
Item 69 (P1) — Per-block corpus sensitivity isolation — GPT-2 FFN

## Context
Cycle 044 (domain robustness) discovered the method is highly corpus-specific:
- Shakespeare→Shakespeare (corrected): +6.29% PPL ✓ works
- WikiText-2→WikiText-2 (corpus-specific): +31.86% PPL ✗ severe
- Shakespeare→WikiText zero-shot: +120.57% PPL ✗ catastrophic
- Gap: 88.7pp between conditions

Cycle 045 (recipe validation) confirmed the gap is real and reproducible.

## Goal
Isolate which of the 4 compressed blocks [7, 5, 4, 1] drives corpus sensitivity. Single-block swaps with Shakespeare vs WikiText-2 probes will identify bottlenecks for corpus robustness.

## Method
For each block independently:
1. Compress block (topk channel selection + lstsq refit) using Shakespeare probes
2. Evaluate on both Shakespeare and WikiText-2 validation sets
3. Compute per-block corpus sensitivity = (WikiText rel PPL) − (Shakespeare rel PPL)
4. Rank blocks by sensitivity delta

Also test with WikiText-2 probes to distinguish channel selection vs refit artifacts.

## Success Criteria
- **Confirmed**: One block accounts for ≥50% of total corpus sensitivity gap → localized, fixable
- **Refuted**: All blocks contribute equally (within ±25% of mean) → systemic, not per-block
- **Inconclusive**: Sensitivity concentrated but switches between channel-selection and refit

## Dependencies
None (eligible immediately)

## Artifacts
- Single-block swap results (4 blocks × 2 probe corpora)
- Per-block sensitivity ranking
- Block-level corpus robustness heatmap
