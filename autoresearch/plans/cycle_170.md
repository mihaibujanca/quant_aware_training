# Cycle 170 Plan — Literature Scan: FFN Compression 2025-2026

## Pre-flight checks
- P1 (premise): We need to know what the field has produced for FFN-only or structured MLP compression post-2024. Our topk+lstsq recipe is hand-crafted; the field may have converged on something better.
- P2 (measurement): Deliverable is an annotated table with >=5 methods. Comparison metric is WikiText-2 PPL at comparable compression ratios to our 6x FFN compression.
- P3 (capacity): Literature search only, no compute needed.
- P4 (pitfalls): Most papers evaluate on LLaMA-7B+, not GPT-2. Need to normalize by compression ratio and note scale differences. Unstructured sparsity results are not comparable (no wall-clock speedup without special hardware).

## Steps
1. Search arxiv for structured FFN pruning, calibration-only compression, activation-aware pruning 2024-2026.
2. For each method: record paper, approach, compression ratio, PPL result, training requirement, FFN-only capability.
3. Compare against our recipe: topk+lstsq = PPL 70.7 at 6x FFN, topk+lstsq+corrections = PPL 59 at 6x FFN + 49K correction params.
4. Identify methods not yet tried in this repo.
5. Write annotated table to results file.

## Expected outcome
Table of >=5 methods with gap analysis. Likely CONFIRMED (field is large, we should find untried approaches) or INCONCLUSIVE (different scales).
