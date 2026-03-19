# Cycle 056 Brief — Compression Ratio Sweep on OpenWebText

## Item
72: **Compression ratio sweep on OpenWebText: finding the viable frontier — GPT-2 FFN**

## Context
Cycle 054 REFUTED k=4@6× compression on broad text (+48% PPL). All prior ratio sweeps (cycles 034–036) used Shakespeare, which tolerates 6× well (+8.5%). The method may have a viability window at lower ratios on broad text. This experiment measures where topk+lstsq actually works on OpenWebText.

## Objective
Single-block swaps at {2×, 3×, 4×, 6×} on blocks [7, 5, 4] using OpenWebText as both probe and eval. 3 seeds, 200 eval seqs per block. Identify the compression frontier on broad text.

## Hypothesis
At least one ratio ≤4× achieves <10% PPL degradation for ≥2/3 blocks on OpenWebText.

## Key Assets
- Compression method: `aleph/lm_geometric_compression.py`
- Recipe: `autoresearch/results/cycle_045_recipe.md`
- Validation template: `experiments/compression_recipe_validation.py`
- Data: HuggingFace `Skylion007/openwebtext` (500 docs, ~250K tokens)

## Success Criteria
- **Confirmed**: ≥1 ratio ≤4× achieves <10% PPL on ≥2/3 blocks
- **Refuted**: even 2× exceeds 10% PPL for majority of blocks
- **Inconclusive**: 2× works but 3× fails (narrow window)

## Expected Duration
~45 min (3 seeds × 3 blocks × 4 ratios = 36 configs)
