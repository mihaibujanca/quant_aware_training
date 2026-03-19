# Cycle 054 Brief — Training-Distribution Probing with OpenWebText

## Item
70: **Training-distribution probing with OpenWebText subset — GPT-2 FFN**

## Context
Cycles 044–045 exposed critical corpus specificity: Shakespeare probes yield +6% PPL cost, but WikiText-2 (broader distribution) yields +32% cost; zero-shot transfer to WikiText reaches +120%. Hypothesis: the method fails on broad text because the probe corpus is narrow. GPT-2 was trained on WebText; OpenWebText is the closest public replica. This experiment tests whether matching the training distribution recovers generalization.

## Objective
Run k=4@6× corrected compression (blocks 7, 5, 4, 1) using OpenWebText as BOTH probe AND eval corpus. Compare PPL degradation to Shakespeare (+6–9%) and WikiText-2 (+32%). If training-distribution probing yields ≤+15% on its own eval set, the method works as intended and corpus specificity is a probe-mismatch artifact, not a fundamental limitation.

## Hypothesis
OpenWebText-probed, OpenWebText-evaluated k=4@6× corrected degradation: **≤+15%**

## Key Assets
- Compression method: `aleph/lm_geometric_compression.py`
- Recipe: `autoresearch/results/cycle_045_recipe.md`
- Validation template: `experiments/compression_recipe_validation.py`
- Data: HuggingFace `Skylion007/openwebtext` (first 10K tokens)

## Success Criteria
- **Confirmed**: +15% or better on training-distribution eval set
- **Refuted**: ≥+30% (training-distribution data cannot save k=4@6×)
- **Inconclusive**: 15–30% range

## Expected Duration
~30 min (similar to cycle 045)
