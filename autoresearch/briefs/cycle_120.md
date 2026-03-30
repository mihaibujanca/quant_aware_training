# Cycle 120 Brief

## Item
137. Cascade corrections at k=8 stacking — GPT-2 FFN

## Context
Cycle 119 confirmed cascade corrections at k=5 (73.8% recovery, 61K params). This item scales to k=8 — adding 3 more blocks to the stacking sequence. Tests whether gradient-based correction remains efficient at deeper compression (k=4→k=5 proved robust; k=8 is the next scaling boundary).

## Dependencies
- Cycle 119 baseline: k=5 with blocks [7,5,4,1,6] at 6× (topk+lstsq+500 Adam), recovery 73.8% ± 1.4%
- Config: GPT-2 medium, 768-d, OWT validation set
- Assumes k=8 expansion uses same block selection heuristic (attention-driven, uniform across heads)

## Key Files
- Script: `/Users/mihai/Projects/quant_aware_training/experiments/cascade_correction_k5_k8_gpt2.py` (exists, merged k=5/k=8 impl from cycle 117)
- Output: `autoresearch/results/cycle_120_cascade_correction_k8.json`
- Prior: `autoresearch/results/cycle_119_cascade_correction_k5.json` (k=5: 73.8%, 61K params)

## Constraints
- r=8 LoRA rank (matches cycle 119 precedent)
- 2000 CE training steps, lr=0.0001
- 3 seeds: 42, 123, 7
- Verdict threshold: **CONFIRMED** if recovery ≥70% (graceful scaling from k=5); **INCONCLUSIVE** if 60–70%; **REFUTED** if <60%

## Handoff
Run cascade_correction_k5_k8_gpt2.py with k_blocks=8 flag. Report per-seed recovery % and total correction params. Compare to k=5 baseline (73.8%) for scaling efficiency check.
