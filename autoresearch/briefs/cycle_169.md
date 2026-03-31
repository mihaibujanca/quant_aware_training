# Cycle 169 — Item 179: Does constant bias shift under compression?

## Context
Cycle 166 confirmed the GELU negative regime is functionally a constant bias (b_leak) per block: 11/12 blocks within +5% PPL. But that was on the float model. Under topk+lstsq compression (k=4 blocks at 6x), upstream blocks produce different hidden activations, which means different channels go negative. b_leak may shift, contributing to the 41%/block cascade.

## Test
1. Load GPT-2, compress k=4 blocks [7,5,4,1] at 6x with topk+lstsq.
2. Compute b_leak_float per block on float inputs (reuse cycle 166 method).
3. Compute b_leak_compressed per block on compressed-model inputs.
4. Measure relative shift: ||b_leak_compressed - b_leak_float|| / ||b_leak_float|| per block.
5. If shift > 20% in any block: fix b_leak to float value (add correction = b_leak_float - b_leak_compressed after activation), measure PPL improvement.

## Criteria
- **CONFIRMED**: shift > 20% relative AND fixing reduces cascade by >= 10%.
- **REFUTED**: shift < 5% — bias is stable.
- **INCONCLUSIVE**: shift exists but fixing doesn't help.

## Dependencies
None. Uses cycle 166's b_leak methodology + standard compression pipeline. ~15 min compute.
