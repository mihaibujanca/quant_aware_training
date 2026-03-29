# Cycle 103 Brief: GELU-ReLU Disagreement Anatomy

## Item
108. GELU-ReLU disagreement anatomy — GPT-2 FFN (analysis)

## Context
Cycle 102 refuted zero-shot GELU→ReLU swap: +8320% PPL catastrophic, block 0 alone +2664%.
This cycle interprets WHY—measure activation zones and quantify downstream signal.

## Dependencies
- Cycle 102 results: `experiments/gelu_relu_swap_gpt2.py`
- GPT-2 pre-quantized checkpoint
- 4096-token mixed-corpus batch

## Key Files
- `experiments/residual_projection_correction_gpt2.py` (existing analysis framework)
- GPT-2 weights in `~/.cache/huggingface/hub/`

## Constraints
- Partition pre-activation z into 3 zones: z >> 0 (agreement), z << 0 (agreement), z ≈ 0 (bump)
- Measure: zone fraction, |GELU(z) - ReLU(z)| vs block output norm, token/channel anatomy, masking effect
- Success: bump <5% energy AND <1% prediction change (noise)
- Failure: bump >20% energy OR >10% prediction change (signal)

## Handoff
Analyze zone occupancy, energy distribution, and masking impact across all 12 FFN blocks.
Visualize: 2D heatmaps (channels × tokens), energy pie charts, prediction flip heatmap.
Outputs to `autoresearch/results/cycle_103_gelu_relu_anatomy.{json,png}`.
