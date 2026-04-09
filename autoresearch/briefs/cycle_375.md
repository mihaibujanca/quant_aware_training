# Cycle 375 Brief — Adapter Dead-Neuron Census and Exact Pruning (Item 383)

## Context
- **Working recipe**: topk (k=512 of 3072) + lstsq + MLP adapter (h=64, ReLU). 3-block PPL: 47.43, 6-block PPL: 65.54.
- **Adapter architecture**: Residual MLP. Forward: `out += W2 @ ReLU(W1 @ h + b1) + b2`. W1 is (64, k), W2 is (768, 64). ReLU activation.
- **GPT-2 FFN dead-neuron census**: closed (c128/c162/c192: 0 dead GELU channels). But adapters use ReLU — Serra/Berzins exact pruning applies directly.
- **No saved adapter weights**: prior experiments (c357, c359, c374) train adapters in-memory without saving checkpoints. Must train fresh.

## Hypothesis
ReLU MLP adapters may be over-provisioned. If a significant fraction of the 64 adapter neurons are dead (never activate on calibration data) or always-active (clampable to linear), the adapter has redundant capacity. Pruning these neurons exactly (no PPL cost) would free capacity for a larger effective hidden size at the same param count.

## Task
1. Train a 6-block compressed model (topk k=512 + lstsq + adapter h=64) using the standard recipe (joint training, 1500 steps, seed 42).
2. After training, run calibration data through each adapter and classify neurons as: dead (max activation = 0 across all calibration tokens), always-active (min pre-ReLU > 0 across all tokens), or mixed.
3. Prune dead neurons (remove rows from W1/b1, columns from W2). Prune always-active neurons (fold into linear term). Measure PPL delta.

## Success Criteria
- **Confirmed**: >= 10% of adapter neurons prunable AND pruning causes < 0.1 PPL delta
- **Refuted**: < 5% prunable OR pruning hurts > 0.5 PPL
- **Inconclusive**: 5-10% prunable or 0.1-0.5 PPL delta

## Key References
- Adapter hidden dim: 64 per block, 6 blocks = 384 total adapter neurons
- Float baseline: 53.89
- 6-block k=512 + adapter: 65.54
