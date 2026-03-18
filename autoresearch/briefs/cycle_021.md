# Cycle 021 Brief — Grokking as Compression Catalyst: Toy ReLU

## Item 39 Summary
Train a spirals ReLU MLP well past convergence. Monitor geometric signals (activation
pattern count, pattern entropy, single-affine residual) at regular checkpoints.
Look for a grokking-like phase transition where partition complexity drops sharply.
If found: compare compression quality (from relu_geometric_compression.py) at
pre-grokking vs post-grokking checkpoints.

## Hypothesis
Grokking induces geometric simplification — the network discards redundant linear
regions and consolidates into a simpler partition, making it easier to compress with
the MASO-based pipeline. Post-grokking, fewer, larger regions → higher compressibility.

## Setup
- Dataset: spirals, n=200 (small → forces grokking), 3 turns
- Architecture: 2→32→32→32→32→1 (4 hidden layers, 32 units each)
- Training: 10000 epochs, weight_decay=0.01, lr=0.001
- Checkpoints: every 500 epochs (20 total)
- Signals: distinct pattern count, pattern entropy, single-affine residual

## Criteria
- **Confirmed**: pattern count drops ≥40% pre→post AND post-grokking compression
  achieves ≥10% better accuracy at matched param budget
- **Refuted**: pattern count stable or compression quality unchanged
- **Inconclusive**: pattern count drops but compression doesn't improve
