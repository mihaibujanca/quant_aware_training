# Cycle 200 Brief: Residual stream rank trajectory under compression (Item 210)

## Item

210 — Residual stream rank trajectory under compression — does topk+lstsq preserve rank collapse?

## Context

- Cycle 198: input rank collapse 269→207→25→1 across blocks 0-3. Blocks 3-11 all input_rank=1.
- Current recipe: BLOCKS=[1,4,7,11], K=512 → PPL ~30 (cycle 196).
- Cycle 199: LN creates full 768D per-token variation; K-sweep still monotone at K=512.
- Hypothesis: compressing block 1 (which has input_rank=207) may break the rank collapse trajectory. If block 1's compressed output no longer collapses to near-rank-1 by block 3, later blocks receive wrong-rank input.
- This would partially explain the cascade barrier: each compressed block disrupts the rank trajectory of subsequent blocks.

## Experiment Design

1. Collect output activations at each block boundary for:
   - Float model
   - Compressed model (BLOCKS=[1,4,7,11], K=512)
2. Compute input_rank_90 at each block boundary (same method as cycle 198: SVD of centered block inputs)
3. Compare rank trajectories: does compression disrupt the 269→1 collapse?
4. Also compute Frobenius distance between float and compressed residual stream at each block.

## Key Files

- `experiments/block0_criticality_cycle198.py` — same hook pattern for collecting block I/O
- `experiments/k_sweep_rank1_cycle199.py` — CompressedMLP, collect_mlp_io, fit_block patterns
- `autoresearch/results/cycle_198_block0_criticality.json` — float per-block input ranks (ground truth)

## Constraints

- MPS: lstsq must be on CPU
- BLOCKS=[1,4,7,11], K=512 (standard recipe)
- N_SEQS=50, SEQ_LEN=64 for rank measurement (same as cycle 198 for comparability)
- Rank measurement: input to each block (before LN-2), i.e., residual stream state
