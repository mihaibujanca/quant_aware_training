# Cycle 370 — Item 378: Selective Compression (b10+b11 Uncompressed)

## Goal
Test whether composition barrier is driven by late-block error: compress 10 blocks {0–9}, leave b10+b11 at float precision.

## Dependencies
- **c366** (cycle 366): Error propagation shows hockey-stick explosion at blocks 10–11. Cumulative error grows slowly (10.0→50.3) through blocks 0–9, then explodes (50.3→72→160) at blocks 10–11. Both b10 and b11 account for 77% of total local FFN error.
- **c367** (cycle 367): Block importance ranking. Block 0 catastrophically critical (+3021 PPL). Block 11 is 2nd most important (+9.12 PPL), block 10 is 7th (+7.31 PPL). All blocks task-critical; no pruning below threshold.

## Hypothesis
If composition barrier is driven by b10/b11 quantization error accumulating through subsequent layers, skipping quantization on these late blocks should break through the 3-block PPL ceiling while maintaining compression on the 10 earlier blocks.

## Method
- **Config**: topk+lstsq+adapter (k=512, h=64) on blocks {0,1,2,3,4,5,6,7,8,9}. Leave b10 and b11 at float32.
- **Training**: Joint adapter training across 10 compressed blocks.
- **Metrics**: Full-model PPL, effective compression ratio (10 compressed + 2 uncompressed blocks).

## Success Criteria
- **Confirmed**: 10-block PPL < 55.0 (within ~1 PPL of float 53.89) — barrier broken.
- **Refuted**: 10-block PPL > 60.0 — composition collapses even without late-block quantization.
- **Inconclusive**: PPL in [55, 60] — partial improvement.

## Status
Ready to execute. Working recipe stable (360 cycles, topk+lstsq+adapter).
