# Cycle 033 Plan — Item 48: Easiest-First Multi-Block Accumulation (Up to 6 Blocks)

## Goal
Re-run easiest-first multi-block accumulation with the current (improved) geometric method from cycles 026-027. Determine whether 4+ blocks can be compressed within 10% PPL tolerance.

## New Script
`experiments/multiblock_accumulation_6block.py`

## Implementation Plan

```python
"""Cycle 033: Easiest-first multi-block accumulation, 6 blocks, improved method."""
# Config:
TARGET_BLOCKS = [7, 5, 4, 1, 3, 6]  # easiest-first from cycle 007
HIDDEN_FRACTION = 0.25               # 4× compression
SELECTION_MODE = "topk"              # best method from cycles 026-027
PROBE_SEQUENCES = 200, holdout 50
SHAKESPEARE_PATH = "data/shakespeare.txt"
RESULTS = "autoresearch/results/cycle_033_accumulation.json"

# Steps:
# 1. Load GPT-2, build probe corpus
# 2. Compute baseline PPL on holdout
# 3. Probe geometry for all 6 target blocks once
# 4. Compress each to 4× (topk selection)
# 5. Accumulate: for k=1..6, create pack with first k blocks, swap, eval PPL
# 6. Save results + plot
```

## Output Artifacts
- `autoresearch/results/cycle_033_accumulation.json`
- `autoresearch/results/cycle_033_accumulation.png`

## Verdict Criteria
- **Confirmed**: k=4 PPL ≤ baseline × 1.10 (≤10% increase)
- **Refuted**: k=3 PPL > baseline × 1.20 (>20% increase at 3 blocks)
- **Inconclusive**: k=3 within 10%, k=4 exceeds 10%

## Commit
`[autoresearch] cycle 033: easiest-first multi-block accumulation 6 blocks`
