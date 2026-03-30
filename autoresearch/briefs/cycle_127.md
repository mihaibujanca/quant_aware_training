# Item 144

SVD of learned corrections: geometric alignment — GPT-2 [INTERPRETABILITY]

# Context

Cycle 125 (item 142) confirmed residual-stream placement (P1) is near-optimal for LoRA corrections (r=8, 49K params, PPL=69.5). Cycle 126 (item 143) was invalidated as a result. Item 144 now analyzes what the learned P1 corrections learn geometrically.

# Dependencies

- **Item 142** ✓ COMPLETED (cycle 125). P1 corrections (r=8) stored in checkpoint from experiment `placement_sweep_k4_cycle125.py` (commit 4371880).
- **Cycle 104 data** ✓ AVAILABLE (`cycle_104_cascade_source.json`): attention output perturbation SVD across transitions.
- **Cycle 110 data** ✓ AVAILABLE (`cycle_110_bump_relevance.json`): bump channel directions per block.

# Key Files

- `/Users/mihai/Projects/quant_aware_training/experiments/placement_sweep_k4_cycle125.py` — P1 trained corrections (seed 42).
- `/Users/mihai/Projects/quant_aware_training/autoresearch/results/cycle_104_cascade_source.json` — attention amplification SVD.
- `/Users/mihai/Projects/quant_aware_training/autoresearch/results/cycle_110_bump_relevance.json` — bump channel data.

# Constraints

- Extract U, V^T from learned LoRA (r=8) at blocks 1, 4, 7, 11.
- SVD each LoRA to extract top-8 singular vectors.
- Compare cosine alignment: (a) vs compression error SVD E_l, (b) vs cycle 104 attention output SVD, (c) vs cycle 110 bump directions.
- Report alignment matrices per block.
- Confirm criterion: alignment with attention amplification cosine > 0.5 in ≥3/4 blocks → CONFIRMED (corrections undo attention error).

# Handoff

Ready to extract P1 correction matrices (r=8, 4 blocks) from cycle 125 checkpoint, compute their SVDs, load reference data from cycles 104/110, and compute pairwise cosine alignment. No blockers.
