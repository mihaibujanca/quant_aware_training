# Item
227 (P1): Frobenius-weighted channel selection — select channels that minimize output Frobenius error.
For each compressed block (BLOCKS=[1,4,7] K=512), pick the K channels minimizing ||float_out - compressed_out||_F across the calibration set, vs the current standard topk by act_std × W_proj_norm.

# Context
- Cycle 207 (item 218, REFUTED): all 4 channel-selection scoring criteria within 1 PPL on single-block PPL — selection was effectively irrelevant at K=512 because lstsq W_out adapts to whichever 512 channels are chosen.
- BUT cycle 207 measured single-block PPL on block 1. The CASCADE effect is what matters. This item tests whether a different selection criterion can reduce cascade damage on the 3-block init PPL.
- Current best: BLOCKS=[1,4,7] K=512 topk init → 61.746 PPL (cycle 206 init), 14.6% above float.

# Dependencies
None.

# Key Files
- experiments/block1_channel_selection_cycle207.py — channel-selection comparison.
- experiments/adam_wout_correction_cycle206.py — base 3-block compressed init pipeline.
- aleph/lm_geometric_compression.py — topk+lstsq construction.

# Constraints
- BLOCKS=[1,4,7], K=512. Same as the canonical cycle 206 init.
- Frobenius-optimal channel selection is generally NP-hard, but practical heuristics:
  (a) Greedy forward: pick channels one at a time to minimize the residual Frobenius error.
  (b) Frobenius-importance: rank channels by their contribution to ||W_proj[c] · z_c||_F across calibration tokens, then pick top-K.
  (c) Held-out-CV variant: split the calibration set, fit the lstsq solution on one half, evaluate Frobenius error on the other.
- Baselines: standard topk (act_std × W_proj_norm).
- Metric: 3-block init PPL vs standard init PPL (61.746).
- Confirmed if: Frobenius-optimal selection improves init PPL by ≥1 PPL (≤60.75).
- Refuted if: improvement <0.3 PPL.
- Inconclusive if: 0.3-1 PPL improvement.
- Use 3 seeds for any randomized selection.
- Apply same lstsq fit afterward (so the only variable is which channels are kept).

# Handoff
Plan agent: design 1-2 selection variants. Recommendation:
- Variant 1: Frobenius-importance per channel = sum of (W_proj[c] · z_c[t])^2 over calibration tokens t. Sort and take top-K.
- Variant 2: Greedy forward selection minimizing remaining Frobenius residual (slower, more accurate).
Apply to all 3 blocks. Compute init PPL (no Adam correction yet — that's a separate test). Compare against standard topk init PPL.

Tight scope. ≤30 min wall time.
