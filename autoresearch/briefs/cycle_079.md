# Cycle 079: Subspace alignment re-run with correct hooks — GPT-2 FFN

## Item
Item 91: Re-run cycle 069 (item 81) with register_forward_hook on mlp.c_fc to capture true FFN input. The original used output_hidden_states[block_idx], which is the block input (pre-attention), not the FFN input (post-attention + ln_2).

## Context
- Cycle 068 (item 80): CONFIRMED. 10/12 blocks have low effective rank (rank_99 <= 500). Rank does NOT predict compression difficulty.
- Cycle 069 (item 81): INCONCLUSIVE due to implementation bug. Jacobians computed from block-level hidden states, missing attention + ln_2 preprocessing. Raw results (8/12 blocks >= 0.8 at k_99) are unreliable.
- Cycle 069 design review flagged: at k=470 in d=768, random baseline alignment is 0.716. Normalized excess is the primary metric.
- This cycle uses k=384 (agenda spec, 8x compression) and register_forward_hook on mlp.c_fc for correct FFN input.

## Dependencies
- Item 80 (cycle 068): CONFIRMED. Results in `autoresearch/results/cycle_068_jacobian_rank_census.json`.
- Cycle 069 script `experiments/ffn_jacobian_subspace_alignment.py` — adapt with hook fix.

## Key Files
- `experiments/ffn_jacobian_subspace_alignment.py` — cycle 069 script (has the bug on line 95)
- `autoresearch/results/cycle_068_jacobian_rank_census.json` — rank stats per block
- Output: `autoresearch/results/cycle_079_subspace_alignment.json`, `cycle_079_subspace_alignment.png`

## Constraints
- Hook on mlp.c_fc (NOT block input). k=384. 256 tokens, 3 seeds [42, 123, 7].
- Random baseline + normalized excess = (actual - random) / (1 - random) as primary metric.
- Confirmed if: raw mean alignment >= 0.8 for >= 10/12 blocks. Refuted if: < 0.5 for majority.
