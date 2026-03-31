# Item 178: Training Dynamics of Negative z — GPT-2 FFN (Scale Comparison)

## Item
Does z_mean negativity scale with model size? Compare GPT-2 small/medium/large z_mean per block on 4096 tokens. Random-init baseline already established in cycle 165.
- **Confirmed**: z_mean ≈ -1 across all three scales (negativity is universal, not scale-specific).
- **Refuted**: z_mean magnitude scales significantly with model size (>2x difference small vs large).
- **Inconclusive**: mixed pattern across scales.

## Context
- Cycle 165: negativity is LEARNED — Wx_mean ≈ -0.9, random-init ≈ 0.0 (shift=0.886). 83-98% of W_up rows anti-aligned with E[x]. This answers the core question of item 178 (random-init vs trained).
- The remaining novel content: scale comparison across GPT-2 small (12L/768), medium (24L/1024), large (36L/1280).
- Cycle 129: isotropy is learned (7-8x increase). Does z_mean show similar universality across scales?

## Dependencies
None. ~10 min compute (3 models, forward pass each).

## Key Files
- `experiments/weight_data_alignment_cycle165.py` — prior z_mean analysis (reuse hooking pattern).
- `autoresearch/results/cycle_165_alignment.json` — GPT-2 small baseline numbers.

## Handoff
Load GPT-2 small/medium/large. Hook mlp.c_fc output for pre-GELU z. Forward 4096 OWT tokens. Report z_mean, Wx_mean per block for each model. Compare across scales.
