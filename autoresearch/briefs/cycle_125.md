# Item 142: Correction Placement Sweep — GPT-2 FFN k=4

## Item
Ablate WHERE to place rank-8 corrections on k=4 FFN-compressed GPT-2.
5 placements: (P1) after FFN, before attention [**cycle 115 baseline**], (P2) after attention, before next FFN, (P3) Q+K projections (additive: Q'=Qx+U_q V_q x), (P4) V+O projections, (P5) P1+P2 combined.
Train each 2000 steps, seed 42. Report PPL.
- **Confirmed**: any non-P1 beats P1 by ≥5% relative PPL.
- **Refuted**: P1 within 2% of best (placement already near-optimal).
- **Inconclusive**: 2–5% differences or unstable ranking.

## Context
- **Cycle 115 baseline (P1)**: r=8 corrections after FFN, before attention. k=4 blocks [7,5,4,1], 6× compression.
  - Mean PPL: 59.1 (seed-42: 69.5). Degradation reduction: 74.3%.
  - 49,152 correction params (r=8 per block, 4 blocks).
- **Cycle 123** (item 140): QK vs V error decomposition. No single component >70% anywhere. V dominates early layers (L2–L4: 65–68%), QK dominates late (L9–L11: 55–58%). Implies both pathways require correction — suggests P1 (after FFN, before attention) may not be optimal if V-space corruption is early.
- **Cascade mechanism**: c≈0.41 attention-driven (cycles 104, 113). Cascade amplification 70%+ in 8/11 layers (cycle 104).

## Dependencies
- **None hard**. Can run independently.
- Cycle 123 results (QK/V decomposition) inform interpretation but not execution.

## Key Files
- Cycle 115 baseline: `autoresearch/results/cycle_115_cascade_correction.json` (PPL, reduction %, per-seed breakdown)
- Cycle 123 QK/V decomposition: `autoresearch/results/cycle_123_qk_v_decomposition.json` (layer-wise dominant source)
- Experimental baseline code: `experiments/cascade_correction_k5_gpt2.py` (cycle 119 variant, adapt for k=4 + 5 placements)

## Constraints
- **Single seed (42) only** for sweep phase. Post-hoc validation with 3 seeds if best placement differs >5% from P1.
- **Same total rank budget**: r=8 per block, 4 blocks, regardless of placement.
- **Model loss only** (not reconstruction MSE). 2000 training steps, standard OWT config.
- **PPL baseline**: float 57.65 (from cycle 115 manifest).

## Handoff
Design a 5-placement ablation (P1–P5). P1 is the known quantity (cycle 115: PPL 69.5 at seed 42, +74.3% recovery). Test P2–P5 at same computational budget. Track: PPL per placement, PPL delta vs P1, ranking stability. If best non-P1 placement beats P1 by >5%, escalate to item 143 (3-seed validation + rank sweep). Else, dismiss as placement-agnostic and move to interpretation (item 144).
