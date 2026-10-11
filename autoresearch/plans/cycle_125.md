# Cycle 125 Plan: Correction Placement Sweep (Item 142)

## Task
Ablate WHERE to place rank-8 corrections on k=4 FFN-compressed GPT-2 (blocks [7,5,4,1]).

## Hypothesis
Cycle 123 found V dominates early layers, QK late. P4 (V+O) or P3 (Q+K) may beat the residual-stream baseline P1 by routing corrections closer to the error source.

## Pre-flight Results
- P1 (duplicate): no prior cycle ran this item.
- P2 (deps): item 140 (cycle 123) completed. No hard blockers.
- P3 (scope): 5 placements x 1 seed = 5 training runs at 2000 steps. Single-cycle.
- P4 (program.md): no modifications needed.
- P5 (null baseline): uncorrected PPL = 95.95 (seed 42), float = 57.65. Degradation = 38.30.
- P6 (architecture): GPT-2 uses fused c_attn (768->2304) for QKV. P3/P4 must hook c_attn output and add corrections to the Q/K/V segments, NOT hook separate Q/K/V modules (they don't exist).
- P7 (seeds): agenda says seed 42 only for sweep. Brief says escalate to 3 seeds if best non-P1 beats P1 by >5%. Run all 5 placements at seed 42; if triggered, re-run top 3 with seeds [42, 123, 7].

## Param Budget (Fairness)
Fixed budget: 12,288 params per block (= 2 x 768 x 8). All placements must match.
- P1: r=8, one correction per block. 12,288/block.
- P2: r=8, one correction per block. 12,288/block.
- P3: Q + K corrections, r=4 each. 2x(2x768x4) = 12,288/block.
- P4: V + O corrections, r=4 each. 12,288/block.
- P5: P1 + P2, r=4 each. 2x(2x768x4) = 12,288/block.

## Steps
1. Copy `experiments/cascade_correction_k5_gpt2.py` to `experiments/placement_sweep_k4_cycle125.py`.
2. Change FFN_BLOCKS to [7, 5, 4, 1] (k=4). Remove k=5 block 6.
3. Implement 5 placement classes, all subclassing a common interface:
   - **P1**: `LowRankCorrection` on block output (existing code). r=8. Hook: `block.register_forward_hook`.
   - **P2**: `LowRankCorrection` after attention residual, before ln_2. r=8. Hook: register a forward pre-hook on `block.ln_2` that applies correction to its input.
   - **P3**: `QKCorrection` — hook `block.attn.c_attn` output. Split output into Q/K/V segments ([:,:,:768], [:,:,768:1536], [:,:,1536:]). Add U_q @ V_q @ x to Q segment, U_k @ V_k @ x to K segment. r=4 each. x = hook input (the ln_1 output).
   - **P4**: `VOCorrection` — two hooks: (a) hook c_attn output, add U_v @ V_v @ x to V segment (r=4), (b) hook c_proj output, add U_o @ V_o @ x to O output (r=4). x = ln_1 output for V, attention output for O.
   - **P5**: P1(r=4) + P2(r=4) combined.
4. For each placement, train corrections 2000 steps, lr=1e-4, Adam, seed 42.
5. Evaluate PPL on 200 OWT sequences. Record: PPL, delta vs P1, delta vs uncorrected.
6. If any non-P1 beats P1 by >=5% relative: re-run top 3 placements with seeds [42, 123, 7].
7. Save results JSON, loss curves plot, PPL bar chart.
8. Apply verdict criteria, update agenda.md, commit.

## Validation
- Sanity: all placements must have PPL < uncorrected (95.95). If any placement has PPL >= uncorrected, the hook is broken.
- P1 reproduction: must match cycle 115 seed-42 value (69.53) within 2 PPL.
- Param count: print and verify 49,152 total (12,288 x 4 blocks) for every placement.

## Criteria (verbatim)
- **Confirmed** if: any non-P1 placement beats P1 by >=5% relative PPL (current placement is suboptimal).
- **Refuted** if: P1 is within 2% of the best alternative (residual stream correction is already near-optimal placement).
- **Inconclusive** if: 2-5% differences or placement ranking is not robust.

## Artifacts
- `experiments/placement_sweep_k4_cycle125.py`
- `autoresearch/results/cycle_125_placement_sweep.json`
- `autoresearch/results/cycle_125_ppl_bar.png`
- `autoresearch/results/cycle_125_loss_curves.png`

## Fallbacks
- If P3/P4 hooks conflict with HF internals (gradient issues with fused c_attn), fall back to wrapper module that replaces `block.attn.forward` entirely, calling original attention with modified QKV.
- If P1 reproduction fails (>2 PPL from 69.53), debug compression pipeline first — do not proceed with sweep on a broken baseline.
