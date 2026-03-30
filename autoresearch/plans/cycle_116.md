# Cycle 116 Plan: Extended-Polish Ablation — GPT-2 FFN

**Item:** 132 | **Script:** `experiments/extended_polish_ablation_gpt2.py` (new)

## Criteria (verbatim from agenda)
- **Confirmed if:** C (corrections) outperforms both B and D by >=5% relative PPL.
- **Refuted if:** B or D reaches within 2% of C.
- **Inconclusive if:** 2-5% gap, or D converges to C given enough steps.

## Pre-flights
**P1 -- Prior:** Cycle 115 CONFIRMED corrections work. This tests whether the mechanism
is the correction structure or just extra optimization budget. Novel question.
**P2 -- Cost:** 4 conditions x 3 seeds. A reuses 500 Adam. B/D are 2500 steps. C is
500 + 2000. ~90 min MPS total.
**P3 -- Abort:** Float PPL outside [25, 65]. Any condition PPL > 200.
**P4 -- Block config:** Use [1,4,7,11] per agenda (not [7,5,4,1] from cycle 115).

## Implementation

1. **Baseline.** Load GPT-2, eval 200 seqs OWT. Record float_ppl. Abort if outside [25,65].

2. **Compress k=4.** Blocks [1,4,7,11] at 6x, topk+lstsq. Save compressed model state
   (before any Adam polish) for reuse across conditions.

3. **Condition A: 500 Adam on W_down.** Standard polish: 500 Adam steps (lr=1e-4) on
   W_down (c_proj.weight + bias) per block, frozen everything else. Eval → ppl_A.

4. **Condition B: 2500 Adam on W_down.** Same as A but 2500 steps total. Log PPL at
   steps {500, 1000, 1500, 2000, 2500} to capture learning curve. Eval → ppl_B.

5. **Condition C: 500 Adam + r=8 corrections 2000 steps.** Polish W_down 500 steps
   (identical to A). Then freeze everything, insert C_l(x) = x + U_l @ (V_l @ x) with
   r=8, U_l init 0. Train corrections 2000 steps (Adam lr=1e-4, batch=4, seq_len=128,
   next-token CE). This is the cycle 115 recipe. Eval → ppl_C.

6. **Condition D: 500 Adam + 2000 more Adam on W_down.** Polish W_down 500 steps
   (identical to A). Then continue polishing W_down for 2000 more steps (same optimizer
   reset, lr=1e-4). Matched total optimization budget to C but no correction params.
   Log PPL at steps {500, 1000, 1500, 2000, 2500} for learning curve. Eval → ppl_D.

7. **Param accounting.** Log for each condition:
   - A: 0 extra params beyond compressed model
   - B: 0 extra params (same W_down, more steps)
   - C: 4 * 2 * 768 * 8 = 49,152 correction params
   - D: 0 extra params (same W_down, more steps)

8. **Verdict logic.**
   - relative_gap_B = |ppl_C - ppl_B| / ppl_C * 100
   - relative_gap_D = |ppl_C - ppl_D| / ppl_C * 100
   - If ppl_C < ppl_B AND ppl_C < ppl_D AND both gaps >= 5%: CONFIRMED
   - If ppl_B within 2% of ppl_C OR ppl_D within 2% of ppl_C: REFUTED
   - If gaps between 2-5%: INCONCLUSIVE

9. **Output.**
   - JSON: `autoresearch/results/cycle_116_extended_polish_ablation.json`
   - Plot 1: Bar chart of final PPL for A/B/C/D with error bars (3 seeds)
   - Plot 2: Learning curves for B and D (PPL at {500,1000,1500,2000,2500})
   - Manifest: `autoresearch/results/cycle_116_manifest.txt`

## Notes
- Block order [1,4,7,11] per agenda item 132, not [7,5,4,1] from cycle 115.
- Conditions A, B, D share the same compressed checkpoint (pre-polish). Each condition
  starts from the same lstsq initialization. Seeds control data sampling order.
- For D, the optimizer is reset at step 500 to match the "500 + 2000 more" framing.
  This ensures D's second phase starts from the same W_down state as C's second phase.
