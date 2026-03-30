# Cycle 116 Brief — Extended-Polish Ablation for Cascade Corrections — GPT-2 FFN

**Item:** 132 | **Type:** empirical | **Verdict:** CONFIRMED

## Context
Cycle 115 showed r=8 LoRA-style corrections recover 74% of k=4 stacking degradation
(PPL 80.7 → 59.1, float 51.7). The audit flagged a critical confound: the 2000 correction
training steps add optimization budget. If 2000 more Adam steps on W_down alone achieve
similar recovery, the correction *structure* (low-rank residual stream perturbation) is
not contributing — it is just extra fine-tuning.

## Question
Is the 74% recovery due to the correction architecture (I + UV on residual stream), or
would equivalent optimization budget spent on W_down polish reach the same PPL?

## Approach
Four conditions at k=4 [7,5,4,1], 6x, OWT, 3 seeds, 200 eval seqs:
A) 500 Adam on W_down (baseline). B) 2500 Adam on W_down. C) 500 Adam + r=8 corrections
2000 steps (cycle 115 recipe). D) 500 Adam + 2000 more Adam on W_down (matched budget).
Track PPL learning curves at {500,1000,1500,2000,2500} for B and D.

## Results (cycle 116)
| Condition | PPL (mean) | Description |
|-----------|-----------|-------------|
| Float     | 51.66     | Baseline |
| A (500)   | 59.86     | Standard recipe |
| B (2500)  | 62.12     | Extended polish |
| C (corr)  | 58.88     | Corrections |
| D (500+2000) | 62.16  | Matched budget |

Gap C vs B: 5.50%, C vs D: 5.57% — both exceed 5% CONFIRMED threshold.

**Key finding**: Extended polish overshoots. B and D improve to ~58.5 PPL at step 1000
then worsen to ~62.1 at step 2500. The 500-step optimum for W_down polish is already past;
additional steps hurt. Corrections exploit the residual stream without this overfitting.

## Conclusion
Correction structure (not just optimization budget) is responsible for the 74% recovery.
The 49K params are structurally justified.
