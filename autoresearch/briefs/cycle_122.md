# Cycle 122: Train MoE Testbed — Tiny Mixtral (Dropout Fix)

## Item
Item 138. Train Tiny Mixtral with dropout to fix overfitting from cycle 121.

## Context
Cycle 121 achieved PPL 65.08 at 15M tokens, then degraded to 84.9 at 30M (clear overfitting). Initial lr sweep {0.0001, 0.0003, 0.0005} too narrow. Verdict: training recipe insufficient (not architecture failure).

## Dependencies
None. Independent training task.

## Key Files
- **Experiment template**: `experiments/moe_train_cycle121.py` (49 layers, reuse for cycle 122)
- **Previous results**: `autoresearch/results/cycle_121_moe_training.json` (best_lr=5e-4, PPL curve shows overfitting at 30M)
- **Tokenizer**: cycle 118 (already exists)
- **Status**: cycle 121 marked INCONCLUSIVE with dropout remedy

## Changes Required
1. Add `attention_dropout=0.1, hidden_dropout_prob=0.1` to MixtralConfig
2. Expand lr sweep: {3e-4, 5e-4, 1e-3, 2e-3} (4 values, was 3)
3. Add gradient accumulation for effective batch 16 (reduce per-step batch)
4. Train to 30M tokens (not 60M; 30M is decision boundary per cycle 121 curve)
5. Save checkpoint only if PPL < 50

## Success Criterion
- **Confirmed**: model reaches val PPL < 50 on WikiText-2
- **Refuted**: PPL > 50 at 30M tokens with dropout + lr sweep

## Handoff
New experiment file or modified cycle 121 file with dropout + wider lr sweep. Run single best seed to 30M tokens. Report: final PPL, val curve, best_lr, whether condition met.
