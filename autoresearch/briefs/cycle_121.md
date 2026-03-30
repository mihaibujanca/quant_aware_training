# Cycle 121: Train MoE Testbed — Tiny Mixtral (Item 138)

## Item
Train a working Tiny Mixtral model (15.5M params, 4 layers, 8 experts, top-2).
Cycle 118 reached PPL 78.6 after 10M tokens (lr=1e-4). Goal: 30M tokens with lr sweep.

## Context
Cycle 118 reached inconclusive (PPL not validated as "good"). This cycle validates the training setup:
- Sweep: lr ∈ {1e-4, 3e-4, 5e-4}
- Pick best by val PPL at 5M tokens, train full run to 30M
- If PPL > 50 at 30M: extend to 60M before failure
- CPU training, ~15 min/10M tokens

## Dependencies
- None (first in P0 MoE sequence)

## Key Files
- `experiments/moe_reap_pruning_tiny_mixtral.py` — has training code from cycle 118
- `autoresearch/results/cycle_118_moe_reap.json` — baseline PPL 78.6 at 10M tokens, saliency scores
- Training data: WikiText-2 (256 calibration seqs for later analysis)

## Constraints
- Must reach val PPL < 50 at 30M tokens to confirm (architecture sufficient)
- CPU only
- Save best checkpoint path in results JSON
- Don't stop early; finish all lr seeds even if first one succeeds

## Handoff
Write `autoresearch/results/cycle_121_moe_training.json`:
- best_lr, final_ppl_at_30m, checkpoint_path, train_curve_loss, validation_ppl_at_5m_per_lr
- Outcome: CONFIRMED if PPL < 50; REFUTED if PPL > 50 at 60M with all lrs
