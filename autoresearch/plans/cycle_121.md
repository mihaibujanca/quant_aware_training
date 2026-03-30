# Plan — Cycle 121: Train MoE Testbed — Tiny Mixtral (Item 138)

## Goal
Train Tiny Mixtral to val PPL < 50. Sweep lr, pick best, train to 30M tokens.

## Time budget
lr sweep at 1M tokens: ~5 min. Full train at 30M: ~45 min. Total: ~50 min.
This exceeds the 30-min cap, but the item explicitly specifies 30M tokens at ~15 min/10M.
Mitigation: sweep at 1M tokens (not 5M) to save ~20 min. Accept the 30M train time.

## Approach
Write `experiments/moe_train_cycle121.py` (new file — never overwrite ran experiments).

### Phase 1: lr sweep (1M tokens each)
- lrs = {1e-4, 3e-4, 5e-4}
- Reuse cycle 118's tokenizer from `experiments/results/cycle_118_tokenizer/`
- Same MixtralConfig (hidden_size=256, 4 layers, 8 experts, top-2, vocab=4096)
- BATCH_SIZE=4, SEQ_LEN=256, cosine schedule, 5% warmup, weight_decay=0.01
- Evaluate val PPL at 1M tokens for each lr
- Print table: lr → val_ppl_at_1M
- Pick lr with lowest val PPL

### Phase 2: full training (30M tokens)
- Train fresh model with best lr for 30M tokens
- Evaluate val PPL every 5M tokens (print + store curve)
- If PPL < 50 at 30M: CONFIRMED, save checkpoint
- If PPL > 50 at 30M: extend to 60M tokens (same cosine schedule, recomputed for 60M)
- Save checkpoint at best val PPL seen during training

### Output
`autoresearch/results/cycle_121_moe_training.json`:
```json
{
  "sweep_results": {"1e-4": ppl, "3e-4": ppl, "5e-4": ppl},
  "best_lr": float,
  "final_ppl_30m": float,
  "final_ppl_60m": float or null,
  "checkpoint_path": "experiments/results/cycle_121_tiny_mixtral.pt",
  "val_ppl_curve": [[tokens_M, ppl], ...],
  "outcome": "CONFIRMED" or "REFUTED"
}
```

## Key code reuse from cycle 118
- Tokenizer loading (lines 80-86)
- pack_texts function (lines 91-101)
- MixtralConfig (lines 109-121)
- Training loop structure (lines 127-200), modified for:
  - Parameterized lr and target_tokens
  - `train_one_run(lr, target_tokens) -> (model, val_ppl, train_curve)` function
  - Val PPL eval every 5M tokens during full train
  - Checkpoint saved with best-ppl tracking

## Verification
- Sweep: 3 PPL values printed; best lr selected automatically
- Full train: val PPL curve monotonically decreasing (mostly)
- Final: PPL < 50 → CONFIRMED; PPL > 50 at 60M → REFUTED
- Checkpoint file exists and loads correctly (quick torch.load check)

## Pre-flight
- [x] P1 single-item: yes
- [x] P2 not duplicate: cycle 118 was inconclusive, this extends it
- [x] P3 deps met: none
- [x] P4 falsifiable: PPL < 50 threshold
- [x] P5 time: ~50 min (over cap, justified by item spec)
- [x] P6 type=implementation: single run per lr, no multi-seed
- [x] P7 scope: training only, no pruning/analysis
