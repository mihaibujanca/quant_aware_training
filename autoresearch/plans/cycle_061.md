# Cycle 061 Plan: Gradient Polish Scaling (Item 77)

## Goal
Extend cycle 059's polish sweep from {0,10,25,50} to {100,200,500} steps, and add joint polish (all 4 W_down matrices updated simultaneously).

## File
Create `experiments/gradient_polish_scaling_gpt2.py` based on `experiments/gradient_polish_stacked_gpt2.py`.

## Changes from cycle 059

1. **Config**: `POLISH_STEPS_LIST = [0, 50, 100, 200, 500]` (0 and 50 are replication baselines).
2. **Joint polish function** `_polish_joint`: unfreeze `c_proj.weight` (and bias) on ALL 4 compressed blocks simultaneously. Single Adam optimizer over all 4 param sets. Same mini-batch loop as `_polish_block_v2`.
3. **Outer loop**: for each seed, for each n_steps, run two variants:
   - `independent`: polish each block sequentially (existing `_polish_block_v2`, same as cycle 059).
   - `joint`: polish all 4 blocks simultaneously via `_polish_joint`.
4. **Results structure**: `polish_ppls[seed][variant][n_steps] = ppl`, variants = `{"independent", "joint"}`.
5. **Aggregation**: compute mean/std over 3 seeds for each (variant, n_steps) pair. Relative improvement is vs the 0-step baseline (shared across variants).
6. **Verdict logic**:
   - CONFIRMED if 500 steps (either variant) achieves <= +8% PPL degradation OR joint beats independent by >= 5pp at matched step count.
   - REFUTED if 200 steps shows < 2pp improvement over 50 steps (plateau reached).
   - INCONCLUSIVE otherwise.

## Key implementation details

- Fresh `swap_compressed_ffns` per (seed, variant, n_steps) to avoid state leakage.
- `_polish_joint` implementation: iterate `model.transformer.h[block_id].mlp.c_proj.weight` for each block_id in TARGET_BLOCKS, collect into one param list, one Adam optimizer.
- Same constants: `POLISH_LR=1e-4`, `BATCH_SIZE=4`, `SEQUENCE_LENGTH=128`, `PROBE_HOLDOUT_SEQS=200`, `EVAL_VAL_SEQS=200`, `SEEDS=[42,123,7]`.
- Track per-step loss curves for both variants to visualize convergence.

## Plot
Two-panel figure (`cycle_061_gradient_polish_scaling.png`):
- Left: PPL delta (%) vs steps, two lines (independent, joint) with error bars.
- Right: relative improvement (%) vs steps, two lines, with horizontal lines at cycle 059 criteria thresholds.

## Output artifacts
- `autoresearch/results/cycle_061_gradient_polish_scaling.json`
- `autoresearch/results/cycle_061_gradient_polish_scaling.png`

## Verification
- 0-step PPL must match cycle 059 baseline (~75.06 mean, +30.2%).
- 50-step independent must replicate cycle 059 (~68.29 mean, +18.4%).
- If 50-step replication deviates > 2pp from cycle 059, flag and investigate.
