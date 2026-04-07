# Item
260 (P0): 6-block [1,2,3,4,7,8] — drop 5, add 3.

# Hypothesis
Block 5 is the hard block; replacing it with block 3 yields a 6-block compression near float (≤55.0 PPL), much better than cycle 221's [1,2,4,5,7,8]=59.5.

# Method
Copy `experiments/adam_6block_earlystop_cycle221.py` to `experiments/six_block_no5_add3_cycle250.py`. Change `BLOCKS=[1,2,4,5,7,8]` → `BLOCKS=[1,2,3,4,7,8]`. All other hyperparameters unchanged.

# Hyperparameters
N_CALIB=200, SEQ_LEN=128, K=512, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024], best-state restore on eval PPL.

# Metrics
Per-seed best eval PPL, mean/std across seeds, compare to cycle 221 (59.515) and cycle 245 5-block (53.826).

# Decision
- Confirmed if best_mean ≤ 55.0.
- Refuted if best_mean ≥ 58.0.
- Inconclusive if 55.0 < best_mean < 58.0.

# Risks
Block 3 may itself be hard; fallback hypothesis is that any 6-block config near 60 PPL indicates a count phenomenon (not specific to block 5).

# Deliverables
- `experiments/six_block_no5_add3_cycle250.py`
- `autoresearch/results/cycle_250_*.json`
- `autoresearch/briefs/cycle_250_result.md`
