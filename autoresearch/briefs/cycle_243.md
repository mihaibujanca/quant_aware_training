# Item
253 (P2): Probe the 6-block 59.5 ceiling — per-block PPL decomposition.
Load cycle 221's best-state joint Adam snapshot for BLOCKS=[1,2,4,5,7,8] K=512. Starting from 59.515 PPL, measure eval PPL contribution of each compressed block by restoring the float FFN for one block at a time. Report 6 numbers (ppl_if_block_X_restored).

# Context
- Cycle 221 (INCONCLUSIVE): 6-block joint Adam → 59.515 ± 0.094.
- Cycle 242 (CONFIRMED): late blocks resist correction more than early blocks. The 59.5 floor is likely dominated by late-ish blocks (7, 8 in [1,2,4,5,7,8]).
- This cycle does direct per-block attribution to find which block(s) contribute most.

# Dependencies
- Cycle 221's best-state snapshot. If not on disk, refit seed 42 only (~10 min).

# Key Files
- experiments/adam_6block_earlystop_cycle221.py — for refit if needed.
- autoresearch/results/cycle_221_*.json — check if it has best-state weights.

# Constraints
- Eval set: same as cycle 221 (`all_ids[200:300]`, SEQ_LEN=128).
- 1 seed (42) for the snapshot.
- For each block X in [1,2,4,5,7,8]: restore float FFN for block X, keep other 5 compressed at their best-state. Eval PPL.
- Also record per-block mean residual MSE at the cycle 221 best state.
- Confirmed if any single block X has ppl_if_X_restored ≤ 56.0 (one block accounts for ≥3.5 of the 6.0 PPL 3→6 gap).
- Refuted if all 6 ΔPPL entries lie in [0.5, 1.5].
- Inconclusive otherwise.

# Handoff
Plan agent: design via reusing cycle 221 setup, refitting at seed 42 only if needed, then doing the per-block restoration scan. New script `experiments/6block_per_block_attribution_cycle243.py`.
