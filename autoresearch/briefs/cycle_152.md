# Item
162 — Progressive stacking (train-then-freeze) — GPT-2 FFN k=4

# Context
- Cycle 138 (item 152): independently trained AE blocks don't compose. Single-block AE beats topk by 3%, but stacked AE loses by 4.7% (PPL 74.4 vs topk 70.7). The composition barrier.
- Cycle 149 (item 159): joint training (all 4 blocks simultaneously) with random init achieves PPL 69.76, beating topk 70.70 by 1.3%. But SVD init gets stuck at 78.2 (overfits to init and can't escape). Verdict: REFUTED (didn't beat topk by >=2%).
- Cycle 151 (item 161): corrections are 93-96% cascade compensation, not local error fixing. The "novel" subspace dominates in 4/4 blocks. This means inter-block error propagation is the dominant problem.
- Cycle 148: CE training objective matters more than architecture (linear AE = ReLU AE within 0.7%).

Progressive stacking trains blocks sequentially: compress block 1 alone (500 steps), freeze it, then compress block 4 with frozen block 1 already swapped in (500 steps), freeze, etc. Each block trains knowing the upstream is already compressed. This is intermediate between independent training (no awareness) and joint training (simultaneous awareness).

# Dependencies
- None. Fork cycle 149 script.

# Key Files
- `experiments/joint_stacking_training_cycle149.py` — fork this, adapt to sequential phases
- `aleph/lm_geometric_compression.py` — compression pipeline

# Constraints
- BLOCKS = [1, 4, 7, 11] — train in this order (upstream to downstream).
- Linear factorization rank-64 (same as cycle 149). Random init (SVD init gets stuck per cycle 149).
- 500 steps per block, 4 phases = 2000 total steps.
- Cosine schedule PER PHASE (not one global schedule).
- AdamW lr=1e-4.
- Seed 42 only (screening).
- 200 OWT eval sequences.
- Eval PPL after each phase (to track how stacking damage evolves).

# Handoff
Fork `experiments/joint_stacking_training_cycle149.py` to `experiments/progressive_stacking_cycle152.py`. Replace the single joint training loop with 4 sequential phases. Each phase: swap in new compressed block, train only that block's params for 500 steps, freeze, eval. Save JSON to `autoresearch/results/cycle_152_progressive_stacking.json`.
