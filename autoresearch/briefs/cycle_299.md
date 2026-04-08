# Item
309 (P1): Cycle 280 retrain decomposition — what does Adam's 8.6 PPL recovery move?
Re-run cycle 280 with extra logging: save B(t) at each 100 steps, compute drift cosines.

# Context
- Cycle 280: block 1 rank-2 retrain, init 57.66 → best 49.03. 8.6 PPL recovery from B + bias fine-tuning.
- Open: does B rotate within SVD subspace (stay-put) or drift into new rank-2 operator?

# Dependencies
- Cycle 280 script.

# Key Files
- experiments/block1_lowrank_cycle280.py.

# Constraints
- Single seed (42 is enough for decomposition).
- Save B(t) every 100 steps during the 4000-step Adam retrain.
- Per snapshot: Frobenius angle B(t) vs B(0), row-space cosine with Vh_original.
- Compute W(t) = A @ B(t), its top-2 right SVs.
- Plot drift trajectory vs eval PPL.

# Handoff
Plan agent: adapt cycle 280 script for extra logging. New script `experiments/cycle280_decomp_cycle299.py`.
