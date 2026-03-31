# Item
159 — Joint stacking-aware compression training — GPT-2 FFN k=4

# Context
- Cycle 138 (item 152): independently trained AE blocks don't compose. AE beats topk 3% single-block, but stacked PPL is 74 vs topk 71. The composition barrier.
- Cycle 148 (item 158): CE training objective matters more than architecture (linear AE = ReLU AE within 0.7%).
- Current working recipe: topk+lstsq+500 polish steps = PPL ~70 stacked.
- Polish trains c_proj weights of each block jointly under stacked CE loss, but only for 500 steps from an lstsq-initialized W_down.
- The hypothesis: training ALL compressed weights jointly under stacked CE loss for an extended budget (2000 steps) lets the optimizer learn to avoid composition-destructive errors that independent training produces.

# Dependencies
- None. Builds on cycle 138 (composition barrier established) and cycle 148 (CE objective is the active ingredient), both completed.

# Key Files
- `experiments/ae_stacking_k4_cycle138.py` — template: `train_params()`, `get_topk_params()`, CompressedFfnPack stacking, BottleneckFFN class
- `experiments/linear_ae_ablation_cycle148.py` — template: `BottleneckFFN` with `use_relu` flag
- `aleph/lm_geometric_compression.py` — teacher loading, probe corpus, swap infra, `_evaluate_model`

# Constraints
- Seed 42 only (screening). 3 conditions, ~45 min total.
- Blocks [1, 4, 7, 11] at 6x compression (k=512 of 3072). Consistent with all prior stacking experiments.
- Rank-r linear factorization: W_ffn (3072x768) ≈ W_enc (3072x64) @ W_dec (64x768), where r = d_bn = 64 for 6x compression ratio.
- SVD initialization: compute SVD of original c_proj weight, take top-64 components for W_enc/W_dec.
- 3 conditions: (a) joint from SVD init, 2000 steps, (b) joint from random init, 2000 steps, (c) topk+lstsq+polish 500 steps (baseline).
- AdamW lr=1e-4 with cosine schedule.
- 200 OWT eval sequences.

# Handoff
Write `experiments/joint_stacking_training_cycle149.py`. Three conditions, seed 42. Reuse `train_params()` pattern from cycle 138 but extended with cosine schedule. Linear factorization (no ReLU) per cycle 148 finding. Save JSON to `autoresearch/results/cycle_149_joint_stacking.json`.
