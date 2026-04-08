# Item
322 (P1): 128-wide MLP on 50D input — extend cycle 307.

# Context
- Cycle 307: 32-wide MLP on 50D input, eff_rank still collapsed to ~1 by layer 4.
- Hypothesis: wider network (128) gives more room for post-collapse eff_rank to stay elevated.

# Dependencies
None.

# Key Files
- experiments/higher_dim_eff_rank_cycle307.py — direct template.

# Constraints
- sklearn.datasets.make_classification(n_samples=2000, n_features=50, n_informative=10).
- MLP: input=50, hidden=128, depth=12, output=1.
- Train to ≥90% train acc.
- 3 seeds.
- Per-layer eff_rank at init + post-training.
- Verdict per item 322.

# Handoff
Plan agent: trivial width swap on cycle 307. New script `experiments/higher_width_mlp_cycle312.py`.
