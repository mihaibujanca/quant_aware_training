# Item
317 (P1): Higher-dim input testbed for eff_rank census.
Repeat cycle 295's eff_rank measurement on a 50D or 100D input task.

# Context
- Cycle 295: spirals/moons (2D input) already collapsed to eff_rank 1 at init — wrong control.
- Cycle 297: trajectory was UPWARD on 2D task.
- Open: on a higher-dim input, is there actually a collapse during training?

# Dependencies
- aleph.make_mlp infrastructure.

# Key Files
- experiments/spirals_moons_eff_rank_cycle295.py — baseline template.

# Constraints
- Use a 50D or 100D input task. Options:
  - Embed the 2D spirals into 50D via a random orthogonal projection (per MEMORY.md note: "100D embedding: cond(T)=564M, canonical space meaningless" — this is a real concern but here we're not doing canonical space, just eff_rank).
  - Use sklearn's make_classification with 50 features.
  - Use a small image dataset feature (MNIST pixels, 784D, but that's expensive).
- Simpler: use sklearn.datasets.make_classification(n_samples=2000, n_features=50, n_informative=10, n_classes=2).
- MLP: 50×{32→32→...→1} depth=12.
- 3 seeds. Train to ≥90% acc. Capture per-layer post-ReLU activations, compute eff_rank at init and post-training.
- Verdict per item 317.

# Handoff
Plan agent: adapt cycle 295 with 50D input. New script `experiments/higher_dim_eff_rank_cycle307.py`.
