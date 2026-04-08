# Item
313 (P2): Apply unified framework to (ReLU², ReLU) and (LeakyReLU², ReLU²) on parameter-golf.

# Context
- Item 312 / cycle 302 implemented the unified framework in PyTorch.
- Item 313 is a cross-project pointer: run the framework on parameter-golf checkpoints.
- Parameter-golf uses MLX (see hat_mlx.py, train_gpt_mlx.py), not PyTorch.

# Pre-flight
- P1: is this a duplicate? No, item 313 is new.
- P2: dependencies? Cycle 302 (done).
- P3: feasible in one cycle?
- P4: modify program.md or prompts? No.

# Feasibility analysis
- Parameter-golf uses MLX. The unified framework (cycle 302 curvature_residual_geometric.py) is PyTorch-only.
- Porting MLX checkpoint → PyTorch requires nontrivial state-dict translation and architecture match.
- Per CLAUDE.md hard rule: "Always use PyTorch for numerical code" — MLX is not PyTorch.
- Item 313 itself says "cross-project — runs against parameter-golf checkpoints, not GPT-2" and "the parameter-golf project doesn't have an autoresearch loop, so the experiment is run manually".
- The item is explicitly labeled as a MANUAL experiment to be run separately from the autoresearch loop.

# Recommendation
PLAN INVALID. Item 313 is a manual cross-project pointer; running it inside the autoresearch loop violates both the hard "PyTorch only" rule and the item's own "run manually" caveat. Move to PAST as inconclusive with reason "cross-project manual item, not loop-compatible".

# Handoff
Plan agent: write PLAN INVALID.
