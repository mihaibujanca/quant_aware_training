# Item
334 (P0): (ReLU², ReLU) and (LeakyReLU², ReLU²) curvature residual decomposition on parameter-golf 13L_640d checkpoint — cross-project, builds on cycle 318 eff_rank profile.

# Context
- Cycle 318 (item 333) closed CONFIRMED-collapse-generalizes: parameter-golf 13L_640d ReLU² has 3 blocks with eff_rank ≤ 20 (b3=4.89, b11=13.40, b12=6.94) and 2 with eff_rank ≥ 100 (b0=187.61, b1=160.08). Same metric on GPT-2 cycle 278 has GPT-2's leading concentration at b1/b2 (1.94/1.05). The U-shape is inverted in early stack but matches in late stack.
- Cycle 302 (item 312) implemented `experiments/curvature_residual_geometric.py` and CONFIRMED the unified curvature residual framework `Δy(x) = W_proj · (σ(z) − σ_lin(z))` on (GELU, ReLU) at GPT-2 small.
- Item 334 is the **trained-from-scratch** measurement: parameter-golf was actually trained with ReLU², so the pre-activation distribution z(x) is correct for ReLU² and LeakyReLU² residuals (unlike item 330 zero-shot patches on GPT-2).
- Open: does the (LeakyReLU², ReLU²) δ_σ residual concentrate to specific blocks, providing a geometric explanation of pgolf leaderboard's empirical "lrelu² > relu² by ~0.005 BPB" finding?

# Dependencies
- Cycle 318 / item 333 result (eff_rank profile per block — used to predict which blocks may show curvature concentration; b3 and b11/b12 are the candidates).
- `autoresearch/cross_project/parameter_golf.md` — REQUIRED READING per item 334's propagation rule.
- `parameter-golf/local_scripts/pgolf_eff_rank_census.py` (cycle 318) — has the working `load_model_v2` and reusable infrastructure (load checkpoint, hook MLP.__call__, val tokens, subprocess).

# Key Files
- `autoresearch/cross_project/parameter_golf.md` (REQUIRED READING)
- `parameter-golf/local_scripts/pgolf_eff_rank_census.py` (cycle 318 — reuse `load_model_v2`)
- `experiments/pgolf_eff_rank_census_cycle318.py` (cycle 318 — wrapper template)
- `experiments/curvature_residual_geometric.py` (cycle 302 — qat reference for the framework, but PyTorch — cannot import in pgolf venv)
- `parameter-golf/train_gpt_mlx.py` line 510: `MLP.__call__(self, x, slope=0.5)` — relu² body is `self.proj((F.relu(h)).square())` after `h = self.fc(x)`

# Constraints
Per item 334 verbatim:
- Three (σ, σ_lin) pairs: (a) ("relu2", "relu"), (b) ("lrelu2", "relu2"), (c) ("gelu", "relu") cross-reference.
- Per block: rank_95(Δy), top-5 SVs, top-5 left singular vectors, ‖Δy‖_F.
- Capture pre-activation `z = MLP.fc(x)` per block by monkey-patching MLP.__call__.
- δ(z) = σ(z) − σ_lin(z) elementwise via numpy on float64.
- Δy = δ @ proj.weight.T per token (MLX `proj.weight` shape is `(out_dim, hidden_dim)` so transpose).
- Subprocess timeout=1800 per call.
- Verdict tree: CONFIRMED-leak-localized if any block has (lrelu²,relu²) `rank_95(Δy) ≤ 50 AND ‖Δy‖_F ≥ 2× median`; CONFIRMED-leak-diffuse if `rank_95(Δy) ≥ 200` everywhere; REFUTED-no-difference if (lrelu²,relu²) ‖Δy‖_F within 5% of (relu²,relu) everywhere; else INCONCLUSIVE.

# Handoff
Plan agent: write `parameter-golf/local_scripts/pgolf_curvature_residual.py` reusing `load_model_v2` and the MLP hook pattern from cycle 318. Capture `fc(x)` per block by computing it inline in a monkey-patched `MLP.__call__` (to avoid double-fc cost). Apply δ_σ(z) over numpy. SVD per block. Then qat-side wrapper invokes 3 times.
