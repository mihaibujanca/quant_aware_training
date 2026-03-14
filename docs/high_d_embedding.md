# High-Dimensional Embedding Experiments

## Motivation

Does geometric compression scale to high-dimensional inputs? Real-world SDFs operate in R^3, but many geometric learning pipelines embed inputs into higher dimensions (positional encoding, learned embeddings). We test whether geometric KAN — which matches the source MLP's Jacobian structure — maintains its advantages when the input dimension grows.

## Setup

**Embedding method**: A fixed random linear projection A ∈ R^{d×3} maps native 3D coordinates into R^d. The source MLP trains on embedded coordinates; all compressed models distill from this embedded-space MLP.

**Models compared**:
- **Source MLP** (h=256, depth=4): the teacher, trained with eikonal regularization
- **Geometric KAN** ([d, 8, 1], L-BFGS + Jacobian matching): preserves first-order structure
- **KAN-scratch** ([d, 8, 1], Adam on raw data): same architecture, no Jacobian constraint
- **Distilled KAN** (Adam on source outputs): distillation without structure
- **Small MLP** (matched param count): conventional baseline

**Metrics**:
- `test_mse`: point-wise reconstruction error on held-out test set
- `jacobian_mse`: MSE between model and source Jacobians
- `effective_rank`: Shannon entropy-based rank of the Jacobian singular value spectrum
- `top3_energy`: fraction of Jacobian energy in the top 3 singular values (should be ~1.0 for a function from R^3)

## Key Findings

### All methods improve with higher d

More dimensions give every model more representational room. MSE decreases monotonically from d=3 to d=100 for all methods.

### Geometric KAN uniquely preserves Jacobian rank

At d=100, geometric KAN maintains effective rank ~2.8 and top3_energy ~0.99 — correctly reflecting that the underlying function only varies along 3 directions in the 100D input space. All other methods spread energy across many directions:

| Model (d=100)    | eff_rank | top3_energy | rank_95 | test_mse   |
|------------------|----------|-------------|---------|------------|
| geometric_kan    | 2.8      | 0.995       | 3       | 2.8e-05    |
| distilled_kan    | 8.1      | 0.642       | 12      | 6.0e-05    |
| kan_scratch      | 9.8      | 0.599       | 15      | 5.1e-06    |
| small_mlp        | —        | —           | —       | 9.1e-06    |

The high rank for KAN-scratch is **not** an L-BFGS overfitting artifact — KAN-scratch uses Adam on 100k points and still has effective rank ~10. Without the Jacobian constraint, KAN splines spread sensitivity across many input directions because there's no pressure to be parsimonious: using small contributions from many directions is easier to optimize than concentrating signal into exactly 3.

### But MSE is 5.6× worse than KAN-scratch

Despite having "correct" Jacobian structure, geometric KAN's point-wise accuracy is significantly worse than KAN-scratch (which has "wrong" structure but better MSE).

### Tension: structural fidelity ≠ function accuracy

This is the core finding. Preserving first-order geometric structure (Jacobian matching) and minimizing point-wise error are genuinely competing objectives — not an optimization difficulty, but a fundamental tradeoff (confirmed by the lambda annealing results below).

## Lambda Annealing Results

**Hypothesis**: annealing the Jacobian constraint (start at λ=0, ramp up) lets L-BFGS find a good MSE basin first, then enforces structure.

Tested schedules on torus at d=3 and d=100:
- `lambda=0` — pure MSE distillation (no Jacobian term)
- `lambda=1, constant` — current behavior (Jacobian matching throughout)
- `lambda=1, warmup` — linear ramp 0→λ over training
- `lambda=1, cooldown` — linear ramp λ→0 over training

Run `experiments/jacobian_annealing_sweep.py` to reproduce.
Archived results: `runs/high_d_embedding_2026-02-23/annealing_results.csv`,
`runs/high_d_embedding_2026-02-23/annealing_results_short_run.csv`,
`runs/high_d_embedding_2026-02-23/annealing_results_overnight.csv`.

### Results at d=3 (native)

All schedules perform similarly — the Jacobian is naturally rank ~2.5 in 3D, so the constraint doesn't fight the optimizer. Cooldown gets marginally best MSE (0.00108 vs 0.00120 for lambda=0).

### Results at d=100

| Schedule        | test_mse   | jac_mse  | eff_rank | top3_energy |
|-----------------|------------|----------|----------|-------------|
| lambda=0        | **0.000018** | 0.700  | 16.3     | 0.50        |
| constant (λ=1)  | 0.000317   | **0.0006** | **2.9** | **0.98**   |
| warmup (0→λ)    | 0.000893   | 0.0019 | 4.1      | 0.91        |
| cooldown (λ→0)  | 0.000460   | 0.0070 | 7.4      | 0.70        |

In this run, **warmup is worse than constant** (2.8× higher MSE), suggesting that ramping the Jacobian term can pull the model away from a low-MSE basin.

**Cooldown is the interesting case in this run**: best constrained MSE (0.000460), with partial structural persistence after removing the constraint (rank 7.4 vs 16.3 unconstrained).

**MSE vs structure is a Pareto tradeoff**, not an optimization problem. Confirmed across both runs (short and full).

### Overnight confirmation (2026-02-23)

Cross-run check at `d=100`:

| Run file | Best constrained schedule | Best constrained MSE | Ratio vs `lambda=0` |
|---|---|---:|---:|
| `annealing_results.csv` | `lambda=1 constant` | 3.17e-4 | 17.9x |
| `annealing_results_short_run.csv` | `lambda=1 cooldown` | 5.70e-3 | 46.0x |
| `annealing_results_overnight.csv` | `lambda=1 warmup` | 5.36e-5 | 4.65x |

Across all three runs, `lambda=0` remains the best absolute MSE at `d=100`.
Annealing changes where you land on the MSE-vs-structure tradeoff curve, but does not remove the tradeoff.
This matches the `kan_last_stand` verdict: under favorable budgets, KAN/Jacobian-constrained variants are still not competitive for pure reconstruction fidelity.

## Interpretation

The Jacobian constraint doesn't just affect optimization dynamics; it changes which solutions are preferred. Across runs, warmup/cooldown can reshuffle the best constrained schedule, but none closes the `lambda=0` MSE gap at high dimension. Structure and accuracy remain competing objectives at high d.

Geometric structure preservation may still be valuable for:
- Physically meaningful gradients (SDF normals, ray marching)
- Out-of-distribution robustness (untested)
- Downstream tasks that depend on first-order structure

But for pure reconstruction fidelity, unconstrained optimization wins.

## Open Questions

1. **Does rank compress with longer training?** KAN-scratch's effective rank might decrease as training converges, naturally finding low-rank solutions.
2. **Does low-rank Jacobian help out-of-distribution?** Test on points outside the training domain.
3. **Is there a sweet spot?** Perhaps λ=0.1 or λ=0.01 gives structure-aware training without crippling MSE.
4. **Adam vs L-BFGS for geometric KAN?** L-BFGS memorizes small fitting sets. Adam with larger batches might avoid the overfitting issue entirely.
5. **Connection to grokking and GrokAlign** ([arXiv:2506.12284](https://arxiv.org/abs/2506.12284)). That paper shows grokking corresponds to Jacobian rank collapse, accelerated by Jacobian norm regularization — structurally similar to our Jacobian matching. KAN-scratch's rank ~10 at d=100 suggests it hasn't "grokked" — it interpolates well without discovering the true 3D structure. Key difference: SDF regression (MSE on distances) penalizes everywhere, leaving no loss plateau for the network to freely reorganize its representations. Classification on sign(SDF) might exhibit rank collapse more readily, since cross-entropy stops caring once you're on the right side of the boundary. Testable: train KAN-scratch on sign(SDF) classification and check whether rank decreases with extended training.

## Projected KAN

**Hypothesis**: The source MLP *does* discover the low-dimensional structure — its Jacobian has effective rank ~3 at d=100. The failure of geometric KAN is that it must simultaneously discover the projection *and* learn the function. Decoupling these two problems should recover both MSE and structural fidelity.

**Pipeline**: MLP (R^d → R) → extract projection P via Jacobian SVD → KAN (R^k → R) on projected inputs.

```
P = top-k right singular vectors of ∇f(X_profile)   # [d, k]
X_proj = X @ P                                        # [N, k]
grad_proj = ∇f(X) @ P                                # [N, k]  (chain rule)
KAN fits g(z) ≈ f(Pz) with Jacobian matching in projected space
```

**Implementation**: `jacobian_projection()` + `ProjectedModel` in `aleph/geometric_compression.py`. Enable via `methods=(..., "projected_kan")` and `projection_rank=k` in `GeometricCompressionConfig`.

**Sweep**: `experiments/projected_kan_sweep.py` — dims ∈ {3, 10, 50, 100}, ranks ∈ {2, 3, 5}.
Archived results: `runs/high_d_embedding_2026-02-23/projected_kan_results.csv`.

**Expected outcomes**:
- d=3, k=3: projected KAN ≈ geometric KAN (same effective input dim, projection is identity-like)
- d=100, k=3: projected KAN MSE close to KAN-scratch with Jacobian rank ≈ 3 (both by construction)
- Compression ratio much better than geometric KAN at high d (k×8 splines vs d×8)

## Last-Stand Overnight Verdict (2026-02-23)

To settle the question decisively, we ran a favorable low-capacity KAN sweep across multiple shapes:

- Script: `experiments/kan_last_stand_overnight.py`
- Shapes: `torus`, `sphere`, `capsule`, `box`, `cylinder`, `notebooks/meshes/bunny.ply`
- Capacity: KAN `[3, 8, 1]` (~335 params), small MLP ~316 params
- Training budget: `epochs=1000`, `kan_steps=1000`, `n_points=120000`, `n_fit=25000`, `n_profile=25000`

Artifacts: `runs/kan_last_stand_2026-02-23/kan_last_stand_results.csv`,
`runs/kan_last_stand_2026-02-23/kan_last_stand_summary.json`,
`runs/kan_last_stand_2026-02-23/kan_last_stand_summary.md`.

### Outcome

`best_kan_mse / small_mlp_mse` by shape:

| Shape | Best KAN / Small MLP |
|---|---:|
| torus | 9.312 |
| sphere | 0.480 |
| capsule | 5.241 |
| box | 36.080 |
| cylinder | 4.651 |
| bunny | 1.405 |

Aggregate:

- Best-KAN wins (<= 0.95x small MLP): **1 / 6**
- Median(best_kan / small_mlp): **4.651**
- Geometric mean(best_kan / small_mlp): **4.204**
- Verdict: **PIVOT**

### Decision

For this task (SDF reconstruction fidelity under small parameter budgets), the KAN direction is not competitive enough. Keep KAN work only if the objective changes to prioritize gradient-structure fidelity over MSE.
