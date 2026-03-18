# Task
Item 20: Single-affine collapse quality — toy ReLU. For each hidden layer in a trained spirals ReLU MLP, fit a single affine map A,b via least squares on probe data, then measure normalized residual r on held-out data. Determine whether any layer is functionally linear (r < 0.15).

# Hypothesis
Verbatim from agenda:
- Confirmed if: at least one hidden layer achieves r < 0.15 on held-out probe data, indicating near-linear behavior.
- Refuted if: all layers have r > 0.5, indicating that single-affine collapse is uniformly poor and the function uses its nonlinearity everywhere.
- Inconclusive if: r is moderate (0.15–0.5) and does not correlate with layer depth or neuron stability statistics.

# Pre-flight Results
- P1 PASS: no prior cycle tested single-affine collapse on toy ReLU. Cycle 009 tested GPT-2 Jacobian variance (item 19), distinct.
- P2 PASS: item 20 depends on nothing.
- P3 PASS: fully specified empirical experiment with clear thresholds.
- P4 PASS: no program/prompt changes needed.

# Steps
1. **Create experiment script** `experiments/single_affine_collapse_toy.py`:
   - Train spirals ReLU MLP (2→64→64→64→64→1, lr=0.001, 2× scale, 3 turns, n=2000) using `aleph.datasets.make_embedded_classification_benchmark` with `IdentityEmbedding` and `aleph.models.MLP`. Train to >92% accuracy (cap at 3000 epochs, early stop if achieved).
   - Split train data: first 50% = fit probe, second 50% = held-out probe.
   - For each hidden layer ℓ (1..4): collect pre-activations h_{ℓ-1} and post-ReLU activations h_ℓ on fit data. Solve least-squares: A, b = argmin ||h_ℓ − A·h_{ℓ-1} − b||_F (use `torch.linalg.lstsq` with bias column).
   - Evaluate on held-out: compute r_ℓ = ||h_ℓ − A·h_{ℓ-1} − b||_F / ||h_ℓ − mean(h_ℓ)||_F.
   - Also compute: fraction of stable-0 (dead) and stable-1 (always-active) neurons per layer on fit data, to correlate with r.
   - Save JSON artifact: `autoresearch/results/cycle_010_single_affine.json` with per-layer r, stability stats, train accuracy.
   - Print verdict: confirmed/refuted/inconclusive per agenda criteria.
2. **Run experiment**: `python experiments/single_affine_collapse_toy.py`
3. **Write summary** to `autoresearch/results/cycle_010_summary.md`: per-layer r table, verdict, stability correlation.
4. **Update agenda**: move item 20 from NEXT to PAST EXPERIMENTS with cycle 010, commit hash, artifact path, outcome.
5. **Commit**: `[autoresearch] cycle 010: single-affine collapse quality on toy ReLU`

# Validation
- Train accuracy must be >92% before measuring residuals (undertrained model is meaningless).
- r must be computed on held-out data only (not fit data) — check that fit/held-out split indices are disjoint.
- Sanity check: r for an identity mapping (A=I, b=0) should be higher than fitted A,b (least squares must improve).
- JSON artifact must contain per-layer r values and the final verdict string.

# Artifacts
- `experiments/single_affine_collapse_toy.py` — experiment script
- `autoresearch/results/cycle_010_single_affine.json` — raw per-layer results
- `autoresearch/results/cycle_010_summary.md` — human-readable summary with verdict

# Fallbacks
- If training fails to reach 92% accuracy: reduce to 3 layers or 32 hidden (simpler config), document the config change.
- If all r values fall in the inconclusive range (0.15–0.5): additionally test with moons dataset (simpler separation) and a 2-layer network (less nonlinearity). Report both.
- If lstsq is numerically unstable (condition number > 1e10): add Tikhonov regularization (ridge) with λ=1e-6 and report the regularized r alongside unregularized.
