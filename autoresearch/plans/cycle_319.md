# Task
Item 334 (P0): (ReLU², ReLU) and (LeakyReLU², ReLU²) curvature residual decomposition on parameter-golf 13L_640d ReLU² checkpoint via cycle 312 unified framework.

# Hypothesis (verbatim)
(a) (ReLU², ReLU) Δy has rank_95 ≪ d_model (640) in some blocks but the rank profile is different from GPT-2's because the curvature region is z>0 not z∈(-3,3). (b) (LeakyReLU², ReLU²) Δy is concentrated in 1-3 specific blocks and has rank_95 ≤ 50 in those blocks (the leaderboard leak is geometrically localized). (c) (GELU, ReLU) cross-reference is similar in shape to GPT-2 cycle 302.

# Pre-flight Results
- P1 not duplicate. Cycle 318 measured eff_rank only; this measures curvature residual subspace per block. Different metric.
- P2 cross-project doc exists; cycle 318 already wrote `load_model_v2`. Checkpoint loads.
- P3 ≤30 min: 3 subprocess calls × ~6 min each (forward pass + per-block SVD on 25600x640).
- P4 no program/prompt mods.

# Steps
1. **Step 0** (REQUIRED): re-read `autoresearch/cross_project/parameter_golf.md` sections 1, 3, 4, 5, 6 (already read in cycle 318; re-state in plan that gotchas 1, 3, 4, 6, 8 still apply).
2. **Step 1**: write `/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_curvature_residual.py`.
   - Args: `--checkpoint`, `--sigma {relu2,lrelu2,gelu}`, `--sigma-lin {relu,relu2}`, `--n-tokens` (default 25600), `--seq-len` (default 1024).
   - Reuse `load_model_v2` from `pgolf_eff_rank_census.py` (import or copy verbatim).
   - Monkey-patch `MLP.__call__` to compute `h = self.fc(x)` once, then capture `np.array(h.astype(mx.float32))` into `captured_z[block_idx[id(self)]]`. Then complete the original forward (relu² body: `self.proj((nn.relu(h)).square())`). DO NOT call original_mlp_call afterwards (would duplicate fc work).
   - For each block: stack the captured z chunks → `Z ∈ R^(N, hidden_l)`. Apply σ and σ_lin elementwise via numpy:
     - relu(z) = max(z, 0)
     - relu2(z) = max(z, 0) ** 2
     - lrelu2(z, slope=0.5) = leaky_relu(z, 0.5) ** 2 (where leaky_relu(z,0.5)=z if z>0 else 0.5*z)
     - gelu(z) = 0.5 z (1 + erf(z/sqrt(2)))
   - δ = σ(z) - σ_lin(z) shape (N, hidden_l).
   - Get `proj_W = np.array(model.blocks[l].mlp.proj.weight.astype(mx.float32))` shape (out_dim=640, hidden_l).
   - Δy = δ @ proj_W.T shape (N, 640).
   - Per block compute: rank_95(Δy) via SVD energy (cumsum of singular values squared / sum), top-5 singular values, top-5 left singular vectors (640-dim), ‖Δy‖_F.
   - Print JSON: `{"checkpoint":..., "sigma":..., "sigma_lin":..., "n_blocks":13, "profile":{l:{rank_95, top5_sv, top5_left_sv (5x640 list), frob_norm, n_tokens, hidden_dim}}}`.
3. **Step 2**: write `experiments/pgolf_curvature_residual_cycle319.py` (qat side).
   - Three (σ, σ_lin) pairs: ("relu2","relu"), ("lrelu2","relu2"), ("gelu","relu").
   - For each: subprocess.run with cwd=PGOLF_ROOT, env DATA_PATH set, timeout=1800. Parse JSON.
   - Combine into `cycle_319_pgolf_curvature_residual.json` with three pair entries.
   - Build summary with side-by-side per-block tables (rank_95, top-5 SVs, ‖Δy‖_F) for the 3 pairs, identify the (lrelu²,relu²) low-rank_95 blocks if any, compare to cycle 318 eff_rank profile.
4. **Step 3**: run `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/pgolf_curvature_residual_cycle319.py`. Verify all 3 pairs return non-empty per-block measurements.
5. **Step 4**: build `cycle_319_summary.md` with verdict per item 334's tree, plus comparison to cycle 318.

# Validation
- pgolf script exits 0 for all 3 pair invocations.
- JSON has 13 blocks per pair.
- For each block, hidden_l matches load_model_v2's per-block mlp_mult: 8 blocks at 1920, 4 blocks at 1280, b12 at 320.
- ‖δ_σ‖ for (lrelu², relu²) on z<0 entries should equal `0.25 * z²` numerically — sanity check via numpy assertion in qat wrapper.
- Trainable params = 0.

# Artifacts
- `experiments/pgolf_curvature_residual_cycle319.py`
- `parameter-golf/local_scripts/pgolf_curvature_residual.py`
- `autoresearch/results/cycle_319_pgolf_curvature_residual.json`
- `autoresearch/results/cycle_319_summary.md`
- `autoresearch/results/cycle_319_manifest.txt`
- `autoresearch/audits/cycle_319.md`, `autoresearch/results/cycle_319_audit.txt`
- `autoresearch/logs/cycle_319_run.log`

# Fallbacks
- If subprocess timeout: reduce n-tokens to 12800 (ratio 0.5x SVD work).
- If pgolf venv has broken numpy/scipy: subprocess with `--n-tokens 6400` minimum sample.
- If `top5_left_sv` JSON payload is too large (5×640×3 = 9600 floats × 13 blocks = 124800): drop to top-3.

# Update bookkeeping
- Move 334 from NEXT/P0 to PAST/Cycle 319.
- TSV row, commit, STATUS.

# APPROVE
