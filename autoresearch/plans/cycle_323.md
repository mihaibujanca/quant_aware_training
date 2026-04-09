# Task
Item 336: Post-`mlp_scale` eff_rank census on parameter-golf. Measure whether per-channel scaling in `Block.__call__` (line 613 of `train_gpt_mlx.py`: `self.mlp_scale[None,None,:] * self.mlp(...)`) changes the eff_rank profile relative to cycle 318's pre-scale capture.

# Hypothesis
Post-`mlp_scale` eff_rank profile is within +/-20% relative of cycle 318's pre-scale profile at every block. Per-channel scaling is a diagonal linear map; eff_rank is invariant under full-rank linear transforms, so unless `mlp_scale` has near-zero entries collapsing channels, it should not change rank.

# Pre-flight Results
- P1 (duplicate): No. PAST EXPERIMENTS section has no post-scale or mlp_scale eff_rank entry.
- P2 (dependency): Satisfied. Checkpoint exists, cycle 318 baseline JSON exists, template script exists.
- P3 (concrete): Yes. Single measurement, clear verdict criteria, ~10 min.
- P4 (program.md): No modifications needed.
- P5 (null baseline): Cycle 318 pre-scale profile is the baseline (b0=187.61 ... b12=6.94).
- P6 (method-architecture): Must monkey-patch `Block.__call__` (not `MLP.__call__`) to capture the post-scale product. Block.__call__ signature is `(self, x, x0, mask=None)` -- the hook must preserve this.
- P7 (statistical power): 25,600 tokens (25 seqs x 1024) is the same as cycle 318; sufficient for covariance estimation at dim=640.

# Steps
0. `cat autoresearch/cross_project/parameter_golf.md` (REQUIRED -- exec must run this first).
1. Write `parameter-golf/local_scripts/pgolf_eff_rank_census_post_scale.py`. Copy `pgolf_eff_rank_census.py`, change the hook target from `MLP.__call__` to `Block.__call__`. Inside the hooked `Block.__call__`, call the original, but also compute `scaled_mlp = self.mlp_scale.astype(x.dtype)[None,None,:] * self.mlp(self.mlp_norm(x_after_attn), slope=self.lrelu_slope)` and capture it. Key detail: `x_after_attn` is not directly available -- instead, run the original `Block.__call__` but intercept the MLP contribution. Cleanest approach: monkey-patch `Block.__call__` to wrap the original, extracting the scaled MLP output as `post_scale = self.mlp_scale.astype(x.dtype)[None,None,:] * mlp_out` where `mlp_out` is captured via a nested `MLP.__call__` hook in the same scope. Use `block_idx = {id(model.blocks[l]): l for l in range(n_blocks)}` keyed on Block id.
2. Write `experiments/pgolf_eff_rank_postscale_cycle323.py` (qat-side wrapper). Subprocess-invoke pgolf script, save JSON to `autoresearch/results/cycle_323_pgolf_postscale_eff_rank.json`. Build `cycle_323_summary.md` with side-by-side table (pre-scale from cycle 318 vs post-scale) and verdict.
3. Run the wrapper. Capture stdout/stderr.
4. Compare per-block: compute `|post - pre| / pre * 100%` for each block. Apply verdict tree.
5. Write `autoresearch/results/cycle_323_summary.md` with table and verdict.
6. Update `autoresearch/agenda.md`: move item 336 to PAST EXPERIMENTS with cycle number and verdict.
7. Commit all artifacts.

# Validation
- JSON schema matches cycle 318: keys `eff_rank`, `rank_95`, `rank_99`, `top5_eigvals`, `n_tokens`, `dim`, `trace` per block.
- `n_blocks` == 13, `n_tokens` ~25600 per block.
- Verdict criteria (verbatim from agenda):
  - **CONFIRMED-rank-invariant** if post-scale and pre-scale eff_rank agree within +/-20% relative at every block: the cycle 318 measurement is in the correct representation, the b3 finding is robust to the choice of capture point. Closes the caveat.
  - **REFUTED-scale-matters** if any block shifts by >=50% relative: the per-channel `mlp_scale` is doing meaningful rank reshaping, the cycle 318 reading should be re-interpreted in the post-scale representation, and the LASER target may shift.
  - **INCONCLUSIVE** otherwise.

# Artifacts
- `parameter-golf/local_scripts/pgolf_eff_rank_census_post_scale.py`
- `experiments/pgolf_eff_rank_postscale_cycle323.py`
- `autoresearch/results/cycle_323_pgolf_postscale_eff_rank.json`
- `autoresearch/results/cycle_323_summary.md`

# Fallbacks
- If Block.__call__ hook fails (signature mismatch with x0/mask): fall back to computing post-scale from the existing MLP hook output by loading `model.blocks[l].mlp_scale` weights directly from the checkpoint and multiplying after capture.
- If subprocess times out: reduce n-tokens to 12800.
