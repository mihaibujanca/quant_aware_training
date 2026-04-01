# Cycle 172 Plan — Item 182: Lstsq regression target alternatives

## Pre-flights
- P1: `_fit_output_projection` takes `hidden [N,k]` and `outputs [N,768]`, solves lstsq. ✓
- P2: `layer.output_sample` = raw FFN output; `layer.input_sample` = residual stream input. ✓
- P3: Token frequencies available from `corpus.probe_holdout` token IDs. ✓
- P4: Baseline PPL ~74 unpolished / ~59 polished (cycle 171 at same config). ✓

## Design
Four lstsq targets, all sharing same topk channel selection (k=512, blocks [1,4,7,11]):

| Target  | Y for lstsq                                 | Rationale                                     |
|---------|----------------------------------------------|-----------------------------------------------|
| raw     | `Y = output_sample` (FFN output)             | Current baseline                              |
| norm_wt | `Y = output_sample / ||input_sample||`       | Scale-invariant residual delta                |
| svd_sub | `Y` projected onto top-64 right-singular     | Geometry-preserving (Jacobian proxy)          |
|         | vectors of `input_sample @ W_float_out.T`    |                                               |
| freq_wt | Weighted lstsq: rows weighted by token freq  | Common tokens matter more                     |

### Implementation notes
- **norm_wt**: divide each row of Y by `||x_i||` before lstsq, then at inference
  multiply c_proj output by `||x||`. But this changes the MLP semantics — instead,
  after lstsq, scale W_out back: `W_final = diag(1/||x||) solution` won't work per-token.
  Correct approach: lstsq minimizes `||H @ W - Y/||x||||}` → `W_hat`. At inference
  the compressed MLP outputs `H_new @ W_hat` which approximates `Y/||x||`. But we need
  `Y`, so this only works if we also multiply by `||x||` at inference. Since `x` IS the
  input to the MLP, we can store `W_hat` and at forward time multiply output by `||x||`.
  **Simpler**: just use `Y_norm = Y * (1/||x||).unsqueeze(1)` in lstsq, then wrap
  the CompressedMLP forward to multiply output by `||x||`.
- **svd_sub**: compute `U, S, Vh = svd(X @ W_out.T)`, take top-64 V directions,
  project `Y` onto them, fit lstsq in that subspace, project back for final W_out.
- **freq_wt**: count token IDs in probe corpus, assign weight per token position,
  use `sqrt(w)` scaling on both H and Y rows (equivalent to weighted least squares).

## Execution
- `experiments/lstsq_target_alt_cycle172.py`
- 3 seeds [42, 137, 7], 200 eval seqs, 500 polish steps
- Blocks [1, 4, 7, 11], hidden=512 (6x compression)
- Report: unpolished + polished PPL per target, per seed
- Output: `autoresearch/results/cycle_172_lstsq_targets.json`

## Risk
- norm_wt changes MLP forward signature — need custom CompressedMLP wrapper
- svd_sub rank-64 may be too low/high — but this is a reasonable default
- freq_wt may collapse to near-uniform for random OWT samples
