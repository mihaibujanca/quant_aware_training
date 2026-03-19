verdict: REVISE
round: 2

## Round 1 issue status

**Issue 1 (blocking) — measure (c) non-orthogonal projector**: Partially fixed. The plan now uses QR decomposition, which is the correct approach. However, the QR is applied to the wrong matrix — see new issue below.

**Issue 2 (minor) — bias subtraction**: Fixed. Step 5 now explicitly includes `- b_refit` in the residual formula.

## New issue — measure (c) dimension mismatch

Step 8 computes `Q, _ = torch.linalg.qr(H_discard.T)` where `H_discard = layer.hidden_sample[:, ~selected]` has shape `(tokens, discarded_neurons)`. This gives Q shape `(discarded_neurons, rank)`. Then the plan says `E_proj = E @ Q @ Q.T` — but E has shape `(tokens, output_dim=768)`, so `E @ Q` requires `output_dim == discarded_neurons`, which is wrong. E lives in output space; Q lives in hidden-neuron space.

The question "does the residual live in the subspace the discarded neurons span" must be answered in output space. The correct approach:

1. Get original c_proj columns for discarded neurons: `W_orig = _standard_weight(layer.handle.c_proj)` shape `(output_dim, hidden_dim)`, then `W_discard = W_orig[:, ~selected]` shape `(output_dim, num_discarded)`.
2. QR on `W_discard`: `Q, _ = torch.linalg.qr(W_discard)` — Q shape `(output_dim, rank)`.
3. Project: `E_proj = E @ Q @ Q.T`, compute `‖E_proj‖_F² / ‖E‖_F²`.

This projects E onto the output-space subspace that discarded neurons contribute to through the original weight matrix, which is the geometrically meaningful question. Fix step 8 to use `W_discard` (weight columns) instead of `H_discard` (activation matrix).

## Remainder

Everything else from round 1 remains sound: hypothesis is well-motivated, SVD rank-90 and token correlation measures are correct, sample size gives 200:1 overdetermination, verdict criteria are appropriate. No other new issues.
