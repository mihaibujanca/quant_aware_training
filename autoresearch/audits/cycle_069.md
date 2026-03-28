status=CORRECTED
old_outcome=INCONCLUSIVE — 8/12 blocks pass raw >= 0.8 at k_99 (just below 10/12 threshold). Spectral structure partially shared, excess 0.28-0.52, middle blocks 5-8 weaker.
new_outcome=UNRELIABLE — Jacobians computed at wrong operating point. Code uses hidden_states[block_idx] (block input) instead of the post-attention, post-ln_2 FFN input. GELU derivatives are therefore wrong for all tokens in all blocks. Alignment numbers cannot be trusted. Cycle 068 correctly used register_forward_hook on mlp.c_fc; this cycle regressed to manual computation from the wrong hidden states.
reason=Critical G1 failure: the function collect_token_jacobians computes z = h @ W_in.T + b_in where h is the block-level hidden state, missing ln_1, attention, residual, and ln_2 transformations before the FFN. This means every Jacobian J(x) is evaluated at a wrong operating point, making the GELU activation pattern (and thus the entire spectral structure) unreliable. Must re-run with hooks on mlp.c_fc as cycle 068 did.
guardrail_flags=G1-implementation-bug,G2-marginal-no-confidence-intervals
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Re-run with register_forward_hook on mlp.c_fc (matching cycle 068 methodology). Add bootstrap CI on alignment values. Keep the same verdict logic and thresholds.
interesting=false
interesting_reason=N/A — results unreliable due to implementation bug
