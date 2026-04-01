# Cycle 195 Audit

- Plan followed: yes (jacfwd+vmap as specified; XtX accumulation for stacked rank)
- Config deviation: N_SEQS reduced from 30 to 5 (memory pressure from concurrent MLX training process); CHUNK=8 (plan said 64, then revised to 2 for OOM, then 8 for efficiency). Cross-validated: 30-seq block-0 run gave 169.9 vs 5-seq 166.8 — consistent within std.
- Output file: autoresearch/results/cycle_195_jacobian_rank.json
- Verdict criterion: n_low_rank_blocks >= 6 → CONFIRMED; all_mean_ranks[-1] > 300 AND overall > 300 → REFUTED. Got n_low_rank=0, overall=472.4 → REFUTED.
- Runtime: 165s total for 12 blocks × 160 tokens
