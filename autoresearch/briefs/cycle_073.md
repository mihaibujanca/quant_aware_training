# Item
84 — Rank-matched adaptive compression — GPT-2 FFN

# Context
Cycle 072 confirmed stacking is the bottleneck (+30% PPL unpolished, +4.9% polished).
Cycle 068 measured per-block effective rank: 10/12 blocks have rank_99 ≤ 500.
Hypothesis: adaptive per-block compression ratios (matched to measured rank) reduce stacking error better than uniform 6× across all blocks.

# Dependencies
Item 80 (cycle 068): **COMPLETE** — effective rank per block measured.
- Block ranks: 1=588.4 (outlier), 2–11≈200–500, 12≈400–500

# Key Files
- `autoresearch/results/cycle_068_jacobian_rank_census.json` — per-block rank_99 values
- `autoresearch/experiments/gpt2_compression.py` — stacking experiment harness (from cycle 061)
- Baseline: uniform 6× yields +30% unpolished, +4.9% polished (cycle 061 results)

# Constraints
- Total parameter budget FIXED to match uniform 6× across k=4 blocks (~6M→1M)
- Map rank_99 → per-block compression ratio: block with rank 200 keeps ~250, rank 500 keeps ~550
- Test stacked k=4 with/without gradient polish (500 Adam steps, lr=1e-4)
- Success: rank-matched ≤+20% unpolished OR ≤+3% polished (vs uniform benchmarks)

# Handoff
Implement adaptive compression mapping from cycle 068 ranks. Compare stacked k=4 PPL: rank-matched vs uniform 6×, both polished/unpolished. Report rank distribution, per-block compression ratios, and final PPL deltas.
