# Item
168. **Topk+lstsq init for joint training — GPT-2 FFN k=4**

# Context
- Random-init joint training gives median 91.1±1.0 across 8 seeds (cycle 155)
- Topk+lstsq solution: single-block ~84.9 PPL, stacked ~95 PPL
- Current best: topk+lstsq+polish = 70.7 PPL (1.58M params)
- Hypothesis: SVD-factorized topk+lstsq gives warm start avoiding bad optimization basins

# Dependencies
- Cycle 155 completed: random-init baseline = 91.1 (median)
- No new data dependencies; uses existing OWT eval 200 sequences

# Key Files
- `/Users/mihai/Projects/quant_aware_training/experiments/joint_multiseed_cycle155.py` — random-init protocol
- `/Users/mihai/Projects/quant_aware_training/experiments/gradient_polish_stacked_gpt2.py` — polish setup
- `/Users/mihai/Projects/quant_aware_training/aleph/lm_geometric_compression.py` — topk selection (line 760)

# Constraints
- **Protocol**: (a) topk+lstsq W_approx per block, (b) SVD → rank-64 W_dec @ W_enc, (c) joint-train stacked CE 2000 steps
- **Success threshold**: PPL < 68 AND std < 2.0 (reliable AND good)
- **Refutation**: PPL within 2% of polish baseline (70.7) → warm start adds nothing
- **Inconclusive**: better than random (91.1) but worse than corrections (59)
- **Seeds**: 3, eval 200 OWT sequences

# Handoff
Implement topk+lstsq initialization for rank-64 joint training. Compare vs cycle 155 baseline (91.1). If PPL improves to <68 with low variance, confirms initialization matters for joint optimization. If not, rules out initialization as solution to seed-dependence.
