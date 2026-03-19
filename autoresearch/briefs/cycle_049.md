# Item
51 — Per-block learned compressor with block-conditioned bottleneck — GPT-2 FFN

# Context
Cycle 030 (item 18) tested a **shared** low-rank bottleneck (768→384→768) trained on blocks 0-7, evaluated zero-shot on blocks 8-11. Result: INCONCLUSIVE. The shared compressor severely underfitted (train PPL mean=163.4 vs baseline=122.4) and beat geometric on only 1/4 held-out blocks. Root cause: each block computes qualitatively different functions; cross-block sharing forces the compressor to memorize an average rather than learn reusable structure. Per-block distillation (baseline upper bound) achieved 122-132 PPL — near baseline — with the same parameter budget.

Item 51 asks: was the failure due to **cross-block sharing** or **insufficient capacity**? Solution: train one lightweight compressor **per block** (separate U,V weights, same 768→384→768 architecture), 500 distill steps each. Hypothesis: per-block training overcomes the underfitting and beats geometric baseline.

# Dependencies
- **Item 14** (rank-only baseline vs geometry): CONFIRMED (cycle 026). Geometric beats SVD-norm in 11/12 blocks at matched FFN param budget. ✓ Available.
- **Item 18** (shared compressor): INCONCLUSIVE (cycle 030). Established the underfitting phenomenon and per-block distillation upper bound. ✓ Available.

No blocked dependencies.

# Key Files
- `experiments/learned_compressor_generalization.py` — item 18 implementation (shared bottleneck). Reference for per-block variant structure.
- `aleph/lm_geometric_compression.py` — geometric baseline + probe corpus, evaluation loop, distillation setup.
- `autoresearch/results/cycle_030_generalization.json` — prior results (shared PPL patterns, per-block distill baseline numbers).

# Constraints
- 12 compressors × 500 steps each = 6K gradient steps total. Feasible in one session.
- Parameter budget must match cycle 030 (384 hidden = same as geometric 384-channel selection) to isolate the sharing-vs-capacity variable.
- Metric: per-block compressor PPL vs geometric baseline PPL on all 12 blocks (not just held-out).
- Success bar: beat geometric on ≥8/12 blocks (criterion: ≥1 PPL point gain).

# Handoff
Item 51 is straightforward: modify cycle-030 code to train independent compressors per block instead of one shared compressor. Key edit: loop over blocks in distillation, maintain separate U,V per block. Evaluate on all 12 blocks (not zero-shot). Expected outcome: per-block training should recover much of the per-block distillation quality (122-132 PPL) if underfitting was the problem. If geometric still wins, the issue is depth/compression-incompatibility rather than sharing.
