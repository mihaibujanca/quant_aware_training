# Item
21 — Single-affine collapse quality — GPT-2 FFN blocks

# Context
Cycle 010 confirmed toy ReLU deep layers are near-linear (r < 0.06 for L2–L4). Now apply same method to GPT-2 FFN blocks to test if any block is functionally linear on Shakespeare probe corpus. If yes: zero-shot depth compression (replace block with single A·x+b, no retraining/distillation). Contrast with cycle 009 refutation (no blocks near-linear via Jacobian variance L metric).

# Dependencies
- Item 20 (cycle 010, **confirmed**) — provides proof-of-concept on toy ReLU
- Item 9 (cycle 004, **confirmed**) — benchmark locked to gpt2 proxy + Shakespeare
- Inverse dependency: if item 21 is refuted, item 23 (Jacobian k-means on GPT-2) must address multi-region clustering instead

# Key Files
- `experiments/single_affine_collapse_toy.py` — reference implementation; adapt for GPT-2
- `aleph/lm_geometric_compression.py` — has tokenizer + GPT-2 loading utilities
- `autoresearch/results/cycle_010_summary.md` — toy baseline (r values per layer)
- `autoresearch/results/cycle_009_summary.md` — prior GPT-2 analysis (Jacobian L metric)

# Constraints
- **Cycle 009 contradiction**: Jacobian linearity L showed no blocks near-linear (L < 0.6 all blocks). New single-affine fit may contradict this → check if methods measure different phenomena.
- Fit on probe split 1 (256 tokens), validate on split 2 (held-out). Report train/held-out gap.
- Perplexity test: actually swap in best 2–3 blocks' affine approximations, measure ppl degradation.
- Criterion: pass if r < 0.15 for ≥2 blocks AND swap ≤5% perplexity increase; refute if r > 0.5 all blocks OR swap > 20% ppl increase.

# Handoff
Apply cycle 010 methodology directly to GPT-2 FFN blocks. For each block: collect probe activations (input h_in, output h_out), fit A,b = `torch.linalg.lstsq([h_in|1] → h_out)` on first half, measure r on second half. Expected outcome: either confirms toy transfer (some blocks r < 0.15, safe swap) or refutes (all r > 0.5, consistent with cycle 009 Jacobian result). If refuted, multi-region methods (items 22–25) become mandatory. If confirmed, provides concrete no-SGD depth-compression baseline.
