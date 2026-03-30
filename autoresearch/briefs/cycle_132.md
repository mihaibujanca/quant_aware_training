# Cycle 132 Brief — Item 126

**Item**: Attention projection pruning on high-amplification heads — GPT-2

**Context**: Cycle 104 confirmed cascade is attention-driven (attn_frac > 1.0 in layers 0-6). Cycle 113 then measured per-head amplification and found it **near-uniform** across all 12 heads in 10/11 transitions (top3_frac 0.29-0.39, Gini 0.06-0.28). Exception: block 10->11 (top3_frac=0.722). This REFUTED the premise of "top-3 amplifying heads" — no surgical head targets exist.

**Complication**: Item 126 was written assuming cycle 113 would find concentrated heads. It didn't. The experiment is still worth running: even without head-level targeting, low-rank SVD of W_V @ W_O applied uniformly to heads in high-amplification layers may reduce the cascade by constraining the attention subspace. The question shifts from "prune the bad heads" to "can rank-reducing V+O projections in high-attn-frac layers help?"

**Design**: Apply rank-r SVD (r in {8, 11, 16}) to W_V @ W_O per head in layers 0-6 (attn_frac > 1.0 from cycle 104). Measure: (a) single-transition cascade amplification factor, (b) stacked k=4 FFN PPL. 3 seeds, 200 eval seqs. ~27 evals total, fits 30 min.

**Key Files**:
- Cycle 104: `autoresearch/results/cycle_104_cascade_source.json` (per-transition attn_frac)
- Cycle 113: `autoresearch/results/cycle_113_head_amplification.json` (per-head uniformity)
- FFN compression baseline: `experiments/cascade_correction_k5_gpt2.py` (adapt)

**Constraints**:
- 3 seeds [42, 123, 7]. No training — SVD is applied directly (no learned params).
- Confirmed if: cascade amp reduced >= 20% relative AND stacked PPL improves >= 2% absolute.
- Refuted if: PPL degrades (attention heads carrying essential info).
- Inconclusive if: amp reduced but PPL unchanged.

**Handoff**: Plan must handle the cycle 113 tension — apply SVD uniformly to all heads in target layers (not top-3), and include layer 10->11 as a separate test (only layer with concentrated heads). Report per-layer amp factors and stacked PPL.
