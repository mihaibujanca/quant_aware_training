# Cycle 015 Brief — Item 26

**Item:** Intrinsic dimension trajectory through depth (GPT-2)

**Context:**
Region-merging approaches have failed (cycles 012–014): k-means on affine maps (item 22, refuted), RTG spectral (items 24–25, inconclusive/refuted). Hypothesis shift: intrinsic dimension may predict compressibility without attempting compression.

**Dependencies:**
- Item 9 (cycle 004): Benchmark lock-in complete; GPT-2 proxy loaded via `load_nanochat_teacher("gpt2")`.
- No blocking dependencies; item 26 is independent.

**Key Files:**
- `experiments/jacobian_variance_atlas.py` — GPT-2 loading pattern (line 95–108): TeacherLoadSpec + ProbeCorpusSpec.
- `aleph/lm_geometric_compression.py` — load_nanochat_teacher, build_probe_corpus, _extract_ffn_handles.
- `autoresearch/results/cycle_005_summary.md` — single-block swap damage curve (reference baseline).

**Constraints:**
- Use TwoNN estimator (Facco et al. 2017): requires only k=2 nearest-neighbor distances, no hyperparameters.
- Collect token reps at all 12 FFN block outputs on probe holdout set (Shakespeare, ~32 seqs, 128 tokens/seq).
- Correlate intrinsic dimension trajectory with cycle 005 compression difficulty (Spearman ≥0.4 target).
- Confirmed if: dim decreases ≥30% over final third of layers AND Spearman ≥0.4.

**Handoff:**
Implement TwoNN on per-layer token representations. Plot dim trajectory and CKA-style heatmap. Compute Spearman vs cycle_005_damage. Store results in `autoresearch/results/cycle_015_intrinsic_dim.json`.
