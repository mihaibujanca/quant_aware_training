# Item
13 — Probe-corpus robustness of geometry signals

# Context
- Cycle 006 refuted stable-channel masks (<1% in most blocks) → skip that sub-signal
- Key signals to test: channel_scores (3072-dim per layer), jacobian_spectrum, selected_indices (top-k selection)
- probe_ffn_geometry() computes all signals from probe data in one pass
- ProbeCorpus has probe_train (64 seqs), probe_holdout (32), eval_val (32)

# Dependencies
- Item 9 (cycle 004): DONE — benchmark locked, proxy gpt2+Shakespeare available
- Item 11 (cycle 006): DONE (refuted) — stable-channel masks moot, but other signals still valid

# Key Files
- `aleph/lm_geometric_compression.py` — `probe_ffn_geometry()`, `derive_ffn_compression()`
- `data/shakespeare.txt` — probe corpus source

# Constraints
- Proxy-only, label clearly
- Split probe data into 2 disjoint halves, probe each independently
- Compare: Spearman rank correlation of channel_scores, cosine of jacobian_spectrum, Jaccard overlap of selected_indices
- 30-minute cap

# Handoff
Write `experiments/probe_corpus_robustness.py`. Split Shakespeare into 2 disjoint probe halves. Run `probe_ffn_geometry` + `derive_ffn_compression` on each half. Compare signals per block.
