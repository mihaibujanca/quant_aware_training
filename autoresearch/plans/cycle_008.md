# Task
Probe-corpus robustness of FFN geometry signals on proxy gpt2+Shakespeare.

# Hypothesis
From agenda item 13:
- **Confirmed if**: for a majority of tested blocks, score rank correlation is `>=0.8` and selected-set overlap is `>=50%` across disjoint probes.
- **Refuted if**: selected-set overlap drops below `20%` and score correlations stay below `0.5` for most blocks.
- **Inconclusive if**: early and late blocks behave very differently.

# Pre-flight Results
- P1: Not a duplicate of anything in PAST.
- P2: Depends on 9 (done) and 11 (done/refuted). Met.
- P3: Concrete and testable.
- P4: No prompt/program modifications needed.

# Steps

1. **Write experiment script** `experiments/probe_corpus_robustness.py`:
   - Load gpt2 teacher
   - Load Shakespeare corpus, build two disjoint probe halves:
     - Half A: first 32 train seqs, first 16 holdout seqs, first 16 val seqs
     - Half B: last 32 train seqs, last 16 holdout seqs, last 16 val seqs
   - For each half, run `probe_ffn_geometry(teacher, corpus_half)` to get an atlas
   - For each half, run `derive_ffn_compression(atlas, budget)` at hf=0.0625 to get selected_indices
   - Per block, compare:
     - Spearman rank correlation of `channel_scores` (3072-dim)
     - Cosine similarity of `jacobian_spectrum`
     - Jaccard overlap of `selected_indices` (top-k sets)
   - Aggregate: fraction of blocks meeting >=0.8 rank corr AND >=50% Jaccard overlap
   - Save results JSON and summary

2. **Run**: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/probe_corpus_robustness.py`

3. **Evaluate** against confirmed/refuted/inconclusive criteria.

4. **Move item 13** from NEXT to PAST, commit, write manifest.

# Validation
- Channel scores should be non-degenerate (not all zeros or constant)
- Both halves should produce comparable teacher perplexity on their respective eval sets
- k=12 blocks tested total

# Fallbacks
- If 32 train seqs per half is too few, use 48/16 split instead
- If jacobian computation is too slow, skip jacobian_spectrum and focus on channel_scores + selected_indices

# Artifacts
- `experiments/probe_corpus_robustness.py`
- `autoresearch/results/cycle_008_robustness.json`
- `autoresearch/results/cycle_008_summary.md`
