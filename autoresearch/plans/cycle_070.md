# Task
Item 64: Corpus-diverse probing — multi-domain channel selection stability (GPT-2 FFN). Measure pairwise Jaccard of topk channel sets and cross-corpus PPL (4×4 matrix) for blocks [7,5,4,1] at 6×.

# Hypothesis
The +120% PPL degradation (cycle 044) is caused by channel selection instability across domains, not the lstsq refit. If Jaccard ≥ 0.7 across corpora but PPL still diverges, the problem is in lstsq coefficients.

# Pre-flight Results
- P1 (duplicate): NOT a duplicate. Cycle 057 measured Jaccard(OWT, Shakespeare)=0.47-0.54 on 3 blocks (2 corpora, no cross-eval matrix). This extends to 4 corpora × 4 blocks with full 4×4 cross-eval.
- P2 (deps): No deps. Item 50 (cycle 067) completed. Item 66 (mixed-corpus refit) blocks on this result.
- P3 (scope): Yes — 4 corpora × 4 blocks × topk + Jaccard + 4×4 PPL fits one cycle.
- P4 (program.md): No modifications needed.
- P5 (null baseline): Random Jaccard for k=512 of 3072: J = k/(2n-k) = 512/5632 ≈ 0.091. Any measured J >> 0.09 is meaningful.
- P6 (method-architecture): GPT-2 uses GELU FFN; topk channel selection uses `channel_scores` from `probe_ffn_geometry`. `selected_indices` stored on `CompressedFfnLayer`. Conv1D transposition handled by `_standard_weight()`.
- P7 (power): 4 corpora × 4 blocks = 16 Jaccard pairs per block (6 unique pairs). 4×4 cross-eval = 16 PPL values per block. No stochastic element in topk selection (deterministic given corpus), so seeds only needed for corpus sampling — use seeds [42, 123, 7] for corpus construction, report mean/std Jaccard.

# Steps
Script: `experiments/corpus_diverse_channel_selection.py`

1. **Corpus setup** (4 text sources, 2048 tokens each = 16 sequences @ seq_len=128):
   - Shakespeare: `data/shakespeare.txt` (exists)
   - WikiText-2: `data/wikitext2_valid.txt` (exists)
   - OpenWebText: `data/openwebtext_sample.txt` (exists)
   - C4: download via `datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)`, save first 200 docs to `data/c4_sample.txt`
   - For each corpus, build `ProbeCorpusSpec(probe_train_sequences=16, probe_holdout_sequences=8, eval_val_sequences=200, sequence_length=128)` with 3 seeds [42,123,7]

2. **Channel selection** (deterministic per corpus+seed):
   - For each (corpus, seed, block_id) in 4×3×4 = 48 runs:
     - `probe_ffn_geometry(teacher, corpus, budget, layer_ids=[block_id])`
     - `derive_ffn_compression(atlas, budget)` with `hidden_fraction=1/6` (k=512)
     - Extract `pack.layers[0].selected_indices` → store as set

3. **Jaccard matrix** (per block):
   - For each block, for each pair of corpora (6 pairs), compute mean Jaccard across 3 seeds
   - Also compute within-corpus cross-seed Jaccard as a stability reference (cycle 067 showed 0.844 at 2048 tokens)

4. **Cross-eval PPL matrix** (4 probe × 4 eval):
   - For each (probe_corpus, seed): compress block, swap into model via `swap_compressed_ffns`
   - Evaluate on all 4 eval corpora → 4×4×3 = 48 PPL values per block
   - Report mean±std across seeds

5. **Baselines**: Uncompressed GPT-2 PPL on all 4 eval corpora (for relative PPL computation).

6. **Outputs**: JSON with Jaccard matrix (block × corpus-pair), PPL matrix (block × probe × eval), baselines. Plot: heatmaps of Jaccard and relative PPL per block.

Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/corpus_diverse_channel_selection.py`

# Validation
- **Confirmed if**: cross-corpus Jaccard of selected channels ≥ 0.7 for ≥ 3/4 blocks AND cross-eval PPL gap between best and worst probe corpus is ≤ 10% for at least 2/4 blocks.
- **Refuted if**: Jaccard < 0.4 for a majority of blocks (channel selection is fundamentally corpus-specific, not just the refit).
- **Inconclusive if**: Jaccard is moderate (0.4–0.7) or high Jaccard does not translate to cross-corpus PPL stability.

Sanity checks: (a) within-corpus cross-seed Jaccard should match cycle 067 (~0.84 at 2048 tokens); (b) all Jaccard >> 0.09 (random baseline); (c) Shakespeare and WikiText-2 cross-corpus Jaccard should be near cycle 057's 0.47-0.54 range.

# Artifacts
- `experiments/corpus_diverse_channel_selection.py`
- `autoresearch/results/cycle_070_corpus_diverse_channels.json`
- `autoresearch/results/cycle_070_corpus_diverse_channels.png`
- `data/c4_sample.txt` (downloaded if missing)
- Update `autoresearch/agenda.md`: move item 64 to PAST EXPERIMENTS with results
- Commit: `git add experiments/corpus_diverse_channel_selection.py autoresearch/results/cycle_070_* data/c4_sample.txt && git commit`

# Fallbacks
- If C4 download fails (HuggingFace auth/connectivity): drop C4, run with 3 corpora (3×3 matrix). 3 corpora still sufficient for the Jaccard threshold (≥ 0.7 in 3/3 pairs).
- If any corpus yields < 16 sequences at seq_len=128: increase `max_documents` to 500.
- If PPL eval is too slow (4 blocks × 4 probes × 4 evals × 3 seeds = 192 evals): single-block swap only (not stacked), reduce eval_val_sequences to 100.
