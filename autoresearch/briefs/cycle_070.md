# Item
**Corpus-diverse probing: multi-domain channel selection stability — GPT-2 FFN**
Run topk channel selection independently on 4 probe corpora (Shakespeare, WikiText-2, C4, OpenWebText) for blocks [7, 5, 4, 1] at 6×. Measure pairwise Jaccard similarity of selected channels and cross-corpus PPL (4×4 matrix).

# Context
Cycle 044 found +120% PPL degradation when compression trained on Shakespeare transferred to WikiText-2 zero-shot. Hypothesis: instability is in *channel selection*, not lstsq refit. Cycle 067 (item 50) showed channel selection requires ≥1024 tokens to reach Jaccard≥0.85 stability (per corpus, same model). This experiment tests *cross-corpus* stability.

# Dependencies
- None. Item 50 (probe corpus size) completed cycle 067.
- Item 66 (mixed-corpus refit) blocks on this result.

# Key Files
- `aleph/lm_geometric_compression.py` — ProbeCorpus, CompressionBudgetSpec, derive_ffn_compression, _evaluate_model
- `experiments/per_block_corpus_sensitivity.py` — template for block-wise probing (cycle 053)
- Results: `autoresearch/logs/experiments.tsv` (manifest), cycle files in logs/

# Constraints
- Use ≥512 tokens per corpus (cycle 067 requires 512+ for stable lstsq)
- Blocks: [7, 5, 4, 1], 6× compression ratio (k=512 of 3072), topk selection
- 4×4 cross-eval: each of 4 probe corpora → compressed model → eval on all 4 corpora

# Handoff
Implement corpus-diverse topk selection loop. Collect Jaccard matrix (4×4 blocks, pairwise similarity) and cross-eval PPL matrix (4 probe × 4 eval). Accept if Jaccard≥0.7 for ≥3/4 blocks AND PPL gap ≤10% for ≥2/4 blocks. Refute if Jaccard<0.4 for majority of blocks. Write cycle_070_run.log, update experiments.tsv with row(s).
