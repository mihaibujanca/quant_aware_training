# Cycle 018 Brief — Item 36: Procrustes Linearity Profile + Identity-Layer Pruning (GPT-2)

## Item
Replicate "Your Transformer is Secretly Linear" (2405.12250) on GPT-2 proxy.
Compute Procrustes linearity score for each of 12 consecutive layer pairs.
Replace top-2 most linear layers with identity. Measure perplexity cost.
Correlate layer ranking with cycle 005 damage ranking.

## Context
Procrustes linearity score = ||X_i^T X_{i+1}||_* / (||X_i||_F ||X_{i+1}||_F)
where ||·||_* is nuclear norm. Score ≈ 1 means the transformation is approximately orthogonal (linear).
Cycle 005 damage ranking: layer 0 hardest, layers 4, 5, 7 easiest (smallest perplexity increase).
Identity replacement: zero out block contribution; residual stream output = input (block(input) = 0).

## Dependencies
- Item 9 (GPT-2 + Shakespeare probe corpus) — COMPLETE
- Cycle 005 damage curve — AVAILABLE in `autoresearch/results/cycle_005_summary.md`

## Key Files
- `aleph/lm_geometric_compression.py` — `load_nanochat_teacher`, `TeacherLoadSpec`, `build_probe_corpus`, `ProbeCorpusSpec`
- `autoresearch/results/cycle_005_summary.md` — layer damage ranking
- Experiment script: `experiments/procrustes_linearity_gpt2.py`

## Constraints
- Teacher: HuggingFace `gpt2`, corpus: `data/shakespeare.txt`
- 13 checkpoints: after embedding + after each of 12 blocks (residual stream)
- Linearity score uses nuclear norm formula above
- Identity = zero out block's residual contribution; input passes through unchanged
- Intervention on top-2 most linear layers; measure perplexity change
- Confirmed: ≥4 layers score ≥0.98 AND identity of top-2 ≤10% ppl increase AND Spearman ≥0.4 with damage
- Refuted: all layers < 0.90

## Handoff
Script: `experiments/procrustes_linearity_gpt2.py`
Results: `autoresearch/results/cycle_018_*.{json,png}`
Manifest: `autoresearch/results/cycle_018_manifest.txt`
