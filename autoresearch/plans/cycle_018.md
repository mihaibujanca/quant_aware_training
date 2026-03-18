# Cycle 018 Plan — Procrustes Linearity Profile + Identity-Layer Pruning (GPT-2)

## Task
Compute Procrustes linearity scores for all 12 GPT-2 block transitions, run identity-replacement
intervention on top-2 most linear blocks, compare ranking with cycle 005 damage curve.

## Hypothesis
≥4 GPT-2 blocks will show linearity ≥0.98 (consistent with "Secretly Linear" paper).
Most linear layers will align with easiest-to-compress layers from cycle 005.
Identity replacement of top-2 linear layers will cost ≤10% perplexity.

## Pre-flight
P1. Verify GPT-2 loads and Shakespeare corpus tokenizes correctly (baseline perplexity ~122).
P2. Verify hook registration collects residual stream at 13 checkpoints (embedding + 12 block outputs).
P3. Verify nuclear norm / Frobenius norm formula returns values in [0, 1].
P4. Verify identity replacement (zero block output) evaluates under no_grad without side effects.

## Steps
1. Load GPT-2 + Shakespeare probe corpus using `load_nanochat_teacher` / `build_probe_corpus`.
2. Register forward hooks on: transformer.wte+wpe output AND each transformer.h[i] output (12 blocks).
3. Run 64 probe sequences through model; collect residual stream tensors X_0..X_12 (shape: [N,T,768]).
4. For each i in 0..11:
   - X = X_i.reshape(-1, 768), Y = X_{i+1}.reshape(-1, 768)
   - Frobenius normalize: X̃ = X / ||X||_F, Ỹ = Y / ||Y||_F
   - Compute M = X̃.T @ Ỹ (768×768)
   - linearity[i] = nuclear_norm(M) / min(N*T, 768)
5. Rank layers by linearity (descending). Print full table.
6. Baseline perplexity: run model normally on holdout sequences.
7. Identity intervention: for top-2 most linear block indices, wrap model so those blocks return zeros.
   - Hook: replace block output tensor with zeros before residual add.
   - Actually: monkey-patch forward to return torch.zeros_like(input) for those blocks.
8. Measure perplexity with identity replacement applied.
9. Spearman correlation with cycle 005 damage: damage_rank = [0,1,2,...,11] ordered by perplexity increase
   from cycle 005 summary (layer 0 worst, layers 4,5,7 best).
10. Save JSON with linearity scores, perplexity results, Spearman value.
11. Save PNG: bar chart of linearity scores, annotated with top-2 identity blocks.

## Validation
- Linearity scores in [0, 1] for all layers.
- Baseline perplexity matches cycle 005 teacher perplexity (~122).
- Confirmed: ≥4 layers ≥0.98, top-2 identity ≤10% ppl increase, Spearman ≥0.4.
- Refuted: all layers < 0.90.

## Artifacts
- `experiments/procrustes_linearity_gpt2.py`
- `autoresearch/results/cycle_018_linearity.json`
- `autoresearch/results/cycle_018_linearity.png`
- `autoresearch/results/cycle_018_manifest.txt`

## Fallbacks
- If nuclear_norm(M)/min(N*T, 768) gives values > 1: use min(N*T, 768) as denominator; if still > 1
  clip and investigate (rank-deficiency may cause this).
- If GPT-2 load fails: check HF cache, use `device='cpu'`.
- If Shakespeare corpus not found at `data/shakespeare.txt`: use `ProbeCorpusSpec` text_chunks fallback.
- If identity hook interferes with KV cache: disable `use_cache=False` in model.generate.
