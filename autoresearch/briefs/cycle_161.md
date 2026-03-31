# Cycle 161 Brief — Centroid-Affinity Channel Selection

## Item
171 — Centroid-affinity channel selection vs activation-norm topk (GPT-2 FFN k=4)

## Context
- Cycle 27 established centroid similarity as strong merge signal (ρ=0.853)
- Current baseline: activation-norm topk → single-block PPL ~85, stacked PPL ~70.7 (with polish)
- Item proposes: select k=4 channels that maximize coverage of centroid embedding space (farthest-point or greedy set cover) instead of activation magnitude

## Dependencies
- None. Independent from cycles 159–160 (which focused on joint training and corrections)

## Key Files
- Baseline config: `/Users/mihai/Projects/quant_aware_training/experiments/topk_baseline_cycle*.py`
- Cycle 27 results: `autoresearch/results/cycle_027_affinity_signals.json` (ρ_centroid=0.853)
- Expected output: single-block PPL per block, stacked PPL, comparison plots

## Constraints
- k=4 fixed (matching baseline)
- Compare on full 12-block GPT-2 stack (not toy models)
- Success threshold: centroid PPL ≥ baseline – 1.5 PPL (confirmed), or failure if within 0.5 PPL
- Inconclusive: better/worse on subset of blocks, net within 0.5–1.5 PPL of baseline

## Handoff
Run centroid affinity selection (farthest-point or greedy set cover on block-wise centroid embeddings) at k=4, measure single + stacked PPL. Hypothesis: geometric coverage beats activation magnitude for channel selection.
