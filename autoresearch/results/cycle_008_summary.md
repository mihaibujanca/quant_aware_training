# Cycle 008: Probe-Corpus Robustness of Geometry Signals

## Verdict

Outcome: **confirmed (proxy)**

All 12/12 GPT-2 FFN blocks pass both robustness criteria (≥7 needed). Channel score rank correlations are ≥0.95 everywhere, selected-set Jaccard overlap ≥0.82 everywhere, Jacobian spectrum cosine ≥0.997 everywhere. The compression geometry signals are highly robust across disjoint probe subsets.

## Setup

- Teacher: `gpt2` (proxy), Shakespeare corpus
- Full corpus: 128 train seqs, 64 holdout, 64 eval (split into 2 disjoint halves)
- Compression budget: 16x (hf=0.0625), top-k selection
- Signals tested: channel_scores Spearman, jacobian_spectrum cosine, selected_indices Jaccard

## Key Results

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| Channel score Spearman | 0.955 (layer 0) | 0.991 (layer 1) | 0.979 |
| Jacobian spectrum cosine | 0.997 (layer 11) | 1.000 (layers 2,4,5,8) | 0.999 |
| Selected-set Jaccard | 0.820 (layers 6,1) | 0.949 (layer 0) | 0.859 |

All blocks select k=192 hidden channels (192/3072 = 6.25% = 1/16).

## Criteria Evaluation

- **Confirmed**: 12/12 blocks have score rank correlation ≥ 0.8 (all ≥ 0.95) AND selected-set overlap ≥ 50% (all ≥ 0.82). This far exceeds the majority threshold.
- NOT inconclusive: early blocks (0-3) and late blocks (8-11) show nearly identical robustness (Spearman: 0.978 vs 0.979, Jaccard: 0.872 vs 0.853).

## Interpretation

1. **Compression signals are not probe-noise**: the same channels are selected regardless of which half of Shakespeare is used. This is a necessary condition for the geometric compression approach to generalize.
2. **Jacobian spectrum is the most robust signal**: cosine similarity ≥0.997, effectively identical across halves.
3. **Layer 0 has the highest channel score variance** (lowest Spearman at 0.955) but the highest selected-set overlap (0.949) — the top channels are stable even when the full ranking is slightly noisier.
4. **This validates the single-probe approach**: existing experiments using one probe corpus are not suffering from probe-specific artifacts.

## Caveats

- Both halves come from the same Shakespeare text. The stronger test (different corpus entirely) is not done here.
- Proxy-only (gpt2+Shakespeare).
- Only tested at one compression budget (16x). Different budgets may show different robustness.

## Artifacts

- `experiments/probe_corpus_robustness.py`
- `autoresearch/results/cycle_008_robustness.json`
