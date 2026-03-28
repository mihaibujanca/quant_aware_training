# Cycle 074: Persistent Homology of Activation Manifold — GPT-2

## Item
Item 28: Persistent homology of GPT-2 hidden states at each block output. Test whether Betti complexity predicts compression difficulty.

## Context
Item 27 (toy ReLU, cycle 047) was marginally refuted: 2/3 transitions vs 80% threshold, only 3 data points. GPT-2 has 12 transitions — much more statistical power. Unblocked.

## Dependencies
- Item 9 (Jacobian variance atlas, cycle 009): refuted (blocked)
- Item 27 (toy ReLU homology, cycle 047): refuted but statistical power limited — **GPT-2 provides statistical power to revisit**

## Key Files
- `experiments/phom_gpt2.py` (new) — compute persistent homology on GPT-2
- `autoresearch/results/cycle_074_phom_gpt2.json`, `.png` (artifacts)
- Related: `experiments/phom_toy_relu.py` (cycle 047 reference)

## Constraints
- **Ripser NOT available**: `which ripser` fails. Must install via pip or homebrew before running.
- PCA to 30D required (ripser input).
- 12 blocks provide Spearman statistical power (vs 3 in toy).
- Confirmed if: Betti decreases ≥60% blocks AND Spearman(drop, 1−difficulty) ≥ 0.35.

## Handoff
Install ripser, run homology on GPT-2 blocks with compression-difficulty ground truth from cycles 068–073. Compare with toy-ReLU results and justify statistical power gain.
