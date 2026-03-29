# Cycle 108 Brief — Pre-activation distribution anatomy (Item 119)

## Status
- **107 cycles completed**: 17 confirmed, 33 refuted, 48 inconclusive, 1 crash
- **Latest**: Cycle 107 (item 117): output-space direct regression — INCONCLUSIVE (audit corrected)
  - Middle blocks nearly linear; early/late strongly nonlinear
  - Topk wins early/late by 2-27%; marginal early blocks hint at linearity gradient

## Task: Item 119 (Interpretability)
**Pre-activation distribution anatomy — GPT-2 FFN**

Cycle 103 found 22-52% of FFN pre-activations in |z| < 0.5 (GELU nonlinear knee), with block 1 at 52%.
Determine whether this concentration is expected (variance-driven) or structurally interesting (active).

### Experiment
- 4096 mixed-corpus tokens, all 12 blocks, 3072 channels each
- Per-channel: fit Gaussian N(μ, σ²), compute zone fractions at T ∈ {0.25, 0.5, 1.0, 1.5, 2.0}
- Compare empirical vs Gaussian-predicted Φ(T/σ) - Φ(-T/σ)
- Report per-block: mean/std of σ, channels with σ < 0.5 ("bump-dominated")
- Correlate per-channel σ with topk importance (Spearman)

### Confirmation criteria
- **Confirmed**: empirical zone fractions match Gaussian ≤5% relative for ≥10/12 blocks
- **Refuted**: empirical exceed Gaussian >20% for ≥6/12 blocks (active concentration)
- **Inconclusive**: mixed or moderate excess (5-20%)

## Key context
- Subspace compression plateau: topk k≤4 at 6× → +5% PPL; k=5 hits +22-24% but with high variance
- Block linearity gradient (107): middle blocks ≈linear, early/late nonlinear
- Dead directions ruled out in cycles 099-103
- P0 next: items 115, 119-123 (cascade, GELU-leak), then P1: 100, 117, 124-125

## Command
```
python3 -c "import sys; print('Ready for cycle 108 work')"
```
