# Cycle 051 Handoff Brief

## Item 67: Lstsq Refit Residual Analysis — GPT-2 FFN

**Objective**: Characterize the lstsq refit residual to explain why compression works on narrow corpora (Shakespeare) but fails on broad ones (WikiText-2).

**Dependencies**: None (eligible to run immediately).

**Context**:
- **Cycle 044 (domain robustness)**: Shakespeare→Shakespeare corrected +6.29%. WikiText-2 zero-shot +120.57% (catastrophic). Even WikiText-specific refit gives +31.86%. **The method is corpus-specific.**
- **Cycle 039 (boundary correction)**: Residual error at block 8 is high-rank (rank-95=392/768). But the affine component is strong enough to recover ~3.5pp PPL loss.
- **Hypothesis**: On Shakespeare, the residual E = f_block(x) − W_refit @ x_selected is low-rank or structured (correctable via affine). On WikiText-2, the residual becomes full-rank (chaotic/uncorrectable).

**Task**: For blocks [7, 5, 4] at 6× compression, compute refit residuals on both probe datasets.
- Measure rank-90 via SVD (is rank low on Shakespeare, high on WikiText-2?).
- Measure token correlation structure (are residuals coherent or noise-like?).
- Check alignment of E with discarded-channel subspace.
- If residual rank explains the corpus gap, it reveals the mechanical failure mode.

**Success Criteria**:
- **Confirmed**: residual rank-90 ≤ 50 on Shakespeare AND ≥ 200 on WikiText-2 for ≥2/3 blocks.
- **Refuted**: residual rank similar on both corpora (corpus specificity is not rank-driven).
- **Inconclusive**: rank differs but doesn't predict PPL gap.

**Key Data Paths**:
- Prior compression configs: `experiments/compression_recipe_validation.py` (Shakespeare+WikiText, blocks 7/5/4).
- Shakespeare probe: embedded in `experiments/domain_robustness_wikitext.py` (condition A baseline).
- WikiText-2 probe: same file (condition B baseline).
