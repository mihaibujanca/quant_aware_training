# Geometric FFN Compression Recipe — GPT-2 Family

## Summary

No-SGD geometric compression of transformer FFN blocks via topk channel selection, least-squares output refit, and residual-stream affine correction. Achieves ~28-30% FFN parameter reduction at +6-9% perplexity on narrow-distribution text. Zero extra parameters at inference.

## IMPORTANT: Corpus Specificity Caveat

**This method is highly corpus-specific.** All "good" numbers below were measured on Shakespeare text. On broader-distribution text (WikiText-2), even corpus-specific compression degrades perplexity by +32%. Zero-shot transfer from Shakespeare to WikiText is catastrophic (+120%). **The probe corpus MUST match the target distribution.** If using the model's training data for probing, results may improve; this has not been tested.

## Recipe

### 1. Select blocks
- Compress: blocks [7, 5, 4, 1] (4 easiest by single-block damage from cycle 005)
- Skip: block 0 (catastrophic +29% on all model sizes), blocks 8-11 (higher damage)
- Do NOT compress more than 4 blocks (k=5 wall: +16% min regardless of ratio/ordering)

### 2. Compress each block
- **Ratio**: 6× (hidden_fraction=1/6). 10× also works with correction but offers marginal additional savings.
- **Channel selection**: topk by data-conditioned activity × output-norm score. Use `CompressionBudgetSpec(hidden_fraction=1/6, selection_mode="topk")`.
- **Output refit**: least-squares refit of W_down on probe token activations.
- **Probe corpus**: 200 sequences, seq_len=128. **Must match eval distribution.**

### 3. Fit residual correction
- **Location**: single affine correction at block 8 input (block after last compressed block)
- **Fitting**: lstsq on ~25K probe tokens. Augmented system: [N, 769] → [N, 768].
- **Application**: forward pre-hook on `model.transformer.h[8]`. Zero inference params.
- Do NOT use multi-point corrections (cascading corrections interfere, cycle 043).

### 4. Expected quality

| Condition | PPL cost | FFN reduction |
|-----------|---------|---------------|
| k=4@6× corrected, narrow-distribution | +6-9% | ~28% |
| k=4@10× corrected, narrow-distribution | +8-9% | ~30% |
| k=4@6× corrected, broad-distribution (WikiText-2) | +32% | ~28% |
| Single-block any ratio | ~0% | ~8% per block |

### 5. Failure modes
- **Block 0**: always catastrophic (+29%), skip it
- **k≥5 blocks**: +16% minimum regardless of per-block ratio or ordering
- **Cross-domain transfer**: +120%, do not transfer compressed blocks across domains
- **Multi-point corrections**: cascading corrections interfere, use single end-point only

## Validated on
- GPT-2-small (12 layers, 768 hidden, 3072 FFN): cycles 033-044
- GPT-2-medium (24 layers, 1024 hidden, 4096 FFN): cycle 041 (single-block only)

## Code
- Library: `aleph/lm_geometric_compression.py`
- Validation script: `experiments/compression_recipe_validation.py`
