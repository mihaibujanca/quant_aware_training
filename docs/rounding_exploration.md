# QAT Rounding Mode Exploration

## Hypothesis

Systematic rounding bias (floor/ceil) should be easier for neural networks to optimize against than nearest rounding's "random flipping" behavior around the 0.5 boundary.

**Rationale**: Floor always rounds down, ceil always rounds up—predictable biases the network could learn to compensate for. Nearest rounding flips based on the fractional part, introducing seemingly random perturbations.

## Experiment

**Setup**:
- Dataset: 2D spirals (2000 samples) embedded into 100D space
- Model: MLP with varying width (4-128) and depth (2-10)
- Rounding modes: baseline (no quantization), nearest, floor, ceil
- 8-bit quantization via PyTorch QAT
- 3000 epochs, Adam optimizer, lr=1e-4
- 3 seeds × 6 widths × 5 depths × 4 modes = 360 runs

## Results

### Overall Accuracy (mean across all configs)

| Mode     | Mean Acc | Std   |
|----------|----------|-------|
| baseline | 0.788    | 0.026 |
| nearest  | 0.746    | 0.023 |
| floor    | 0.738    | 0.030 |
| ceil     | 0.738    | 0.038 |

**Hypothesis refuted.** Nearest rounding outperforms floor/ceil by ~1%.

### Key Findings

1. **Nearest is best AND most consistent**: Lowest variance across seeds (0.023 vs 0.030-0.038). The "random" rounding produces more predictable outcomes than systematic bias.

2. **No depth accumulation effect**: The floor/ceil penalty doesn't grow with network depth. At depth=10, floor/ceil actually slightly outperform nearest. The "bias compounds through layers" theory doesn't hold.

3. **Differences shrink for well-trained models**: Among configs achieving >70% accuracy, all QAT modes cluster within 2% of each other (90.4-92.0%).

4. **Win rate**: Nearest wins 38/90 configs outright. Floor+ceil together win 52/90, but their wins are scattered across different configurations with no clear pattern.

5. **Width matters more than rounding mode**: The quantization penalty is largest at width=16 (10% gap to baseline) and smallest at width=64-128 (3-4% gap).

## Interpretation

The network apparently prefers **unbiased noise it can average over** rather than **systematic drift it must compensate for**.

Nearest rounding has E[error] ≈ 0 (symmetric around true value). Floor/ceil have E[error] ≠ 0, introducing a consistent bias that shifts activation distributions. This bias doesn't "compound" through layers in a simple way, but it does make optimization harder.

## Conclusion

**Not worth pursuing further.** Nearest rounding (PyTorch's default) is the right choice—it's both more accurate on average and more consistent across random seeds. The intuition about predictable bias being easier to optimize was incorrect.

## Files

- `experiment.py`: Experiment runner with TensorBoard logging
- `notebooks/analysis.ipynb`: Detailed analysis notebook
- `runs/experiment_20260118_151239/summary.csv`: Raw results
- `aleph/quantization.py`: Custom FakeQuantize implementation with configurable rounding
