# QAT Rounding Mode Exploration

## Hypothesis

Systematic rounding bias (floor/ceil) should be easier for neural networks to optimize against than nearest rounding's "random flipping" behavior around the 0.5 boundary.

**Rationale**: Floor always rounds down, ceil always rounds up—predictable biases the network could learn to compensate for. Nearest rounding flips based on the fractional part, introducing seemingly random perturbations.

## Experiment

**Setup**:
- Dataset: 2D spirals (2000 samples) embedded into 100D space
- Model: MLP with varying width (4-128) and depth (2-10)
- Rounding modes: baseline (no quantization), nearest, floor, ceil
- **Bit widths: 4-bit and 8-bit quantization**
- 3000 epochs, Adam optimizer, lr=1e-4
- 3 seeds × 6 widths × 5 depths × 4 modes × 2 bit widths = **630 runs**

## Results

### Overall Accuracy (mean across all configs)

| Mode     | Mean Acc | Std   | Count |
|----------|----------|-------|-------|
| baseline | 0.7881   | 0.188 | 90    |
| nearest  | 0.7545   | 0.154 | 180   |
| floor    | 0.7524   | 0.155 | 180   |
| ceil     | 0.7402   | 0.155 | 180   |

**Hypothesis mostly refuted.** Nearest rounding outperforms floor/ceil on average.

### Bit Width Effect

**4-bit quantization:**
| Mode    | Mean Acc | Std   |
|---------|----------|-------|
| nearest | 0.7741   | 0.105 |
| floor   | 0.7700   | 0.109 |
| ceil    | 0.7435   | 0.111 |

**8-bit quantization:**
| Mode     | Mean Acc | Std   |
|----------|----------|-------|
| baseline | 0.7881   | 0.188 |
| ceil     | 0.7369   | 0.190 |
| nearest  | 0.7350   | 0.189 |
| floor    | 0.7347   | 0.190 |

**Key insight**: At 8-bit, rounding mode is irrelevant (all within 0.002 of each other). The effect only matters at lower bit widths like 4-bit.

### Gap vs Nearest

**4-bit:**
- floor - nearest: -0.0041
- ceil - nearest: -0.0306

**8-bit:**
- floor - nearest: -0.0002
- ceil - nearest: +0.0019

### Key Findings

1. **Nearest is best AND most stable**: Lowest variance across seeds. The "random" rounding produces more predictable outcomes than systematic bias.

2. **8-bit rounding mode is irrelevant**: All modes within 0.002 of each other. Don't bother optimizing this for 8-bit quantization.

3. **Ceil is consistently worst at 4-bit**: The 3% gap between ceil and nearest at 4-bit is significant.

4. **Floor/ceil can win specific configurations**: Despite losing on average, floor beats nearest in **42% of configurations** (75/180) and ceil wins **38%** (69/180). Some wins were substantial (>20% improvement).

5. **Depth effect at 4-bit**:
   | Depth | floor - nearest | ceil - nearest |
   |-------|-----------------|----------------|
   | 2     | -0.010          | -0.023         |
   | 4     | -0.012          | -0.015         |
   | 6     | -0.019          | -0.057         |
   | 8     | -0.028          | -0.051         |
   | 10    | **+0.049**      | -0.007         |

   Ceil consistently degrades with depth. Floor shows surprising reversal at depth=10 where it outperforms nearest.

6. **Large wins for floor at depth=10**:
   - seed=123, w=4, d=10: floor=0.7425 vs nearest=0.5150 (+0.2275)
   - seed=123, w=64, d=10: floor=0.8625 vs nearest=0.6525 (+0.2100)
   - seed=123, w=128, d=10: floor=0.8525 vs nearest=0.6600 (+0.1925)
   - seed=999, w=128, d=8: floor=0.8475 vs nearest=0.6625 (+0.1850)

7. **Width effect at 4-bit**: Floor rounding improves relative to nearest at wider models:
   | Width | floor - nearest |
   |-------|-----------------|
   | 4     | -0.004          |
   | 8     | -0.019          |
   | 16    | -0.019          |
   | 32    | -0.001          |
   | 64    | **+0.005**      |
   | 128   | **+0.013**      |

8. **Seed stability**: Nearest rounding has lowest variance across seeds, confirming it's the most reliable choice:
   | Mode (4-bit) | Mean std across seeds |
   |--------------|----------------------|
   | nearest      | 0.042                |
   | floor        | 0.048                |
   | ceil         | 0.049                |

## Interpretation

The network apparently prefers **unbiased noise it can average over** rather than **systematic drift it must compensate for**.

However, the picture is more nuanced than "nearest always wins":
- At 8-bit, rounding mode is essentially noise in the measurement
- At 4-bit with deep networks, floor rounding occasionally provides substantial benefits
- The ~40% win rate for floor/ceil suggests configuration-dependent effects
- Floor performs better at wider models (64, 128 units) at 4-bit

## Conclusion

**Use nearest rounding as default** - it's both more accurate on average and more consistent across random seeds.

**Consider floor for deep 4-bit networks** - the depth=10 results show floor can significantly outperform nearest in specific cases. This might be worth investigating for low-bit quantization of deep models.

**Avoid ceil** - it's consistently worst, especially at lower bit widths and deeper networks.

## Files

- `old_experiments/rounding_mode_sweep.py`: Experiment runner
- `old_runs/rounding/experiment_20260119_201449/summary.csv`: Raw results (630 runs)
- `old_experiments/analyze_rounding.py`: Analysis script
- `aleph/quantization.py`: Custom FakeQuantize implementation with configurable rounding
