# Transformer Correction Layer Sweep

## Experiment Overview

Large-scale sweep testing learned correction layers for quantized transformers on Shakespeare character-level language modeling.

### Configuration

- **Seeds**: 42, 123, 999
- **Bits**: 2, 4, 8
- **d_model**: 64, 128, 256
- **n_layers**: 2, 4, 6
- **correct_every**: 1, 2, 4 (correction frequency)
- **correction_hidden**: 0, 32, 64 (0 = linear correction)
- **Total runs**: 243
- **Total time**: ~55 hours

## Key Results

### Recovery by Bit Width

| Bit Width | Mean Recovery | Std  | >80% Recovery |
|-----------|--------------|------|---------------|
| 2-bit     | 54.2%        | 15.2%| 1.2%          |
| 4-bit     | 82.7%        | 4.7% | 75.3%         |
| 8-bit     | 101.4%       | 1.8% | 100%          |

**8-bit achieves >100% recovery** - the correction layer actually improves upon the float model! This suggests the correction is learning useful regularization.

### Perplexity Analysis

| Bit Width | Float PPL | Quantized PPL | Corrected PPL | Recovery |
|-----------|-----------|---------------|---------------|----------|
| 2-bit     | 8.3       | 51.1          | 19.8          | 54%      |
| 4-bit     | 8.3       | 12.7          | 8.8           | 83%      |
| 8-bit     | 8.3       | 9.9           | 8.2           | 101%     |

At 4-bit, correction reduces perplexity from 12.7 to 8.8 (nearly matching float's 8.3).

## Surprising Finding: Linear Correction Beats MLP

| Bit Width | Linear (h=0) | MLP (h>0) | Difference |
|-----------|--------------|-----------|------------|
| 2-bit     | 63.5%        | 49.6%     | **-13.9%** |
| 4-bit     | 86.0%        | 81.1%     | **-4.9%**  |
| 8-bit     | 101.0%       | 101.6%    | +0.5%      |

**Linear correction significantly outperforms MLP correction at 2-bit and 4-bit.** This suggests:
1. Error correction is fundamentally a linear operation (adding back what was lost)
2. MLP correction may overfit with limited training data
3. Simpler is better for this task

### Correction Hidden Size Comparison (4-bit)

| Hidden Size | Mean Recovery |
|-------------|---------------|
| 0 (linear)  | 86.0%         |
| 32          | 79.1%         |
| 64          | 83.0%         |

The pattern holds: linear > 64-hidden > 32-hidden.

## Model Size Effects

### Recovery by d_model

**2-bit** - Smaller models work better:
| d_model | Recovery |
|---------|----------|
| 64      | 70.2%    |
| 128     | 49.9%    |
| 256     | 42.6%    |

**4-bit** - Larger models work better:
| d_model | Recovery |
|---------|----------|
| 64      | 78.9%    |
| 128     | 84.0%    |
| 256     | 85.3%    |

### Recovery by n_layers

**2-bit** - Shallower models work better:
| n_layers | Recovery |
|----------|----------|
| 2        | 70.2%    |
| 4        | 49.9%    |
| 6        | 42.6%    |

**4-bit** - Deeper models work better:
| n_layers | Recovery |
|----------|----------|
| 2        | 78.9%    |
| 4        | 84.0%    |
| 6        | 85.3%    |

**Interpretation**: At 2-bit, quantization error is so severe that larger/deeper models accumulate more uncorrectable error. At 4-bit, the correction layers have enough signal to work with, and more correction opportunities (more layers) helps.

## Correction Frequency

**2-bit**: Less frequent correction is better
| correct_every | Recovery |
|---------------|----------|
| 1             | 50.0%    |
| 2             | 53.9%    |
| 4             | 58.7%    |

**4-bit**: More frequent correction is better
| correct_every | Recovery |
|---------------|----------|
| 1             | 84.7%    |
| 2             | 83.4%    |
| 4             | 80.1%    |

## Best Configurations

### Top Performers (all 8-bit, >100% recovery)
1. seed=999, d=64, L=2, every=4, h=0: **107.2%**
2. seed=999, d=64, L=2, every=4, h=32: **106.6%**
3. seed=42, d=64, L=2, every=2, h=0: **106.5%**

### Best 4-bit Configuration
- d_model=256, n_layers=6, correct_every=1, correction_hidden=0
- Recovery: ~92%

### Best 2-bit Configuration
- d_model=64, n_layers=2, correct_every=4, correction_hidden=0
- Recovery: ~81%

## Conclusions

1. **Use linear correction (hidden=0)** - it outperforms MLP correction at all bit widths except 8-bit where they're equal.

2. **4-bit is the sweet spot** - 83% mean recovery with low variance, practical for deployment.

3. **2-bit requires smaller models** - the quantization error is too severe for large transformers, but small models (d=64, L=2) can achieve ~70% recovery.

4. **8-bit correction > float** - the correction layer provides beneficial regularization, achieving >100% recovery.

5. **Correction frequency depends on bit width**:
   - 2-bit: correct less often (every 4)
   - 4-bit: correct more often (every 1-2)
   - 8-bit: doesn't matter

## Files

- `runs/transformer_sweep_20260120_003848/summary.csv`: Raw results
- `old_experiments/transformer_correction_sweep.py`: Experiment runner
