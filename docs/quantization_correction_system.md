# Quantization and Learned Correction System

A comprehensive technical document describing the quantization approach and learned correction layers for recovering accuracy lost during neural network quantization.

## Table of Contents

1. [Overview](#overview)
2. [Quantization Fundamentals](#quantization-fundamentals)
3. [Fake Quantization](#fake-quantization)
4. [Calibration](#calibration)
5. [Learned Correction Layers](#learned-correction-layers)
6. [Training Procedure](#training-procedure)
7. [Quantized Correction Layers](#quantized-correction-layers)
8. [Architecture Details](#architecture-details)
9. [Implementation Reference](#implementation-reference)

---

## Overview

This system implements **post-training quantization with learned correction layers**. The key insight is that quantization introduces systematic errors that accumulate through network layers. By inserting small correction networks at regular intervals, we can predict and compensate for these errors, recovering much of the accuracy lost to quantization.

### Key Components

1. **Fake Quantization**: Simulates quantization effects while maintaining float32 computation for gradient flow
2. **Symmetric Quantization**: Uses zero_point=0 for simpler arithmetic
3. **Per-tensor Activation Quantization**: One scale factor per layer activation
4. **Learned Correction Networks**: Small networks that predict corrections from accumulated quantization error
5. **Distillation-based Training**: Corrections are trained to match float model outputs, not task labels

---

## Quantization Fundamentals

### Bit Width Support

The system supports 2-bit, 4-bit, and 8-bit quantization with signed integer ranges:

| Bits | Range | quant_min | quant_max |
|------|-------|-----------|-----------|
| 2    | [-2, 1] | -2 | 1 |
| 4    | [-8, 7] | -8 | 7 |
| 8    | [-128, 127] | -128 | 127 |

```python
SUPPORTED_BITS = {
    2: (-2, 1),
    4: (-8, 7),
    8: (-128, 127),
}
```

### Quantization Formula

For a floating-point value `x`, the quantized-then-dequantized value is:

```
x_scaled = x / scale + zero_point
x_rounded = round(x_scaled)
x_clamped = clamp(x_rounded, quant_min, quant_max)
x_dequant = (x_clamped - zero_point) * scale
```

**Quantization error**: `error = x - x_dequant`

### Symmetric Quantization

This implementation uses **symmetric quantization** with `zero_point = 0`:

```
scale = max(|x_min|, |x_max|) / quant_max
x_quant = clamp(round(x / scale), quant_min, quant_max)
x_dequant = x_quant * scale
```

Benefits:
- Simpler arithmetic (no zero_point subtraction)
- Zero maps exactly to zero
- Efficient for activations that are roughly symmetric around zero (post-ReLU activations are not symmetric, but this is acceptable)

---

## Fake Quantization

### Purpose

Fake quantization simulates quantization effects while keeping values in float32. This enables:
1. **Gradient computation**: Backpropagation works because values remain differentiable
2. **Quantization-aware training**: Model can adapt to quantization noise during training
3. **Evaluation**: Accurately predicts quantized model behavior

### Implementation

```python
def fake_quantize(x, scale, zero_point, num_bits=8):
    """Fake quantize: quantize then dequantize (stays in float32)."""
    quant_min, quant_max = get_quant_range(num_bits)

    x_scaled = x / scale + zero_point
    x_rounded = torch.round(x_scaled)
    x_clamped = torch.clamp(x_rounded, quant_min, quant_max)
    x_dequant = (x_clamped - zero_point) * scale

    return x_dequant
```

### Straight-Through Estimator (STE)

The rounding operation `round()` has zero gradient almost everywhere. To enable gradient flow, we use the **Straight-Through Estimator**:

```python
x_rounded = x_scaled + (torch.round(x_scaled) - x_scaled).detach()
```

This computes:
- **Forward pass**: Uses the rounded value
- **Backward pass**: Gradient flows through as if no rounding occurred

### Fake Quantize with Error

For correction layers, we need both the quantized value and the quantization error:

```python
def fake_quantize_with_error(x, scale, zero_point, num_bits=8):
    """Fake quantize with STE and return quantization error."""
    quant_min, quant_max = get_quant_range(num_bits)

    x_scaled = x / scale + zero_point
    x_rounded = torch.round(x_scaled)
    x_ste = x_scaled + (x_rounded - x_scaled).detach()  # STE
    x_clamped = torch.clamp(x_ste, quant_min, quant_max)
    x_dequant = (x_clamped - zero_point) * scale
    error = x - x_dequant  # What we lost

    return x_dequant, error
```

---

## Calibration

### Purpose

Calibration determines the **scale factors** for each quantization point by analyzing activation ranges on representative data.

### Algorithm

For each layer activation tensor:

```python
def compute_scales(activations, num_bits=8):
    _, quant_max = get_quant_range(num_bits)

    scale_factors = []
    for acts in activations:
        abs_max = max(abs(acts.min()), abs(acts.max()))
        scale = abs_max / quant_max if abs_max > 0 else 1.0
        scale_factors.append(scale)

    return scale_factors
```

### Calibration Procedure

1. **Run float model** on calibration data (typically training set or subset)
2. **Collect activations** at each quantization point
3. **Compute scale** for each activation: `scale = max(|min|, |max|) / quant_max`
4. **Store scales** for use during quantized inference

### What Gets Quantized

In this implementation, we quantize **activations only**, not weights:
- After each linear layer + activation function
- The output is quantized before being passed to the next layer

This is simpler than full weight+activation quantization and is effective for demonstrating the correction approach.

---

## Learned Correction Layers

### Concept

Quantization error accumulates through layers. At regular intervals, we insert a small **correction network** that:
1. Takes the **accumulated quantization error** as input
2. Predicts a **correction term** to add to the quantized activation
3. Resets the error accumulator

### Error Accumulation

Between correction points, quantization errors are summed:

```python
error_accum = None

for i, layer in enumerate(backbone):
    x = relu(layer(x))
    x, err = fake_quantize_with_error(x, scales[i], 0.0, num_bits)

    # Accumulate error
    error_accum = err if error_accum is None else error_accum + err

    # Apply correction at designated positions
    if i in correction_positions:
        correction = correction_net(error_accum)
        x = x + correction
        error_accum = None  # Reset accumulator
```

### Correction Network Architecture

The `CorrectionNet` can be either:

**Linear (hidden_size=0)**:
```python
self.layers = nn.Linear(size, size)
```

**MLP (hidden_size>0)**:
```python
self.layers = nn.Sequential(
    nn.Linear(size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, size),
)
```

### Zero Initialization

The correction network's output layer is initialized to **zero weights and zero bias**:

```python
nn.init.zeros_(self.layers[2].weight)
nn.init.zeros_(self.layers[2].bias)
```

This ensures:
- **Untrained corrections do nothing**: The model starts as if no correction exists
- **Gradual learning**: Corrections deviate from zero only when beneficial
- **Safe default**: If training fails, the model isn't worse than uncorrected

### Correction Frequency

Corrections are applied every N quantization points, controlled by `correction_every_n`:

```python
correction_positions = [i for i in range(depth) if (i + 1) % correction_every_n == 0]
```

Example with `depth=6, correction_every_n=2`:
- Corrections at layers: 1, 3, 5 (after layers 2, 4, 6)
- 3 correction networks total

---

## Training Procedure

### Overview

Training follows a **two-phase approach**:

1. **Train float model** on the task (classification, reconstruction, etc.)
2. **Freeze backbone**, train only correction layers using **distillation**

### Why Distillation?

Training corrections on the original task loss can lead to:
- Corrections learning task-specific improvements (not error correction)
- Recovery rates >100% (corrections "cheating")
- Unstable training

**Distillation loss** ensures corrections only recover what quantization lost:

```python
# Teacher: float model output
with torch.no_grad():
    teacher_logits = model.forward_float(X_train)

# Student: quantized + corrected output
student_logits = model.forward_with_correction(X_train, scales, num_bits)

# Loss: match the teacher
loss = F.mse_loss(student_logits, teacher_logits)
```

### Hybrid Distillation (Output + Layer)

Optionally, we can add **per-layer supervision** to help corrections learn faster:

```
Loss = MSE(output_corrected, output_float) + λ * Σ MSE(activation_corrected[i], activation_float[i])
```

The `λ` parameter balances output-level and layer-level supervision:
- `λ=0`: Output-only distillation
- `λ>0`: Hybrid distillation with per-layer supervision

### Loss Normalization

Losses are normalized by **target variance** to ensure comparable scales:

```python
output_norm = teacher_logits.var().item()
layer_norms = {k: teacher_activations[k].var().item() for k in teacher_activations}

output_loss = F.mse_loss(student_logits, teacher_logits) / output_norm
layer_loss = sum(F.mse_loss(acts[k], teacher_acts[k]) / layer_norms[k]
                 for k in acts) / len(acts)
```

### QAT for Correction Layers

The correction networks themselves can be quantized for fully-quantized inference. Training uses **Quantization-Aware Training (QAT)**:

**Phase 1: Warmup (FP corrections)**
- Train corrections in float32 for ~25% of epochs
- Establishes reasonable weight values

**Phase 2: QAT (quantized corrections)**
- Quantize correction network activations during training
- Uses STE for gradient flow
- Periodically re-calibrate correction scales

```python
# Warmup phase
for _ in range(warmup_epochs):
    out = model.forward_with_correction(X, scales, num_bits, quantize_correction=False)
    loss = F.mse_loss(out, teacher_logits)
    loss.backward()
    optimizer.step()

# Calibrate correction networks
model.calibrate_corrections(X, scales, num_bits)

# QAT phase
for _ in range(qat_epochs):
    out = model.forward_with_correction(X, scales, num_bits, quantize_correction=True)
    loss = F.mse_loss(out, teacher_logits)
    loss.backward()
    optimizer.step()

    # Re-calibrate periodically
    if epoch % 50 == 0:
        model.calibrate_corrections(X, scales, num_bits)
```

---

## Quantized Correction Layers

### Motivation

For deployment, we want the entire inference path quantized, including corrections.

### Implementation

The `CorrectionNet.forward()` method supports optional quantization:

```python
def forward(self, error, quantize=False, num_bits=8):
    if not quantize or self.hidden_size == 0:
        return self.layers(error)

    # Quantized forward
    h = F.relu(self.layers[0](error))
    if 'hidden' in self._scales:
        h = fake_quantize(h, self._scales['hidden'], 0.0, num_bits)

    out = self.layers[2](h)
    if 'output' in self._scales:
        out = fake_quantize(out, self._scales['output'], 0.0, num_bits)

    return out
```

### Correction Calibration

Each correction network has its own scale factors, calibrated on the errors it will receive:

```python
def calibrate_corrections(self, x, scales, num_bits=8):
    error_accum = None

    with torch.no_grad():
        for i, layer in enumerate(backbone):
            x = relu(layer(x))
            _, err = fake_quantize_with_error(x, scales[i], 0.0, num_bits)
            error_accum = err if error_accum is None else error_accum + err

            if str(i) in self.corrections:
                # Calibrate this correction on the errors it will see
                self.corrections[str(i)].calibrate(error_accum, num_bits)
                error_accum = None
```

### Linear vs MLP Corrections (Empirical Findings)

Surprisingly, **linear corrections outperform MLP corrections** in most cases:

| Bit Width | Linear (h=0) | MLP (h>0) | Difference |
|-----------|--------------|-----------|------------|
| 2-bit     | 63.5%        | 49.6%     | -13.9%     |
| 4-bit     | 86.0%        | 81.1%     | -4.9%      |
| 8-bit     | 101.0%       | 101.6%    | +0.5%      |

Possible explanations:
- Error correction is fundamentally linear (adding back what was lost)
- MLP corrections may overfit on limited training data
- Simpler models are more robust to quantization

---

## Architecture Details

### MLP with Correction

```
Input → [Linear → ReLU → FakeQuant]×N → Linear → Output
              ↓
         CorrectionNet (every M layers)
```

**Quantization points**: After each ReLU (N points for N hidden layers)

**Correction input**: Accumulated error since last correction

### Autoencoder with Correction

```
Encoder:  Input → [Linear → ReLU → FakeQuant]×K → Linear → Latent
                       ↓
                  CorrectionNet

Decoder:  Latent → FakeQuant → [Linear → ReLU → FakeQuant]×K → Linear → Output
                                     ↓
                                CorrectionNet
```

**Special handling**: Error accumulator resets at latent layer (dimension change)

### Transformer with Correction

```
Input → Embed + PosEmbed → [TransformerBlock]×L → LayerNorm → Linear → Output

TransformerBlock:
    x → LN → Attention → + → FakeQuant → (correction?) →
                         ↑              ↓
                         x    accumulated error

    → LN → FFN → + → FakeQuant → (correction?)
              ↑
              x
```

**Quantization points**: 2 per transformer block (after attention, after FFN)

**Correction positions**: Determined by `correction_every_n` across all quant points

---

## Implementation Reference

### Key Files

| File | Description |
|------|-------------|
| `aleph/quantization.py` | Fake quantization, calibration, scale computation |
| `aleph/models.py` | Model architectures with correction support |
| `experiments/lambda_sweep.py` | Training with hybrid distillation + QAT |
| `experiments/layer_distillation.py` | Detailed distillation experiments |

### Key Functions

**Quantization**:
- `fake_quantize(x, scale, zero_point, num_bits)` - Basic fake quantization
- `fake_quantize_with_error(x, scale, zero_point, num_bits)` - Returns quantized value and error
- `compute_scales(activations, num_bits)` - Calibration scale computation

**Model Methods**:
- `model.forward(x)` - Float forward pass
- `model.forward_quantized(x, scales, num_bits)` - Quantized, no correction
- `model.forward_with_correction(x, scales, num_bits, quantize_correction)` - Quantized with correction
- `model.calibrate(x, num_bits)` - Calibrate backbone scales
- `model.calibrate_corrections(x, scales, num_bits)` - Calibrate correction network scales
- `model.get_float_activations(x)` - Get teacher targets for distillation

### Hyperparameters

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| `num_bits` | 4 | Quantization bit width |
| `correction_every_n` | 2 | Layers between corrections |
| `correction_hidden` | 0 or 32 | Correction network hidden size (0=linear) |
| `correction_epochs` | 300-1000 | Training epochs for corrections |
| `lr` | 1e-4 | Correction training learning rate |
| `warmup_ratio` | 0.25 | Fraction of epochs for FP warmup |
| `layer_loss_weight` (λ) | 0.0-1.0 | Weight for per-layer distillation loss |

---

## Known Failure Modes

1. **Wrong path through ReLU**: Quantized values may trigger different ReLU decisions than float. Correction happens AFTER this damage — the activation pattern is already different.

2. **Dimension changes reset error accumulator**: When layer sizes change (e.g., at the autoencoder latent layer), accumulated error vectors can't be summed across the boundary. The accumulator must reset.

3. **Overfitting correction layers**: Small correction networks trained on limited data may not generalize. This is especially problematic for MLP corrections with hidden layers.

---

## Summary

This quantization correction system provides:

1. **Flexible bit-width support** (2, 4, 8-bit)
2. **Learned correction** that recovers 50-100%+ of quantization accuracy loss
3. **Distillation-based training** for stable, bounded recovery
4. **Quantized corrections** for fully-quantized deployment
5. **Simple linear corrections** that often outperform complex MLPs

The approach is particularly effective at 4-bit (80%+ recovery) and provides useful improvements even at 2-bit (50%+ recovery on simpler architectures).
