"""
Oracle Correction Experiment

Tests whether periodic error correction can recover quantization losses.

Setup:
1. Train float32 model
2. Quantize to int8 (post-training)
3. At inference, run float and quantized in parallel
4. Every N layers, correct quantized with oracle error (float - quant)

Compare: FP32 baseline vs INT8 naive vs INT8 with oracle correction
"""

import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from aleph.datasets import make_spirals, embed_dataset_in_high_dimensional_space
from aleph.visualization import plot_decision_boundary


class MLPWithCorrection(nn.Module):
    """MLP that supports oracle correction during inference."""

    def __init__(self, input_size, hidden_size, output_size, depth):
        super().__init__()
        self.depth = depth

        # Build layers separately (not Sequential) so we can intervene
        self.layers = nn.ModuleList()
        in_features = input_size
        for _ in range(depth):
            self.layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        self.layers.append(nn.Linear(hidden_size, output_size))

        self.relu = nn.ReLU()

    def forward(self, x):
        """Standard forward pass (float32)."""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def forward_quantized(self, x, scale_factors, zero_points, num_bits=8):
        """Forward pass with fake quantization, no correction."""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.relu(layer(x))
            x = fake_quantize(x, scale_factors[i], zero_points[i], num_bits=num_bits)
        x = self.layers[-1](x)
        return x

    def forward_with_oracle_correction(self, x, scale_factors, zero_points, correct_every_n=3, num_bits=8):
        """
        Forward pass with oracle correction every N layers.

        Runs float and quantized paths in parallel.
        Every N layers, corrects quantized path: x_quant += (x_float - x_quant)
        """
        x_float = x.clone()
        x_quant = x.clone()

        layers_since_correction = 0

        for i, layer in enumerate(self.layers[:-1]):
            # Both paths through same layer
            x_float = self.relu(layer(x_float))
            x_quant = self.relu(layer(x_quant))

            # Quantize the quantized path
            x_quant = fake_quantize(x_quant, scale_factors[i], zero_points[i], num_bits=num_bits)

            layers_since_correction += 1

            # Oracle correction every N layers
            if layers_since_correction >= correct_every_n:
                error = x_float - x_quant
                x_quant = x_quant + error  # = x_float
                layers_since_correction = 0

        # Final layer (no ReLU, no quantization)
        x_float = self.layers[-1](x_float)
        x_quant = self.layers[-1](x_quant)

        return x_quant, x_float


def fake_quantize(x, scale, zero_point, num_bits=8):
    """Fake quantize: quantize then dequantize (stays in float32)."""
    if num_bits == 8:
        quant_min, quant_max = -128, 127
    elif num_bits == 4:
        quant_min, quant_max = -8, 7
    else:
        raise ValueError(f"Unsupported num_bits: {num_bits}")

    x_scaled = x / scale + zero_point
    x_rounded = torch.round(x_scaled)
    x_clamped = torch.clamp(x_rounded, quant_min, quant_max)
    x_dequant = (x_clamped - zero_point) * scale

    return x_dequant


def calibrate_quantization(model, calibration_data, num_bits=8):
    """
    Calibrate scale/zero_point for each layer using min-max.

    Returns lists of scale factors and zero points, one per hidden layer.
    """
    model.eval()
    activations = [[] for _ in range(len(model.layers) - 1)]

    with torch.no_grad():
        x = calibration_data
        for i, layer in enumerate(model.layers[:-1]):
            x = model.relu(layer(x))
            activations[i].append(x.clone())

    # Determine quantization range based on bits
    if num_bits == 8:
        quant_max = 127
    elif num_bits == 4:
        quant_max = 7
    else:
        raise ValueError(f"Unsupported num_bits: {num_bits}")

    scale_factors = []
    zero_points = []

    for i, acts in enumerate(activations):
        all_acts = torch.cat(acts, dim=0)
        min_val = all_acts.min().item()
        max_val = all_acts.max().item()

        # Symmetric quantization around zero
        abs_max = max(abs(min_val), abs(max_val))
        scale = abs_max / quant_max if abs_max > 0 else 1.0
        zero_point = 0.0

        scale_factors.append(scale)
        zero_points.append(zero_point)

    return scale_factors, zero_points


def evaluate(model, X, y, mode='float', scale_factors=None, zero_points=None, correct_every_n=3, num_bits=8):
    """Evaluate model accuracy in different modes."""
    model.eval()
    with torch.no_grad():
        if mode == 'float':
            logits = model(X)
        elif mode == 'quantized':
            logits = model.forward_quantized(X, scale_factors, zero_points, num_bits=num_bits)
        elif mode == 'corrected':
            logits, _ = model.forward_with_oracle_correction(
                X, scale_factors, zero_points, correct_every_n, num_bits=num_bits)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()

    return acc


def main():
    parser = argparse.ArgumentParser(description='Oracle Correction Experiment')
    parser.add_argument('--depth', type=int, default=6, help='Number of hidden layers')
    parser.add_argument('--width', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--correct-every', type=int, default=3, help='Correct every N layers')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--target-dim', type=int, default=100, help='Embedding dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--bits', type=int, default=8, help='Quantization bits (4 or 8)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Data setup
    print("Setting up data...")
    X_2d, y = make_spirals(n_samples=2000, noise=0.3, n_turns=3, random_state=args.seed)
    X_high, embedding = embed_dataset_in_high_dimensional_space(
        X_2d, target_dim=args.target_dim, random_state=args.seed)

    X_train_2d, X_test_2d, y_train, y_test = train_test_split(
        X_2d, y, test_size=0.2, random_state=args.seed)
    X_train = torch.tensor(embedding.transform(X_train_2d), dtype=torch.float32)
    X_test = torch.tensor(embedding.transform(X_test_2d), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print(f"Data: {X_2d.shape} -> {X_high.shape}")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Create model
    model = MLPWithCorrection(
        input_size=args.target_dim,
        hidden_size=args.width,
        output_size=2,
        depth=args.depth
    )
    print(f"\nModel: depth={args.depth}, width={args.width}")
    print(f"Correction every {args.correct_every} layers")

    # Train in float32
    print(f"\n{'='*60}")
    print("Training (float32)...")
    print('='*60)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            model.eval()
            train_acc = evaluate(model, X_train, y_train, mode='float')
            test_acc = evaluate(model, X_test, y_test, mode='float')
            print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.4f} | "
                  f"Train: {train_acc:.3f} | Test: {test_acc:.3f}")
            model.train()

    # Calibrate quantization
    print(f"\n{'='*60}")
    print("Calibrating quantization...")
    print('='*60)

    scale_factors, zero_points = calibrate_quantization(model, X_train, num_bits=args.bits)
    for i, (s, z) in enumerate(zip(scale_factors, zero_points)):
        print(f"  Layer {i+1}: scale={s:.6f}, zero_point={z:.1f}")

    # Evaluate all modes
    print(f"\n{'='*60}")
    print("Results")
    print('='*60)

    acc_float = evaluate(model, X_test, y_test, mode='float')
    acc_quant = evaluate(model, X_test, y_test, mode='quantized',
                         scale_factors=scale_factors, zero_points=zero_points,
                         num_bits=args.bits)

    print(f"\n{'Mode':<30} {'Test Accuracy':>12}")
    print("-" * 44)
    print(f"{'FP32 (baseline)':<30} {acc_float:>12.4f}")
    print(f"{'INT8 (no correction)':<30} {acc_quant:>12.4f}")

    # Test different correction frequencies
    for n in [1, 2, 3, args.depth]:
        if n > args.depth:
            continue
        acc_corrected = evaluate(model, X_test, y_test, mode='corrected',
                                 scale_factors=scale_factors, zero_points=zero_points,
                                 correct_every_n=n, num_bits=args.bits)
        label = f"INT8 + oracle (every {n} layers)"
        print(f"{label:<30} {acc_corrected:>12.4f}")

    # Summary
    print(f"\nQuantization gap: {acc_float - acc_quant:.4f}")
    acc_best_corrected = evaluate(model, X_test, y_test, mode='corrected',
                                  scale_factors=scale_factors, zero_points=zero_points,
                                  correct_every_n=1, num_bits=args.bits)
    print(f"Recovery with oracle (N=1): {acc_best_corrected - acc_quant:.4f}")

    # Plot decision boundaries
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Float model
    plot_decision_boundary(model, X_test_2d, y_test.numpy(),
                           f"FP32 (acc={acc_float:.3f})", axes[0], embedding=embedding)

    # Quantized model - need a wrapper for plotting
    class QuantizedWrapper:
        def __init__(self, model, sf, zp, bits):
            self.model = model
            self.sf = sf
            self.zp = zp
            self.bits = bits

        def eval(self):
            self.model.eval()

        def __call__(self, x):
            return self.model.forward_quantized(x, self.sf, self.zp, num_bits=self.bits)

    quant_wrapper = QuantizedWrapper(model, scale_factors, zero_points, args.bits)
    plot_decision_boundary(quant_wrapper, X_test_2d, y_test.numpy(),
                           f"INT8 naive (acc={acc_quant:.3f})", axes[1], embedding=embedding)

    # Corrected model
    class CorrectedWrapper:
        def __init__(self, model, sf, zp, n, bits):
            self.model = model
            self.sf = sf
            self.zp = zp
            self.n = n
            self.bits = bits

        def eval(self):
            self.model.eval()

        def __call__(self, x):
            out, _ = self.model.forward_with_oracle_correction(x, self.sf, self.zp, self.n, num_bits=self.bits)
            return out

    acc_corrected_3 = evaluate(model, X_test, y_test, mode='corrected',
                               scale_factors=scale_factors, zero_points=zero_points,
                               correct_every_n=args.correct_every, num_bits=args.bits)
    corrected_wrapper = CorrectedWrapper(model, scale_factors, zero_points, args.correct_every, args.bits)
    plot_decision_boundary(corrected_wrapper, X_test_2d, y_test.numpy(),
                           f"INT8 + oracle N={args.correct_every} (acc={acc_corrected_3:.3f})",
                           axes[2], embedding=embedding)

    plt.tight_layout()
    plt.savefig("oracle_correction.png", dpi=150)
    print(f"\nSaved decision boundary plot to 'oracle_correction.png'")


if __name__ == "__main__":
    main()
