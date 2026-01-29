"""
Oracle Correction Experiment

Tests whether periodic error correction can recover quantization losses.

Setup:
1. Train float32 model
2. Quantize to int8 (post-training)
3. At inference, run float and quantized in parallel
4. Every N layers, correct quantized with oracle error (float - quant)

Compare: FP32 baseline vs INT8/INT4 naive vs INT8/INT4 with oracle correction
"""

import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from aleph.datasets import make_spirals, embed_dataset_in_high_dimensional_space
from aleph.models import MLPWithCorrection, MLPWithLearnedCorrection
from aleph.quantization import calibrate_quantization
from aleph.visualization import (
    plot_decision_boundary,
    QuantizedWrapper,
    OracleWrapper,
    LearnedCorrectionWrapper,
)

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


def estimate_mlp_macs(input_size, hidden_size, output_size, depth):
    """Estimate MACs for a forward pass through the MLP (ignores activations)."""
    macs = input_size * hidden_size
    macs += (depth - 1) * hidden_size * hidden_size
    macs += hidden_size * output_size
    return macs


def format_size_bytes(num_bytes):
    return f"{num_bytes / (1024 ** 2):.2f} MB"


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
    parser.add_argument('--learned-epochs', type=int, default=600, help='Training epochs for learned correction')
    parser.add_argument('--learned-lr', type=float, default=1e-4, help='Learning rate for learned correction')
    parser.add_argument('--correction-hidden', type=int, default=32, help='Hidden size for correction MLP')
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

    # Efficiency estimates (weights only)
    total_params = sum(p.numel() for p in model.parameters())
    macs = estimate_mlp_macs(args.target_dim, args.width, 2, args.depth)
    fp32_bytes = total_params * 4
    quant_bytes = total_params * (args.bits / 8)

    print(f"Params: {total_params:,} | MACs (per fwd): {macs:,}")
    print(f"Model size FP32: {format_size_bytes(fp32_bytes)} | "
          f"INT{args.bits}: {format_size_bytes(quant_bytes)} | "
          f"Savings: {fp32_bytes / quant_bytes:.1f}x")

    acc_float = evaluate(model, X_test, y_test, mode='float')
    acc_quant = evaluate(model, X_test, y_test, mode='quantized',
                         scale_factors=scale_factors, zero_points=zero_points,
                         num_bits=args.bits)

    print(f"\n{'Mode':<30} {'Test Accuracy':>12}")
    print("-" * 44)
    print(f"{'FP32 (baseline)':<30} {acc_float:>12.4f}")
    print(f"{f'INT{args.bits} (no correction)':<30} {acc_quant:>12.4f}")

    # Train learned correction model
    print(f"\n{'='*60}")
    print("Training learned correction...")
    print('='*60)
    learned_model = MLPWithLearnedCorrection(
        input_size=args.target_dim,
        hidden_size=args.width,
        output_size=2,
        depth=args.depth,
        correction_every_n=args.correct_every,
        correction_hidden=args.correction_hidden,
    )
    learned_model.load_state_dict(model.state_dict(), strict=False)

    learned_criterion = nn.CrossEntropyLoss()
    learned_optimizer = torch.optim.Adam(learned_model.parameters(), lr=args.learned_lr)

    learned_model.train()
    for epoch in range(args.learned_epochs):
        learned_optimizer.zero_grad()
        logits = learned_model.forward_quantized_with_correction(
            X_train, scale_factors, zero_points, num_bits=args.bits
        )
        loss = learned_criterion(logits, y_train)
        loss.backward()
        learned_optimizer.step()

        if (epoch + 1) % 200 == 0:
            learned_model.eval()
            preds = learned_model.forward_quantized_with_correction(
                X_test, scale_factors, zero_points, num_bits=args.bits
            )
            acc = (preds.argmax(1) == y_test).float().mean().item()
            print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.4f} | Test: {acc:.3f}")
            learned_model.train()

    learned_model.eval()
    with torch.no_grad():
        logits = learned_model.forward_quantized_with_correction(
            X_test, scale_factors, zero_points, num_bits=args.bits
        )
        acc_learned = (logits.argmax(1) == y_test).float().mean().item()
    print(f"{f'INT{args.bits} + learned correction':<30} {acc_learned:>12.4f}")

    acc_oracle = evaluate(model, X_test, y_test, mode='corrected',
                          scale_factors=scale_factors, zero_points=zero_points,
                          correct_every_n=args.correct_every, num_bits=args.bits)

    # Efficiency comparison (learned correction adds parameters/MACs)
    learned_params = sum(p.numel() for p in learned_model.parameters())
    learned_bytes = learned_params * (args.bits / 8)
    learned_macs = macs + (
        len(learned_model.correction_layers)
        * (args.width * args.correction_hidden + args.correction_hidden * args.width)
    )

    print(f"\n{'Mode':<30} {'Accuracy':>10} {'Params':>12} {'MACs':>12} {'Model Size':>14}")
    print("-" * 86)
    print(f"{'FP32 baseline':<30} {acc_float:>10.4f} {total_params:>12,} {macs:>12,} {format_size_bytes(fp32_bytes):>14}")
    print(f"{f'INT{args.bits} naive':<30} {acc_quant:>10.4f} {total_params:>12,} {macs:>12,} {format_size_bytes(quant_bytes):>14}")
    print(f"{f'INT{args.bits} oracle':<30} {acc_oracle:>10.4f} {total_params:>12,} {macs * 2:>12,} {format_size_bytes(fp32_bytes):>14}")
    print(f"{f'INT{args.bits} learned':<30} {acc_learned:>10.4f} {learned_params:>12,} {learned_macs:>12,} {format_size_bytes(learned_bytes):>14}")

    # Test different correction frequencies
    for n in [1, 2, 3, args.depth]:
        if n > args.depth:
            continue
        acc_corrected = evaluate(model, X_test, y_test, mode='corrected',
                                 scale_factors=scale_factors, zero_points=zero_points,
                                 correct_every_n=n, num_bits=args.bits)
        label = f"INT{args.bits} + oracle (every {n} layers)"
        print(f"{label:<30} {acc_corrected:>12.4f}")

    # Summary
    print(f"\nQuantization gap: {acc_float - acc_quant:.4f}")
    acc_best_corrected = evaluate(model, X_test, y_test, mode='corrected',
                                  scale_factors=scale_factors, zero_points=zero_points,
                                  correct_every_n=1, num_bits=args.bits)
    print(f"Recovery with oracle (N=1): {acc_best_corrected - acc_quant:.4f}")

    # Plot decision boundaries
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # Float model
    plot_decision_boundary(model, X_test_2d, y_test.numpy(),
                           f"FP32 (acc={acc_float:.3f})", axes[0], embedding=embedding)

    # Quantized model
    quant_wrapper = QuantizedWrapper(model, scale_factors, zero_points, args.bits)
    plot_decision_boundary(quant_wrapper, X_test_2d, y_test.numpy(),
                           f"INT{args.bits} naive (acc={acc_quant:.3f})", axes[1], embedding=embedding)

    # Oracle corrected model
    acc_corrected_3 = evaluate(model, X_test, y_test, mode='corrected',
                               scale_factors=scale_factors, zero_points=zero_points,
                               correct_every_n=args.correct_every, num_bits=args.bits)
    oracle_wrapper = OracleWrapper(model, scale_factors, zero_points, args.correct_every, args.bits)
    plot_decision_boundary(oracle_wrapper, X_test_2d, y_test.numpy(),
                           f"INT{args.bits} + oracle N={args.correct_every} (acc={acc_corrected_3:.3f})",
                           axes[2], embedding=embedding)

    # Learned correction model
    learned_wrapper = LearnedCorrectionWrapper(learned_model, scale_factors, zero_points, args.bits)
    plot_decision_boundary(learned_wrapper, X_test_2d, y_test.numpy(),
                           f"INT{args.bits} + learned (acc={acc_learned:.3f})",
                           axes[3], embedding=embedding)

    plt.tight_layout()
    plt.savefig("oracle_correction.png", dpi=150)
    print(f"\nSaved decision boundary plot to 'oracle_correction.png'")


if __name__ == "__main__":
    main()
