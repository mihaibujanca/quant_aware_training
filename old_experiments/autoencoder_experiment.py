"""
Autoencoder Quantization Correction Experiment

Tests learned correction layers on reconstruction task with MNIST.
"""

import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from aleph.datasets import load_mnist_flat
from aleph.models import AutoencoderWithCorrection
from aleph.quantization import calibrate_model


def evaluate_reconstruction(model, X, mode='float', scale_factors=None, zero_points=None, num_bits=8):
    """Evaluate reconstruction MSE."""
    model.eval()
    with torch.no_grad():
        if mode == 'float':
            recon = model(X)
        elif mode == 'quantized':
            recon = model.forward_quantized(X, scale_factors, zero_points, num_bits=num_bits)
        elif mode == 'corrected':
            recon = model.forward_quantized_with_correction(X, scale_factors, zero_points, num_bits=num_bits)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        mse = nn.functional.mse_loss(recon, X).item()
    return mse


def main():
    parser = argparse.ArgumentParser(description='Autoencoder Correction Experiment')
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 128], help='Hidden layer sizes')
    parser.add_argument('--latent', type=int, default=32, help='Latent dimension')
    parser.add_argument('--correct-every', type=int, default=2, help='Correct every N layers')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--bits', type=int, default=8, help='Quantization bits (4 or 8)')
    parser.add_argument('--learned-epochs', type=int, default=10, help='Epochs for learned correction')
    parser.add_argument('--correction-hidden', type=int, default=32, help='Hidden size for correction MLP')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, test_loader = load_mnist_flat(batch_size=args.batch_size)

    # Get a batch for calibration and visualization
    X_test, _ = next(iter(test_loader))
    X_test = X_test.to(device)

    input_size = 784  # 28x28

    # Create model
    model = AutoencoderWithCorrection(
        input_size=input_size,
        hidden_sizes=args.hidden,
        latent_size=args.latent,
        correction_every_n=args.correct_every,
        correction_hidden=args.correction_hidden,
    ).to(device)

    print(f"Model: hidden={args.hidden}, latent={args.latent}")
    print(f"Correction every {args.correct_every} layers")
    print(f"Correction layers: {len(model.correction_layers)}")

    # Train float model
    print(f"\n{'='*60}")
    print("Training autoencoder (float32)...")
    print('='*60)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            recon = model(X_batch)
            loss = criterion(recon, X_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            test_mse = evaluate_reconstruction(model, X_test, mode='float')
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_loss:.6f} | Test MSE: {test_mse:.6f}")

    # Calibrate
    print(f"\n{'='*60}")
    print("Calibrating quantization...")
    print('='*60)

    X_calib, _ = next(iter(train_loader))
    X_calib = X_calib.to(device)
    scale_factors, zero_points = calibrate_model(model, X_calib, num_bits=args.bits)
    print(f"Calibrated {len(scale_factors)} layers")

    # Evaluate before correction training
    mse_float = evaluate_reconstruction(model, X_test, mode='float')
    mse_quant = evaluate_reconstruction(model, X_test, mode='quantized',
                                        scale_factors=scale_factors, zero_points=zero_points,
                                        num_bits=args.bits)

    print(f"\n{'Mode':<35} {'MSE':>12}")
    print("-" * 50)
    print(f"{'FP32 (baseline)':<35} {mse_float:>12.6f}")
    print(f"{f'INT{args.bits} (no correction)':<35} {mse_quant:>12.6f}")

    # Train correction layers (freeze base model)
    print(f"\n{'='*60}")
    print("Training correction layers...")
    print('='*60)

    # Freeze base model, train only correction layers
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False

    corr_optimizer = torch.optim.Adam(model.correction_layers.parameters(), lr=args.lr)

    for epoch in range(args.learned_epochs):
        model.train()
        total_loss = 0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            corr_optimizer.zero_grad()
            recon = model.forward_quantized_with_correction(
                X_batch, scale_factors, zero_points, num_bits=args.bits)
            loss = criterion(recon, X_batch)
            loss.backward()
            corr_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        test_mse = evaluate_reconstruction(model, X_test, mode='corrected',
                                           scale_factors=scale_factors, zero_points=zero_points,
                                           num_bits=args.bits)
        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_loss:.6f} | Test MSE: {test_mse:.6f}")

    # Final evaluation
    mse_corrected = evaluate_reconstruction(model, X_test, mode='corrected',
                                            scale_factors=scale_factors, zero_points=zero_points,
                                            num_bits=args.bits)

    print(f"\n{'='*60}")
    print("Final Results")
    print('='*60)
    print(f"\n{'Mode':<35} {'MSE':>12} {'vs FP32':>12}")
    print("-" * 60)
    print(f"{'FP32 (baseline)':<35} {mse_float:>12.6f} {'-':>12}")
    print(f"{f'INT{args.bits} (no correction)':<35} {mse_quant:>12.6f} {f'+{(mse_quant-mse_float)/mse_float*100:.1f}%':>12}")
    print(f"{f'INT{args.bits} + learned correction':<35} {mse_corrected:>12.6f} {f'+{(mse_corrected-mse_float)/mse_float*100:.1f}%':>12}")

    recovery = (mse_quant - mse_corrected) / (mse_quant - mse_float) * 100 if mse_quant != mse_float else 0
    print(f"\nRecovery: {recovery:.1f}% of quantization gap")

    # Visualize reconstructions
    fig, axes = plt.subplots(4, 10, figsize=(15, 6))

    model.eval()
    with torch.no_grad():
        recon_float = model(X_test[:10])
        recon_quant = model.forward_quantized(X_test[:10], scale_factors, zero_points, num_bits=args.bits)
        recon_corr = model.forward_quantized_with_correction(X_test[:10], scale_factors, zero_points, num_bits=args.bits)

    for i in range(10):
        axes[0, i].imshow(X_test[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].imshow(recon_float[i].cpu().view(28, 28), cmap='gray')
        axes[2, i].imshow(recon_quant[i].cpu().view(28, 28), cmap='gray')
        axes[3, i].imshow(recon_corr[i].cpu().view(28, 28), cmap='gray')
        for ax in axes[:, i]:
            ax.axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=10)
    axes[1, 0].set_ylabel('FP32', fontsize=10)
    axes[2, 0].set_ylabel(f'INT{args.bits}', fontsize=10)
    axes[3, 0].set_ylabel(f'INT{args.bits}+Corr', fontsize=10)

    plt.tight_layout()
    plt.savefig('autoencoder_correction.png', dpi=150)
    print(f"\nSaved reconstruction plot to 'autoencoder_correction.png'")


if __name__ == "__main__":
    main()
