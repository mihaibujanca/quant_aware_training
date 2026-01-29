"""
Autoencoder Quantization Correction Sweep

Comprehensive experiment varying architecture, quantization, and correction parameters.
"""

import os
import csv
import time
import itertools
from datetime import datetime

import torch
import torch.nn as nn

from aleph.datasets import load_mnist_flat
from aleph.models import AutoencoderWithCorrection
from aleph.quantization import calibrate_model

# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================
SEEDS = [42, 123, 999]
BIT_WIDTHS = [2, 4, 8]
ARCHITECTURES = [
    # (hidden_sizes, latent_size)
    ([128], 16),             # shallow
    ([256, 128], 32),        # medium
    ([512, 256, 128], 64),   # deep
]
CORRECTION_EVERY = [1, 2, 3]
CORRECTION_HIDDEN = [0, 32, 64]  # 0 = linear only
LEARNED_LRS = [1e-3]

EPOCHS = 15
LEARNED_EPOCHS = 15
BATCH_SIZE = 256
LR = 1e-3

# Output
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"runs/autoencoder_sweep_{TIMESTAMP}"
os.makedirs(LOG_DIR, exist_ok=True)


def evaluate_mse(model, X, mode='float', scale_factors=None, zero_points=None, num_bits=8):
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


def run_experiment(seed, bits, hidden_sizes, latent_size, correct_every, correction_hidden,
                   learned_lr, train_loader, test_X, device):
    """Run a single experiment configuration."""
    torch.manual_seed(seed)

    # Skip invalid configs (correct_every > num correctable layers)
    num_encoder_layers = len(hidden_sizes)
    num_decoder_layers = len(hidden_sizes)
    if correct_every > max(num_encoder_layers, num_decoder_layers):
        return None

    model = AutoencoderWithCorrection(
        input_size=784,
        hidden_sizes=hidden_sizes,
        latent_size=latent_size,
        correction_every_n=correct_every,
        correction_hidden=correction_hidden,
    ).to(device)

    num_correction_layers = len(model.correction_layers)
    if num_correction_layers == 0:
        return None  # No correction layers for this config

    # Train float model
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            recon = model(X_batch)
            loss = criterion(recon, X_batch)
            loss.backward()
            optimizer.step()

    # Calibrate
    X_calib, _ = next(iter(train_loader))
    X_calib = X_calib.to(device)
    scale_factors, zero_points = calibrate_model(model, X_calib, num_bits=bits)

    # Evaluate float and quantized
    mse_float = evaluate_mse(model, test_X, mode='float')
    mse_quant = evaluate_mse(model, test_X, mode='quantized',
                             scale_factors=scale_factors, zero_points=zero_points, num_bits=bits)

    # Train correction layers
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False

    if len(list(model.correction_layers.parameters())) > 0:
        corr_optimizer = torch.optim.Adam(model.correction_layers.parameters(), lr=learned_lr)

        model.train()
        for epoch in range(LEARNED_EPOCHS):
            for X_batch, _ in train_loader:
                X_batch = X_batch.to(device)
                corr_optimizer.zero_grad()
                recon = model.forward_quantized_with_correction(
                    X_batch, scale_factors, zero_points, num_bits=bits)
                loss = criterion(recon, X_batch)
                loss.backward()
                corr_optimizer.step()

    mse_corrected = evaluate_mse(model, test_X, mode='corrected',
                                 scale_factors=scale_factors, zero_points=zero_points, num_bits=bits)

    # Compute metrics
    quant_gap = mse_quant - mse_float
    recovery = (mse_quant - mse_corrected) / quant_gap * 100 if quant_gap > 0 else 0
    quant_degradation = (mse_quant - mse_float) / mse_float * 100
    corrected_degradation = (mse_corrected - mse_float) / mse_float * 100

    return {
        "seed": seed,
        "bits": bits,
        "hidden_sizes": str(hidden_sizes),
        "latent_size": latent_size,
        "correct_every": correct_every,
        "correction_hidden": correction_hidden,
        "learned_lr": learned_lr,
        "num_correction_layers": num_correction_layers,
        "mse_float": mse_float,
        "mse_quant": mse_quant,
        "mse_corrected": mse_corrected,
        "quant_degradation_pct": quant_degradation,
        "corrected_degradation_pct": corrected_degradation,
        "recovery_pct": recovery,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    train_loader, test_loader = load_mnist_flat(batch_size=BATCH_SIZE)

    test_X, _ = next(iter(test_loader))
    test_X = test_X.to(device)

    # Generate configs
    configs = list(itertools.product(
        SEEDS, BIT_WIDTHS, ARCHITECTURES, CORRECTION_EVERY, CORRECTION_HIDDEN, LEARNED_LRS
    ))
    total_configs = len(configs)

    print(f"\n{'='*60}")
    print(f"Autoencoder Correction Sweep")
    print(f"{'='*60}")
    print(f"Seeds: {SEEDS}")
    print(f"Bit widths: {BIT_WIDTHS}")
    print(f"Architectures: {len(ARCHITECTURES)}")
    print(f"Correction every: {CORRECTION_EVERY}")
    print(f"Correction hidden: {CORRECTION_HIDDEN}")
    print(f"Learned LRs: {LEARNED_LRS}")
    print(f"Max configs: {total_configs}")
    print(f"Output: {LOG_DIR}")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()
    completed = 0

    for i, (seed, bits, (hidden_sizes, latent_size), correct_every, correction_hidden, learned_lr) in enumerate(configs):
        run_start = time.time()

        result = run_experiment(
            seed, bits, hidden_sizes, latent_size, correct_every, correction_hidden,
            learned_lr, train_loader, test_X, device
        )

        if result is None:
            continue

        results.append(result)
        completed += 1

        run_time = time.time() - run_start
        elapsed = time.time() - start_time
        eta = (elapsed / completed) * (total_configs - i - 1) if completed > 0 else 0

        print(f"[{completed}/{total_configs}] seed={seed}, bits={bits}, "
              f"arch={hidden_sizes}, corr_every={correct_every}, corr_h={correction_hidden}, lr={learned_lr}")
        print(f"    MSE: float={result['mse_float']:.6f}, quant={result['mse_quant']:.6f}, "
              f"corrected={result['mse_corrected']:.6f}, recovery={result['recovery_pct']:.1f}%")
        print(f"    time={run_time:.1f}s, ETA={eta/60:.1f}min")

    # Save results
    csv_path = f"{LOG_DIR}/summary.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"Sweep complete!")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Completed runs: {completed}")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*60}")

    # Print summary stats
    if results:
        import statistics
        recoveries = [r['recovery_pct'] for r in results]
        print(f"\nRecovery stats:")
        print(f"  Mean: {statistics.mean(recoveries):.1f}%")
        print(f"  Median: {statistics.median(recoveries):.1f}%")
        print(f"  Min: {min(recoveries):.1f}%")
        print(f"  Max: {max(recoveries):.1f}%")


if __name__ == "__main__":
    main()
