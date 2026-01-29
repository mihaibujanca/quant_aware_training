"""
Transformer Quantization Correction Sweep

Character-level language modeling on Shakespeare to test correction layers on generation.
"""

import os
import csv
import time
import itertools
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from aleph.datasets import load_shakespeare
from aleph.models import TransformerWithCorrection
from aleph.quantization import calibrate_model

# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================
SEEDS = [42, 123, 999]
BIT_WIDTHS = [2, 4, 8]
ARCHITECTURES = [
    # (d_model, n_heads, n_layers, d_ff)
    (64, 4, 2, 256),      # tiny
    (128, 4, 4, 512),     # small
    (256, 8, 6, 1024),    # medium
]
CORRECTION_EVERY = [1, 2, 4]
CORRECTION_HIDDEN = [0, 32, 64]  # 0 = linear only

EPOCHS = 10
LEARNED_EPOCHS = 5
BATCH_SIZE = 64
SEQ_LEN = 128
LR = 3e-4

# Output
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"runs/transformer_sweep_{TIMESTAMP}"
os.makedirs(LOG_DIR, exist_ok=True)


def evaluate_loss(model, X, Y, mode='float', scale_factors=None, zero_points=None, num_bits=8):
    """Evaluate cross-entropy loss."""
    model.eval()
    with torch.no_grad():
        if mode == 'float':
            logits = model(X)
        elif mode == 'quantized':
            logits = model.forward_quantized(X, scale_factors, zero_points, num_bits=num_bits)
        elif mode == 'corrected':
            logits = model.forward_quantized_with_correction(X, scale_factors, zero_points, num_bits=num_bits)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
    return loss.item()


def run_experiment(seed, bits, d_model, n_heads, n_layers, d_ff, correct_every, correction_hidden,
                   train_X, train_Y, test_X, test_Y, vocab_size, device):
    """Run a single experiment configuration."""
    torch.manual_seed(seed)

    # Total quantization points = n_layers * 2 (attn + ffn)
    total_quant_points = n_layers * 2
    if correct_every > total_quant_points:
        return None

    model = TransformerWithCorrection(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=SEQ_LEN,
        dropout=0.1,
        correction_every_n=correct_every,
        correction_hidden=correction_hidden,
    ).to(device)

    num_correction_layers = len(model.correction_layers)
    if num_correction_layers == 0:
        return None

    # Train float model
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    n_batches = len(train_X) // BATCH_SIZE

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(train_X))
        train_X = train_X[perm]
        train_Y = train_Y[perm]

        for i in range(n_batches):
            batch_X = train_X[i*BATCH_SIZE:(i+1)*BATCH_SIZE].to(device)
            batch_Y = train_Y[i*BATCH_SIZE:(i+1)*BATCH_SIZE].to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch_Y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Calibrate
    X_calib = train_X[:BATCH_SIZE].to(device)
    scale_factors, zero_points = calibrate_model(model, X_calib, num_bits=bits)

    # Evaluate float and quantized
    test_X_dev = test_X.to(device)
    test_Y_dev = test_Y.to(device)

    loss_float = evaluate_loss(model, test_X_dev, test_Y_dev, mode='float')
    loss_quant = evaluate_loss(model, test_X_dev, test_Y_dev, mode='quantized',
                               scale_factors=scale_factors, zero_points=zero_points, num_bits=bits)

    # Train correction layers
    for name, param in model.named_parameters():
        if 'correction_layers' not in name:
            param.requires_grad = False

    if len(list(model.correction_layers.parameters())) > 0:
        corr_optimizer = torch.optim.Adam(model.correction_layers.parameters(), lr=LR)

        model.train()
        for epoch in range(LEARNED_EPOCHS):
            perm = torch.randperm(len(train_X))
            train_X = train_X[perm]
            train_Y = train_Y[perm]

            for i in range(n_batches):
                batch_X = train_X[i*BATCH_SIZE:(i+1)*BATCH_SIZE].to(device)
                batch_Y = train_Y[i*BATCH_SIZE:(i+1)*BATCH_SIZE].to(device)

                corr_optimizer.zero_grad()
                logits = model.forward_quantized_with_correction(
                    batch_X, scale_factors, zero_points, num_bits=bits
                )
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch_Y.reshape(-1))
                loss.backward()
                corr_optimizer.step()

    loss_corrected = evaluate_loss(model, test_X_dev, test_Y_dev, mode='corrected',
                                   scale_factors=scale_factors, zero_points=zero_points, num_bits=bits)

    # Compute metrics
    quant_gap = loss_quant - loss_float
    recovery = (loss_quant - loss_corrected) / quant_gap * 100 if quant_gap > 0 else 0
    quant_degradation = (loss_quant - loss_float) / loss_float * 100
    corrected_degradation = (loss_corrected - loss_float) / loss_float * 100

    # Perplexity
    ppl_float = torch.exp(torch.tensor(loss_float)).item()
    ppl_quant = torch.exp(torch.tensor(loss_quant)).item()
    ppl_corrected = torch.exp(torch.tensor(loss_corrected)).item()

    return {
        "seed": seed,
        "bits": bits,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "correct_every": correct_every,
        "correction_hidden": correction_hidden,
        "num_correction_layers": num_correction_layers,
        "loss_float": loss_float,
        "loss_quant": loss_quant,
        "loss_corrected": loss_corrected,
        "ppl_float": ppl_float,
        "ppl_quant": ppl_quant,
        "ppl_corrected": ppl_corrected,
        "quant_degradation_pct": quant_degradation,
        "corrected_degradation_pct": corrected_degradation,
        "recovery_pct": recovery,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    train_X, train_Y, test_X, test_Y, vocab_size, char_to_idx, idx_to_char = load_shakespeare(seq_len=SEQ_LEN)

    print(f"Vocab size: {vocab_size}")
    print(f"Train sequences: {len(train_X)}, Test sequences: {len(test_X)}")

    # Generate configs
    configs = list(itertools.product(
        SEEDS, BIT_WIDTHS, ARCHITECTURES, CORRECTION_EVERY, CORRECTION_HIDDEN
    ))
    total_configs = len(configs)

    print(f"\n{'='*60}")
    print(f"Transformer Correction Sweep (Shakespeare)")
    print(f"{'='*60}")
    print(f"Seeds: {SEEDS}")
    print(f"Bit widths: {BIT_WIDTHS}")
    print(f"Architectures: {len(ARCHITECTURES)}")
    print(f"Correction every: {CORRECTION_EVERY}")
    print(f"Correction hidden: {CORRECTION_HIDDEN}")
    print(f"Max configs: {total_configs}")
    print(f"Output: {LOG_DIR}")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()
    completed = 0

    for i, (seed, bits, (d_model, n_heads, n_layers, d_ff), correct_every, correction_hidden) in enumerate(configs):
        run_start = time.time()

        result = run_experiment(
            seed, bits, d_model, n_heads, n_layers, d_ff, correct_every, correction_hidden,
            train_X, train_Y, test_X, test_Y, vocab_size, device
        )

        if result is None:
            continue

        results.append(result)
        completed += 1

        run_time = time.time() - run_start
        elapsed = time.time() - start_time
        eta = (elapsed / completed) * (total_configs - i - 1) if completed > 0 else 0

        print(f"[{completed}/{total_configs}] seed={seed}, bits={bits}, "
              f"arch=({d_model},{n_heads},{n_layers},{d_ff}), corr_every={correct_every}, corr_h={correction_hidden}")
        print(f"    Loss: float={result['loss_float']:.4f}, quant={result['loss_quant']:.4f}, "
              f"corrected={result['loss_corrected']:.4f}, recovery={result['recovery_pct']:.1f}%")
        print(f"    PPL: float={result['ppl_float']:.1f}, quant={result['ppl_quant']:.1f}, "
              f"corrected={result['ppl_corrected']:.1f}")
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
