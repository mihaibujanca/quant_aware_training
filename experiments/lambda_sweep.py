"""
Lambda sweep for hybrid distillation across tasks.

Sweeps:
- λ: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 (6 values)
- bits: 2, 4, 8 (3 values)
- correction_hidden: 0, 32, 64 (3 values)
- seeds: 42, 123, 999 (3 values)
- Total: 162 runs per task

Usage:
    python experiments/lambda_sweep.py
    python experiments/lambda_sweep.py --with_transformer
"""

import argparse

import json
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from aleph.models import (
    CorrectionNet,
    MLPWithCorrection,
    TransformerWithCorrection,
    AutoencoderWithCorrection,
)
from aleph.training import train_with_qat
from aleph.datasets import (
    make_spirals,
    embed_dataset_in_high_dimensional_space,
    load_mnist_flat,
    load_shakespeare,
)


# =============================================================================
# Task runners
# =============================================================================

def run_classification(num_bits, layer_loss_weight, correction_hidden=0, target_dim=100, seed=42):
    """Classification on spirals. Architecture: depth=6, width=64."""
    torch.manual_seed(seed)

    X_2d, y = make_spirals(n_samples=2000, noise=0.3, n_turns=3, random_state=seed)
    _, embedding = embed_dataset_in_high_dimensional_space(X_2d, target_dim=target_dim, random_state=seed)
    X_train_2d, X_test_2d, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=seed)
    X_train = torch.tensor(embedding.transform(X_train_2d), dtype=torch.float32)
    X_test = torch.tensor(embedding.transform(X_test_2d), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    model = MLPWithCorrection(target_dim, 64, 2, 6, correction_every_n=2, correction_hidden=correction_hidden)

    # Train float
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5000):
        opt.zero_grad()
        F.cross_entropy(model(X_train), y_train).backward()
        opt.step()

    scales = model.calibrate(X_train, num_bits)

    model.eval()
    with torch.no_grad():
        acc_float = (model(X_test).argmax(1) == y_test).float().mean().item()
        acc_quant = (model.forward_quantized(X_test, scales, num_bits=num_bits).argmax(1) == y_test).float().mean().item()

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = False

    with torch.no_grad():
        teacher_acts, teacher_logits = model.get_float_activations(X_train)

    # Train correction (more epochs for high-dim classification)
    train_with_qat(model, X_train, teacher_logits, teacher_acts, scales, num_bits, layer_loss_weight,
                   correction_epochs=1000)

    model.eval()
    with torch.no_grad():
        acc_corrected = (model.forward_with_correction(X_test, scales, num_bits, quantize_correction=True)
                        .argmax(1) == y_test).float().mean().item()

    gap = acc_float - acc_quant
    delta = acc_corrected - acc_quant
    recovery = (delta / gap * 100) if gap != 0 else 0

    return {
        "acc_float": acc_float,
        "acc_quant": acc_quant,
        "acc_corrected": acc_corrected,
        "gap": gap,
        "delta": delta,
        "recovery": recovery,
    }


def run_autoencoder(num_bits, layer_loss_weight, correction_hidden=32, seed=42):
    """Autoencoder on MNIST. Architecture: [256,128] -> 32 -> [128,256]."""
    torch.manual_seed(seed)

    train_loader, test_loader = load_mnist_flat(batch_size=256)
    X_test, _ = next(iter(test_loader))

    # Collect more training data for correction (4 batches = 1024 samples)
    X_train_batches = []
    for i, (X_batch, _) in enumerate(train_loader):
        X_train_batches.append(X_batch)
        if i >= 3:
            break
    X_train = torch.cat(X_train_batches, dim=0)

    model = AutoencoderWithCorrection(784, [256, 128], 32, correction_every_n=1, correction_hidden=correction_hidden)

    # Train float
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(30):
        for X_batch, _ in train_loader:
            opt.zero_grad()
            F.mse_loss(model(X_batch), X_batch).backward()
            opt.step()

    scales = model.calibrate(X_train, num_bits)

    model.eval()
    with torch.no_grad():
        mse_float = F.mse_loss(model(X_test), X_test).item()
        mse_quant = F.mse_loss(model.forward_quantized(X_test, scales, num_bits=num_bits), X_test).item()

    # Freeze backbone
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False

    with torch.no_grad():
        teacher_acts, teacher_out = model.get_float_activations(X_train)

    # Train correction
    train_with_qat(model, X_train, teacher_out, teacher_acts, scales, num_bits, layer_loss_weight)

    model.eval()
    with torch.no_grad():
        mse_corrected = F.mse_loss(model.forward_with_correction(X_test, scales, num_bits, quantize_correction=True), X_test).item()

    gap = mse_quant - mse_float
    delta = mse_quant - mse_corrected
    recovery = (delta / gap * 100) if gap != 0 else 0

    return {
        "mse_float": mse_float,
        "mse_quant": mse_quant,
        "mse_corrected": mse_corrected,
        "gap": gap,
        "delta": delta,
        "recovery": recovery,
    }


def run_transformer(num_bits, layer_loss_weight, correction_hidden=128, seed=42):
    """Transformer language modeling on Shakespeare."""
    torch.manual_seed(seed)

    train_X, train_Y, test_X, test_Y, vocab_size, _, _ = load_shakespeare(seq_len=128)

    model = TransformerWithCorrection(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=8,
        d_ff=512,
        max_seq_len=128,
        dropout=0.1,
        correction_every_n=2,
        correction_hidden=correction_hidden,
    )

    # Train float
    batch_size = 64
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    n_batches = len(train_X) // batch_size
    for _ in range(10):
        perm = torch.randperm(len(train_X))
        train_X_shuffled = train_X[perm]
        train_Y_shuffled = train_Y[perm]
        for i in range(n_batches):
            batch_X = train_X_shuffled[i * batch_size:(i + 1) * batch_size]
            batch_Y = train_Y_shuffled[i * batch_size:(i + 1) * batch_size]
            opt.zero_grad()
            logits = model(batch_X)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_Y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # Use more samples for calibration and correction training
    n_correction_samples = min(256, len(train_X))
    X_corr = train_X[:n_correction_samples]

    # Calibrate scales
    scales = model.calibrate(X_corr, num_bits)

    model.eval()
    with torch.no_grad():
        loss_float = F.cross_entropy(
            model(test_X).reshape(-1, vocab_size), test_Y.reshape(-1)
        ).item()
        loss_quant = F.cross_entropy(
            model.forward_quantized(test_X, scales, num_bits=num_bits).reshape(-1, vocab_size),
            test_Y.reshape(-1)
        ).item()

    # Freeze backbone
    for name, p in model.named_parameters():
        if "correction" not in name:
            p.requires_grad = False

    with torch.no_grad():
        teacher_acts, teacher_logits = model.get_float_activations(X_corr)

    # Train correction
    train_with_qat(
        model,
        X_corr,
        teacher_logits,
        teacher_acts,
        scales,
        num_bits,
        layer_loss_weight,
    )

    model.eval()
    with torch.no_grad():
        loss_corrected = F.cross_entropy(
            model.forward_with_correction(test_X, scales, num_bits, quantize_correction=True)
            .reshape(-1, vocab_size),
            test_Y.reshape(-1),
        ).item()

    gap = loss_quant - loss_float
    delta = loss_quant - loss_corrected
    recovery = (delta / gap * 100) if gap != 0 else 0

    return {
        "loss_float": loss_float,
        "loss_quant": loss_quant,
        "loss_corrected": loss_corrected,
        "gap": gap,
        "delta": delta,
        "recovery": recovery,
    }


# =============================================================================
# Main sweep
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_transformer", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    args = parser.parse_args()

    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bits_list = [2, 4, 8]
    hidden_sizes = [0, 32, 64]

    results = {"classification": [], "autoencoder": [], "transformer": []}

    print("=" * 70)
    print("Lambda Sweep: Hybrid Distillation")
    print("=" * 70)

    # tasks = ["classification", "autoencoder"]
    tasks = []
    if args.with_transformer:
        tasks.append("transformer")

    for task in tasks:
        print(f"\n{'='*70}")
        print(f"Task: {task.upper()}")
        print("=" * 70)

        for bits in bits_list:
            for corr_h in hidden_sizes:
                print(f"\n  {bits}-bit, correction_hidden={corr_h}:")
                for seed in args.seeds:
                    for lam in lambdas:
                        if task == "classification":
                            r = run_classification(bits, lam, correction_hidden=corr_h, target_dim=10000, seed=seed)
                        elif task == "transformer":
                            r = run_transformer(bits, lam, correction_hidden=corr_h, seed=seed)
                        else:
                            r = run_autoencoder(bits, lam, correction_hidden=corr_h, seed=seed)

                        r["bits"] = bits
                        r["lambda"] = lam
                        r["correction_hidden"] = corr_h
                        r["seed"] = seed
                        r["task"] = task
                        results[task].append(r)

                        print(
                            f"    seed={seed} λ={lam:.1f}: "
                            f"gap={r['gap']:.4f}, "
                            f"delta={r['delta']:.4f}, "
                            f"recovery={r['recovery']:.1f}%"
                        )

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for task in tasks:
        print(f"\n{task.upper()}:")
        for corr_h in hidden_sizes:
            print(f"\n  correction_hidden={corr_h}:")
            print(f"  {'bits':<6} {'λ=0.0':<8} {'λ=0.2':<8} {'λ=0.4':<8} {'λ=0.6':<8} {'λ=0.8':<8} {'λ=1.0':<8}")
            for bits in bits_list:
                row = [r for r in results[task] if r["bits"] == bits and r["correction_hidden"] == corr_h]
                recoveries = []
                for lam in lambdas:
                    vals = [r["recovery"] for r in row if r["lambda"] == lam]
                    avg = sum(vals) / len(vals) if vals else 0.0
                    recoveries.append(f"{avg:.1f}%")
                print(
                    f"  {bits:<6} {recoveries[0]:<8} {recoveries[1]:<8} {recoveries[2]:<8} "
                    f"{recoveries[3]:<8} {recoveries[4]:<8} {recoveries[5]:<8}"
                )

    # Save results
    with open("lambda_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to lambda_sweep_results.json")


if __name__ == "__main__":
    main()
