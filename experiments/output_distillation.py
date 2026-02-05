"""
Output-level Distillation for Correction Networks

Loss = MSE(final_logits_corrected, final_logits_float)

Correction networks are trained to produce corrections that, after propagating
through all remaining layers, result in final outputs matching the float model.

Compares:
1. No correction (baseline)
2. Full precision correction layers
3. Quantized correction layers (same bit width as backbone)

Usage:
    python experiments/output_distillation.py

See also:
    layer_distillation.py - per-layer supervision (more direct learning signal)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from aleph.models import CorrectionNet, MLPWithCorrection
from aleph.quantization import fake_quantize, fake_quantize_with_error, get_quant_range
from aleph.datasets import make_spirals, embed_dataset_in_high_dimensional_space

log = logging.getLogger(__name__)


def run_experiment(
    num_bits=4,
    input_size=100,
    hidden_size=64,
    depth=6,
    correction_every_n=2,
    correction_hidden=32,
    n_samples=2000,
    epochs=500,
    correction_epochs=300,
    seed=42,
):
    """Run the quantized vs unquantized correction comparison."""
    from sklearn.model_selection import train_test_split

    torch.manual_seed(seed)

    # Data
    X_2d, y = make_spirals(n_samples=n_samples, noise=0.3, n_turns=3, random_state=seed)
    _, embedding = embed_dataset_in_high_dimensional_space(X_2d, target_dim=input_size, random_state=seed)

    X_train_2d, X_test_2d, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=seed)
    X_train = torch.tensor(embedding.transform(X_train_2d), dtype=torch.float32)
    X_test = torch.tensor(embedding.transform(X_test_2d), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    log.info(f"Config: {num_bits}-bit, depth={depth}, width={hidden_size}, correction_every={correction_every_n}")

    # Model
    model = MLPWithCorrection(input_size, hidden_size, 2, depth, correction_every_n, correction_hidden)

    # Train float
    log.info("Training float model...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(X_train), y_train).backward()
        opt.step()

    # Calibrate backbone
    scales = model.calibrate(X_train, num_bits)

    # Baseline accuracies
    model.eval()
    with torch.no_grad():
        acc_float = (model(X_test).argmax(1) == y_test).float().mean().item()
        acc_quant = (model.forward_quantized(X_test, scales, num_bits=num_bits).argmax(1) == y_test).float().mean().item()

    log.info(f"Float: {acc_float:.4f}, Quantized: {acc_quant:.4f}")

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = False

    # Teacher logits for distillation
    with torch.no_grad():
        teacher = model(X_train)

    # Train correction (full precision)
    log.info("Training correction (full precision)...")
    opt = torch.optim.Adam(model.corrections.parameters(), lr=1e-4)
    for _ in range(correction_epochs):
        opt.zero_grad()
        student = model.forward_with_correction(X_train, scales, num_bits, quantize_correction=False)
        F.mse_loss(student, teacher).backward()
        opt.step()

    with torch.no_grad():
        acc_fp = (model.forward_with_correction(X_test, scales, num_bits, quantize_correction=False)
                  .argmax(1) == y_test).float().mean().item()
    log.info(f"With FP correction: {acc_fp:.4f}")

    # Calibrate correction networks
    model.calibrate_corrections(X_train, scales, num_bits)

    # Evaluate quantized correction (no finetune)
    with torch.no_grad():
        acc_quant_raw = (model.forward_with_correction(X_test, scales, num_bits, quantize_correction=True)
                         .argmax(1) == y_test).float().mean().item()
    log.info(f"With quantized correction (no finetune): {acc_quant_raw:.4f}")

    # Finetune with quantized correction
    log.info("Finetuning with quantized correction...")
    opt = torch.optim.Adam(model.corrections.parameters(), lr=1e-5)
    for _ in range(correction_epochs // 2):
        opt.zero_grad()
        student = model.forward_with_correction(X_train, scales, num_bits, quantize_correction=True)
        F.mse_loss(student, teacher).backward()
        opt.step()

    model.calibrate_corrections(X_train, scales, num_bits)

    with torch.no_grad():
        acc_quant_ft = (model.forward_with_correction(X_test, scales, num_bits, quantize_correction=True)
                        .argmax(1) == y_test).float().mean().item()
    log.info(f"With quantized correction (finetuned): {acc_quant_ft:.4f}")

    # Recovery rates
    gap = acc_float - acc_quant
    rec_fp = (acc_fp - acc_quant) / gap * 100 if gap > 0 else 0
    rec_quant_raw = (acc_quant_raw - acc_quant) / gap * 100 if gap > 0 else 0
    rec_quant_ft = (acc_quant_ft - acc_quant) / gap * 100 if gap > 0 else 0

    log.info(f"Recovery: FP={rec_fp:.1f}%, Quant(raw)={rec_quant_raw:.1f}%, Quant(ft)={rec_quant_ft:.1f}%")

    return {
        "acc_float": acc_float,
        "acc_quant": acc_quant,
        "acc_fp_correction": acc_fp,
        "acc_quant_correction_raw": acc_quant_raw,
        "acc_quant_correction_ft": acc_quant_ft,
        "recovery_fp": rec_fp,
        "recovery_quant_raw": rec_quant_raw,
        "recovery_quant_ft": rec_quant_ft,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("Quantized Correction Layer Experiment")
    print("=" * 60)

    for bits in [4, 2]:
        print(f"\n{'='*60}\n{bits}-bit\n{'='*60}")
        results = run_experiment(
            num_bits=bits,
            input_size=100,
            depth=6,
            hidden_size=64,
            correction_every_n=2,
            correction_hidden=32,
            epochs=500,
            correction_epochs=300,
        )
        print("\nResults:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
