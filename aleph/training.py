"""Training utilities for quantization-aware correction networks."""

import torch
import torch.nn.functional as F


def train_with_qat(model, X_train, teacher_logits, teacher_activations, scales, num_bits,
                   layer_loss_weight, correction_epochs=300, lr=1e-4):
    """Train correction layers with hybrid distillation + QAT.

    Works with any model that has:
    - corrections property (nn.ModuleDict of CorrectionNet)
    - forward_with_correction(x, scales, num_bits, quantize_correction, return_intermediates)
    - calibrate_corrections(x, scales, num_bits)
    """
    output_norm = teacher_logits.var().item()
    layer_norms = {k: teacher_activations[k].var().item() for k in teacher_activations}

    opt = torch.optim.Adam(model.corrections.parameters(), lr=lr)

    # Warmup (FP)
    warmup = correction_epochs // 4
    for _ in range(warmup):
        opt.zero_grad()
        out, acts = model.forward_with_correction(X_train, scales, num_bits,
                                                   quantize_correction=False, return_intermediates=True)
        output_loss = F.mse_loss(out, teacher_logits) / output_norm
        layer_loss = sum(F.mse_loss(acts[k], teacher_activations[k]) / layer_norms[k]
                        for k in acts) / len(acts) if acts else 0
        loss = output_loss + layer_loss_weight * layer_loss
        loss.backward()
        opt.step()

    model.calibrate_corrections(X_train, scales, num_bits)

    # QAT
    for i in range(correction_epochs - warmup):
        opt.zero_grad()
        out, acts = model.forward_with_correction(X_train, scales, num_bits,
                                                   quantize_correction=True, return_intermediates=True)
        output_loss = F.mse_loss(out, teacher_logits) / output_norm
        layer_loss = sum(F.mse_loss(acts[k], teacher_activations[k]) / layer_norms[k]
                        for k in acts) / len(acts) if acts else 0
        loss = output_loss + layer_loss_weight * layer_loss
        loss.backward()
        opt.step()
        if (i + 1) % 50 == 0:
            model.calibrate_corrections(X_train, scales, num_bits)

    model.calibrate_corrections(X_train, scales, num_bits)
