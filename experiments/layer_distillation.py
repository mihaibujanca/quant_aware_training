"""
Hybrid Distillation for Correction Networks

Loss = MSE(final_logits) + λ * Σ MSE(activation_corrected[i], activation_float[i])

Combines output-level distillation (primary goal) with per-layer supervision
(regularization / optimization aid).

The layer-wise term helps correction networks learn by providing more direct
gradients, while the output term ensures the final result matches.

Usage:
    python experiments/layer_distillation.py

See also:
    output_distillation.py - output-only supervision (baseline)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from aleph.quantization import fake_quantize, fake_quantize_with_error, get_quant_range
from aleph.datasets import make_spirals, embed_dataset_in_high_dimensional_space

log = logging.getLogger(__name__)


class CorrectionNet(nn.Module):
    """
    Small network that predicts a correction from accumulated quantization error.
    Can optionally quantize its own activations for fully-quantized inference.
    """

    def __init__(self, size, hidden_size=32):
        super().__init__()

        if hidden_size and hidden_size > 0:
            self.layers = nn.Sequential(
                nn.Linear(size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, size),
            )
        else:
            self.layers = nn.Linear(size, size)

        self.hidden_size = hidden_size
        self._scales = {}

    def forward(self, error, quantize=False, num_bits=8):
        if not quantize or self.hidden_size == 0:
            return self.layers(error)

        h = F.relu(self.layers[0](error))
        if 'hidden' in self._scales:
            h = fake_quantize(h, self._scales['hidden'], 0.0, num_bits=num_bits)
        out = self.layers[2](h)
        if 'output' in self._scales:
            out = fake_quantize(out, self._scales['output'], 0.0, num_bits=num_bits)
        return out

    def calibrate(self, sample_errors, num_bits=8):
        _, quant_max = get_quant_range(num_bits)

        with torch.no_grad():
            if self.hidden_size and self.hidden_size > 0:
                h = F.relu(self.layers[0](sample_errors))
                abs_max = max(abs(h.min().item()), abs(h.max().item()))
                self._scales['hidden'] = abs_max / quant_max if abs_max > 0 else 1.0
                out = self.layers[2](h)
            else:
                out = self.layers(sample_errors)

            abs_max = max(abs(out.min().item()), abs(out.max().item()))
            self._scales['output'] = abs_max / quant_max if abs_max > 0 else 1.0


class MLPWithCorrection(nn.Module):
    """MLP classifier with correction networks at regular intervals."""

    def __init__(self, input_size, hidden_size, num_classes, depth,
                 correction_every_n=2, correction_hidden=32):
        super().__init__()

        layers = []
        in_dim = input_size
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_size))
            in_dim = hidden_size
        self.backbone = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

        self.correction_positions = [i for i in range(depth) if (i + 1) % correction_every_n == 0]
        self.corrections = nn.ModuleDict({
            str(i): CorrectionNet(hidden_size, correction_hidden)
            for i in self.correction_positions
        })

    def forward(self, x):
        """Standard float forward."""
        for layer in self.backbone:
            x = self.relu(layer(x))
        return self.head(x)

    def forward_quantized(self, x, scales, num_bits=8):
        """Quantized forward, no correction."""
        for i, layer in enumerate(self.backbone):
            x = self.relu(layer(x))
            x = fake_quantize(x, scales[i], 0.0, num_bits=num_bits)
        return self.head(x)

    def get_float_activations(self, x):
        """Get float activations at each correction point."""
        activations = {}
        for i, layer in enumerate(self.backbone):
            x = self.relu(layer(x))
            if str(i) in self.corrections:
                activations[str(i)] = x.clone()
        return activations, self.head(x)

    def forward_with_correction(self, x, scales, num_bits=8, quantize_correction=False,
                                 return_intermediates=False):
        """
        Quantized forward with correction networks.

        If return_intermediates=True, also returns activations at correction points
        for computing layer-wise distillation loss.
        """
        error_accum = None
        intermediates = {} if return_intermediates else None

        for i, layer in enumerate(self.backbone):
            x = self.relu(layer(x))
            x, err = fake_quantize_with_error(x, scales[i], 0.0, num_bits=num_bits)

            error_accum = err if error_accum is None else error_accum + err

            if str(i) in self.corrections:
                correction = self.corrections[str(i)](error_accum, quantize=quantize_correction, num_bits=num_bits)
                x = x + correction
                if return_intermediates:
                    intermediates[str(i)] = x.clone()
                error_accum = None

        logits = self.head(x)

        if return_intermediates:
            return logits, intermediates
        return logits

    def calibrate(self, x, num_bits=8):
        _, quant_max = get_quant_range(num_bits)
        scales = []

        with torch.no_grad():
            for layer in self.backbone:
                x = self.relu(layer(x))
                abs_max = max(abs(x.min().item()), abs(x.max().item()))
                scales.append(abs_max / quant_max if abs_max > 0 else 1.0)

        return scales

    def calibrate_corrections(self, x, scales, num_bits=8):
        error_accum = None

        with torch.no_grad():
            for i, layer in enumerate(self.backbone):
                x = self.relu(layer(x))
                _, err = fake_quantize_with_error(x, scales[i], 0.0, num_bits=num_bits)

                error_accum = err if error_accum is None else error_accum + err

                if str(i) in self.corrections:
                    self.corrections[str(i)].calibrate(error_accum, num_bits=num_bits)
                    error_accum = None


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
    layer_loss_weight=1.0,  # λ for layer-wise loss
    seed=42,
):
    """
    Compare hybrid distillation (output + layer) vs output-only distillation.
    """
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

    log.info(f"Config: {num_bits}-bit, depth={depth}, width={hidden_size}, λ={layer_loss_weight}")

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
        acc_quant = (model.forward_quantized(X_test, scales, num_bits).argmax(1) == y_test).float().mean().item()

    log.info(f"Float: {acc_float:.4f}, Quantized: {acc_quant:.4f}")

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = False

    # Get teacher targets (float activations and logits)
    with torch.no_grad():
        teacher_activations, teacher_logits = model.get_float_activations(X_train)

    # Compute normalization factors (variance of targets) - same scale for both methods
    output_norm = teacher_logits.var().item()
    layer_norms = {k: teacher_activations[k].var().item() for k in teacher_activations}

    # Save initial correction weights for fair comparison
    torch.manual_seed(seed + 1000)  # Different seed for correction init
    for corr in model.corrections.values():
        corr._scales = {}
        for layer in corr.layers if isinstance(corr.layers, nn.Sequential) else [corr.layers]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    initial_correction_state = {k: v.clone() for k, v in model.corrections.state_dict().items()}

    # --- Train with OUTPUT-ONLY distillation + QAT (baseline) ---
    log.info("Training correction (output-only + QAT)...")

    opt = torch.optim.Adam(model.corrections.parameters(), lr=1e-4)

    # Warmup with FP
    warmup = correction_epochs // 4
    for _ in range(warmup):
        opt.zero_grad()
        student_logits = model.forward_with_correction(X_train, scales, num_bits, quantize_correction=False)
        loss = F.mse_loss(student_logits, teacher_logits) / output_norm
        loss.backward()
        opt.step()

    # Calibrate then QAT
    model.calibrate_corrections(X_train, scales, num_bits)

    for _ in range(correction_epochs - warmup):
        opt.zero_grad()
        student_logits = model.forward_with_correction(X_train, scales, num_bits, quantize_correction=True)
        loss = F.mse_loss(student_logits, teacher_logits) / output_norm
        loss.backward()
        opt.step()
        if (_ + 1) % 50 == 0:
            model.calibrate_corrections(X_train, scales, num_bits)

    model.calibrate_corrections(X_train, scales, num_bits)

    with torch.no_grad():
        acc_output_qat = (model.forward_with_correction(X_test, scales, num_bits, quantize_correction=True)
                          .argmax(1) == y_test).float().mean().item()
    log.info(f"Output-only + QAT: {acc_output_qat:.4f}")

    # --- Train with HYBRID distillation + QAT for correction layers ---
    log.info(f"Training correction (hybrid + QAT, λ={layer_loss_weight})...")

    # Restore to same initial state as output-only for fair comparison
    model.corrections.load_state_dict(initial_correction_state)
    for corr in model.corrections.values():
        corr._scales = {}

    opt = torch.optim.Adam(model.corrections.parameters(), lr=1e-4)

    # Phase 1: Warmup with FP correction to get reasonable weights
    warmup_epochs = correction_epochs // 4
    for _ in range(warmup_epochs):
        opt.zero_grad()
        student_logits, student_activations = model.forward_with_correction(
            X_train, scales, num_bits, quantize_correction=False, return_intermediates=True
        )
        # Normalized losses (MSE / variance = relative error)
        output_loss = F.mse_loss(student_logits, teacher_logits) / output_norm
        layer_loss = sum(F.mse_loss(student_activations[k], teacher_activations[k]) / layer_norms[k]
                        for k in student_activations) / len(student_activations)
        loss = output_loss + layer_loss_weight * layer_loss
        loss.backward()
        opt.step()

    # Calibrate correction networks before QAT
    model.calibrate_corrections(X_train, scales, num_bits)

    # Phase 2: QAT - train with quantized corrections (STE for gradients)
    qat_epochs = correction_epochs - warmup_epochs
    for _ in range(qat_epochs):
        opt.zero_grad()

        student_logits, student_activations = model.forward_with_correction(
            X_train, scales, num_bits, quantize_correction=True, return_intermediates=True
        )

        # Normalized losses
        output_loss = F.mse_loss(student_logits, teacher_logits) / output_norm
        layer_loss = sum(F.mse_loss(student_activations[k], teacher_activations[k]) / layer_norms[k]
                        for k in student_activations) / len(student_activations)

        # Combined loss
        loss = output_loss + layer_loss_weight * layer_loss
        loss.backward()
        opt.step()

        # Re-calibrate periodically during QAT
        if (_ + 1) % 50 == 0:
            model.calibrate_corrections(X_train, scales, num_bits)

    # Final calibration
    model.calibrate_corrections(X_train, scales, num_bits)

    with torch.no_grad():
        acc_hybrid_qat = (model.forward_with_correction(X_test, scales, num_bits, quantize_correction=True)
                          .argmax(1) == y_test).float().mean().item()
    log.info(f"Hybrid + QAT: {acc_hybrid_qat:.4f}")

    # Recovery rates
    gap = acc_float - acc_quant
    rec_output_qat = (acc_output_qat - acc_quant) / gap * 100 if gap > 0 else 0
    rec_hybrid_qat = (acc_hybrid_qat - acc_quant) / gap * 100 if gap > 0 else 0

    log.info(f"Recovery: output+QAT={rec_output_qat:.1f}%, hybrid+QAT={rec_hybrid_qat:.1f}%")

    return {
        "acc_float": acc_float,
        "acc_quant": acc_quant,
        "acc_output_qat": acc_output_qat,
        "acc_hybrid_qat": acc_hybrid_qat,
        "recovery_output_qat": rec_output_qat,
        "recovery_hybrid_qat": rec_hybrid_qat,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("Hybrid Distillation Experiment")
    print("Output loss + λ * Layer loss")
    print("=" * 60)

    # Focus on 4-bit for iteration, λ from 0.0 to 1.0
    for lam in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        print(f"\n{'='*60}")
        print(f"4-bit, λ={lam}")
        print("=" * 60)

        results = run_experiment(
            num_bits=4,
            input_size=100,
            depth=6,
            hidden_size=64,
            correction_every_n=2,
            correction_hidden=32,
            epochs=500,
            correction_epochs=300,
            layer_loss_weight=lam,
        )

        print(f"\nRecovery: output+QAT={results['recovery_output_qat']:.1f}%, "
              f"hybrid+QAT={results['recovery_hybrid_qat']:.1f}%")
