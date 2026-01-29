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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from aleph.quantization import fake_quantize, fake_quantize_with_error, get_quant_range
from aleph.datasets import (
    make_spirals,
    embed_dataset_in_high_dimensional_space,
    load_mnist_flat,
    load_shakespeare,
)


# =============================================================================
# Models (copied from layer_distillation.py to keep self-contained)
# =============================================================================

class CorrectionNet(nn.Module):
    def __init__(self, size, hidden_size=32):
        super().__init__()
        if hidden_size and hidden_size > 0:
            self.layers = nn.Sequential(
                nn.Linear(size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, size),
            )
            # Initialize output layer to zero so correction starts as identity
            nn.init.zeros_(self.layers[2].weight)
            nn.init.zeros_(self.layers[2].bias)
        else:
            self.layers = nn.Linear(size, size)
            nn.init.zeros_(self.layers.weight)
            nn.init.zeros_(self.layers.bias)
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
    """For classification task."""
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
        for layer in self.backbone:
            x = self.relu(layer(x))
        return self.head(x)

    def forward_quantized(self, x, scales, num_bits=8):
        for i, layer in enumerate(self.backbone):
            x = self.relu(layer(x))
            x = fake_quantize(x, scales[i], 0.0, num_bits=num_bits)
        return self.head(x)

    def get_float_activations(self, x):
        activations = {}
        for i, layer in enumerate(self.backbone):
            x = self.relu(layer(x))
            if str(i) in self.corrections:
                activations[str(i)] = x.clone()
        return activations, self.head(x)

    def forward_with_correction(self, x, scales, num_bits=8, quantize_correction=False, return_intermediates=False):
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


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feedforward."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerWithCorrection(nn.Module):
    """For transformer language modeling task."""
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=512,
                 max_seq_len=256, dropout=0.1, correction_every_n=2, correction_hidden=32):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # Correction layers at intervals (2 quant points per layer: attn, ffn)
        self.corrections = nn.ModuleDict()
        self.correction_positions = []
        quant_point = 0
        for layer_idx in range(n_layers):
            for sub in ['attn', 'ffn']:
                if (quant_point + 1) % correction_every_n == 0:
                    key = f"{layer_idx}_{sub}"
                    self.corrections[key] = CorrectionNet(d_model, correction_hidden)
                    self.correction_positions.append(key)
                quant_point += 1

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.dropout_layer(self.embedding(x) + self.pos_embedding(pos))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.ln_f(x))

    def forward_quantized(self, x, scales, num_bits=8):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.dropout_layer(self.embedding(x) + self.pos_embedding(pos))
        sf_idx = 0
        for layer in self.layers:
            # Attention
            x_ln = layer.ln1(x)
            attn_out, _ = layer.attn(x_ln, x_ln, x_ln)
            x = x + layer.dropout(attn_out)
            x = fake_quantize(x, scales[sf_idx], 0.0, num_bits=num_bits)
            sf_idx += 1
            # FFN
            x = x + layer.ffn(layer.ln2(x))
            x = fake_quantize(x, scales[sf_idx], 0.0, num_bits=num_bits)
            sf_idx += 1
        return self.head(self.ln_f(x))

    def get_float_activations(self, x):
        """Get float activations at correction points."""
        activations = {}
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.dropout_layer(self.embedding(x) + self.pos_embedding(pos))
        for layer_idx, layer in enumerate(self.layers):
            x_ln = layer.ln1(x)
            attn_out, _ = layer.attn(x_ln, x_ln, x_ln)
            x = x + layer.dropout(attn_out)
            if f"{layer_idx}_attn" in self.corrections:
                activations[f"{layer_idx}_attn"] = x.clone()
            x = x + layer.ffn(layer.ln2(x))
            if f"{layer_idx}_ffn" in self.corrections:
                activations[f"{layer_idx}_ffn"] = x.clone()
        return activations, self.head(self.ln_f(x))

    def forward_with_correction(self, x, scales, num_bits=8, quantize_correction=False, return_intermediates=False):
        intermediates = {} if return_intermediates else None
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.dropout_layer(self.embedding(x) + self.pos_embedding(pos))
        error_accum = None
        sf_idx = 0

        for layer_idx, layer in enumerate(self.layers):
            # Attention
            x_ln = layer.ln1(x)
            attn_out, _ = layer.attn(x_ln, x_ln, x_ln)
            x = x + layer.dropout(attn_out)
            x, err = fake_quantize_with_error(x, scales[sf_idx], 0.0, num_bits=num_bits)
            sf_idx += 1
            error_accum = err if error_accum is None else error_accum + err

            key = f"{layer_idx}_attn"
            if key in self.corrections:
                correction = self.corrections[key](error_accum, quantize=quantize_correction, num_bits=num_bits)
                x = x + correction
                if return_intermediates:
                    intermediates[key] = x.clone()
                error_accum = None

            # FFN
            x = x + layer.ffn(layer.ln2(x))
            x, err = fake_quantize_with_error(x, scales[sf_idx], 0.0, num_bits=num_bits)
            sf_idx += 1
            error_accum = err if error_accum is None else error_accum + err

            key = f"{layer_idx}_ffn"
            if key in self.corrections:
                correction = self.corrections[key](error_accum, quantize=quantize_correction, num_bits=num_bits)
                x = x + correction
                if return_intermediates:
                    intermediates[key] = x.clone()
                error_accum = None

        logits = self.head(self.ln_f(x))
        if return_intermediates:
            return logits, intermediates
        return logits

    def calibrate(self, x, num_bits=8):
        _, quant_max = get_quant_range(num_bits)
        scales = []
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        with torch.no_grad():
            x = self.dropout_layer(self.embedding(x) + self.pos_embedding(pos))
            for layer in self.layers:
                x_ln = layer.ln1(x)
                attn_out, _ = layer.attn(x_ln, x_ln, x_ln)
                x = x + layer.dropout(attn_out)
                abs_max = max(abs(x.min().item()), abs(x.max().item()))
                scales.append(abs_max / quant_max if abs_max > 0 else 1.0)
                x = x + layer.ffn(layer.ln2(x))
                abs_max = max(abs(x.min().item()), abs(x.max().item()))
                scales.append(abs_max / quant_max if abs_max > 0 else 1.0)
        return scales

    def calibrate_corrections(self, x, scales, num_bits=8):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        error_accum = None
        sf_idx = 0
        with torch.no_grad():
            x = self.dropout_layer(self.embedding(x) + self.pos_embedding(pos))
            for layer_idx, layer in enumerate(self.layers):
                x_ln = layer.ln1(x)
                attn_out, _ = layer.attn(x_ln, x_ln, x_ln)
                x = x + layer.dropout(attn_out)
                _, err = fake_quantize_with_error(x, scales[sf_idx], 0.0, num_bits=num_bits)
                sf_idx += 1
                error_accum = err if error_accum is None else error_accum + err
                if f"{layer_idx}_attn" in self.corrections:
                    self.corrections[f"{layer_idx}_attn"].calibrate(error_accum, num_bits=num_bits)
                    error_accum = None
                x = x + layer.ffn(layer.ln2(x))
                _, err = fake_quantize_with_error(x, scales[sf_idx], 0.0, num_bits=num_bits)
                sf_idx += 1
                error_accum = err if error_accum is None else error_accum + err
                if f"{layer_idx}_ffn" in self.corrections:
                    self.corrections[f"{layer_idx}_ffn"].calibrate(error_accum, num_bits=num_bits)
                    error_accum = None


class AutoencoderWithCorrection(nn.Module):
    """For autoencoder task."""
    def __init__(self, input_size, hidden_sizes, latent_size, correction_every_n=2, correction_hidden=32):
        super().__init__()
        # Encoder
        self.encoder = nn.ModuleList()
        in_dim = input_size
        for h in hidden_sizes:
            self.encoder.append(nn.Linear(in_dim, h))
            in_dim = h
        self.encoder.append(nn.Linear(in_dim, latent_size))

        # Decoder
        self.decoder = nn.ModuleList()
        in_dim = latent_size
        for h in reversed(hidden_sizes):
            self.decoder.append(nn.Linear(in_dim, h))
            in_dim = h
        self.decoder.append(nn.Linear(in_dim, input_size))

        self.relu = nn.ReLU()

        # Correction positions (in encoder, excluding latent)
        encoder_positions = [i for i in range(len(hidden_sizes)) if (i + 1) % correction_every_n == 0]
        decoder_positions = [i for i in range(len(hidden_sizes)) if (i + 1) % correction_every_n == 0]

        self.encoder_corrections = nn.ModuleDict({
            f"enc_{i}": CorrectionNet(hidden_sizes[i], correction_hidden) for i in encoder_positions
        })
        self.decoder_corrections = nn.ModuleDict({
            f"dec_{i}": CorrectionNet(list(reversed(hidden_sizes))[i], correction_hidden) for i in decoder_positions
        })
        self.corrections = nn.ModuleDict({**self.encoder_corrections, **self.decoder_corrections})
        self.hidden_sizes = hidden_sizes

    def forward(self, x):
        for layer in self.encoder[:-1]:
            x = self.relu(layer(x))
        x = self.encoder[-1](x)  # latent
        for layer in self.decoder[:-1]:
            x = self.relu(layer(x))
        x = self.decoder[-1](x)
        return x

    def get_float_activations(self, x):
        activations = {}
        for i, layer in enumerate(self.encoder[:-1]):
            x = self.relu(layer(x))
            if f"enc_{i}" in self.encoder_corrections:
                activations[f"enc_{i}"] = x.clone()
        x = self.encoder[-1](x)
        for i, layer in enumerate(self.decoder[:-1]):
            x = self.relu(layer(x))
            if f"dec_{i}" in self.decoder_corrections:
                activations[f"dec_{i}"] = x.clone()
        return activations, self.decoder[-1](x)

    def calibrate(self, x, num_bits=8):
        _, quant_max = get_quant_range(num_bits)
        scales = []
        with torch.no_grad():
            for layer in self.encoder[:-1]:
                x = self.relu(layer(x))
                abs_max = max(abs(x.min().item()), abs(x.max().item()))
                scales.append(abs_max / quant_max if abs_max > 0 else 1.0)
            x = self.encoder[-1](x)
            abs_max = max(abs(x.min().item()), abs(x.max().item()))
            scales.append(abs_max / quant_max if abs_max > 0 else 1.0)
            for layer in self.decoder[:-1]:
                x = self.relu(layer(x))
                abs_max = max(abs(x.min().item()), abs(x.max().item()))
                scales.append(abs_max / quant_max if abs_max > 0 else 1.0)
        return scales

    def forward_quantized(self, x, scales, num_bits=8):
        idx = 0
        for layer in self.encoder[:-1]:
            x = self.relu(layer(x))
            x = fake_quantize(x, scales[idx], 0.0, num_bits=num_bits)
            idx += 1
        x = self.encoder[-1](x)
        x = fake_quantize(x, scales[idx], 0.0, num_bits=num_bits)
        idx += 1
        for layer in self.decoder[:-1]:
            x = self.relu(layer(x))
            x = fake_quantize(x, scales[idx], 0.0, num_bits=num_bits)
            idx += 1
        return self.decoder[-1](x)

    def forward_with_correction(self, x, scales, num_bits=8, quantize_correction=False, return_intermediates=False):
        intermediates = {} if return_intermediates else None
        idx = 0
        error_accum = None

        for i, layer in enumerate(self.encoder[:-1]):
            x = self.relu(layer(x))
            x, err = fake_quantize_with_error(x, scales[idx], 0.0, num_bits=num_bits)
            idx += 1
            error_accum = err if error_accum is None else error_accum + err
            if f"enc_{i}" in self.encoder_corrections:
                correction = self.encoder_corrections[f"enc_{i}"](error_accum, quantize=quantize_correction, num_bits=num_bits)
                x = x + correction
                if return_intermediates:
                    intermediates[f"enc_{i}"] = x.clone()
                error_accum = None

        x = self.encoder[-1](x)
        x, _ = fake_quantize_with_error(x, scales[idx], 0.0, num_bits=num_bits)
        idx += 1
        error_accum = None

        for i, layer in enumerate(self.decoder[:-1]):
            x = self.relu(layer(x))
            x, err = fake_quantize_with_error(x, scales[idx], 0.0, num_bits=num_bits)
            idx += 1
            error_accum = err if error_accum is None else error_accum + err
            if f"dec_{i}" in self.decoder_corrections:
                correction = self.decoder_corrections[f"dec_{i}"](error_accum, quantize=quantize_correction, num_bits=num_bits)
                x = x + correction
                if return_intermediates:
                    intermediates[f"dec_{i}"] = x.clone()
                error_accum = None

        output = self.decoder[-1](x)
        if return_intermediates:
            return output, intermediates
        return output

    def calibrate_corrections(self, x, scales, num_bits=8):
        idx = 0
        error_accum = None
        with torch.no_grad():
            for i, layer in enumerate(self.encoder[:-1]):
                x = self.relu(layer(x))
                _, err = fake_quantize_with_error(x, scales[idx], 0.0, num_bits=num_bits)
                idx += 1
                error_accum = err if error_accum is None else error_accum + err
                if f"enc_{i}" in self.encoder_corrections:
                    self.encoder_corrections[f"enc_{i}"].calibrate(error_accum, num_bits=num_bits)
                    error_accum = None
            x = self.encoder[-1](x)
            _, _ = fake_quantize_with_error(x, scales[idx], 0.0, num_bits=num_bits)
            idx += 1
            error_accum = None
            for i, layer in enumerate(self.decoder[:-1]):
                x = self.relu(layer(x))
                _, err = fake_quantize_with_error(x, scales[idx], 0.0, num_bits=num_bits)
                idx += 1
                error_accum = err if error_accum is None else error_accum + err
                if f"dec_{i}" in self.decoder_corrections:
                    self.decoder_corrections[f"dec_{i}"].calibrate(error_accum, num_bits=num_bits)
                    error_accum = None


# =============================================================================
# Training utilities
# =============================================================================

def train_with_qat(model, X_train, teacher_logits, teacher_activations, scales, num_bits,
                   layer_loss_weight, correction_epochs=300, lr=1e-4):
    """Train correction layers with hybrid distillation + QAT."""

    # Normalization factors
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
        acc_quant = (model.forward_quantized(X_test, scales, num_bits).argmax(1) == y_test).float().mean().item()

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
        mse_quant = F.mse_loss(model.forward_quantized(X_test, scales, num_bits), X_test).item()

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


def run_transformer(num_bits, layer_loss_weight, correction_hidden=32, seed=42):
    """Transformer language modeling on Shakespeare."""
    torch.manual_seed(seed)

    train_X, train_Y, test_X, test_Y, vocab_size, _, _ = load_shakespeare(seq_len=128)

    model = TransformerWithCorrection(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
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
            model.forward_quantized(test_X, scales, num_bits).reshape(-1, vocab_size),
            test_Y.reshape(-1)
        ).item()

    # Freeze backbone
    for name, p in model.named_parameters():
        if "corrections" not in name:
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

    tasks = ["classification", "autoencoder"]
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
