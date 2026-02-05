import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub

from aleph.quantization import fake_quantize, fake_quantize_with_error, get_quant_range


class MLP(nn.Module):
    """Multi-Layer Perceptron compatible with PyTorch Quantization Aware Training."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        depth: int = 1,
        activation_fn: type = nn.ReLU,
    ):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        layers = []
        in_features = input_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation_fn())
            in_features = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.layers(x)
        x = self.dequant(x)
        return x


class CorrectionNet(nn.Module):
    """Small network that predicts a correction from accumulated quantization error.

    Supports optional quantization of its own activations for fully-quantized inference.
    Works as either a linear correction (hidden_size=0) or MLP correction (hidden_size>0).
    """

    def __init__(self, size, hidden_size=32):
        super().__init__()
        if hidden_size and hidden_size > 0:
            self.layers = nn.Sequential(
                nn.Linear(size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, size),
            )
            nn.init.zeros_(self.layers[2].weight)
            nn.init.zeros_(self.layers[2].bias)
        else:
            self.layers = nn.Linear(size, size)
            nn.init.zeros_(self.layers.weight)
            nn.init.zeros_(self.layers.bias)

        self.hidden_size = hidden_size
        self._scales = {}

    def forward(self, error, quantize=False, num_bits=8):
        if not quantize:
            return self.layers(error)

        # Linear case: also quantize output
        if self.hidden_size == 0:
            out = self.layers(error)
            if 'output' in self._scales:
                out = fake_quantize(out, self._scales['output'], 0.0, num_bits=num_bits)
            return out

        # MLP case: quantize hidden and output
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


# Backward compat alias
CorrectionMLP = CorrectionNet


def _zero_points_or_default(zero_points, n):
    """Return zero_points list, defaulting to 0.0 if None (symmetric quant)."""
    if zero_points is None:
        return [0.0] * n
    return zero_points


class MLPWithCorrection(nn.Module):
    """MLP that supports oracle and learned correction during inference.

    Supports two calling conventions:
    - Experiment-style: forward_with_correction(x, scales, num_bits, ...)
    - run_experiment-style: forward_quantized(x, scale_factors, zero_points, num_bits)
    """

    def __init__(self, input_size, hidden_size, output_size, depth,
                 correction_every_n=2, correction_hidden=32):
        super().__init__()
        self.depth = depth

        # Backbone layers
        layers = []
        in_features = input_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        self.backbone = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # For backward compat with run_experiment.py (uses self.layers)
        self.layers = nn.ModuleList(list(self.backbone) + [self.head])

        # Correction networks
        self.correction_positions = [i for i in range(depth) if (i + 1) % correction_every_n == 0]
        self.correction_layers = nn.ModuleDict({
            str(i): CorrectionNet(hidden_size, correction_hidden)
            for i in self.correction_positions
        })

    @property
    def corrections(self):
        return self.correction_layers

    def forward(self, x):
        """Standard float forward."""
        for layer in self.backbone:
            x = self.relu(layer(x))
        return self.head(x)

    def get_quant_activations(self, x):
        """Get activations at quantization points for calibration."""
        activations = []
        for layer in self.backbone:
            x = self.relu(layer(x))
            activations.append(x.clone())
        return activations

    def get_float_activations(self, x):
        """Get float activations at each correction point."""
        activations = {}
        for i, layer in enumerate(self.backbone):
            x = self.relu(layer(x))
            if str(i) in self.correction_layers:
                activations[str(i)] = x.clone()
        return activations, self.head(x)

    def forward_quantized(self, x, scale_factors, zero_points=None, num_bits=8):
        """Forward pass with fake quantization, no correction."""
        zp = _zero_points_or_default(zero_points, self.depth)
        for i, layer in enumerate(self.backbone):
            x = self.relu(layer(x))
            x = fake_quantize(x, scale_factors[i], zp[i], num_bits=num_bits)
        return self.head(x)

    def forward_with_oracle_correction(self, x, scale_factors, zero_points=None,
                                       correct_every_n=3, num_bits=8):
        """Forward pass with oracle correction every N layers."""
        zp = _zero_points_or_default(zero_points, self.depth)
        x_float = x.clone()
        x_quant = x.clone()

        layers_since_correction = 0

        for i, layer in enumerate(self.backbone):
            x_float = self.relu(layer(x_float))
            x_quant = self.relu(layer(x_quant))
            x_quant = fake_quantize(x_quant, scale_factors[i], zp[i], num_bits=num_bits)

            layers_since_correction += 1
            if layers_since_correction >= correct_every_n:
                error = x_float - x_quant
                x_quant = x_quant + error
                layers_since_correction = 0

        x_float = self.head(x_float)
        x_quant = self.head(x_quant)

        return x_quant, x_float

    def forward_with_correction(self, x, scales, num_bits=8, quantize_correction=False,
                                return_intermediates=False, zero_points=None):
        """Forward pass with quantization and learned correction."""
        zp = _zero_points_or_default(zero_points, self.depth)
        error_accum = None
        intermediates = {} if return_intermediates else None

        for i, layer in enumerate(self.backbone):
            x = self.relu(layer(x))
            x, err = fake_quantize_with_error(x, scales[i], zp[i], num_bits=num_bits)
            error_accum = err if error_accum is None else error_accum + err

            if str(i) in self.correction_layers:
                correction = self.correction_layers[str(i)](
                    error_accum, quantize=quantize_correction, num_bits=num_bits
                )
                x = x + correction
                if return_intermediates:
                    intermediates[str(i)] = x.clone()
                error_accum = None

        logits = self.head(x)
        if return_intermediates:
            return logits, intermediates
        return logits

    # Alias for run_experiment.py backward compat
    def forward_quantized_with_correction(self, x, scale_factors, zero_points=None, num_bits=8):
        return self.forward_with_correction(x, scale_factors, num_bits=num_bits, zero_points=zero_points)

    def calibrate(self, x, num_bits=8):
        """Calibrate backbone quantization scales."""
        _, quant_max = get_quant_range(num_bits)
        scales = []
        with torch.no_grad():
            for layer in self.backbone:
                x = self.relu(layer(x))
                abs_max = max(abs(x.min().item()), abs(x.max().item()))
                scales.append(abs_max / quant_max if abs_max > 0 else 1.0)
        return scales

    def calibrate_corrections(self, x, scales, num_bits=8):
        """Calibrate correction networks by collecting sample errors."""
        error_accum = None
        with torch.no_grad():
            for i, layer in enumerate(self.backbone):
                x = self.relu(layer(x))
                _, err = fake_quantize_with_error(x, scales[i], 0.0, num_bits=num_bits)
                error_accum = err if error_accum is None else error_accum + err
                if str(i) in self.correction_layers:
                    self.correction_layers[str(i)].calibrate(error_accum, num_bits=num_bits)
                    error_accum = None


class MLPWithLearnedCorrection(nn.Module):
    """MLP that learns to correct accumulated quantization errors.

    Used by run_experiment.py for classification. Uses CorrectionMLP (no quantize support).
    """

    def __init__(self, input_size, hidden_size, output_size, depth, correction_every_n=3, correction_hidden=32):
        super().__init__()
        self.depth = depth
        self.correction_every_n = correction_every_n

        self.layers = nn.ModuleList()
        in_features = input_size
        for _ in range(depth):
            self.layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        self.layers.append(nn.Linear(hidden_size, output_size))

        self.relu = nn.ReLU()

        self.correction_indices = [
            idx for idx in range(depth) if (idx + 1) % correction_every_n == 0
        ]
        self.correction_layers = nn.ModuleList(
            [CorrectionNet(hidden_size, hidden_size=correction_hidden) for _ in self.correction_indices]
        )
        self._correction_index_map = {
            layer_idx: corr_idx for corr_idx, layer_idx in enumerate(self.correction_indices)
        }

    def forward_quantized_with_correction(self, x, scale_factors, zero_points=None, num_bits=8):
        """Forward pass with quantization and learned correction."""
        zp = _zero_points_or_default(zero_points, self.depth)
        error_accum = None

        for i, layer in enumerate(self.layers[:-1]):
            x = self.relu(layer(x))
            x_quant, err = fake_quantize_with_error(
                x, scale_factors[i], zp[i], num_bits=num_bits
            )
            x = x_quant
            if error_accum is None:
                error_accum = err.clone()
            else:
                error_accum = error_accum + err

            if i in self._correction_index_map:
                corr_layer = self.correction_layers[self._correction_index_map[i]]
                x = x + corr_layer(error_accum)
                error_accum = torch.zeros_like(x)

        x = self.layers[-1](x)
        return x


class AutoencoderWithCorrection(nn.Module):
    """Autoencoder that supports learned quantization correction."""

    def __init__(self, input_size, hidden_sizes, latent_size, correction_every_n=2, correction_hidden=32):
        super().__init__()
        self.correction_every_n = correction_every_n
        self.hidden_sizes = hidden_sizes

        # Encoder layers
        self.encoder = nn.ModuleList()
        in_features = input_size
        for h in hidden_sizes:
            self.encoder.append(nn.Linear(in_features, h))
            in_features = h
        self.encoder.append(nn.Linear(in_features, latent_size))

        # Decoder layers
        self.decoder = nn.ModuleList()
        in_features = latent_size
        for h in reversed(hidden_sizes):
            self.decoder.append(nn.Linear(in_features, h))
            in_features = h
        self.decoder.append(nn.Linear(in_features, input_size))

        self.relu = nn.ReLU()

        # Correction positions
        encoder_sizes = hidden_sizes + [latent_size]
        decoder_sizes = list(reversed(hidden_sizes))
        encoder_positions = [i for i in range(len(hidden_sizes)) if (i + 1) % correction_every_n == 0]
        decoder_positions = [i for i in range(len(hidden_sizes)) if (i + 1) % correction_every_n == 0]

        self.encoder_corrections = nn.ModuleDict({
            f"enc_{i}": CorrectionNet(encoder_sizes[i], correction_hidden)
            for i in encoder_positions
        })
        self.decoder_corrections = nn.ModuleDict({
            f"dec_{i}": CorrectionNet(decoder_sizes[i], correction_hidden)
            for i in decoder_positions
        })

        # Combined view for parameter access
        self.correction_layers = nn.ModuleDict({
            **self.encoder_corrections, **self.decoder_corrections
        })

    @property
    def corrections(self):
        return self.correction_layers

    def forward(self, x):
        """Standard forward pass (float32)."""
        for layer in self.encoder[:-1]:
            x = self.relu(layer(x))
        x = self.encoder[-1](x)
        for layer in self.decoder[:-1]:
            x = self.relu(layer(x))
        x = self.decoder[-1](x)
        return x

    def get_quant_activations(self, x):
        """Get activations at quantization points for calibration."""
        activations = []
        for layer in self.encoder[:-1]:
            x = self.relu(layer(x))
            activations.append(x.clone())
        x = self.encoder[-1](x)
        activations.append(x.clone())
        for layer in self.decoder[:-1]:
            x = self.relu(layer(x))
            activations.append(x.clone())
        return activations

    def get_float_activations(self, x):
        """Get float activations at correction points."""
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

    def forward_quantized(self, x, scale_factors, zero_points=None, num_bits=8):
        """Forward pass with fake quantization, no correction."""
        n_sf = len(self.encoder) - 1 + 1 + len(self.decoder) - 1
        zp = _zero_points_or_default(zero_points, n_sf)
        sf_idx = 0

        for layer in self.encoder[:-1]:
            x = self.relu(layer(x))
            x = fake_quantize(x, scale_factors[sf_idx], zp[sf_idx], num_bits=num_bits)
            sf_idx += 1
        x = self.encoder[-1](x)
        x = fake_quantize(x, scale_factors[sf_idx], zp[sf_idx], num_bits=num_bits)
        sf_idx += 1

        for layer in self.decoder[:-1]:
            x = self.relu(layer(x))
            x = fake_quantize(x, scale_factors[sf_idx], zp[sf_idx], num_bits=num_bits)
            sf_idx += 1
        x = self.decoder[-1](x)
        return x

    def forward_with_correction(self, x, scales, num_bits=8, quantize_correction=False,
                                return_intermediates=False, zero_points=None):
        """Forward pass with quantization and learned correction."""
        n_sf = len(self.encoder) - 1 + 1 + len(self.decoder) - 1
        zp = _zero_points_or_default(zero_points, n_sf)
        intermediates = {} if return_intermediates else None
        idx = 0
        error_accum = None

        for i, layer in enumerate(self.encoder[:-1]):
            x = self.relu(layer(x))
            x, err = fake_quantize_with_error(x, scales[idx], zp[idx], num_bits=num_bits)
            idx += 1
            error_accum = err if error_accum is None else error_accum + err

            if f"enc_{i}" in self.encoder_corrections:
                correction = self.encoder_corrections[f"enc_{i}"](
                    error_accum, quantize=quantize_correction, num_bits=num_bits
                )
                x = x + correction
                if return_intermediates:
                    intermediates[f"enc_{i}"] = x.clone()
                error_accum = None

        x = self.encoder[-1](x)
        x, _ = fake_quantize_with_error(x, scales[idx], zp[idx], num_bits=num_bits)
        idx += 1
        error_accum = None

        for i, layer in enumerate(self.decoder[:-1]):
            x = self.relu(layer(x))
            x, err = fake_quantize_with_error(x, scales[idx], zp[idx], num_bits=num_bits)
            idx += 1
            error_accum = err if error_accum is None else error_accum + err

            if f"dec_{i}" in self.decoder_corrections:
                correction = self.decoder_corrections[f"dec_{i}"](
                    error_accum, quantize=quantize_correction, num_bits=num_bits
                )
                x = x + correction
                if return_intermediates:
                    intermediates[f"dec_{i}"] = x.clone()
                error_accum = None

        output = self.decoder[-1](x)
        if return_intermediates:
            return output, intermediates
        return output

    # Alias for run_experiment.py backward compat
    def forward_quantized_with_correction(self, x, scale_factors, zero_points=None, num_bits=8):
        return self.forward_with_correction(x, scale_factors, num_bits=num_bits, zero_points=zero_points)

    def calibrate(self, x, num_bits=8):
        """Calibrate backbone quantization scales."""
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

    def calibrate_corrections(self, x, scales, num_bits=8):
        """Calibrate correction networks."""
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


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hd)

        # Causal attention
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerWithCorrection(nn.Module):
    """Small transformer for character-level generation with learned quantization correction."""

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=256,
        dropout=0.1,
        correction_every_n=2,
        correction_hidden=32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.correction_every_n = correction_every_n

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # Correction layers: one per correction point
        # Each transformer block has 2 quantization points (attn, ffn)
        self.correction_layers = nn.ModuleDict()
        quant_point = 0
        for layer_idx in range(n_layers):
            for sub in ['attn', 'ffn']:
                if (quant_point + 1) % correction_every_n == 0:
                    self.correction_layers[f"{layer_idx}_{sub}"] = CorrectionNet(
                        d_model, hidden_size=correction_hidden
                    )
                quant_point += 1

    @property
    def corrections(self):
        return self.correction_layers

    def _embed(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        return self.dropout(self.embedding(x) + self.pos_embedding(pos))

    def forward(self, x):
        """Standard forward pass (float32)."""
        x = self._embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.ln_f(x))

    def get_quant_activations(self, x):
        """Get activations at quantization points for calibration."""
        activations = []
        x = self._embed(x)
        for layer in self.layers:
            x_ln = layer.ln1(x)
            attn_out = layer.attn(x_ln)
            x = x + attn_out
            activations.append(x.clone())
            x_ln = layer.ln2(x)
            ffn_out = layer.ffn(x_ln)
            x = x + ffn_out
            activations.append(x.clone())
        return activations

    def get_float_activations(self, x):
        """Get float activations at correction points."""
        activations = {}
        x = self._embed(x)
        for layer_idx, layer in enumerate(self.layers):
            x_ln = layer.ln1(x)
            attn_out = layer.attn(x_ln)
            x = x + attn_out
            if f"{layer_idx}_attn" in self.correction_layers:
                activations[f"{layer_idx}_attn"] = x.clone()
            x_ln = layer.ln2(x)
            ffn_out = layer.ffn(x_ln)
            x = x + ffn_out
            if f"{layer_idx}_ffn" in self.correction_layers:
                activations[f"{layer_idx}_ffn"] = x.clone()
        return activations, self.head(self.ln_f(x))

    def forward_quantized(self, x, scale_factors, zero_points=None, num_bits=8):
        """Forward pass with fake quantization, no correction."""
        n_sf = self.n_layers * 2
        zp = _zero_points_or_default(zero_points, n_sf)
        x = self._embed(x)
        sf_idx = 0

        for layer in self.layers:
            x_ln = layer.ln1(x)
            attn_out = layer.attn(x_ln)
            x = x + attn_out
            x = fake_quantize(x, scale_factors[sf_idx], zp[sf_idx], num_bits=num_bits)
            sf_idx += 1

            x_ln = layer.ln2(x)
            ffn_out = layer.ffn(x_ln)
            x = x + ffn_out
            x = fake_quantize(x, scale_factors[sf_idx], zp[sf_idx], num_bits=num_bits)
            sf_idx += 1

        return self.head(self.ln_f(x))

    def forward_with_correction(self, x, scales, num_bits=8, quantize_correction=False,
                                return_intermediates=False, zero_points=None):
        """Forward pass with quantization and learned correction."""
        n_sf = self.n_layers * 2
        zp = _zero_points_or_default(zero_points, n_sf)
        intermediates = {} if return_intermediates else None
        x = self._embed(x)
        error_accum = None
        sf_idx = 0

        for layer_idx, layer in enumerate(self.layers):
            # Attention
            x_ln = layer.ln1(x)
            attn_out = layer.attn(x_ln)
            x = x + attn_out
            x, err = fake_quantize_with_error(x, scales[sf_idx], zp[sf_idx], num_bits=num_bits)
            sf_idx += 1
            error_accum = err if error_accum is None else error_accum + err

            key = f"{layer_idx}_attn"
            if key in self.correction_layers:
                correction = self.correction_layers[key](
                    error_accum, quantize=quantize_correction, num_bits=num_bits
                )
                x = x + correction
                if return_intermediates:
                    intermediates[key] = x.clone()
                error_accum = None

            # FFN
            x_ln = layer.ln2(x)
            ffn_out = layer.ffn(x_ln)
            x = x + ffn_out
            x, err = fake_quantize_with_error(x, scales[sf_idx], zp[sf_idx], num_bits=num_bits)
            sf_idx += 1
            error_accum = err if error_accum is None else error_accum + err

            key = f"{layer_idx}_ffn"
            if key in self.correction_layers:
                correction = self.correction_layers[key](
                    error_accum, quantize=quantize_correction, num_bits=num_bits
                )
                x = x + correction
                if return_intermediates:
                    intermediates[key] = x.clone()
                error_accum = None

        logits = self.head(self.ln_f(x))
        if return_intermediates:
            return logits, intermediates
        return logits

    # Alias for run_experiment.py backward compat
    def forward_quantized_with_correction(self, x, scale_factors, zero_points=None, num_bits=8):
        return self.forward_with_correction(x, scale_factors, num_bits=num_bits, zero_points=zero_points)

    def calibrate(self, x, num_bits=8):
        """Calibrate backbone quantization scales."""
        _, quant_max = get_quant_range(num_bits)
        scales = []
        x = self._embed(x)
        with torch.no_grad():
            for layer in self.layers:
                x_ln = layer.ln1(x)
                attn_out = layer.attn(x_ln)
                x = x + attn_out
                abs_max = max(abs(x.min().item()), abs(x.max().item()))
                scales.append(abs_max / quant_max if abs_max > 0 else 1.0)
                x_ln = layer.ln2(x)
                ffn_out = layer.ffn(x_ln)
                x = x + ffn_out
                abs_max = max(abs(x.min().item()), abs(x.max().item()))
                scales.append(abs_max / quant_max if abs_max > 0 else 1.0)
        return scales

    def calibrate_corrections(self, x, scales, num_bits=8):
        """Calibrate correction networks."""
        x = self._embed(x)
        error_accum = None
        sf_idx = 0
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.layers):
                x_ln = layer.ln1(x)
                attn_out = layer.attn(x_ln)
                x = x + attn_out
                _, err = fake_quantize_with_error(x, scales[sf_idx], 0.0, num_bits=num_bits)
                sf_idx += 1
                error_accum = err if error_accum is None else error_accum + err
                if f"{layer_idx}_attn" in self.correction_layers:
                    self.correction_layers[f"{layer_idx}_attn"].calibrate(error_accum, num_bits=num_bits)
                    error_accum = None
                x_ln = layer.ln2(x)
                ffn_out = layer.ffn(x_ln)
                x = x + ffn_out
                _, err = fake_quantize_with_error(x, scales[sf_idx], 0.0, num_bits=num_bits)
                sf_idx += 1
                error_accum = err if error_accum is None else error_accum + err
                if f"{layer_idx}_ffn" in self.correction_layers:
                    self.correction_layers[f"{layer_idx}_ffn"].calibrate(error_accum, num_bits=num_bits)
                    error_accum = None
