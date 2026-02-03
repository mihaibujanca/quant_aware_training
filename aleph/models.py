import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub

from aleph.quantization import fake_quantize, fake_quantize_with_error


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


class MLPWithCorrection(nn.Module):
    """MLP that supports oracle correction during inference."""

    def __init__(self, input_size, hidden_size, output_size, depth):
        super().__init__()
        self.depth = depth

        self.layers = nn.ModuleList()
        in_features = input_size
        for _ in range(depth):
            self.layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        self.layers.append(nn.Linear(hidden_size, output_size))

        self.relu = nn.ReLU()

    def forward(self, x):
        """Standard forward pass (float32)."""
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def get_quant_activations(self, x):
        """Get activations at quantization points for calibration."""
        activations = []
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            activations.append(x.clone())
        return activations

    def forward_quantized(self, x, scale_factors, zero_points, num_bits=8):
        """Forward pass with fake quantization, no correction."""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.relu(layer(x))
            x = fake_quantize(x, scale_factors[i], zero_points[i], num_bits=num_bits)
        x = self.layers[-1](x)
        return x

    def forward_with_oracle_correction(self, x, scale_factors, zero_points, correct_every_n=3, num_bits=8):
        """
        Forward pass with oracle correction every N layers.

        Runs float and quantized paths in parallel.
        Every N layers, corrects quantized path: x_quant += (x_float - x_quant)
        """
        x_float = x.clone()
        x_quant = x.clone()

        layers_since_correction = 0

        for i, layer in enumerate(self.layers[:-1]):
            x_float = self.relu(layer(x_float))
            x_quant = self.relu(layer(x_quant))
            x_quant = fake_quantize(x_quant, scale_factors[i], zero_points[i], num_bits=num_bits)

            layers_since_correction += 1
            if layers_since_correction >= correct_every_n:
                error = x_float - x_quant
                x_quant = x_quant + error
                layers_since_correction = 0

        x_float = self.layers[-1](x_float)
        x_quant = self.layers[-1](x_quant)

        return x_quant, x_float


class AutoencoderWithCorrection(nn.Module):
    """Autoencoder that supports learned quantization correction."""

    def __init__(self, input_size, hidden_sizes, latent_size, correction_every_n=2, correction_hidden=32):
        super().__init__()
        self.correction_every_n = correction_every_n

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

        # Layer output sizes (for correction layers)
        # Encoder: hidden_sizes + [latent]
        # Decoder: reversed(hidden_sizes) (not output)
        encoder_sizes = hidden_sizes + [latent_size]
        decoder_sizes = list(reversed(hidden_sizes))

        # Create correction layers at specified intervals
        self.encoder_correction = nn.ModuleDict()
        for i, size in enumerate(encoder_sizes[:-1]):  # exclude latent
            if (i + 1) % correction_every_n == 0:
                self.encoder_correction[str(i)] = CorrectionMLP(size, hidden_size=correction_hidden)

        self.decoder_correction = nn.ModuleDict()
        for i, size in enumerate(decoder_sizes):
            if (i + 1) % correction_every_n == 0:
                self.decoder_correction[str(i)] = CorrectionMLP(size, hidden_size=correction_hidden)

        self.correction_layers = nn.ModuleList(
            list(self.encoder_correction.values()) + list(self.decoder_correction.values())
        )

    def forward(self, x):
        """Standard forward pass (float32)."""
        # Encode
        for i, layer in enumerate(self.encoder[:-1]):
            x = self.relu(layer(x))
        x = self.encoder[-1](x)  # latent (no relu)

        # Decode
        for i, layer in enumerate(self.decoder[:-1]):
            x = self.relu(layer(x))
        x = self.decoder[-1](x)  # output (no relu)
        return x

    def get_quant_activations(self, x):
        """Get activations at quantization points for calibration."""
        activations = []
        # Encoder
        for layer in self.encoder[:-1]:
            x = self.relu(layer(x))
            activations.append(x.clone())
        x = self.encoder[-1](x)
        activations.append(x.clone())  # latent
        # Decoder (except output)
        for layer in self.decoder[:-1]:
            x = self.relu(layer(x))
            activations.append(x.clone())
        return activations

    def forward_quantized(self, x, scale_factors, zero_points, num_bits=8):
        """Forward pass with fake quantization, no correction."""
        from aleph.quantization import fake_quantize
        sf_idx = 0

        for i, layer in enumerate(self.encoder[:-1]):
            x = self.relu(layer(x))
            x = fake_quantize(x, scale_factors[sf_idx], zero_points[sf_idx], num_bits=num_bits)
            sf_idx += 1
        x = self.encoder[-1](x)
        x = fake_quantize(x, scale_factors[sf_idx], zero_points[sf_idx], num_bits=num_bits)
        sf_idx += 1

        for i, layer in enumerate(self.decoder[:-1]):
            x = self.relu(layer(x))
            x = fake_quantize(x, scale_factors[sf_idx], zero_points[sf_idx], num_bits=num_bits)
            sf_idx += 1
        x = self.decoder[-1](x)
        return x

    def forward_quantized_with_correction(self, x, scale_factors, zero_points, num_bits=8):
        """Forward pass with quantization and learned correction."""
        from aleph.quantization import fake_quantize_with_error
        error_accum = None
        sf_idx = 0

        # Encoder (except latent layer)
        for i, layer in enumerate(self.encoder[:-1]):
            x = self.relu(layer(x))
            x, err = fake_quantize_with_error(x, scale_factors[sf_idx], zero_points[sf_idx], num_bits=num_bits)
            sf_idx += 1

            if error_accum is None or error_accum.shape[-1] != err.shape[-1]:
                error_accum = err.clone()
            else:
                error_accum = error_accum + err

            if str(i) in self.encoder_correction:
                x = x + self.encoder_correction[str(i)](error_accum)
                error_accum = None

        # Latent
        x = self.encoder[-1](x)
        x, err = fake_quantize_with_error(x, scale_factors[sf_idx], zero_points[sf_idx], num_bits=num_bits)
        sf_idx += 1

        # Decoder (except output layer)
        error_accum = None
        for i, layer in enumerate(self.decoder[:-1]):
            x = self.relu(layer(x))
            x, err = fake_quantize_with_error(x, scale_factors[sf_idx], zero_points[sf_idx], num_bits=num_bits)
            sf_idx += 1

            if error_accum is None or error_accum.shape[-1] != err.shape[-1]:
                error_accum = err.clone()
            else:
                error_accum = error_accum + err

            if str(i) in self.decoder_correction:
                x = x + self.decoder_correction[str(i)](error_accum)
                error_accum = None

        x = self.decoder[-1](x)
        return x


class CorrectionMLP(nn.Module):
    """Small MLP to predict corrections from accumulated quantization error."""

    def __init__(self, size, hidden_size=32):
        super().__init__()
        if hidden_size and hidden_size > 0:
            self.net = nn.Sequential(
                nn.Linear(size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, size),
            )
            # Start as identity/no-op correction to avoid hurting baseline behavior.
            nn.init.zeros_(self.net[2].weight)
            nn.init.zeros_(self.net[2].bias)
        else:
            self.net = nn.Linear(size, size)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)

    def forward(self, x):
        return self.net(x)


class MLPWithLearnedCorrection(nn.Module):
    """MLP that learns to correct accumulated quantization errors."""

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
            [CorrectionMLP(hidden_size, hidden_size=correction_hidden) for _ in self.correction_indices]
        )
        self._correction_index_map = {
            layer_idx: corr_idx for corr_idx, layer_idx in enumerate(self.correction_indices)
        }

    def forward_quantized_with_correction(self, x, scale_factors, zero_points, num_bits=8):
        """Forward pass with quantization and learned correction."""
        error_accum = None

        for i, layer in enumerate(self.layers[:-1]):
            x = self.relu(layer(x))
            x_quant, err = fake_quantize_with_error(
                x, scale_factors[i], zero_points[i], num_bits=num_bits
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
        # We correct every N quantization points
        self.correction_layers = nn.ModuleDict()
        quant_point = 0
        for layer_idx in range(n_layers):
            for sub in ['attn', 'ffn']:
                if (quant_point + 1) % correction_every_n == 0:
                    self.correction_layers[f"{layer_idx}_{sub}"] = CorrectionMLP(
                        d_model, hidden_size=correction_hidden
                    )
                quant_point += 1

    def forward(self, x):
        """Standard forward pass (float32)."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.dropout(self.embedding(x) + self.pos_embedding(pos))

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def get_quant_activations(self, x):
        """Get activations at quantization points for calibration."""
        activations = []
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.dropout(self.embedding(x) + self.pos_embedding(pos))

        for layer in self.layers:
            # After attention
            x_ln = layer.ln1(x)
            attn_out = layer.attn(x_ln)
            x = x + attn_out
            activations.append(x.clone())
            # After FFN
            x_ln = layer.ln2(x)
            ffn_out = layer.ffn(x_ln)
            x = x + ffn_out
            activations.append(x.clone())
        return activations

    def forward_quantized(self, x, scale_factors, zero_points, num_bits=8):
        """Forward pass with fake quantization, no correction."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.dropout(self.embedding(x) + self.pos_embedding(pos))
        sf_idx = 0

        for layer in self.layers:
            # Attention
            x = layer.ln1(x)
            attn_out = layer.attn(x)
            x = x + attn_out
            x = fake_quantize(x, scale_factors[sf_idx], zero_points[sf_idx], num_bits=num_bits)
            sf_idx += 1

            # FFN
            x = layer.ln2(x)
            ffn_out = layer.ffn(x)
            x = x + ffn_out
            x = fake_quantize(x, scale_factors[sf_idx], zero_points[sf_idx], num_bits=num_bits)
            sf_idx += 1

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def forward_quantized_with_correction(self, x, scale_factors, zero_points, num_bits=8):
        """Forward pass with quantization and learned correction."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.dropout(self.embedding(x) + self.pos_embedding(pos))
        error_accum = None
        sf_idx = 0

        for layer_idx, layer in enumerate(self.layers):
            # Attention
            x_ln = layer.ln1(x)
            attn_out = layer.attn(x_ln)
            x = x + attn_out
            x, err = fake_quantize_with_error(x, scale_factors[sf_idx], zero_points[sf_idx], num_bits=num_bits)
            sf_idx += 1

            if error_accum is None:
                error_accum = err.clone()
            else:
                error_accum = error_accum + err

            key = f"{layer_idx}_attn"
            if key in self.correction_layers:
                x = x + self.correction_layers[key](error_accum)
                error_accum = None

            # FFN
            x_ln = layer.ln2(x)
            ffn_out = layer.ffn(x_ln)
            x = x + ffn_out
            x, err = fake_quantize_with_error(x, scale_factors[sf_idx], zero_points[sf_idx], num_bits=num_bits)
            sf_idx += 1

            if error_accum is None:
                error_accum = err.clone()
            else:
                error_accum = error_accum + err

            key = f"{layer_idx}_ffn"
            if key in self.correction_layers:
                x = x + self.correction_layers[key](error_accum)
                error_accum = None

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


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
