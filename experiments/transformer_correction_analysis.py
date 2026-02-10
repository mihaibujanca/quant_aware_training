#!/usr/bin/env python3
"""
Per-Block Transformer Correction Analysis (Phase 1)

Measures six properties of quantization error in transformer FFN blocks:
1. SVD decay of per-block oracle corrections
2. Cross-block structure (principal angles between correction subspaces)
3. Correction cascade (does correcting block L-1 change rank at L?)
4. Metric vs topological error fraction per block
5. Residual stream containment (error propagation vs MLPs)
6. c_local effectiveness (-E_L @ a per-block)

Architecture: Minimal GPT on Shakespeare (char-level), quantize FFN weights to 4-bit.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aleph.datasets import load_shakespeare
from aleph.models import TransformerBlock


# =============================================================================
# Config
# =============================================================================

DEFAULT_CONFIG = dict(
    # Model
    d_model=128,
    d_ff=512,
    n_heads=4,
    n_layers=4,
    max_seq_len=128,
    dropout=0.1,
    # Training
    lr=3e-4,
    epochs=20,
    batch_size=64,
    # Quantization
    num_bits=4,
    # Analysis
    n_calibration=2048,
)

TEST_CONFIG = dict(
    d_model=64,
    d_ff=256,
    n_heads=4,
    n_layers=2,
    max_seq_len=64,
    dropout=0.1,
    lr=3e-4,
    epochs=3,
    batch_size=64,
    num_bits=4,
    n_calibration=256,
)


# =============================================================================
# Minimal GPT (reuses TransformerBlock from aleph.models)
# =============================================================================

class MiniGPT(nn.Module):
    """Minimal GPT for char-level Shakespeare."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def forward_with_hooks(self, idx):
        """Forward returning residual stream after each block's FFN."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        residuals = []
        for block in self.blocks:
            x = block(x)
            residuals.append(x.clone())
        logits = self.head(self.ln_f(x))
        return logits, residuals


# =============================================================================
# Weight quantization utilities
# =============================================================================

def quantize_tensor_uniform(W: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Symmetric uniform quantization of a weight tensor."""
    qmax = 2 ** (num_bits - 1) - 1
    scale = W.abs().max() / qmax
    if scale == 0:
        return W.clone()
    W_q = torch.round(W / scale).clamp(-qmax, qmax) * scale
    return W_q


def quantize_ffn_weights(block: TransformerBlock, num_bits: int):
    """Quantize FFN W1 and W2 in-place, return originals for restoration."""
    ffn = block.ffn
    # ffn is Sequential: Linear, GELU, Linear, Dropout
    W1_orig = ffn[0].weight.data.clone()
    W2_orig = ffn[2].weight.data.clone()

    ffn[0].weight.data = quantize_tensor_uniform(W1_orig, num_bits)
    ffn[2].weight.data = quantize_tensor_uniform(W2_orig, num_bits)

    return W1_orig, W2_orig


def restore_ffn_weights(block: TransformerBlock, W1_orig, W2_orig):
    """Restore original FFN weights."""
    block.ffn[0].weight.data = W1_orig
    block.ffn[2].weight.data = W2_orig


# =============================================================================
# Training
# =============================================================================

def train_model(model, train_X, train_Y, test_X, test_Y, cfg, device):
    """Train the model, return final train/test perplexity."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    n_train = train_X.shape[0]
    bs = cfg["batch_size"]

    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, bs):
            batch_idx = perm[i:i + bs]
            xb = train_X[batch_idx].to(device)
            yb = train_Y[batch_idx].to(device)

            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_ppl = math.exp(total_loss / n_batches)

        # Test perplexity
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            n_test_batches = 0
            for i in range(0, test_X.shape[0], bs):
                xb = test_X[i:i + bs].to(device)
                yb = test_Y[i:i + bs].to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
                test_loss += loss.item()
                n_test_batches += 1
            test_ppl = math.exp(test_loss / n_test_batches)

        print(f"  Epoch {epoch + 1}/{cfg['epochs']}: train_ppl={train_ppl:.1f}, test_ppl={test_ppl:.1f}")

    return train_ppl, test_ppl


# =============================================================================
# Analysis: collect residual streams
# =============================================================================

@torch.no_grad()
def collect_residual_streams(model, X, device, bs=64):
    """Run float model, return per-block residual stream tensors.

    Returns list of (N_tokens, d_model) tensors, one per block.
    Tokens are flattened across batch and sequence dimensions.
    """
    model.eval()
    all_residuals = [[] for _ in range(model.n_layers)]

    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs].to(device)
        _, residuals = model.forward_with_hooks(xb)
        for L, r in enumerate(residuals):
            # r: (B, T, d_model) -> flatten to (B*T, d_model)
            all_residuals[L].append(r.reshape(-1, r.shape[-1]).cpu())

    return [torch.cat(rs, dim=0) for rs in all_residuals]


@torch.no_grad()
def collect_quantized_residual_streams(model, X, device, num_bits, bs=64):
    """Run model with ALL FFN weights quantized, return per-block residuals.

    Quantizes all blocks simultaneously (realistic scenario).
    """
    # Quantize all blocks
    originals = []
    for block in model.blocks:
        originals.append(quantize_ffn_weights(block, num_bits))

    residuals = collect_residual_streams(model, X, device, bs)

    # Restore
    for block, (W1, W2) in zip(model.blocks, originals):
        restore_ffn_weights(block, W1, W2)

    return residuals


@torch.no_grad()
def collect_residuals_with_correction_at(model, X, device, num_bits,
                                         correct_block, float_residuals, bs=64):
    """Run quantized model but inject oracle correction at one specific block.

    After block `correct_block`, replace the quantized residual with the float one.
    Returns per-block residual streams for all blocks.
    """
    model.eval()
    # Quantize all blocks
    originals = []
    for block in model.blocks:
        originals.append(quantize_ffn_weights(block, num_bits))

    all_residuals = [[] for _ in range(model.n_layers)]
    token_offset = 0

    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs].to(device)
        B, T = xb.shape
        n_tokens = B * T

        pos = torch.arange(T, device=device).unsqueeze(0)
        x = model.drop(model.tok_emb(xb) + model.pos_emb(pos))

        for L, block in enumerate(model.blocks):
            x = block(x)

            if L == correct_block:
                # Replace with float residual (oracle correction)
                float_r = float_residuals[L][token_offset:token_offset + n_tokens]
                x = float_r.to(device).reshape(B, T, -1)

            all_residuals[L].append(x.reshape(-1, x.shape[-1]).cpu())

        token_offset += n_tokens

    # Restore
    for block, (W1, W2) in zip(model.blocks, originals):
        restore_ffn_weights(block, W1, W2)

    return [torch.cat(rs, dim=0) for rs in all_residuals]


# =============================================================================
# Analysis 1: SVD of oracle corrections
# =============================================================================

def analyze_svd(corrections, d_model):
    """SVD analysis of per-block correction matrices.

    corrections: list of (N, d_model) tensors (C_L = float - quant)
    Returns per-block: singular values, ranks at 90/95/99% energy.
    """
    results = []
    for L, C in enumerate(corrections):
        # C: (N, d_model). SVD to get spectrum.
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        S = S.cpu()
        energy = (S ** 2).cumsum(0) / (S ** 2).sum()

        rank_90 = int((energy < 0.90).sum()) + 1
        rank_95 = int((energy < 0.95).sum()) + 1
        rank_99 = int((energy < 0.99).sum()) + 1

        results.append({
            "block": L,
            "singular_values": S.tolist()[:min(64, d_model)],  # truncate for JSON
            "total_energy": float((S ** 2).sum()),
            "rank_90": rank_90,
            "rank_95": rank_95,
            "rank_99": rank_99,
            "top_1_frac": float(S[0] ** 2 / (S ** 2).sum()) if S.numel() > 0 else 0,
            "top_8_frac": float((S[:8] ** 2).sum() / (S ** 2).sum()) if S.numel() >= 8 else 1.0,
            "top_32_frac": float((S[:min(32, len(S))] ** 2).sum() / (S ** 2).sum()),
        })

    return results


# =============================================================================
# Analysis 2: Cross-block principal angles
# =============================================================================

def principal_angles(V1, V2):
    """Compute principal angles between column spaces of V1 and V2.

    V1, V2: (d_model, k) orthonormal basis matrices.
    Returns angles in degrees.
    """
    # SVD of V1^T @ V2 gives cos(angles) as singular values
    M = V1.T @ V2
    _, S, _ = torch.linalg.svd(M, full_matrices=False)
    S = S.clamp(0, 1)  # numerical safety
    angles = torch.acos(S) * 180.0 / math.pi
    return angles


def analyze_cross_block(corrections, d_model, k=16):
    """Compute principal angles between top-k correction subspaces across blocks.

    Returns matrix of mean angles (n_layers x n_layers).
    """
    n_layers = len(corrections)
    # Get top-k right singular vectors for each block
    bases = []
    for C in corrections:
        _, _, Vh = torch.linalg.svd(C, full_matrices=False)
        V_k = Vh[:min(k, Vh.shape[0])].T  # (d_model, k)
        bases.append(V_k)

    angle_matrix = torch.zeros(n_layers, n_layers)
    for i in range(n_layers):
        for j in range(n_layers):
            angles = principal_angles(bases[i], bases[j])
            angle_matrix[i, j] = angles.mean()

    return {
        "k": k,
        "mean_angle_matrix": angle_matrix.tolist(),
        "min_angle_matrix": [
            [float(principal_angles(bases[i], bases[j]).min())
             for j in range(n_layers)]
            for i in range(n_layers)
        ],
    }


# =============================================================================
# Analysis 3: Correction cascade
# =============================================================================

def analyze_cascade(model, X, device, num_bits, float_residuals, quant_residuals, d_model):
    """Does correcting block L-1 change the correction needed at block L?

    For each block L>0, compare:
    - C_L without any correction
    - C_L with oracle correction at L-1
    """
    n_layers = model.n_layers
    results = []

    for correct_at in range(n_layers - 1):
        # Run with correction at `correct_at`
        corrected_residuals = collect_residuals_with_correction_at(
            model, X, device, num_bits, correct_at, float_residuals
        )

        L = correct_at + 1  # measure at next block
        C_original = float_residuals[L] - quant_residuals[L]
        C_after_correction = float_residuals[L] - corrected_residuals[L]

        orig_norm = C_original.norm().item()
        after_norm = C_after_correction.norm().item()
        reduction = 1.0 - after_norm / orig_norm if orig_norm > 0 else 0.0

        results.append({
            "corrected_block": correct_at,
            "measured_block": L,
            "error_norm_before": orig_norm,
            "error_norm_after": after_norm,
            "error_reduction_frac": reduction,
        })

    return results


# =============================================================================
# Analysis 4: Metric vs topological decomposition
# =============================================================================

@torch.no_grad()
def analyze_metric_topological(model, X, device, num_bits, bs=64):
    """Per-block metric vs topological error decomposition.

    For each block: check if GELU sign patterns agree between float and quantized.
    Metric error = error where signs agree (linearly correctable).
    Topological error = error where signs disagree (hyperplane crossing).
    """
    model.eval()

    # We need to hook into the FFN internals
    # FFN: x_ln -> W1 -> GELU -> W2 -> (+ residual)
    # "Sign pattern" = sign of the pre-GELU activation (W1 @ x_ln + b1)

    n_layers = model.n_layers

    # Quantize all blocks
    originals = []
    for block in model.blocks:
        originals.append(quantize_ffn_weights(block, num_bits))

    # Collect quantized pre-GELU activations
    quant_pre_gelu = [[] for _ in range(n_layers)]
    quant_residuals = [[] for _ in range(n_layers)]

    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs].to(device)
        B, T = xb.shape
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = model.drop(model.tok_emb(xb) + model.pos_emb(pos))

        for L, block in enumerate(model.blocks):
            x = x + block.attn(block.ln1(x))
            x_ln = block.ln2(x)
            # FFN: Linear -> GELU -> Linear -> Dropout
            pre_gelu = block.ffn[0](x_ln)  # (B, T, d_ff)
            quant_pre_gelu[L].append(pre_gelu.reshape(-1, pre_gelu.shape[-1]).cpu())
            ffn_out = block.ffn[3](block.ffn[2](block.ffn[1](pre_gelu)))
            x = x + ffn_out
            quant_residuals[L].append(x.reshape(-1, x.shape[-1]).cpu())

    # Restore float weights
    for block, (W1, W2) in zip(model.blocks, originals):
        restore_ffn_weights(block, W1, W2)

    # Collect float pre-GELU activations
    float_pre_gelu = [[] for _ in range(n_layers)]
    float_residuals = [[] for _ in range(n_layers)]

    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs].to(device)
        B, T = xb.shape
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = model.drop(model.tok_emb(xb) + model.pos_emb(pos))

        for L, block in enumerate(model.blocks):
            x = x + block.attn(block.ln1(x))
            x_ln = block.ln2(x)
            pre_gelu = block.ffn[0](x_ln)
            float_pre_gelu[L].append(pre_gelu.reshape(-1, pre_gelu.shape[-1]).cpu())
            ffn_out = block.ffn[3](block.ffn[2](block.ffn[1](pre_gelu)))
            x = x + ffn_out
            float_residuals[L].append(x.reshape(-1, x.shape[-1]).cpu())

    # Compare sign patterns
    results = []
    for L in range(n_layers):
        f_pg = torch.cat(float_pre_gelu[L], dim=0)
        q_pg = torch.cat(quant_pre_gelu[L], dim=0)
        f_res = torch.cat(float_residuals[L], dim=0)
        q_res = torch.cat(quant_residuals[L], dim=0)

        # GELU "sign": positive vs negative (GELU(x) ≈ x for x>>0, ≈ 0 for x<<0)
        sign_agree = (f_pg > 0) == (q_pg > 0)  # (N, d_ff)
        agree_frac = sign_agree.float().mean().item()

        # Decompose FFN output error into metric and topological components.
        # For each token, the FFN output is W2 @ GELU(W1 @ x + b1) + b2.
        # Where GELU signs agree, the error is a smooth function of W perturbation
        # (metric/linear). Where they disagree, the error has a discontinuity
        # (topological).
        #
        # Approach: for each token, zero out the d_ff dims where signs disagree,
        # recompute FFN output difference. That gives the "metric-only" error.
        C = f_res - q_res  # (N, d_model) total error

        # Metric component: error that would occur if only the agree-dims contributed
        # We need W2 to project back. Use float W2.
        W2 = block.ffn[2].weight.data  # (d_model, d_ff)
        b2 = block.ffn[2].bias.data if block.ffn[2].bias is not None else None

        # Difference in FFN hidden activations
        delta_a = block.ffn[1](q_pg.to(device)) - block.ffn[1](f_pg.to(device))  # (N, d_ff)

        # Mask: zero out disagreeing dims
        agree_mask = sign_agree.to(device).float()  # (N, d_ff)
        delta_a_metric = delta_a * agree_mask
        delta_a_topo = delta_a * (1 - agree_mask)

        # Project through W2 to get d_model-space error components
        # (This is an approximation — ignores W1 quantization error on agree dims)
        metric_error_proj = F.linear(delta_a_metric, W2, None)  # (N, d_model)
        topo_error_proj = F.linear(delta_a_topo, W2, None)

        metric_energy = (metric_error_proj ** 2).sum().item()
        topo_energy = (topo_error_proj ** 2).sum().item()
        total_energy = metric_energy + topo_energy

        # Also track per-token: fraction with any sign disagreement
        any_disagree = (~sign_agree).any(dim=1)

        results.append({
            "block": L,
            "gelu_sign_agreement_frac": agree_frac,
            "tokens_with_any_sign_flip": any_disagree.float().mean().item(),
            "metric_error_frac": metric_energy / total_energy if total_energy > 0 else 0,
            "topo_error_frac": topo_energy / total_energy if total_energy > 0 else 0,
        })

    return results


# =============================================================================
# Analysis 5: Residual stream containment
# =============================================================================

def analyze_containment(float_residuals, quant_residuals):
    """How much does error grow through layers?

    Compare to MLPs where error compounds multiplicatively.
    In transformers with residuals, error is additive to the stream.
    """
    results = []
    for L in range(len(float_residuals)):
        C = float_residuals[L] - quant_residuals[L]
        stream_norm = float_residuals[L].norm(dim=1).mean().item()
        error_norm = C.norm(dim=1).mean().item()
        relative_error = error_norm / stream_norm if stream_norm > 0 else 0

        results.append({
            "block": L,
            "stream_norm": stream_norm,
            "error_norm": error_norm,
            "relative_error": relative_error,
        })

    # Error growth ratio: layer L / layer 0
    if results[0]["error_norm"] > 0:
        for r in results:
            r["error_growth_vs_L0"] = r["error_norm"] / results[0]["error_norm"]
    else:
        for r in results:
            r["error_growth_vs_L0"] = 0.0

    return results


# =============================================================================
# Analysis 6: c_local effectiveness
# =============================================================================

@torch.no_grad()
def analyze_c_local(model, X, device, num_bits, float_residuals, bs=64):
    """Test c_local = -E_L @ a correction per block.

    E_L = W_q - W_float for the FFN down-projection (W2).
    a = GELU(W1 @ x_ln + b1) = the FFN hidden activation.
    c_local = -E_L @ a approximates the local FFN output error.
    """
    model.eval()
    n_layers = model.n_layers
    results = []

    for L in range(n_layers):
        block = model.blocks[L]
        W2_float = block.ffn[2].weight.data.clone()
        W2_quant = quantize_tensor_uniform(W2_float, num_bits)
        E_L = W2_quant - W2_float  # (d_model, d_ff)

        W1_float = block.ffn[0].weight.data.clone()
        W1_quant = quantize_tensor_uniform(W1_float, num_bits)

        # Collect: run float model up to block L, get FFN hidden activation
        # Then compute c_local = -E_L @ a
        c_local_norms = []
        actual_error_norms = []
        corrected_error_norms = []

        # We need the input to block L's FFN (post-attn residual)
        # Re-run the float model to get this
        for i in range(0, X.shape[0], bs):
            xb = X[i:i + bs].to(device)
            B, T = xb.shape
            pos = torch.arange(T, device=device).unsqueeze(0)
            x = model.drop(model.tok_emb(xb) + model.pos_emb(pos))

            for l in range(L):
                x = model.blocks[l](x)

            # Now at block L input
            x_post_attn = x + block.attn(block.ln1(x))
            x_ln = block.ln2(x_post_attn)

            # Float FFN output
            a_float = block.ffn[1](block.ffn[0](x_ln))  # GELU(W1 @ x_ln + b1)
            ffn_out_float = block.ffn[2](a_float)  # W2 @ a + b2

            # Quantized FFN output (using quantized weights)
            pre_gelu_q = F.linear(x_ln, W1_quant, block.ffn[0].bias)
            a_quant = block.ffn[1](pre_gelu_q)  # GELU
            ffn_out_quant = F.linear(a_quant, W2_quant, block.ffn[2].bias)

            # Actual FFN output error (what enters residual stream)
            actual_error = ffn_out_quant - ffn_out_float  # (B, T, d_model)

            # c_local = -(E_W2 @ a_float + E_W1 effect)
            # Simplest: c_local = -(ffn_out_quant - ffn_out_float) is trivially perfect
            # The useful thing: can we approximate with just -E_W2 @ a_float?
            c_local = -F.linear(a_float, E_L, None)  # (B, T, d_model)

            corrected_error = actual_error + c_local  # residual after c_local

            actual_error_norms.append(actual_error.reshape(-1, actual_error.shape[-1]).norm(dim=1).cpu())
            c_local_norms.append(c_local.reshape(-1, c_local.shape[-1]).norm(dim=1).cpu())
            corrected_error_norms.append(corrected_error.reshape(-1, corrected_error.shape[-1]).norm(dim=1).cpu())

        actual_norm = torch.cat(actual_error_norms).mean().item()
        c_local_norm = torch.cat(c_local_norms).mean().item()
        corrected_norm = torch.cat(corrected_error_norms).mean().item()

        results.append({
            "block": L,
            "actual_ffn_error_norm": actual_norm,
            "c_local_norm": c_local_norm,
            "residual_after_c_local": corrected_norm,
            "error_reduction_frac": 1.0 - corrected_norm / actual_norm if actual_norm > 0 else 0,
        })

    return results


# =============================================================================
# Perplexity evaluation
# =============================================================================

@torch.no_grad()
def eval_perplexity(model, X, Y, device, bs=64):
    """Evaluate perplexity."""
    model.eval()
    total_loss = 0.0
    n = 0
    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs].to(device)
        yb = Y[i:i + bs].to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        total_loss += loss.item()
        n += 1
    return math.exp(total_loss / n)


@torch.no_grad()
def eval_quantized_perplexity(model, X, Y, device, num_bits, bs=64):
    """Evaluate perplexity with all FFN weights quantized."""
    originals = []
    for block in model.blocks:
        originals.append(quantize_ffn_weights(block, num_bits))

    ppl = eval_perplexity(model, X, Y, device, bs)

    for block, (W1, W2) in zip(model.blocks, originals):
        restore_ffn_weights(block, W1, W2)
    return ppl


# =============================================================================
# Plotting
# =============================================================================

def make_plots(results, output_dir):
    """Generate analysis plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # 1. SVD decay curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for svd in results["svd"]:
        S = torch.tensor(svd["singular_values"])
        energy = (S ** 2).cumsum(0) / (S ** 2).sum()
        ax.plot(range(1, len(energy) + 1), energy.numpy(), label=f"Block {svd['block']}")
    ax.set_xlabel("Number of singular values")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_title("SVD Decay of Per-Block Oracle Corrections")
    ax.legend()
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.99, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "svd_decay.png"), dpi=150)
    plt.close(fig)

    # 2. Cross-block similarity heatmap
    angle_matrix = torch.tensor(results["cross_block"]["mean_angle_matrix"])
    n = angle_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(angle_matrix.numpy(), cmap="RdYlBu", vmin=0, vmax=90)
    ax.set_xlabel("Block")
    ax.set_ylabel("Block")
    ax.set_title(f"Mean Principal Angles (top-{results['cross_block']['k']} subspaces)")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{angle_matrix[i, j]:.0f}°",
                    ha="center", va="center", fontsize=9)
    fig.colorbar(im, label="Mean angle (degrees)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cross_block_angles.png"), dpi=150)
    plt.close(fig)

    # 3. Metric vs topological bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    blocks = [r["block"] for r in results["metric_topo"]]
    metric = [r["metric_error_frac"] for r in results["metric_topo"]]
    topo = [r["topo_error_frac"] for r in results["metric_topo"]]
    width = 0.35
    x = range(len(blocks))
    ax.bar([i - width / 2 for i in x], metric, width, label="Metric (same GELU sign)")
    ax.bar([i + width / 2 for i in x], topo, width, label="Topological (sign flip)")
    ax.set_xlabel("Block")
    ax.set_ylabel("Error energy fraction")
    ax.set_title("Metric vs Topological Error Decomposition")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metric_vs_topo.png"), dpi=150)
    plt.close(fig)

    # 4. Error containment across layers
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    blocks = [r["block"] for r in results["containment"]]
    error_norms = [r["error_norm"] for r in results["containment"]]
    relative = [r["relative_error"] for r in results["containment"]]

    axes[0].bar(blocks, error_norms)
    axes[0].set_xlabel("Block")
    axes[0].set_ylabel("Mean error norm")
    axes[0].set_title("Absolute Error per Block")

    axes[1].bar(blocks, relative)
    axes[1].set_xlabel("Block")
    axes[1].set_ylabel("Error / stream norm")
    axes[1].set_title("Relative Error per Block")

    fig.suptitle("Residual Stream Error Containment")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_containment.png"), dpi=150)
    plt.close(fig)

    # 5. c_local effectiveness
    fig, ax = plt.subplots(figsize=(8, 5))
    blocks = [r["block"] for r in results["c_local"]]
    actual = [r["actual_ffn_error_norm"] for r in results["c_local"]]
    residual = [r["residual_after_c_local"] for r in results["c_local"]]
    x = range(len(blocks))
    width = 0.35
    ax.bar([i - width / 2 for i in x], actual, width, label="FFN error (uncorrected)")
    ax.bar([i + width / 2 for i in x], residual, width, label="Residual after c_local")
    ax.set_xlabel("Block")
    ax.set_ylabel("Mean norm")
    ax.set_title("c_local = -E_W2 @ a Correction Effectiveness")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "c_local_effectiveness.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {output_dir}/")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Transformer per-block correction analysis")
    parser.add_argument("--test", action="store_true", help="Quick test mode (2 layers, 3 epochs)")
    parser.add_argument("--n-layers", type=int, default=None, help="Override n_layers")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--plots-dir", type=str, default=None, help="Plot output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    cfg = TEST_CONFIG.copy() if args.test else DEFAULT_CONFIG.copy()
    if args.n_layers:
        cfg["n_layers"] = args.n_layers

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")
    print(f"Config: {json.dumps(cfg, indent=2)}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n=== Loading Shakespeare ===")
    train_X, train_Y, test_X, test_Y, vocab_size, _, _ = load_shakespeare(
        seq_len=cfg["max_seq_len"]
    )
    print(f"Vocab: {vocab_size}, Train: {train_X.shape}, Test: {test_X.shape}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n=== Training Model ===")
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    t0 = time.time()
    train_ppl, test_ppl = train_model(model, train_X, train_Y, test_X, test_Y, cfg, device)
    train_time = time.time() - t0
    print(f"Training done in {train_time:.1f}s. Final test_ppl={test_ppl:.1f}")

    # Verify task performance
    if test_ppl > 20:
        print(f"WARNING: test_ppl={test_ppl:.1f} is high — model may be undertrained")

    # Quantized perplexity
    quant_ppl = eval_quantized_perplexity(model, test_X, test_Y, device, cfg["num_bits"])
    print(f"Quantized (4-bit FFN weights) test_ppl={quant_ppl:.1f}")
    ppl_degradation = quant_ppl - test_ppl
    print(f"Perplexity degradation: +{ppl_degradation:.1f}")

    # ── Calibration data ──────────────────────────────────────────────────
    n_cal = min(cfg["n_calibration"], train_X.shape[0])
    cal_X = train_X[:n_cal]
    print(f"\nCalibration set: {cal_X.shape[0]} sequences, "
          f"{cal_X.shape[0] * cal_X.shape[1]:,} tokens")

    # ── Collect residual streams ──────────────────────────────────────────
    print("\n=== Collecting Residual Streams ===")
    model.eval()
    # Disable dropout for deterministic analysis
    model.drop.p = 0.0
    for block in model.blocks:
        block.ffn[3].p = 0.0  # FFN dropout
        block.attn.dropout.p = 0.0

    t0 = time.time()
    float_residuals = collect_residual_streams(model, cal_X, device)
    quant_residuals = collect_quantized_residual_streams(model, cal_X, device, cfg["num_bits"])
    print(f"Collected in {time.time() - t0:.1f}s")

    corrections = [f - q for f, q in zip(float_residuals, quant_residuals)]
    for L, C in enumerate(corrections):
        print(f"  Block {L}: error_norm={C.norm():.2f}, "
              f"mean_token_error={C.norm(dim=1).mean():.4f}")

    # ── Analysis 1: SVD ──────────────────────────────────────────────────
    print("\n=== Analysis 1: SVD Decay ===")
    svd_results = analyze_svd(corrections, cfg["d_model"])
    for s in svd_results:
        print(f"  Block {s['block']}: rank_95={s['rank_95']}, rank_99={s['rank_99']}, "
              f"top-8={s['top_8_frac']:.1%}, top-32={s['top_32_frac']:.1%}")

    # ── Analysis 2: Cross-block structure ────────────────────────────────
    print("\n=== Analysis 2: Cross-Block Principal Angles ===")
    k = min(16, cfg["d_model"])
    cross_results = analyze_cross_block(corrections, cfg["d_model"], k=k)
    angle_matrix = torch.tensor(cross_results["mean_angle_matrix"])
    for i in range(cfg["n_layers"]):
        angles_str = ", ".join(f"{angle_matrix[i, j]:.1f}°" for j in range(cfg["n_layers"]))
        print(f"  Block {i}: [{angles_str}]")

    # ── Analysis 3: Cascade ──────────────────────────────────────────────
    print("\n=== Analysis 3: Correction Cascade ===")
    cascade_results = analyze_cascade(
        model, cal_X, device, cfg["num_bits"], float_residuals, quant_residuals, cfg["d_model"]
    )
    for c in cascade_results:
        print(f"  Correct block {c['corrected_block']} → block {c['measured_block']}: "
              f"error {c['error_norm_before']:.2f} → {c['error_norm_after']:.2f} "
              f"({c['error_reduction_frac']:.1%} reduction)")

    # ── Analysis 4: Metric vs topological ────────────────────────────────
    print("\n=== Analysis 4: Metric vs Topological ===")
    mt_results = analyze_metric_topological(model, cal_X, device, cfg["num_bits"])
    for r in mt_results:
        print(f"  Block {r['block']}: GELU agree={r['gelu_sign_agreement_frac']:.1%}, "
              f"tokens_w_flip={r['tokens_with_any_sign_flip']:.1%}, "
              f"metric_error={r['metric_error_frac']:.1%}, "
              f"topo_error={r['topo_error_frac']:.1%}")

    # ── Analysis 5: Containment ──────────────────────────────────────────
    print("\n=== Analysis 5: Residual Stream Containment ===")
    contain_results = analyze_containment(float_residuals, quant_residuals)
    for r in contain_results:
        print(f"  Block {r['block']}: error={r['error_norm']:.4f}, "
              f"stream={r['stream_norm']:.2f}, "
              f"relative={r['relative_error']:.4f}, "
              f"growth_vs_L0={r['error_growth_vs_L0']:.2f}x")

    # ── Analysis 6: c_local ──────────────────────────────────────────────
    print("\n=== Analysis 6: c_local Effectiveness ===")
    c_local_results = analyze_c_local(model, cal_X, device, cfg["num_bits"], float_residuals)
    for r in c_local_results:
        print(f"  Block {r['block']}: ffn_error={r['actual_ffn_error_norm']:.4f}, "
              f"after_c_local={r['residual_after_c_local']:.4f}, "
              f"reduction={r['error_reduction_frac']:.1%}")

    # ── Assemble results ─────────────────────────────────────────────────
    all_results = {
        "config": cfg,
        "performance": {
            "train_ppl": train_ppl,
            "test_ppl": test_ppl,
            "quant_ppl": quant_ppl,
            "ppl_degradation": ppl_degradation,
            "n_params": n_params,
            "train_time_s": train_time,
        },
        "svd": svd_results,
        "cross_block": cross_results,
        "cascade": cascade_results,
        "metric_topo": mt_results,
        "containment": contain_results,
        "c_local": c_local_results,
    }

    # Save JSON
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "..", "docs",
        "transformer_correction_analysis.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Plots
    if not args.no_plots:
        plots_dir = args.plots_dir or os.path.join(
            os.path.dirname(__file__), "..", "plots", "transformer_correction"
        )
        make_plots(all_results, plots_dir)

    return all_results


if __name__ == "__main__":
    main()
