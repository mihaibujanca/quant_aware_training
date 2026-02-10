"""Correction distillation for variable-dimension architectures.

Extends Experiment 2 (correction distillation) from fixed-width MLPs to:
- Autoencoders: 784->256->128->32->128->256->784 (MNIST reconstruction)
- Transformer FFN paths: 128->512->128->512->... (Shakespeare char-level)

The core challenge: layers have different dimensions, so the CorrectionNet
must adapt. We use per-layer projections with a shared core MLP.

Usage:
    python experiments/correction_distillation_architectures.py --test   # quick validation
    python experiments/correction_distillation_architectures.py          # full overnight
    python experiments/correction_distillation_architectures.py --only autoencoder
    python experiments/correction_distillation_architectures.py --only transformer
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aleph.datasets import load_mnist_flat, load_shakespeare
from aleph.models import TransformerWithCorrection
from aleph.qgeom import ForwardTrace

# =============================================================================
# Config
# =============================================================================

BITS = 4
DELTA = 1.0 / (2 ** (BITS - 1))
SEED = 42
RESULTS_PATH = Path(__file__).resolve().parent.parent / "docs" / "correction_distillation_architectures.json"


def quantize_weights(W: torch.Tensor) -> torch.Tensor:
    return torch.round(W / DELTA) * DELTA


# =============================================================================
# Shared forward pass utilities (from canonical_overnight.py, with activation param)
# =============================================================================


def extract_linear_layers(module_list):
    """Extract (weights, biases) from a sequence of modules, keeping only nn.Linear."""
    weights, biases = [], []
    for m in module_list:
        if isinstance(m, nn.Linear):
            weights.append(m.weight.detach().clone())
            biases.append(m.bias.detach().clone())
    return weights, biases


@torch.no_grad()
def forward_pass(x, weights, biases, activation="relu"):
    """Manual forward pass collecting pre/post activations."""
    a = x
    pre_acts, post_acts = [], []
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = F.linear(a, W, b)
        is_last = (i == len(weights) - 1)
        if is_last:
            a = z
        elif activation == "relu":
            a = F.relu(z)
        elif activation == "gelu":
            a = F.gelu(z)
        else:
            a = F.relu(z)
        pre_acts.append(z)
        post_acts.append(a)
    return ForwardTrace(pre_acts=pre_acts, post_acts=post_acts)


@torch.no_grad()
def compute_oracle_corrections(x, weights, biases, weights_q, float_trace,
                                activation="relu"):
    """Compute oracle correction at each hidden layer."""
    a = x
    epsilon = torch.zeros_like(a)
    corrections = []
    for i in range(len(weights)):
        is_last = (i == len(weights) - 1)
        E = weights_q[i] - weights[i]
        W, Wq, b = weights[i], weights_q[i], biases[i]
        C = -F.linear(a, E) - F.linear(epsilon, W)
        z = F.linear(a, Wq, b) + C
        if is_last:
            a_new = z
        elif activation == "relu":
            a_new = F.relu(z)
        elif activation == "gelu":
            a_new = F.gelu(z)
        else:
            a_new = F.relu(z)
        epsilon = a_new - float_trace.post_acts[i]
        if not is_last:
            corrections.append(C)
        a = a_new
    return corrections


# =============================================================================
# VariableDimCorrectionNet — per-layer projections + shared core
# =============================================================================


class VariableDimCorrectionNet(nn.Module):
    """Correction network for architectures with variable layer dimensions.

    Architecture:
        proj_in[L]:  concat(z, c_local, embed) in dim_L space -> core_dim
        shared core: core_dim -> core_hidden -> core_dim
        proj_out[L]: core_dim -> dim_L
        skip:        output = proj_out(core(proj_in(...))) + c_local
    """

    def __init__(self, layer_dims, mode='combined', core_dim=64,
                 core_hidden=None, embed_dim=8, skip_local=False):
        """
        Args:
            layer_dims: list of hidden dimensions for each correctable layer
            mode: 'combined' (c_local + embedding), 'local' (c_local only),
                  'embedding' (embedding only)
            core_dim: dimension of shared projection space
            core_hidden: hidden dim of shared core MLP (defaults to core_dim)
            embed_dim: layer embedding dimension
            skip_local: if True, add c_local to output (skip connection)
        """
        super().__init__()
        self.mode = mode
        self.skip_local = skip_local
        self.layer_dims = layer_dims
        n_layers = len(layer_dims)

        if core_hidden is None:
            core_hidden = core_dim

        # Layer embedding
        if mode in ('embedding', 'combined'):
            self.layer_embed = nn.Embedding(n_layers, embed_dim)

        # Per-layer input projections: concat(z, [c_local], [embed]) -> core_dim
        self.proj_in = nn.ModuleList()
        for dim_L in layer_dims:
            in_dim = dim_L  # z is always present
            if mode in ('local', 'combined'):
                in_dim += dim_L  # c_local
            if mode in ('embedding', 'combined'):
                in_dim += embed_dim
            self.proj_in.append(nn.Linear(in_dim, core_dim))

        # Shared core MLP
        self.core = nn.Sequential(
            nn.Linear(core_dim, core_hidden),
            nn.ReLU(),
            nn.Linear(core_hidden, core_dim),
        )

        # Per-layer output projections: core_dim -> dim_L
        self.proj_out = nn.ModuleList()
        for dim_L in layer_dims:
            self.proj_out.append(nn.Linear(core_dim, dim_L))

    def forward(self, z, layer_idx, c_local=None):
        """
        Args:
            z: pre-activation tensor, shape (batch, dim_L)
            layer_idx: int, which hidden layer (indexes into layer_dims)
            c_local: local correction tensor, shape (batch, dim_L)
        """
        parts = [z]
        if self.mode in ('local', 'combined'):
            assert c_local is not None
            parts.append(c_local)
        if self.mode in ('embedding', 'combined'):
            idx = torch.full((z.shape[0],), layer_idx, dtype=torch.long,
                             device=z.device)
            parts.append(self.layer_embed(idx))

        h = self.proj_in[layer_idx](torch.cat(parts, dim=-1))
        h = self.core(h)
        out = self.proj_out[layer_idx](h)

        if self.skip_local and c_local is not None:
            out = out + c_local
        return out

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Corrected forward passes (variable-dim aware)
# =============================================================================


@torch.no_grad()
def direct_local_forward(x, weights_q, biases, error_matrices, activation="relu"):
    """Apply c_local = -E_L @ a directly (no neural network)."""
    a = x
    for L in range(len(weights_q)):
        z = F.linear(a, weights_q[L], biases[L])
        is_last = (L == len(weights_q) - 1)
        if not is_last:
            c_local = -F.linear(a, error_matrices[L])
            z = z + c_local
        if is_last:
            a = z
        elif activation == "relu":
            a = F.relu(z)
        elif activation == "gelu":
            a = F.gelu(z)
        else:
            a = F.relu(z)
    return a


def corrected_forward(x, weights_q, biases, error_matrices, net,
                      activation="relu"):
    """Forward pass with learned correction network."""
    a = x
    corrections = []
    for L in range(len(weights_q)):
        z = F.linear(a, weights_q[L], biases[L])
        is_last = (L == len(weights_q) - 1)
        if not is_last:
            c_local = -F.linear(a, error_matrices[L])
            C = net(z, L, c_local=c_local)
            z = z + C
            corrections.append(C)
        if is_last:
            a = z
        elif activation == "relu":
            a = F.relu(z)
        elif activation == "gelu":
            a = F.gelu(z)
        else:
            a = F.relu(z)
    return a, corrections


# =============================================================================
# Training
# =============================================================================


def train_correction(net, X_t, weights, biases, weights_q, error_matrices,
                     float_trace, activation="relu",
                     phase1_epochs=1000, phase2_epochs=500,
                     phase1_lr=1e-3, phase2_lr=1e-4):
    """Train a VariableDimCorrectionNet through Phase 1 + Phase 2."""
    n_hidden = len(weights) - 1

    # Oracle corrections (targets)
    oracle_corrections = compute_oracle_corrections(
        X_t, weights, biases, weights_q, float_trace, activation=activation)
    oracle_targets = [C.detach() for C in oracle_corrections]

    # Precompute teacher-forced inputs
    z_tf, c_local_tf = [], []
    for L in range(n_hidden):
        a_prev = X_t if L == 0 else float_trace.post_acts[L - 1]
        z_tf.append(F.linear(a_prev, weights_q[L], biases[L]).detach())
        c_local_tf.append((-F.linear(a_prev, error_matrices[L])).detach())

    # Phase 1: teacher forcing
    optimizer = torch.optim.Adam(net.parameters(), lr=phase1_lr)
    for epoch in range(phase1_epochs):
        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for L in range(n_hidden):
            C_pred = net(z_tf[L], L, c_local=c_local_tf[L])
            loss = loss + F.mse_loss(C_pred, oracle_targets[L])
        loss = loss / n_hidden
        loss.backward()
        optimizer.step()

    # Phase 2: autoregressive (minimize output error)
    float_output = float_trace.post_acts[-1].detach()
    optimizer = torch.optim.Adam(net.parameters(), lr=phase2_lr)
    for epoch in range(phase2_epochs):
        optimizer.zero_grad()
        output, _ = corrected_forward(X_t, weights_q, biases,
                                       error_matrices, net,
                                       activation=activation)
        loss = F.mse_loss(output, float_output)
        loss.backward()
        optimizer.step()

    return net


# =============================================================================
# Autoencoder experiment
# =============================================================================


def make_autoencoder(input_dim, hidden_sizes, latent_size):
    """Build encoder->decoder as a single nn.Sequential of Linear+ReLU layers."""
    layers = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
        in_dim = h
    layers.extend([nn.Linear(in_dim, latent_size), nn.ReLU()])
    in_dim = latent_size
    for h in reversed(hidden_sizes):
        layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
        in_dim = h
    layers.append(nn.Linear(in_dim, input_dim))  # no activation on output
    return nn.Sequential(*layers)


def train_autoencoder(model, train_loader, epochs=20):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for X_batch, _ in train_loader:
            loss = F.mse_loss(model(X_batch), X_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * X_batch.size(0)
            n += X_batch.size(0)
        if (epoch + 1) % max(1, epochs // 4) == 0:
            print(f"    Epoch {epoch+1}/{epochs}: MSE = {total_loss/n:.6f}")
    model.eval()
    return total_loss / n


def get_hidden_layer_dims(weights):
    """Get output dimensions of all hidden (non-last) layers."""
    return [W.shape[0] for W in weights[:-1]]


def count_sequential_params(model):
    """Count parameters in a nn.Sequential model."""
    return sum(p.numel() for p in model.parameters())


def run_autoencoder(test_mode=False, core_dims=None):
    """Autoencoder experiment: MNIST reconstruction."""
    print("\n" + "#" * 70)
    print("#  AUTOENCODER (MNIST) — Correction Distillation")
    print("#" * 70)

    if test_mode:
        ae_epochs = 3
        p1_epochs, p2_epochs = 50, 25
        if core_dims is None:
            core_dims = [64]
        ae_configs = [([256, 128], 32, "AE_256_128_32")]
    else:
        ae_epochs = 20
        p1_epochs, p2_epochs = 1000, 500
        if core_dims is None:
            core_dims = [32, 64, 128]
        ae_configs = [
            ([256, 128], 32, "AE_256_128_32"),
            ([128, 64], 16, "AE_128_64_16"),
        ]

    torch.manual_seed(SEED)
    train_loader, test_loader = load_mnist_flat(batch_size=256)

    # Get a batch of test data for analysis
    test_X, _ = next(iter(test_loader))

    results = {}

    for hidden_sizes, latent, label in ae_configs:
        print(f"\n--- {label} ---")
        torch.manual_seed(SEED)
        model = make_autoencoder(784, hidden_sizes, latent)
        task_params = count_sequential_params(model)

        print(f"  Training autoencoder ({ae_epochs} epochs)...")
        train_autoencoder(model, train_loader, epochs=ae_epochs)

        # Extract weights
        weights, biases = extract_linear_layers(model)
        weights_q = [quantize_weights(W) for W in weights]
        error_matrices = [Wq - W for W, Wq in zip(weights, weights_q)]
        n_hidden = len(weights) - 1

        # Architecture info
        arch = [weights[0].shape[1]] + [W.shape[0] for W in weights]
        hidden_dims = get_hidden_layer_dims(weights)
        print(f"  Architecture: {' -> '.join(str(d) for d in arch)}")
        print(f"  Hidden layer dims: {hidden_dims}")
        print(f"  Task params: {task_params}")

        # Float and quantized baselines
        float_trace = forward_pass(test_X, weights, biases)
        quant_trace = forward_pass(test_X, weights_q, biases)
        mse_float = F.mse_loss(float_trace.post_acts[-1], test_X).item()
        mse_quant = F.mse_loss(quant_trace.post_acts[-1], test_X).item()
        print(f"  Float MSE:     {mse_float:.6f}")
        print(f"  Quantized MSE: {mse_quant:.6f}")

        # Direct c_local (analytical ceiling)
        output_direct = direct_local_forward(test_X, weights_q, biases,
                                              error_matrices)
        mse_direct = F.mse_loss(output_direct, test_X).item()
        ratio_direct = mse_direct / mse_float if mse_float > 0 else float('inf')
        print(f"  Direct c_local MSE: {mse_direct:.6f} (ratio to float: {ratio_direct:.4f})")

        config_results = {
            "name": label,
            "architecture": arch,
            "hidden_dims": hidden_dims,
            "task_params": task_params,
            "mse_float": mse_float,
            "mse_quant": mse_quant,
            "mse_direct_local": mse_direct,
            "ratio_direct_local": ratio_direct,
            "variants": {},
        }

        # Learned correction variants
        for core_dim in core_dims:
            variant_label = f"combined+skip_core{core_dim}"
            print(f"\n  Training {variant_label}...")
            torch.manual_seed(SEED)
            net = VariableDimCorrectionNet(
                hidden_dims, mode='combined', core_dim=core_dim,
                embed_dim=8, skip_local=True,
            )
            net = train_correction(
                net, test_X, weights, biases, weights_q, error_matrices,
                float_trace, activation="relu",
                phase1_epochs=p1_epochs, phase2_epochs=p2_epochs,
            )
            with torch.no_grad():
                output_corr, _ = corrected_forward(
                    test_X, weights_q, biases, error_matrices, net)
                mse_corr = F.mse_loss(output_corr, test_X).item()

            n_p = net.n_params()
            ratio = mse_corr / mse_float if mse_float > 0 else float('inf')
            config_results["variants"][variant_label] = {
                "mse": mse_corr,
                "ratio_to_float": ratio,
                "params": n_p,
                "params_pct": round(100 * n_p / task_params, 1),
            }
            print(f"    MSE: {mse_corr:.6f}  ratio: {ratio:.4f}  "
                  f"params: {n_p} ({100*n_p/task_params:.1f}% of task)")

        results[label] = config_results

    return results


# =============================================================================
# Transformer FFN experiment
# =============================================================================


def extract_transformer_ffn_weights(model):
    """Extract FFN linear layers from transformer blocks as a flat sequence."""
    weights, biases = [], []
    for block in model.layers:
        for m in block.ffn:
            if isinstance(m, nn.Linear):
                weights.append(m.weight.detach().clone())
                biases.append(m.bias.detach().clone())
    return weights, biases


def count_transformer_ffn_params(model):
    """Count parameters in FFN layers only."""
    params = 0
    for block in model.layers:
        for m in block.ffn:
            if isinstance(m, nn.Linear):
                params += m.weight.numel() + m.bias.numel()
    return params


def run_transformer(test_mode=False, core_dims=None):
    """Transformer FFN experiment: Shakespeare, FFN path correction."""
    print("\n" + "#" * 70)
    print("#  TRANSFORMER FFN (Shakespeare) — Correction Distillation")
    print("#" * 70)

    if test_mode:
        tf_epochs = 2
        p1_epochs, p2_epochs = 50, 25
        if core_dims is None:
            core_dims = [64]
        tf_configs = [
            (128, 4, 4, 512, "TF_4L"),
        ]
    else:
        tf_epochs = 10
        p1_epochs, p2_epochs = 1000, 500
        if core_dims is None:
            core_dims = [64, 128]
        tf_configs = [
            (128, 4, 4, 512, "TF_4L"),
            (128, 4, 8, 512, "TF_8L"),
        ]

    torch.manual_seed(SEED)
    train_X, train_Y, test_X, test_Y, vocab_size, _, _ = load_shakespeare(seq_len=128)

    results = {}

    for d_model, n_heads, n_layers, d_ff, label in tf_configs:
        print(f"\n--- {label} (d={d_model}, h={n_heads}, L={n_layers}, ff={d_ff}) ---")
        torch.manual_seed(SEED)
        model = TransformerWithCorrection(
            vocab_size=vocab_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, max_seq_len=256, dropout=0.1,
            correction_every_n=999,  # no correction layers
        )

        # Train
        print(f"  Training transformer ({tf_epochs} epochs)...")
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        n_batches = len(train_X) // 64
        model.train()
        for epoch in range(tf_epochs):
            perm = torch.randperm(len(train_X))
            epoch_loss = 0
            for i in range(n_batches):
                batch_X = train_X[perm[i*64:(i+1)*64]]
                batch_Y = train_Y[perm[i*64:(i+1)*64]]
                logits = model(batch_X)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size),
                                        batch_Y.reshape(-1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()
            if (epoch + 1) % max(1, tf_epochs // 3) == 0:
                ppl = torch.exp(torch.tensor(epoch_loss / n_batches)).item()
                print(f"    Epoch {epoch+1}/{tf_epochs}: PPL = {ppl:.1f}")
        model.eval()

        # Verify training
        with torch.no_grad():
            test_logits = model(test_X[:256])
            loss_float = F.cross_entropy(
                test_logits.reshape(-1, vocab_size),
                test_Y[:256].reshape(-1),
            ).item()
        ppl_float = torch.exp(torch.tensor(loss_float)).item()
        print(f"  Float test PPL: {ppl_float:.1f}")

        # Extract FFN weights
        ffn_weights, ffn_biases = extract_transformer_ffn_weights(model)
        ffn_weights_q = [quantize_weights(W) for W in ffn_weights]
        error_matrices = [Wq - W for W, Wq in zip(ffn_weights, ffn_weights_q)]
        n_hidden = len(ffn_weights) - 1

        arch = [ffn_weights[0].shape[1]] + [W.shape[0] for W in ffn_weights]
        hidden_dims = get_hidden_layer_dims(ffn_weights)
        ffn_params = count_transformer_ffn_params(model)
        print(f"  FFN architecture: {' -> '.join(str(d) for d in arch)}")
        print(f"  FFN hidden dims: {hidden_dims}")
        print(f"  FFN params: {ffn_params}")

        # FFN input: residual stream at middle position after embedding
        with torch.no_grad():
            h = model._embed(test_X[:256])
            ffn_input = h[:, h.shape[1] // 2, :]  # (batch, d_model)

        # Float and quantized baselines
        float_trace = forward_pass(ffn_input, ffn_weights, ffn_biases,
                                    activation="gelu")
        quant_trace = forward_pass(ffn_input, ffn_weights_q, ffn_biases,
                                    activation="gelu")
        float_output = float_trace.post_acts[-1]
        quant_output = quant_trace.post_acts[-1]
        err_quant = (quant_output - float_output).norm(dim=-1).mean().item()
        print(f"  FFN output error (quantized): {err_quant:.4f}")

        # Direct c_local
        output_direct = direct_local_forward(ffn_input, ffn_weights_q,
                                              ffn_biases, error_matrices,
                                              activation="gelu")
        err_direct = (output_direct - float_output).norm(dim=-1).mean().item()
        print(f"  FFN output error (direct c_local): {err_direct:.4f}")

        config_results = {
            "name": label,
            "ffn_architecture": arch,
            "hidden_dims": hidden_dims,
            "ffn_params": ffn_params,
            "ppl_float": ppl_float,
            "err_quant": err_quant,
            "err_direct_local": err_direct,
            "variants": {},
        }

        # Learned correction variants
        for core_dim in core_dims:
            variant_label = f"combined+skip_core{core_dim}"
            print(f"\n  Training {variant_label}...")
            torch.manual_seed(SEED)
            net = VariableDimCorrectionNet(
                hidden_dims, mode='combined', core_dim=core_dim,
                embed_dim=8, skip_local=True,
            )
            net = train_correction(
                net, ffn_input, ffn_weights, ffn_biases,
                ffn_weights_q, error_matrices, float_trace,
                activation="gelu",
                phase1_epochs=p1_epochs, phase2_epochs=p2_epochs,
            )
            with torch.no_grad():
                output_corr, _ = corrected_forward(
                    ffn_input, ffn_weights_q, ffn_biases,
                    error_matrices, net, activation="gelu")
                err_corr = (output_corr - float_output).norm(dim=-1).mean().item()

            n_p = net.n_params()
            config_results["variants"][variant_label] = {
                "error": err_corr,
                "params": n_p,
                "params_pct": round(100 * n_p / ffn_params, 1),
            }
            print(f"    Error: {err_corr:.4f}  params: {n_p} "
                  f"({100*n_p/ffn_params:.1f}% of FFN)")

        results[label] = config_results

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Correction distillation for variable-dimension architectures")
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode (small epochs)")
    parser.add_argument("--only", choices=["autoencoder", "transformer"],
                        help="Run only one experiment type")
    args = parser.parse_args()

    start = time.time()
    all_results = {}

    experiments = {
        "autoencoder": lambda: run_autoencoder(test_mode=args.test),
        "transformer": lambda: run_transformer(test_mode=args.test),
    }

    if args.only:
        experiments = {args.only: experiments[args.only]}

    for name, fn in experiments.items():
        print(f"\n\n{'*' * 70}")
        print(f"*  Starting: {name}")
        print(f"{'*' * 70}")
        t0 = time.time()
        try:
            all_results[name] = fn()
            elapsed = time.time() - t0
            print(f"\n  [{name}] Completed in {elapsed:.0f}s")
        except Exception as e:
            print(f"\n  [{name}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    # Summary
    total_time = time.time() - start
    all_results["_meta"] = {
        "bits": BITS,
        "delta": DELTA,
        "seed": SEED,
        "test_mode": args.test,
        "total_time_s": round(total_time, 1),
    }

    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")

    # Autoencoder summary
    if "autoencoder" in all_results and "error" not in all_results["autoencoder"]:
        print(f"\n--- Autoencoder ---")
        print(f"{'Config':<20s}  {'Float MSE':>10s}  {'Quant MSE':>10s}  "
              f"{'Direct':>10s}  {'Ratio':>6s}  {'Best Learned':>12s}  "
              f"{'Ratio':>6s}  {'Params':>8s}  {'%Task':>6s}")
        print("-" * 100)
        for label, r in all_results["autoencoder"].items():
            best_v = None
            for vk, vv in r["variants"].items():
                if best_v is None or vv["mse"] < best_v["mse"]:
                    best_v = vv
            if best_v:
                print(f"{label:<20s}  {r['mse_float']:10.6f}  {r['mse_quant']:10.6f}  "
                      f"{r['mse_direct_local']:10.6f}  {r['ratio_direct_local']:6.3f}  "
                      f"{best_v['mse']:12.6f}  {best_v['ratio_to_float']:6.3f}  "
                      f"{best_v['params']:>8d}  {best_v['params_pct']:5.1f}%")

    # Transformer summary
    if "transformer" in all_results and "error" not in all_results["transformer"]:
        print(f"\n--- Transformer FFN ---")
        print(f"{'Config':<20s}  {'PPL':>6s}  {'Quant Err':>10s}  "
              f"{'Direct':>10s}  {'Best Learned':>12s}  "
              f"{'Params':>8s}  {'%FFN':>6s}")
        print("-" * 80)
        for label, r in all_results["transformer"].items():
            best_v = None
            for vk, vv in r["variants"].items():
                if best_v is None or vv["error"] < best_v["error"]:
                    best_v = vv
            if best_v:
                print(f"{label:<20s}  {r['ppl_float']:6.1f}  {r['err_quant']:10.4f}  "
                      f"{r['err_direct_local']:10.4f}  {best_v['error']:12.4f}  "
                      f"{best_v['params']:>8d}  {best_v['params_pct']:5.1f}%")

    print(f"\nTotal time: {total_time:.0f}s")

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
