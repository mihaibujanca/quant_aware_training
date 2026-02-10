"""Canonical error analysis across model types: classification, reconstruction, transformer.

Confirms that the canonical error decomposition findings (error compounding,
propagated dominance, oracle correction, bottleneck absorption) hold across
architectures — not just spirals MLPs.

Usage:
    python experiments/canonical_overnight.py                    # all experiments
    python experiments/canonical_overnight.py --only classification
    python experiments/canonical_overnight.py --only autoencoder
    python experiments/canonical_overnight.py --only transformer

Results are saved to docs/canonical_overnight_results.json and printed to stdout.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aleph.datasets import (
    embed_dataset_in_high_dimensional_space,
    load_mnist_flat,
    load_shakespeare,
    make_spirals,
)
from aleph.qgeom.canonical import (
    CanonicalSpaceTracker,
    ForwardTrace,
    ReLUDisagreementTracker,
    error_attribution,
)

# =============================================================================
# Config
# =============================================================================

BITS = 4
DELTA = 1.0 / (2 ** (BITS - 1))
SEED = 42
RESULTS_PATH = Path(__file__).resolve().parent.parent / "docs" / "canonical_overnight_results.json"


def quantize_weights(W: torch.Tensor) -> torch.Tensor:
    """Delta (grid) quantization: snap to nearest multiple of DELTA."""
    return torch.round(W / DELTA) * DELTA


# =============================================================================
# Generic canonical analysis (works for any sequence of linear layers)
# =============================================================================


def extract_linear_layers(module_list) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Extract (weights, biases) from a list of modules, keeping only nn.Linear."""
    weights, biases = [], []
    for m in module_list:
        if isinstance(m, nn.Linear):
            weights.append(m.weight.detach().clone())
            biases.append(m.bias.detach().clone())
    return weights, biases


@torch.no_grad()
def forward_pass(x: torch.Tensor, weights, biases, activation="relu") -> ForwardTrace:
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
def perfect_correction(x, weights, weights_q, biases, float_trace,
                       correct_at=None, activation="relu"):
    """Oracle correction: undo both local and propagated error at each layer."""
    if correct_at is None:
        correct_at = set(range(len(weights)))

    errors = [Wq - W for W, Wq in zip(weights, weights_q)]
    a = x
    epsilon = torch.zeros_like(a)
    pre_acts, post_acts = [], []

    for i in range(len(weights)):
        is_last = (i == len(weights) - 1)
        E, W, Wq, b = errors[i], weights[i], weights_q[i], biases[i]

        if i in correct_at:
            C = -F.linear(a, E) - F.linear(epsilon, W)
            z = F.linear(a, Wq, b) + C
        else:
            z = F.linear(a, Wq, b)

        if is_last:
            a_new = z
        elif activation == "relu":
            a_new = F.relu(z)
        elif activation == "gelu":
            a_new = F.gelu(z)
        else:
            a_new = F.relu(z)

        epsilon = a_new - float_trace.post_acts[i]
        pre_acts.append(z)
        post_acts.append(a_new)
        a = a_new

    return ForwardTrace(pre_acts, post_acts)


def run_canonical_analysis(name, x, weights, biases, activation="relu"):
    """Run full canonical error analysis on a sequence of linear layers.

    Returns a dict with all metrics: attribution, correction residuals,
    partial correction, ReLU disagreement, geometric properties.
    """
    weights_q = [quantize_weights(W) for W in weights]
    n_layers = len(weights)

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  {n_layers} layers, activation={activation}")
    arch = [weights[0].shape[1]] + [W.shape[0] for W in weights]
    print(f"  Architecture: {' -> '.join(str(d) for d in arch)}")
    print(f"{'='*70}")

    # Forward passes
    ft = forward_pass(x, weights, biases, activation)
    qt = forward_pass(x, weights_q, biases, activation)

    # --- Error attribution ---
    tracker = CanonicalSpaceTracker(weights)
    attrib = error_attribution(x, weights, weights_q, ft, qt, tracker)

    print(f"\n--- Error Attribution ---")
    print(f"{'Layer':<7s} {'Shape':<12s} {'Local':<10s} {'Propagated':<12s} {'Total':<10s} {'%Prop':<8s}")
    print("-" * 62)

    attrib_data = []
    for r in attrib:
        lo = r['local_output'].norm(dim=-1).mean().item()
        po = r['propagated_output'].norm(dim=-1).mean().item()
        to = r['total_output'].norm(dim=-1).mean().item()
        pct = po / (lo + po) * 100 if (lo + po) > 0 else 0
        layer_shape = tuple(weights[r['layer']].shape)
        print(f"L{r['layer']:<6d} {str(layer_shape):<12s} {lo:<10.4f} {po:<12.4f} {to:<10.4f} {pct:<8.0f}")
        attrib_data.append({
            "layer": r['layer'], "shape": list(layer_shape),
            "local": lo, "propagated": po, "total": to, "pct_propagated": pct,
        })

    # --- Perfect correction ---
    ct = perfect_correction(x, weights, weights_q, biases, ft, activation=activation)

    print(f"\n--- Perfect Correction Residual ---")
    correction_data = []
    for i in range(n_layers):
        ue = (qt.post_acts[i] - ft.post_acts[i]).norm(dim=-1).mean().item()
        ce = (ct.post_acts[i] - ft.post_acts[i]).norm(dim=-1).mean().item()
        print(f"  L{i}: uncorrected={ue:.4f}, corrected={ce:.2e}")
        correction_data.append({"layer": i, "uncorrected": ue, "corrected": ce})

    # --- Partial correction ---
    print(f"\n--- Partial Correction (output error) ---")
    partial_data = {}
    strategies = [
        (set(range(n_layers)), "All layers"),
        ({0}, "Layer 0 only"),
        ({n_layers - 1}, f"Output layer only (L{n_layers-1})"),
        (set(), "No correction"),
    ]
    if n_layers > 3:
        mid = n_layers // 2
        strategies.insert(2, ({mid}, f"Middle layer only (L{mid})"))

    for correct_at, label in strategies:
        rt = perfect_correction(x, weights, weights_q, biases, ft,
                                correct_at=correct_at, activation=activation)
        err = (rt.post_acts[-1] - ft.post_acts[-1]).norm(dim=-1).mean().item()
        print(f"  {label:<35s} -> output error = {err:.4f}")
        partial_data[label] = err

    # --- ReLU disagreement (only for relu activation) ---
    relu_data = []
    if activation == "relu":
        relu_t = ReLUDisagreementTracker(ft, qt)
        print(f"\n--- ReLU Disagreement ---")
        for i, frac in enumerate(relu_t.fractions):
            print(f"  Layer {i}: {frac*100:.1f}%")
            relu_data.append({"layer": i, "disagreement_pct": frac * 100})

    # --- Geometric metrics ---
    print(f"\n--- Geometric Metrics ---")
    print(f"{'Layer':<7s} {'||E||_2':<10s} {'||W||_2':<10s} {'cond(T_L)':<12s}")
    print("-" * 42)
    geo_data = []
    for i, (W, Wq) in enumerate(zip(weights, weights_q)):
        E = Wq - W
        T = tracker.cumulative_transform(i)
        e_norm = torch.linalg.norm(E, ord=2).item()
        w_norm = torch.linalg.norm(W, ord=2).item()
        cond = torch.linalg.cond(T).item()
        print(f"L{i:<6d} {e_norm:<10.4f} {w_norm:<10.4f} {cond:<12.1f}")
        geo_data.append({"layer": i, "E_norm": e_norm, "W_norm": w_norm, "cond_T": cond})

    # Summary
    output_err = attrib_data[-1]["total"]
    output_pct_prop = attrib_data[-1]["pct_propagated"]
    correction_residual = correction_data[-1]["corrected"]
    output_only_err = partial_data[f"Output layer only (L{n_layers-1})"]
    amplification = output_err / attrib_data[0]["total"] if attrib_data[0]["total"] > 0 else 0

    summary = {
        "output_error": output_err,
        "amplification": amplification,
        "pct_propagated_output": output_pct_prop,
        "correction_residual": correction_residual,
        "output_only_correction_error": output_only_err,
    }

    print(f"\n--- Summary ---")
    print(f"  Output error:          {output_err:.4f} ({amplification:.0f}x amplification)")
    print(f"  % propagated at output: {output_pct_prop:.0f}%")
    print(f"  Correction residual:   {correction_residual:.2e}")
    print(f"  Output-only correction: {output_only_err:.4f}")

    return {
        "name": name,
        "n_layers": n_layers,
        "architecture": arch,
        "activation": activation,
        "summary": summary,
        "attribution": attrib_data,
        "correction": correction_data,
        "partial_correction": partial_data,
        "relu_disagreement": relu_data,
        "geometric": geo_data,
    }


# =============================================================================
# Classification: MLP on spirals (2D and 100D)
# =============================================================================

def make_mlp(input_dim, hidden_dim, output_dim, depth):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def train_classifier(model, X, y, epochs=5000, lr=0.001):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        loss = loss_fn(model(X_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = ((torch.sigmoid(model(X_t)) > 0.5).float() == y_t).float().mean().item()
    return acc


def run_classification():
    """Classification experiment: spirals dataset, 2D and 100D."""
    print("\n" + "#" * 70)
    print("#  CLASSIFICATION (spirals, MLP)")
    print("#" * 70)

    torch.manual_seed(SEED)

    X_data, y_data = make_spirals(n_samples=2000, noise=0.5, n_turns=3, random_state=SEED)
    X_data *= 2.0
    X_t = torch.tensor(X_data, dtype=torch.float32)

    results = {}

    # --- 2D model ---
    for hidden_dim, depth, lr, label in [
        (32, 12, 0.001, "2D_deep"),
        (8, 4, 0.01, "2D_shallow"),
    ]:
        torch.manual_seed(SEED)
        model = make_mlp(2, hidden_dim, 1, depth)
        acc = train_classifier(model, X_data, y_data, epochs=5000, lr=lr)
        print(f"\n[{label}] Float accuracy: {acc:.1%}")

        weights, biases = extract_linear_layers(model)
        results[label] = run_canonical_analysis(
            f"Classification {label} ({2}->{hidden_dim}x{depth}->1)",
            X_t, weights, biases,
        )
        results[label]["float_accuracy"] = acc

    # --- 100D embedding ---
    X_high, embedding = embed_dataset_in_high_dimensional_space(X_data, target_dim=100, random_state=SEED)
    X_high_t = torch.tensor(X_high, dtype=torch.float32)

    for hidden_dim, depth, lr, label in [
        (32, 12, 0.001, "100D_deep"),
    ]:
        torch.manual_seed(SEED)
        model = make_mlp(100, hidden_dim, 1, depth)
        acc = train_classifier(model, X_high, y_data, epochs=5000, lr=lr)
        print(f"\n[{label}] Float accuracy: {acc:.1%}")

        weights, biases = extract_linear_layers(model)
        results[label] = run_canonical_analysis(
            f"Classification {label} (100->{hidden_dim}x{depth}->1)",
            X_high_t, weights, biases,
        )
        results[label]["float_accuracy"] = acc

    return results


# =============================================================================
# Autoencoder: MNIST reconstruction
# =============================================================================

def make_autoencoder(input_dim, hidden_sizes, latent_size):
    """Build encoder→decoder as a single nn.Sequential of Linear+ReLU layers."""
    layers = []
    # Encoder
    in_dim = input_dim
    for h in hidden_sizes:
        layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
        in_dim = h
    layers.extend([nn.Linear(in_dim, latent_size), nn.ReLU()])
    # Decoder
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
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: MSE = {total_loss/n:.6f}")
    model.eval()
    return total_loss / n


def run_autoencoder():
    """Autoencoder experiment: MNIST, encoder→latent→decoder as sequential MLP."""
    print("\n" + "#" * 70)
    print("#  AUTOENCODER (MNIST)")
    print("#" * 70)

    torch.manual_seed(SEED)
    train_loader, test_loader = load_mnist_flat(batch_size=256)

    # Use a batch of test data for canonical analysis
    test_X, _ = next(iter(test_loader))

    results = {}

    for hidden_sizes, latent, epochs, label in [
        ([256, 128], 32, 20, "AE_256_128_32"),
        ([128, 64], 16, 20, "AE_128_64_16"),
    ]:
        torch.manual_seed(SEED)
        model = make_autoencoder(784, hidden_sizes, latent)

        print(f"\n[{label}] Training autoencoder...")
        final_mse = train_autoencoder(model, train_loader, epochs=epochs)

        with torch.no_grad():
            mse_float = F.mse_loss(model(test_X), test_X).item()
        print(f"  Test MSE (float): {mse_float:.6f}")

        weights, biases = extract_linear_layers(model)
        weights_q = [quantize_weights(W) for W in weights]
        with torch.no_grad():
            qt = forward_pass(test_X, weights_q, biases)
            mse_quant = F.mse_loss(qt.post_acts[-1], test_X).item()
        print(f"  Test MSE (quantized): {mse_quant:.6f}")

        results[label] = run_canonical_analysis(
            f"Autoencoder {label} (784->{'->'.join(str(h) for h in hidden_sizes)}->{latent}->...->784)",
            test_X, weights, biases,
        )
        results[label]["mse_float"] = mse_float
        results[label]["mse_quant"] = mse_quant

    return results


# =============================================================================
# Transformer: Shakespeare character-level
# =============================================================================

def extract_transformer_ffn_weights(model):
    """Extract FFN linear layers from transformer blocks as a flat sequence.

    Each transformer block's FFN has two linear layers (up-project + down-project).
    We extract these in order to run canonical analysis on the FFN path.
    This ignores attention (which is harder to decompose) and focuses on the
    FFN sub-network that applies a pointwise nonlinear transform at each position.
    """
    weights, biases = [], []
    for block in model.layers:
        # FFN: Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)
        for m in block.ffn:
            if isinstance(m, nn.Linear):
                weights.append(m.weight.detach().clone())
                biases.append(m.bias.detach().clone())
    return weights, biases


@torch.no_grad()
def transformer_layer_errors(model, x):
    """Compute per-block activation error for the full transformer (incl. attention).

    Runs both float and quantized forward passes through the full model,
    collecting residual stream states after each block.
    """
    # Collect all linear layer weights
    all_linears = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and 'correction' not in name:
            all_linears.append((name, m))

    # Quantize all weights
    original_weights = {}
    for name, m in all_linears:
        original_weights[name] = m.weight.data.clone()

    # Float forward — collect residual stream after each block
    model.eval()
    float_states = []
    h = model._embed(x)
    for block in model.layers:
        h = block(h)
        float_states.append(h.clone())
    float_logits = model.head(model.ln_f(h))

    # Quantize weights
    for name, m in all_linears:
        m.weight.data = quantize_weights(m.weight.data)

    # Quantized forward
    quant_states = []
    h = model._embed(x)
    for block in model.layers:
        h = block(h)
        quant_states.append(h.clone())
    quant_logits = model.head(model.ln_f(h))

    # Restore
    for name, m in all_linears:
        m.weight.data = original_weights[name]

    # Compute per-block errors
    block_errors = []
    for i, (fs, qs) in enumerate(zip(float_states, quant_states)):
        err = (qs - fs).norm(dim=-1).mean().item()
        block_errors.append({"block": i, "residual_stream_error": err})

    logit_err = (quant_logits - float_logits).norm(dim=-1).mean().item()

    return block_errors, logit_err, float_logits, quant_logits


def run_transformer():
    """Transformer experiment: Shakespeare, FFN canonical analysis + full-model errors."""
    print("\n" + "#" * 70)
    print("#  TRANSFORMER (Shakespeare)")
    print("#" * 70)

    torch.manual_seed(SEED)
    train_X, train_Y, test_X, test_Y, vocab_size, _, _ = load_shakespeare(seq_len=128)

    results = {}

    for d_model, n_heads, n_layers, d_ff, epochs, label in [
        (128, 4, 4, 512, 10, "transformer_4L"),
        (128, 4, 8, 512, 15, "transformer_8L"),
    ]:
        torch.manual_seed(SEED)
        from aleph.models import TransformerWithCorrection

        model = TransformerWithCorrection(
            vocab_size=vocab_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, max_seq_len=256, dropout=0.1,
            correction_every_n=999,  # no correction layers
        )

        # Train
        print(f"\n[{label}] Training transformer (d={d_model}, h={n_heads}, L={n_layers})...")
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        n_batches = len(train_X) // 64

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(train_X))
            epoch_loss = 0
            for i in range(n_batches):
                batch_X = train_X[perm[i*64:(i+1)*64]]
                batch_Y = train_Y[perm[i*64:(i+1)*64]]
                logits = model(batch_X)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_Y.reshape(-1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                ppl = torch.exp(torch.tensor(epoch_loss / n_batches)).item()
                print(f"  Epoch {epoch+1}/{epochs}: PPL = {ppl:.1f}")

        model.eval()
        with torch.no_grad():
            test_logits = model(test_X[:256])
            loss_float = F.cross_entropy(
                test_logits.reshape(-1, vocab_size),
                test_Y[:256].reshape(-1),
            ).item()
        ppl_float = torch.exp(torch.tensor(loss_float)).item()
        print(f"  Float test PPL: {ppl_float:.1f}")

        # Full-model per-block error analysis
        print(f"\n--- Full-Model Block Errors (residual stream) ---")
        block_errors, logit_err, _, _ = transformer_layer_errors(model, test_X[:256])
        for be in block_errors:
            print(f"  Block {be['block']}: residual stream error = {be['residual_stream_error']:.4f}")
        print(f"  Logit error: {logit_err:.4f}")

        # FFN-only canonical analysis
        # Extract a sample of hidden states to feed through the FFN chain
        # We use the residual stream after embedding as input to the FFN analysis
        with torch.no_grad():
            h = model._embed(test_X[:256])
            # Take residual stream at a single position (middle) for FFN analysis
            # Shape: (batch, d_model)
            ffn_input = h[:, h.shape[1] // 2, :]

        ffn_weights, ffn_biases = extract_transformer_ffn_weights(model)

        if len(ffn_weights) > 0:
            ffn_result = run_canonical_analysis(
                f"Transformer FFN path ({label})",
                ffn_input, ffn_weights, ffn_biases, activation="gelu",
            )
        else:
            ffn_result = {}

        results[label] = {
            "name": label,
            "ppl_float": ppl_float,
            "block_errors": block_errors,
            "logit_error": logit_err,
            "ffn_canonical": ffn_result,
        }

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Canonical error analysis overnight run")
    parser.add_argument("--only", choices=["classification", "autoencoder", "transformer"],
                        help="Run only one experiment type")
    args = parser.parse_args()

    start = time.time()
    all_results = {}

    experiments = {
        "classification": run_classification,
        "autoencoder": run_autoencoder,
        "transformer": run_transformer,
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

    # Save results
    total_time = time.time() - start
    all_results["_meta"] = {
        "bits": BITS,
        "delta": DELTA,
        "seed": SEED,
        "total_time_s": total_time,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\n{'='*70}")
    print(f"All experiments completed in {total_time:.0f}s")
    print(f"Results saved to {RESULTS_PATH}")
    print(f"{'='*70}")

    # Print cross-architecture summary
    print(f"\n{'='*70}")
    print(f"  CROSS-ARCHITECTURE SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Model':<30s} {'Output Err':<12s} {'Amplif':<8s} {'%Prop':<8s} {'Corr Resid':<12s} {'Out-Only':<10s}")
    print("-" * 82)

    for exp_name, exp_results in all_results.items():
        if exp_name.startswith("_"):
            continue
        if isinstance(exp_results, dict) and "error" in exp_results:
            continue
        for sub_name, sub in exp_results.items():
            if not isinstance(sub, dict) or "summary" not in sub:
                # Transformer results have a different structure
                if isinstance(sub, dict) and "ffn_canonical" in sub and isinstance(sub["ffn_canonical"], dict) and "summary" in sub["ffn_canonical"]:
                    s = sub["ffn_canonical"]["summary"]
                    print(f"{sub_name + ' (FFN)':<30s} {s['output_error']:<12.4f} {s['amplification']:<8.0f}x "
                          f"{s['pct_propagated_output']:<8.0f} {s['correction_residual']:<12.2e} "
                          f"{s['output_only_correction_error']:<10.4f}")
                continue
            s = sub["summary"]
            print(f"{sub_name:<30s} {s['output_error']:<12.4f} {s['amplification']:<8.0f}x "
                  f"{s['pct_propagated_output']:<8.0f} {s['correction_residual']:<12.2e} "
                  f"{s['output_only_correction_error']:<10.4f}")


if __name__ == "__main__":
    main()
