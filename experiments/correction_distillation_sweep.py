"""Correction distillation sweep across configurations.

Validates Experiment 2 findings (distilled correction networks) across:
- Datasets: spirals, moons
- Depths: 4, 8, 12
- Widths: 8, 32
- High-dimensional embedding (100D) for spirals depth-12

For each config, trains 4 correction variants:
- direct_local: analytical c_local = -E_L @ a (0 learned params, ceiling)
- embedding: layer embedding only (genuinely deployable)
- local+skip: c_local + learned residual
- combined+skip: c_local + embedding + learned residual

Reports accuracy and param counts (absolute + % of task network).

Usage: python experiments/correction_distillation_sweep.py
"""

import json
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons

sys.path.insert(0, ".")
from aleph.datasets import make_spirals, embed_dataset_in_high_dimensional_space
from aleph.qgeom import ForwardTrace

BITS = 4
DELTA = 1.0 / (2 ** (BITS - 1))
SEED = 42
EPOCHS = 5000
MIN_ACCURACY = 0.85

# Correction network hyperparameters
NET_HIDDEN = 64
EMBED_DIM = 8
PHASE1_EPOCHS = 1000
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 500
PHASE2_LR = 1e-4


def make_mlp(hidden_dim, depth, input_dim=2, output_dim=1):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def count_model_params(hidden_dim, depth, input_dim=2, output_dim=1):
    """Count parameters in the task MLP (without building it)."""
    # First layer: input_dim -> hidden_dim (weight + bias)
    params = hidden_dim * input_dim + hidden_dim
    # Hidden layers: hidden_dim -> hidden_dim
    params += (depth - 1) * (hidden_dim * hidden_dim + hidden_dim)
    # Output layer: hidden_dim -> output_dim
    params += output_dim * hidden_dim + output_dim
    return params


def train_model(model, X, y, lr, epochs=EPOCHS):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_t)) > 0.5).float()
        acc = (preds == y_t).float().mean().item()
    return model, acc


def extract_weights(model, delta=DELTA):
    weights, biases = [], []
    for module in model:
        if isinstance(module, nn.Linear):
            weights.append(module.weight.detach().clone())
            biases.append(module.bias.detach().clone())
    weights_q = [torch.round(W / delta) * delta for W in weights]
    return weights, biases, weights_q


@torch.no_grad()
def forward_pass(x, weights, biases):
    a = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    pre_acts, post_acts = [], []
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = F.linear(a, W, b)
        a = z if i == len(weights) - 1 else F.relu(z)
        pre_acts.append(z)
        post_acts.append(a)
    return ForwardTrace(pre_acts=pre_acts, post_acts=post_acts)


@torch.no_grad()
def accuracy(logits, y):
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == y).float().mean().item()


@torch.no_grad()
def direct_local_forward(x, weights_q, biases, error_matrices):
    """Apply c_local = -E_L @ a directly (no neural network)."""
    a = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    for L in range(len(weights_q)):
        z = F.linear(a, weights_q[L], biases[L])
        is_last = (L == len(weights_q) - 1)
        if not is_last:
            c_local = -F.linear(a, error_matrices[L])
            z = z + c_local
        a = z if is_last else F.relu(z)
    return a


class CorrectionNet(nn.Module):
    def __init__(self, hidden_dim, n_layers, mode='combined',
                 net_hidden=NET_HIDDEN, embed_dim=EMBED_DIM,
                 skip_local=False):
        super().__init__()
        self.mode = mode
        self.skip_local = skip_local
        in_dim = hidden_dim  # z is always present
        if mode in ('embedding', 'combined'):
            self.layer_embed = nn.Embedding(n_layers, embed_dim)
            in_dim += embed_dim
        if mode in ('local', 'combined'):
            in_dim += hidden_dim  # c_local
        self.net = nn.Sequential(
            nn.Linear(in_dim, net_hidden),
            nn.ReLU(),
            nn.Linear(net_hidden, hidden_dim),
        )

    def forward(self, z, layer_idx, c_local=None):
        parts = [z]
        if self.mode in ('local', 'combined'):
            assert c_local is not None
            parts.append(c_local)
        if self.mode in ('embedding', 'combined'):
            idx = torch.full((z.shape[0],), layer_idx, dtype=torch.long,
                             device=z.device)
            parts.append(self.layer_embed(idx))
        out = self.net(torch.cat(parts, dim=1))
        if self.skip_local and c_local is not None:
            out = out + c_local
        return out

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def corrected_forward(x, weights_q, biases, error_matrices, net):
    a = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    corrections = []
    for L in range(len(weights_q)):
        z = F.linear(a, weights_q[L], biases[L])
        is_last = (L == len(weights_q) - 1)
        if not is_last:
            c_local = -F.linear(a, error_matrices[L])
            C = net(z, L, c_local=c_local)
            z = z + C
            corrections.append(C)
        a = z if is_last else F.relu(z)
    return a, corrections


@torch.no_grad()
def compute_oracle_corrections(x, weights, biases, weights_q, float_trace):
    a = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    epsilon = torch.zeros_like(a)
    corrections = []
    for i in range(len(weights)):
        is_last = (i == len(weights) - 1)
        E = weights_q[i] - weights[i]
        W, Wq, b = weights[i], weights_q[i], biases[i]
        C = -F.linear(a, E) - F.linear(epsilon, W)
        z = F.linear(a, Wq, b) + C
        a_new = z if is_last else F.relu(z)
        epsilon = a_new - float_trace.post_acts[i]
        if not is_last:
            corrections.append(C)
        a = a_new
    return corrections


def train_correction(net, X_t, weights_q, biases, error_matrices,
                     oracle_targets, float_trace, n_hidden):
    """Train a CorrectionNet through Phase 1 + Phase 2."""
    # Precompute teacher-forced inputs
    z_tf, c_local_tf = [], []
    for L in range(n_hidden):
        a_prev = X_t if L == 0 else float_trace.post_acts[L - 1]
        z_tf.append(F.linear(a_prev, weights_q[L], biases[L]).detach())
        c_local_tf.append((-F.linear(a_prev, error_matrices[L])).detach())

    # Phase 1: teacher forcing
    optimizer = torch.optim.Adam(net.parameters(), lr=PHASE1_LR)
    for _ in range(PHASE1_EPOCHS):
        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for L in range(n_hidden):
            C_pred = net(z_tf[L], L, c_local=c_local_tf[L])
            loss = loss + F.mse_loss(C_pred, oracle_targets[L])
        loss = loss / n_hidden
        loss.backward()
        optimizer.step()

    # Phase 2: autoregressive
    float_logits = float_trace.post_acts[-1].detach()
    y_t = torch.tensor(
        [0.0], dtype=torch.float32  # placeholder, accuracy measured after
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=PHASE2_LR)
    for _ in range(PHASE2_EPOCHS):
        optimizer.zero_grad()
        logits, _ = corrected_forward(X_t, weights_q, biases,
                                      error_matrices, net)
        loss = F.mse_loss(logits, float_logits)
        loss.backward()
        optimizer.step()

    return net


# VARIANTS: (label, mode, skip_local)
VARIANTS = [
    ("embedding",     "embedding", False),
    ("local+skip",    "local",     True),
    ("combined+skip", "combined",  True),
]


def run_config(name, dataset, hidden_dim, depth, input_dim=2, lr=0.001,
               embed_dim=None):
    """Run one configuration and return summary dict."""
    torch.manual_seed(SEED)

    # Dataset
    if dataset == "spirals":
        X, y = make_spirals(n_samples=2000, noise=0.5, n_turns=3, random_state=SEED)
        X *= 2.0
    elif dataset == "moons":
        X, y = make_moons(n_samples=2000, noise=0.15, random_state=SEED)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    actual_input_dim = input_dim
    if embed_dim is not None:
        X, _ = embed_dataset_in_high_dimensional_space(X, target_dim=embed_dim,
                                                        random_state=SEED)
        actual_input_dim = embed_dim

    # Train
    model = make_mlp(hidden_dim, depth, input_dim=actual_input_dim)
    model, acc = train_model(model, X, y, lr=lr)

    if acc < MIN_ACCURACY:
        print(f"  SKIP {name}: accuracy {acc:.1%} < {MIN_ACCURACY:.0%}")
        return {"name": name, "skipped": True, "accuracy": round(acc, 4)}

    weights, biases, weights_q = extract_weights(model)
    error_matrices = [Wq - W for W, Wq in zip(weights, weights_q)]
    n_hidden = len(weights) - 1

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    float_trace = forward_pass(X_t, weights, biases)
    quant_trace = forward_pass(X_t, weights_q, biases)

    acc_quant = accuracy(quant_trace.post_acts[-1], y_t)

    # Direct c_local (analytical ceiling)
    logits_direct = direct_local_forward(X_t, weights_q, biases, error_matrices)
    acc_direct = accuracy(logits_direct, y_t)

    # Oracle corrections for training
    oracle_corrections = compute_oracle_corrections(
        X_t, weights, biases, weights_q, float_trace)
    oracle_targets = [C.detach() for C in oracle_corrections]

    # Task network params
    task_params = count_model_params(hidden_dim, depth,
                                     input_dim=actual_input_dim)

    # Float ablation helper
    zero_errors = [torch.zeros_like(E) for E in error_matrices]

    # Train each variant
    variant_results = {}
    for label, mode, skip in VARIANTS:
        torch.manual_seed(SEED)
        net = CorrectionNet(hidden_dim, n_hidden, mode, skip_local=skip)
        net = train_correction(net, X_t, weights_q, biases, error_matrices,
                               oracle_targets, float_trace, n_hidden)
        with torch.no_grad():
            logits, _ = corrected_forward(X_t, weights_q, biases,
                                          error_matrices, net)
            acc_v = accuracy(logits, y_t)
            # Float ablation
            logits_f, _ = corrected_forward(X_t, weights, biases,
                                            zero_errors, net)
            acc_float_abl = accuracy(logits_f, y_t)

        n_p = net.n_params()
        variant_results[label] = {
            "accuracy": round(acc_v, 4),
            "params": n_p,
            "params_pct": round(100 * n_p / task_params, 1),
            "float_ablation": round(acc_float_abl, 4),
        }
        print(f"    {label:<16s}: {acc_v:.1%}  ({n_p} params, "
              f"{100*n_p/task_params:.1f}% of task)")

    return {
        "name": name,
        "skipped": False,
        "dataset": dataset,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "input_dim": actual_input_dim,
        "task_params": task_params,
        "accuracy_float": round(acc, 4),
        "accuracy_quant": round(acc_quant, 4),
        "accuracy_direct_local": round(acc_direct, 4),
        "variants": variant_results,
    }


CONFIGS = [
    # (name, dataset, hidden_dim, depth, lr, embed_dim)
    ("spirals_32x12",  "spirals", 32, 12, 0.001, None),
    ("spirals_32x8",   "spirals", 32,  8, 0.001, None),
    ("spirals_32x4",   "spirals", 32,  4, 0.001, None),
    ("spirals_8x12",   "spirals",  8, 12, 0.001, None),
    ("spirals_8x4",    "spirals",  8,  4, 0.001, None),
    ("moons_32x12",    "moons",   32, 12, 0.001, None),
    ("moons_32x4",     "moons",   32,  4, 0.001, None),
    ("moons_8x4",      "moons",    8,  4, 0.001, None),
    ("spirals_100d_32x12", "spirals", 32, 12, 0.001, 100),
]


def main():
    print(f"Correction distillation sweep: {BITS}-bit, delta={DELTA}")
    print(f"Configs: {len(CONFIGS)}")
    print(f"Training: P1={PHASE1_EPOCHS}ep, P2={PHASE2_EPOCHS}ep")
    print("=" * 90)

    results = []
    for cfg_name, dataset, hdim, depth, lr, embed_dim in CONFIGS:
        t0 = time.time()
        print(f"\n--- {cfg_name} ({dataset}, {hdim}x{depth}"
              f"{f', embed={embed_dim}' if embed_dim else ''}) ---")
        r = run_config(cfg_name, dataset, hdim, depth, lr=lr,
                       embed_dim=embed_dim)
        dt = time.time() - t0
        results.append(r)

        if r["skipped"]:
            continue

        print(f"  Float: {r['accuracy_float']:.1%}  Quant: {r['accuracy_quant']:.1%}"
              f"  Direct c_local: {r['accuracy_direct_local']:.1%}  ({dt:.1f}s)")

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'Config':<22s}  {'Float':>6s}  {'Quant':>6s}  {'Direct':>6s}  "
          f"{'Embed':>6s}  {'L+Skip':>6s}  {'C+Skip':>6s}  "
          f"{'TaskP':>6s}  {'CorrP':>6s}  {'%Task':>5s}")
    print("-" * 100)
    for r in results:
        if r["skipped"]:
            print(f"{r['name']:<22s}  {r['accuracy']:.1%}  {'SKIP':>6s}")
            continue
        v = r["variants"]
        best_key = max(v, key=lambda k: v[k]["accuracy"])
        best = v[best_key]
        print(f"{r['name']:<22s}  {r['accuracy_float']:5.1%}  "
              f"{r['accuracy_quant']:5.1%}  "
              f"{r['accuracy_direct_local']:5.1%}  "
              f"{v.get('embedding', {}).get('accuracy', 0):5.1%}  "
              f"{v.get('local+skip', {}).get('accuracy', 0):5.1%}  "
              f"{v.get('combined+skip', {}).get('accuracy', 0):5.1%}  "
              f"{r['task_params']:>6d}  "
              f"{best['params']:>6d}  "
              f"{best['params_pct']:4.1f}%")

    # Float ablation summary
    print(f"\n{'Config':<22s}  {'Float':>6s}  {'Embed/F':>7s}  {'L+S/F':>6s}  {'C+S/F':>6s}")
    print("-" * 55)
    for r in results:
        if r["skipped"]:
            continue
        v = r["variants"]
        print(f"{r['name']:<22s}  {r['accuracy_float']:5.1%}  "
              f"{v.get('embedding', {}).get('float_ablation', 0):5.1%}  "
              f"{v.get('local+skip', {}).get('float_ablation', 0):5.1%}  "
              f"{v.get('combined+skip', {}).get('float_ablation', 0):5.1%}")

    # Save
    out_path = "docs/correction_distillation_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
