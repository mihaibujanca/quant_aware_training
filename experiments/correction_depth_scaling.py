"""Correction distillation: depth scaling and correction network capacity.

Studies how correction quality degrades with task network depth and whether
a larger correction network compensates.

Fixed: spirals dataset, width 32, 4-bit quantization.
Varies:
- Task network depth: 8, 12, 16, 20, 24
- Correction network hidden size: 32, 64, 128, 256

For each (depth, corr_hidden) pair, trains combined+skip (best variant)
and reports accuracy. Also runs embedding-only and direct c_local as
reference points.

Usage: python experiments/correction_depth_scaling.py
"""

import json
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from aleph.datasets import make_spirals
from aleph.qgeom import ForwardTrace

BITS = 4
DELTA = 1.0 / (2 ** (BITS - 1))
SEED = 42
HIDDEN_DIM = 32
EMBED_DIM = 8
MIN_ACCURACY = 0.85

# Training epochs scale with depth to ensure convergence
BASE_EPOCHS = 5000
PHASE1_EPOCHS = 1000
PHASE2_EPOCHS = 500
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4


def make_mlp(hidden_dim, depth, input_dim=2, output_dim=1):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def count_params(hidden_dim, depth, input_dim=2, output_dim=1):
    return (hidden_dim * input_dim + hidden_dim +
            (depth - 1) * (hidden_dim ** 2 + hidden_dim) +
            output_dim * hidden_dim + output_dim)


def train_model(model, X_t, y_t, lr, epochs):
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
    a = x
    pre_acts, post_acts = [], []
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = F.linear(a, W, b)
        a = z if i == len(weights) - 1 else F.relu(z)
        pre_acts.append(z)
        post_acts.append(a)
    return ForwardTrace(pre_acts=pre_acts, post_acts=post_acts)


@torch.no_grad()
def accuracy(logits, y):
    return ((torch.sigmoid(logits) > 0.5).float() == y).float().mean().item()


@torch.no_grad()
def direct_local_forward(x, weights_q, biases, error_matrices):
    a = x
    for L in range(len(weights_q)):
        z = F.linear(a, weights_q[L], biases[L])
        is_last = (L == len(weights_q) - 1)
        if not is_last:
            z = z - F.linear(a, error_matrices[L])
        a = z if is_last else F.relu(z)
    return a


class CorrectionNet(nn.Module):
    def __init__(self, hidden_dim, n_layers, mode='combined',
                 net_hidden=64, embed_dim=EMBED_DIM, skip_local=False):
        super().__init__()
        self.mode = mode
        self.skip_local = skip_local
        in_dim = hidden_dim
        if mode in ('embedding', 'combined'):
            self.layer_embed = nn.Embedding(n_layers, embed_dim)
            in_dim += embed_dim
        if mode in ('local', 'combined'):
            in_dim += hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, net_hidden),
            nn.ReLU(),
            nn.Linear(net_hidden, hidden_dim),
        )

    def forward(self, z, layer_idx, c_local=None):
        parts = [z]
        if self.mode in ('local', 'combined'):
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
        a = z if is_last else F.relu(z)
    return a, corrections


@torch.no_grad()
def compute_oracle_corrections(x, weights, biases, weights_q, float_trace):
    a = x
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
    # Teacher-forced inputs
    z_tf, c_local_tf = [], []
    for L in range(n_hidden):
        a_prev = X_t if L == 0 else float_trace.post_acts[L - 1]
        z_tf.append(F.linear(a_prev, weights_q[L], biases[L]).detach())
        c_local_tf.append((-F.linear(a_prev, error_matrices[L])).detach())

    # Phase 1
    optimizer = torch.optim.Adam(net.parameters(), lr=PHASE1_LR)
    for _ in range(PHASE1_EPOCHS):
        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for L in range(n_hidden):
            C_pred = net(z_tf[L], L, c_local=c_local_tf[L])
            loss = loss + F.mse_loss(C_pred, oracle_targets[L])
        (loss / n_hidden).backward()
        optimizer.step()

    # Phase 2
    float_logits = float_trace.post_acts[-1].detach()
    optimizer = torch.optim.Adam(net.parameters(), lr=PHASE2_LR)
    for _ in range(PHASE2_EPOCHS):
        optimizer.zero_grad()
        logits, _ = corrected_forward(X_t, weights_q, biases,
                                      error_matrices, net)
        F.mse_loss(logits, float_logits).backward()
        optimizer.step()

    return net


def run_depth(depth, corr_hiddens, X_t, y_t):
    """Train a task network at given depth, then test correction variants."""
    torch.manual_seed(SEED)
    epochs = BASE_EPOCHS + (depth - 8) * 500  # more epochs for deeper
    lr = 0.001

    model = make_mlp(HIDDEN_DIM, depth)
    model, acc = train_model(model, X_t, y_t, lr=lr, epochs=epochs)
    task_params = count_params(HIDDEN_DIM, depth)

    if acc < MIN_ACCURACY:
        print(f"  depth={depth}: accuracy {acc:.1%} — SKIP (undertrained)")
        return {"depth": depth, "skipped": True, "accuracy_float": round(acc, 4)}

    weights, biases, weights_q = extract_weights(model)
    error_matrices = [Wq - W for W, Wq in zip(weights, weights_q)]
    n_hidden = len(weights) - 1

    float_trace = forward_pass(X_t, weights, biases)
    quant_trace = forward_pass(X_t, weights_q, biases)

    acc_quant = accuracy(quant_trace.post_acts[-1], y_t)
    logits_direct = direct_local_forward(X_t, weights_q, biases, error_matrices)
    acc_direct = accuracy(logits_direct, y_t)

    oracle_corrections = compute_oracle_corrections(
        X_t, weights, biases, weights_q, float_trace)
    oracle_targets = [C.detach() for C in oracle_corrections]

    print(f"  depth={depth}: float={acc:.1%}, quant={acc_quant:.1%}, "
          f"direct_c_local={acc_direct:.1%}, task_params={task_params}")

    # Embedding-only (single size, for reference)
    torch.manual_seed(SEED)
    net_embed = CorrectionNet(HIDDEN_DIM, n_hidden, mode='embedding',
                              net_hidden=64, skip_local=False)
    net_embed = train_correction(net_embed, X_t, weights_q, biases,
                                 error_matrices, oracle_targets,
                                 float_trace, n_hidden)
    with torch.no_grad():
        logits_e, _ = corrected_forward(X_t, weights_q, biases,
                                        error_matrices, net_embed)
    acc_embed = accuracy(logits_e, y_t)
    embed_params = net_embed.n_params()
    print(f"    embedding (h=64): {acc_embed:.1%} ({embed_params} params, "
          f"{100*embed_params/task_params:.1f}%)")

    # Combined+skip at varying correction hidden sizes
    corr_results = {}
    for ch in corr_hiddens:
        torch.manual_seed(SEED)
        net = CorrectionNet(HIDDEN_DIM, n_hidden, mode='combined',
                            net_hidden=ch, skip_local=True)
        net = train_correction(net, X_t, weights_q, biases, error_matrices,
                               oracle_targets, float_trace, n_hidden)
        with torch.no_grad():
            logits_c, _ = corrected_forward(X_t, weights_q, biases,
                                            error_matrices, net)
        acc_c = accuracy(logits_c, y_t)
        n_p = net.n_params()
        corr_results[ch] = {
            "accuracy": round(acc_c, 4),
            "params": n_p,
            "params_pct_of_task": round(100 * n_p / task_params, 1),
        }
        print(f"    combined+skip (h={ch:>3d}): {acc_c:.1%} "
              f"({n_p} params, {100*n_p/task_params:.1f}%)")

    return {
        "depth": depth,
        "skipped": False,
        "accuracy_float": round(acc, 4),
        "accuracy_quant": round(acc_quant, 4),
        "accuracy_direct_local": round(acc_direct, 4),
        "accuracy_embedding": round(acc_embed, 4),
        "embedding_params": embed_params,
        "task_params": task_params,
        "combined_skip_by_hidden": corr_results,
    }


def main():
    print(f"Correction depth scaling: spirals, width={HIDDEN_DIM}, {BITS}-bit")
    print("=" * 90)

    # Dataset (fixed)
    torch.manual_seed(SEED)
    X, y = make_spirals(n_samples=2000, noise=0.5, n_turns=3, random_state=SEED)
    X *= 2.0
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    depths = [4, 8, 12, 16, 20, 24]
    corr_hiddens = [32, 64, 128, 256]

    results = []
    for depth in depths:
        t0 = time.time()
        print(f"\n--- Depth {depth} ---")
        r = run_depth(depth, corr_hiddens, X_t, y_t)
        dt = time.time() - t0
        r["time_s"] = round(dt, 1)
        results.append(r)
        print(f"  ({dt:.1f}s)")

    # Summary table
    print("\n" + "=" * 100)
    hdr = f"{'Depth':>5s}  {'Float':>6s}  {'Quant':>6s}  {'Direct':>6s}  {'Embed':>6s}"
    for ch in corr_hiddens:
        hdr += f"  {'h='+str(ch):>6s}"
    hdr += f"  {'TaskP':>7s}"
    print(hdr)
    print("-" * 100)

    for r in results:
        if r["skipped"]:
            print(f"{r['depth']:5d}  {r['accuracy_float']:5.1%}  {'SKIP':>6s}")
            continue
        line = (f"{r['depth']:5d}  {r['accuracy_float']:5.1%}  "
                f"{r['accuracy_quant']:5.1%}  "
                f"{r['accuracy_direct_local']:5.1%}  "
                f"{r['accuracy_embedding']:5.1%}")
        for ch in corr_hiddens:
            cr = r['combined_skip_by_hidden'].get(ch, {})
            line += f"  {cr.get('accuracy', 0):5.1%}"
        line += f"  {r['task_params']:>7d}"
        print(line)

    # Param count table
    print(f"\n{'Depth':>5s}", end="")
    for ch in corr_hiddens:
        print(f"  {'h='+str(ch)+' P':>8s}  {'%task':>5s}", end="")
    print()
    print("-" * (5 + 15 * len(corr_hiddens)))
    for r in results:
        if r["skipped"]:
            continue
        print(f"{r['depth']:5d}", end="")
        for ch in corr_hiddens:
            cr = r['combined_skip_by_hidden'].get(ch, {})
            print(f"  {cr.get('params', 0):>8d}  {cr.get('params_pct_of_task', 0):4.1f}%",
                  end="")
        print()

    out_path = "docs/correction_depth_scaling.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
