"""Geometric error decomposition sweep across configurations.

Validates Experiment 3 findings (metric vs topological distortion) across:
- Datasets: spirals, moons
- Depths: 4, 8, 12
- Widths: 8, 32
- High-dimensional embedding (100D) for spirals depth-12

For each config, reports:
- metric_frac: fraction of ||error||^2 from metric distortion (linear ceiling)
- topo_frac: fraction from topological distortion (hyperplane crossings)
- rank_95: SVD rank for 95% metric recovery
- disagree%: neuron disagreement rate
- var_ratio: variance carried by flipped neurons vs all neurons

Usage: python experiments/geometric_decomposition_sweep.py
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
from aleph.qgeom import CanonicalSpaceTracker, ForwardTrace, ReLUDisagreementTracker

BITS = 4
DELTA = 1.0 / (2 ** (BITS - 1))
SEED = 42
EPOCHS = 5000
MIN_ACCURACY = 0.85


def make_mlp(hidden_dim, depth, input_dim=2, output_dim=1):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


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
    """Compute classification accuracy from logits."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == y).float().mean().item()


@torch.no_grad()
def metric_corrected_forward(x, weights, biases, weights_q, float_trace):
    """Forward pass that removes metric error at each hidden layer.

    At each hidden layer, we subtract the metric component of the post-ReLU
    error (eps on neurons where ReLU agrees). This simulates a perfect linear
    correction that fixes all metric distortion but leaves topological
    distortion (hyperplane crossings) untouched.
    """
    a = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    for i, (Wq, b) in enumerate(zip(weights_q, biases)):
        z = F.linear(a, Wq, b)
        is_last = (i == len(weights_q) - 1)
        if is_last:
            a = z
        else:
            a_q = F.relu(z)
            # Compute metric error: post-ReLU error on agreeing neurons
            z_float = float_trace.pre_acts[i]
            a_float = float_trace.post_acts[i]
            agree = (z_float > 0) == (z > 0)  # same half-space
            eps_metric = (a_q - a_float) * agree.float()
            # Subtract metric error — leaves only topological error
            a = a_q - eps_metric
    return a


@torch.no_grad()
def geometric_decomposition(float_trace, quant_trace, weights, weights_q):
    """Per-layer metric vs topological decomposition for hidden layers."""
    n_hidden = len(weights) - 1
    results = []
    for i in range(n_hidden):
        z_float = float_trace.pre_acts[i]
        z_quant = quant_trace.pre_acts[i]
        a_float = float_trace.post_acts[i]
        a_quant = quant_trace.post_acts[i]

        eps_post = a_quant - a_float
        agree = (z_float > 0) == (z_quant > 0)
        disagree = ~agree

        eps_metric = eps_post * agree.float()
        eps_topo = eps_post * disagree.float()

        norm_total = torch.linalg.norm(eps_post, dim=1)
        norm_metric = torch.linalg.norm(eps_metric, dim=1)
        norm_topo = torch.linalg.norm(eps_topo, dim=1)

        total_sq = (norm_total ** 2).mean()
        metric_sq = (norm_metric ** 2).mean()
        metric_frac = (metric_sq / total_sq).item() if total_sq > 0 else 0.0
        topo_frac = 1.0 - metric_frac

        disagree_frac = disagree.float().mean().item()

        # SVD rank of metric error
        U, S, Vh = torch.linalg.svd(eps_metric, full_matrices=False)
        cumvar = torch.cumsum(S ** 2, dim=0)
        total_var = cumvar[-1]
        if total_var > 0:
            explained = cumvar / total_var
            rank_95 = int((explained < 0.95).sum().item()) + 1
        else:
            rank_95 = 0

        # Variance ratio of flipped neurons
        float_var = a_float.var(dim=0)
        disagree_ever = disagree.any(dim=0)
        if disagree_ever.any():
            var_ratio = (float_var[disagree_ever].mean() / float_var.mean()).item() \
                if float_var.mean() > 0 else 0.0
        else:
            var_ratio = 0.0

        E = weights_q[i] - weights[i]
        results.append({
            "layer": i,
            "metric_frac": round(metric_frac, 4),
            "topo_frac": round(topo_frac, 4),
            "disagree_frac": round(disagree_frac, 4),
            "rank_95": rank_95,
            "var_ratio": round(var_ratio, 3),
            "norm_total_mean": round(norm_total.mean().item(), 4),
            "E_spectral_norm": round(torch.linalg.norm(E, ord=2).item(), 4),
        })
    return results


@torch.no_grad()
def oracle_corrections_and_svd(x, weights, biases, weights_q, float_trace):
    """Compute oracle corrections and their SVD at each hidden layer."""
    a = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    epsilon = torch.zeros_like(a)
    bases = []
    for i in range(len(weights)):
        is_last = (i == len(weights) - 1)
        E = weights_q[i] - weights[i]
        W, Wq, b = weights[i], weights_q[i], biases[i]
        C = -F.linear(a, E) - F.linear(epsilon, W)
        z = F.linear(a, Wq, b) + C
        a_new = z if is_last else F.relu(z)
        epsilon = a_new - float_trace.post_acts[i]
        if not is_last:
            U, S, Vh = torch.linalg.svd(C, full_matrices=False)
            bases.append((U, S, Vh))
        a = a_new
    return bases


@torch.no_grad()
def forward_rank_k(x, weights_q, biases, bases, ranks):
    """Forward pass applying rank-k oracle correction (pre-ReLU) at each hidden layer."""
    a = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    for i, (Wq, b) in enumerate(zip(weights_q, biases)):
        z = F.linear(a, Wq, b)
        is_last = (i == len(weights_q) - 1)
        if not is_last:
            U, S, Vh = bases[i]
            k = ranks[i]
            if k > 0 and k <= len(S):
                z = z + (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]
        a = z if is_last else F.relu(z)
    return a


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
        print(f"  SKIP {name}: accuracy {acc:.1%} < {MIN_ACCURACY:.0%} (undertrained)")
        return {"name": name, "skipped": True, "accuracy": round(acc, 4)}

    weights, biases, weights_q = extract_weights(model)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    float_trace = forward_pass(X_t, weights, biases)
    quant_trace = forward_pass(X_t, weights_q, biases)

    # Task performance: float, quantized, metric-corrected
    acc_quant = accuracy(quant_trace.post_acts[-1], y_t)
    metric_corr_logits = metric_corrected_forward(X_t, weights, biases, weights_q,
                                                   float_trace)
    acc_metric_corr = accuracy(metric_corr_logits, y_t)

    # Rank-k correction: oracle SVD + accuracy sweep
    bases = oracle_corrections_and_svd(X_t, weights, biases, weights_q, float_trace)
    n_hidden = len(bases)
    rank_accs = {}
    for k in [1, 2, 3, 5, hidden_dim]:
        k_use = min(k, hidden_dim)
        logits = forward_rank_k(X_t, weights_q, biases, bases, [k_use] * n_hidden)
        rank_accs[k_use] = round(accuracy(logits, y_t), 4)

    # Metric-predicted rank strategy (from Experiment 3)
    per_layer = geometric_decomposition(float_trace, quant_trace, weights, weights_q)
    predicted_ranks = [d['rank_95'] for d in per_layer]
    logits_pred = forward_rank_k(X_t, weights_q, biases, bases, predicted_ranks)
    acc_predicted = round(accuracy(logits_pred, y_t), 4)

    # Aggregate: mean across layers
    mean_metric = sum(d['metric_frac'] for d in per_layer) / len(per_layer)
    mean_topo = sum(d['topo_frac'] for d in per_layer) / len(per_layer)
    mean_rank = sum(d['rank_95'] for d in per_layer) / len(per_layer)
    mean_disagree = sum(d['disagree_frac'] for d in per_layer) / len(per_layer)
    mean_var_ratio = sum(d['var_ratio'] for d in per_layer) / len(per_layer)
    max_disagree = max(d['disagree_frac'] for d in per_layer)

    # First-layer and last-hidden-layer details
    first = per_layer[0]
    last = per_layer[-1]

    return {
        "name": name,
        "skipped": False,
        "accuracy": round(acc, 4),
        "accuracy_quant": round(acc_quant, 4),
        "accuracy_metric_corrected": round(acc_metric_corr, 4),
        "accuracy_rank_k": rank_accs,
        "accuracy_metric_predicted": acc_predicted,
        "predicted_ranks": predicted_ranks,
        "dataset": dataset,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "input_dim": actual_input_dim,
        "n_hidden_layers": len(per_layer),
        "mean_metric_frac": round(mean_metric, 4),
        "mean_topo_frac": round(mean_topo, 4),
        "mean_rank_95": round(mean_rank, 1),
        "mean_disagree": round(mean_disagree, 4),
        "max_disagree": round(max_disagree, 4),
        "mean_var_ratio": round(mean_var_ratio, 3),
        "first_layer": first,
        "last_hidden_layer": last,
        "per_layer": per_layer,
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
    print(f"Geometric decomposition sweep: {BITS}-bit, delta={DELTA}")
    print(f"Configs: {len(CONFIGS)}")
    print("=" * 90)

    results = []
    for name, dataset, hdim, depth, lr, embed_dim in CONFIGS:
        t0 = time.time()
        print(f"\n--- {name} ({dataset}, {hdim}×{depth}"
              f"{f', embed={embed_dim}' if embed_dim else ''}) ---")
        r = run_config(name, dataset, hdim, depth, lr=lr, embed_dim=embed_dim)
        dt = time.time() - t0
        results.append(r)

        if r["skipped"]:
            continue

        rk = r['accuracy_rank_k']
        print(f"  Float: {r['accuracy']:.1%}  Quant: {r['accuracy_quant']:.1%}  "
              f"Metric-corr: {r['accuracy_metric_corrected']:.1%}  ({dt:.1f}s)")
        rank_str = "  ".join(f"r{k}={v:.1%}" for k, v in sorted(rk.items()))
        print(f"  Rank-k: {rank_str}")
        print(f"  Predicted-rank: {r['accuracy_metric_predicted']:.1%}  "
              f"ranks={r['predicted_ranks']}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Config':<22s}  {'Float':>6s}  {'Quant':>6s}  {'MCorr':>6s}  "
          f"{'Rk1':>5s}  {'Rk3':>5s}  {'Rk5':>5s}  {'Pred':>5s}  {'Full':>5s}")
    print("-" * 80)
    for r in results:
        if r["skipped"]:
            print(f"{r['name']:<22s}  {r['accuracy']:5.1%}  {'SKIP':>6s}")
            continue
        rk = r['accuracy_rank_k']
        hdim = r['hidden_dim']
        print(f"{r['name']:<22s}  {r['accuracy']:5.1%}  "
              f"{r['accuracy_quant']:5.1%}  "
              f"{r['accuracy_metric_corrected']:5.1%}  "
              f"{rk.get(1, 0):4.1%}  "
              f"{rk.get(3, 0):4.1%}  "
              f"{rk.get(5, rk.get(hdim, 0)):4.1%}  "
              f"{r['accuracy_metric_predicted']:4.1%}  "
              f"{rk.get(hdim, 0):4.1%}")

    # Save
    out_path = "docs/geometric_decomposition_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
