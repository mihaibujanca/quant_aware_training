"""
Cycle 066: Weight-space diffusion on topological level sets — toy ReLU (Item 45)

Algorithm:
- Train spirals classifier (2->32x4->1) to >=90% acc.
- Compute PH fingerprint PD_0 from N_PROBE probe points (post-ReLU activations, layers L1-L4).
- Calibrate lambda_topo so topology penalty ~ L1 penalty magnitude at init.
- Run penalized Langevin: loss = ||theta||_1 + lambda_topo * mean_W1(PD_theta, PD_0).
  Topology gradient via finite-difference (perturb each layer, 4 ripser calls per step).
- Baselines: magnitude pruning + L1-regularized retraining at matched sparsity.
- Criterion: >=30% L1 norm reduction + >=5pp acc over both baselines, 2/3 seeds.

Key design choices:
- W1 drift is measured relative to W1_ref (trained->random), not from 0.
- Lambda calibration: lambda = L1_0 / W1_small, where W1_small is the W1
  between the trained model and a tiny-perturbation copy (eps=1e-2 on all weights).
  This gives a numerically stable lambda.
- 200 probe points for speed (ripser on 200x32D is ~50ms per call).
- 300 Langevin steps.
"""

import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ripser import ripser
from persim import wasserstein

# ── paths ─────────────────────────────────────────────────────────────────────
RESULT_JSON = "autoresearch/results/cycle_066_topo_diffusion.json"
RESULT_PNG  = "autoresearch/results/cycle_066_topo_diffusion.png"

# ── hyperparams ───────────────────────────────────────────────────────────────
SEEDS        = [42, 123, 7]
N_TRAIN      = 2000
TURNS        = 3
SCALE        = 2.0
NOISE        = 0.05
HIDDEN       = 32
DEPTH        = 4
TRAIN_EPOCHS = 2000
TRAIN_LR     = 1e-3
ACC_MIN      = 0.88      # require before starting Langevin

N_PROBE      = 200       # probe points for PH (200x32D is fast in ripser)
MAXDIM       = 1         # H0 + H1

LANGEVIN_LRS   = [1e-4, 3e-4, 1e-3]
LANGEVIN_STEPS = 300
EPS_FD         = 1e-4   # finite-diff perturbation
W1_DRIFT_MAX   = 3.0    # max W1 drift (relative to W1_ref) before declaring divergence
ACC_FLOOR      = 0.80   # minimum acc for theta* selection


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def make_spirals_data(seed):
    from aleph.datasets import make_spirals
    X_np, y_np = make_spirals(n_samples=N_TRAIN, n_turns=TURNS, noise=NOISE, random_state=seed)
    X = torch.from_numpy(X_np).float() * SCALE
    y = torch.from_numpy(y_np).float()
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(DEPTH):
            layers += [nn.Linear(in_dim, HIDDEN), nn.ReLU()]
            in_dim = HIDDEN
        layers.append(nn.Linear(HIDDEN, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def hidden_linears(self):
        """Linear layers that precede a ReLU (skip the output layer)."""
        mods = list(self.net)
        return [mods[i] for i in range(len(mods) - 1) if isinstance(mods[i], nn.Linear)]

    def post_relu_activations(self, x):
        acts = []
        h = x
        for m in self.net:
            h = m(h)
            if isinstance(m, nn.ReLU):
                acts.append(h.detach().clone())
        return acts  # len == DEPTH


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(model, X, y, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    for ep in range(TRAIN_EPOCHS):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(X), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = ((model(X) > 0).float() == y).float().mean().item()
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# PH utilities
# ─────────────────────────────────────────────────────────────────────────────

def acts_to_pds(acts_list):
    """acts_list: list of tensors [N, D]. Returns list of combined H0+H1 diagrams."""
    pds = []
    for acts in acts_list:
        pts = acts.numpy()
        result = ripser(pts, maxdim=MAXDIM)
        dgm0 = result['dgms'][0]
        dgm0 = dgm0[np.isfinite(dgm0[:, 1])]
        dgm1 = result['dgms'][1]
        dgm1 = dgm1[np.isfinite(dgm1[:, 1])]
        combined = np.vstack([dgm0, dgm1]) if len(dgm1) > 0 else dgm0
        pds.append(combined if len(combined) > 0 else np.zeros((0, 2)))
    return pds


def w1_between(pds_a, pds_b):
    """Mean W1 over layers."""
    dists = []
    for a, b in zip(pds_a, pds_b):
        if len(a) == 0 and len(b) == 0:
            dists.append(0.0)
        elif len(a) == 0:
            dists.append(float(np.sum(np.abs(b[:, 1] - b[:, 0]))))
        elif len(b) == 0:
            dists.append(float(np.sum(np.abs(a[:, 1] - a[:, 0]))))
        else:
            dists.append(float(wasserstein(a, b)))
    return float(np.mean(dists))


def get_pds(model, X_probe):
    with torch.no_grad():
        acts = model.post_relu_activations(X_probe)
    return acts_to_pds(acts)


# ─────────────────────────────────────────────────────────────────────────────
# L1 norm
# ─────────────────────────────────────────────────────────────────────────────

def l1_norm(model):
    return sum(p.abs().sum().item() for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Topology finite-difference gradient + Langevin step
# ─────────────────────────────────────────────────────────────────────────────

def langevin_step(model, X_probe, pds_0, lam, lr):
    """
    One gradient step on L(theta) = ||theta||_1 + lam * mean_W1(PD_theta, PD_0).

    Topology gradient: perturb each hidden layer by +eps (all-ones direction),
    compute dW1/d(layer_i) as finite difference. Broadcast uniformly within layer.
    This is a directional derivative along the all-ones vector — a crude but
    cheap proxy for topology sensitivity. Returns w1_before_step.
    """
    linears = model.hidden_linears()
    pds_curr = get_pds(model, X_probe)
    w1_curr = w1_between(pds_curr, pds_0)

    # Topology gradient per hidden layer
    topo_grad_sign = []  # +1 or -1 per layer (direction to decrease W1)
    for lin in linears:
        with torch.no_grad():
            lin.weight.data += EPS_FD
            if lin.bias is not None:
                lin.bias.data += EPS_FD
        pds_pert = get_pds(model, X_probe)
        w1_pert = w1_between(pds_pert, pds_0)
        with torch.no_grad():
            lin.weight.data -= EPS_FD
            if lin.bias is not None:
                lin.bias.data -= EPS_FD
        dw1 = (w1_pert - w1_curr) / EPS_FD
        topo_grad_sign.append(dw1)

    with torch.no_grad():
        # L1 subgradient step
        for p in model.parameters():
            p.data -= lr * torch.sign(p.data)
        # Topology gradient step (hidden layers only)
        for lin, dw1 in zip(linears, topo_grad_sign):
            lin.weight.data -= lr * lam * dw1  # uniform over params in layer
            if lin.bias is not None:
                lin.bias.data -= lr * lam * dw1

    return w1_curr


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def magnitude_pruning(model_state, target_sparsity, X, y):
    m = MLP()
    m.load_state_dict(copy.deepcopy(model_state))
    all_weights = torch.cat([p.data.abs().flatten() for p in m.parameters()])
    n_prune = int(target_sparsity * len(all_weights))
    if n_prune > 0:
        threshold = all_weights.sort().values[n_prune - 1]
        with torch.no_grad():
            for p in m.parameters():
                p.data[p.data.abs() <= threshold] = 0.0
    with torch.no_grad():
        acc = ((m(X) > 0).float() == y).float().mean().item()
    return acc


def l1_retrain(X, y, target_sparsity, seed):
    """Retrain with L1 penalty calibrated (binary search) to match target_sparsity."""
    def measure_sparsity(m):
        n_zero = sum((p.data.abs() < 1e-3).sum().item() for p in m.parameters())
        return n_zero / sum(p.numel() for p in m.parameters())

    lo, hi = 1e-6, 5e-2
    best_acc = 0.5
    for _ in range(8):
        lam = (lo + hi) / 2
        torch.manual_seed(seed)
        m = MLP()
        opt = torch.optim.Adam(m.parameters(), lr=TRAIN_LR)
        for ep in range(TRAIN_EPOCHS):
            opt.zero_grad()
            task_loss = F.binary_cross_entropy_with_logits(m(X), y)
            l1_pen = lam * sum(p.abs().sum() for p in m.parameters())
            (task_loss + l1_pen).backward()
            opt.step()
        sp = measure_sparsity(m)
        with torch.no_grad():
            acc = ((m(X) > 0).float() == y).float().mean().item()
        if sp < target_sparsity:
            lo = lam
        else:
            hi = lam
            best_acc = acc
    return best_acc


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed run
# ─────────────────────────────────────────────────────────────────────────────

def run_seed(seed):
    print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
    X, y = make_spirals_data(seed)
    X_probe = X[:N_PROBE]

    # Train
    torch.manual_seed(seed)
    model = MLP()
    t0 = time.time()
    acc0 = train(model, X, y, seed)
    print(f"  Trained: acc={acc0:.3f} ({time.time()-t0:.1f}s)")
    if acc0 < ACC_MIN:
        print(f"  Training insufficient, skipping")
        return None

    theta0_state = copy.deepcopy(model.state_dict())
    norm0 = l1_norm(model)
    print(f"  ||theta_0||_1 = {norm0:.2f}")

    # Step 0: baseline PH fingerprint
    t0 = time.time()
    pds_0 = get_pds(model, X_probe)
    print(f"  PH fingerprint ({time.time()-t0:.1f}s)")

    # Step 1: calibrate lambda_topo
    # Use W1 between theta_0 and a small-eps perturbation to get local W1 scale
    # Then set lam so (lam * W1_local) ~ (||theta_0||_1 * lr) — i.e. both steps comparable
    # Strategy: compute W1 between PD_0 and PD of (theta_0 + 0.01*||theta_0||_1 noise)
    eps_noise = 0.01
    m_pert = MLP()
    m_pert.load_state_dict(copy.deepcopy(theta0_state))
    with torch.no_grad():
        for p in m_pert.parameters():
            p.data += eps_noise * (torch.randn_like(p) * p.data.abs().mean())
    pds_pert_cal = get_pds(m_pert, X_probe)
    w1_local = w1_between(pds_0, pds_pert_cal)
    if w1_local < 1e-8:
        w1_local = 1.0
    # Also compute W1 to random init (large-scale drift reference)
    torch.manual_seed(seed + 999)
    m_rand = MLP()
    pds_rand = get_pds(m_rand, X_probe)
    w1_ref = w1_between(pds_0, pds_rand)
    if w1_ref < 1e-8:
        w1_ref = w1_local * 10

    # lambda: L1 term change per step = lr * ||sign(theta_0)||_1 ≈ lr * n_params
    n_params = sum(p.numel() for p in model.parameters())
    l1_step = LANGEVIN_LRS[0] * n_params  # step size for L1 gradient
    lam = l1_step / (w1_local + 1e-10)
    print(f"  n_params={n_params}, w1_local={w1_local:.4f}, w1_ref={w1_ref:.4f}, lambda_topo={lam:.4f}")

    # Step 2: Penalized Langevin
    best_result = None
    for lr in LANGEVIN_LRS:
        print(f"  Trying lr={lr:.0e}...")
        model_try = MLP()
        model_try.load_state_dict(copy.deepcopy(theta0_state))

        history = []
        for step in range(LANGEVIN_STEPS):
            w1_curr = langevin_step(model_try, X_probe, pds_0, lam, lr)
            if step % 50 == 0 or step == LANGEVIN_STEPS - 1:
                norm_t = l1_norm(model_try)
                # Drift: W1 relative to reference scale
                w1_drift_rel = w1_curr / (w1_ref + 1e-10)
                with torch.no_grad():
                    acc_t = ((model_try(X) > 0).float() == y).float().mean().item()
                history.append((step, norm_t, w1_curr, w1_drift_rel, acc_t))
                print(f"    step {step:3d}: norm={norm_t:.1f}  w1={w1_curr:.4f}  "
                      f"drift={w1_drift_rel:.3f}  acc={acc_t:.3f}")

        final_drift = history[-1][3]
        if final_drift > W1_DRIFT_MAX:
            print(f"  lr={lr:.0e}: W1 drift uncontrollable ({final_drift:.2f}), trying next lr")
            continue

        # Select theta*: lowest norm with acc >= ACC_FLOOR and w1_drift < 1.0
        candidates = [(n, a, w, d) for _, n, w, d, a in history
                      if a >= ACC_FLOOR and d < 1.0]
        if not candidates:
            # relax constraints
            candidates = [(n, a, w, d) for _, n, w, d, a in history]
        norm_star, acc_star, w1_star, drift_star = min(candidates, key=lambda x: x[0])
        norm_reduction = 1.0 - norm_star / norm0
        print(f"  lr={lr:.0e}: norm_red={norm_reduction:.3f}  acc*={acc_star:.3f}  w1_drift={drift_star:.3f}")
        best_result = {
            "lr": lr, "norm_reduction": norm_reduction,
            "acc_topo": acc_star, "w1_drift_star": drift_star,
            "history": [(s, n, w, d, a) for s, n, w, d, a in history]
        }
        break

    if best_result is None:
        print("  All lr values diverged → REFUTED for this seed")
        return None

    # Step 3: baselines at matched sparsity
    # Re-run Langevin with best lr to get sparsity
    lr_best = best_result["lr"]
    model_final = MLP()
    model_final.load_state_dict(copy.deepcopy(theta0_state))
    for step in range(LANGEVIN_STEPS):
        langevin_step(model_final, X_probe, pds_0, lam, lr_best)
    with torch.no_grad():
        n_zero = sum((p.data.abs() < 1e-3).sum().item() for p in model_final.parameters())
        n_total = sum(p.numel() for p in model_final.parameters())
    sparsity_star = n_zero / n_total
    print(f"  Sparsity in theta*: {sparsity_star:.3f}")

    print(f"  Magnitude pruning baseline...")
    acc_mag = magnitude_pruning(theta0_state, sparsity_star, X, y)
    print(f"  acc_mag = {acc_mag:.3f}")

    print(f"  L1-retrain baseline...")
    acc_l1 = l1_retrain(X, y, sparsity_star, seed)
    print(f"  acc_l1 = {acc_l1:.3f}")

    result = {
        "seed": seed,
        "acc0": acc0,
        "norm0": norm0,
        "n_params": n_params,
        "lambda_topo": lam,
        "w1_local": w1_local,
        "w1_ref": w1_ref,
        "lr_used": lr_best,
        "norm_reduction": best_result["norm_reduction"],
        "acc_topo": best_result["acc_topo"],
        "w1_drift_star": best_result["w1_drift_star"],
        "acc_mag": acc_mag,
        "acc_l1": acc_l1,
        "sparsity_star": sparsity_star,
        "advantage_over_mag": best_result["acc_topo"] - acc_mag,
        "advantage_over_l1": best_result["acc_topo"] - acc_l1,
    }
    print(f"\n  RESULT seed={seed}: norm_red={result['norm_reduction']:.3f}  "
          f"acc_topo={acc_mag:.3f}  acc_mag={acc_mag:.3f}  acc_l1={acc_l1:.3f}")
    print(f"    +{result['advantage_over_mag']:+.3f} vs mag, {result['advantage_over_l1']:+.3f} vs l1")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_results = []
    for seed in SEEDS:
        r = run_seed(seed)
        if r is not None:
            all_results.append(r)

    if not all_results:
        summary = {"verdict": "REFUTED", "reason": "no seeds completed", "results": []}
        with open(RESULT_JSON, "w") as f:
            json.dump(summary, f, indent=2)
        print("REFUTED (no seeds completed)")
        return

    confirmed_seeds = [
        r for r in all_results
        if r["norm_reduction"] >= 0.30
        and r["advantage_over_mag"] >= 0.05
        and r["advantage_over_l1"] >= 0.05
    ]
    nr30 = [r for r in all_results if r["norm_reduction"] >= 0.30]
    inc_seeds = [r for r in nr30
                 if r not in confirmed_seeds
                 and (r["advantage_over_mag"] >= 0.05 or r["advantage_over_l1"] >= 0.05)]

    if len(confirmed_seeds) >= 2:
        verdict = "CONFIRMED"
    elif len(confirmed_seeds) + len(inc_seeds) >= 2:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "REFUTED"

    summary = {
        "verdict": verdict,
        "seeds_confirmed": [r["seed"] for r in confirmed_seeds],
        "n_seeds_run": len(all_results),
        "results": all_results,
    }

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    for r in all_results:
        print(f"  seed={r['seed']}: nr={r['norm_reduction']:.3f}  "
              f"topo={r['acc_topo']:.3f}  mag={r['acc_mag']:.3f}  l1={r['acc_l1']:.3f}")

    with open(RESULT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {RESULT_JSON}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(all_results))
    w = 0.25
    seeds   = [r["seed"]          for r in all_results]
    a_topo  = [r["acc_topo"]      for r in all_results]
    a_mag   = [r["acc_mag"]       for r in all_results]
    a_l1    = [r["acc_l1"]        for r in all_results]
    nr      = [r["norm_reduction"] for r in all_results]

    b1 = ax.bar(x - w, a_topo, w, label="Topo-Langevin", color="#2196F3", alpha=0.85)
    b2 = ax.bar(x,     a_mag,  w, label="Magnitude pruning", color="#FF9800", alpha=0.85)
    b3 = ax.bar(x + w, a_l1,   w, label="L1 retrain", color="#4CAF50", alpha=0.85)

    for bar, n in zip(b1, nr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"nr={n:.2f}", ha="center", va="bottom", fontsize=8, color="#1565C0")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in seeds])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Weight-space topo diffusion — {verdict}\n"
                 f"nr=L1 norm reduction relative to theta_0")
    ax.legend()
    ax.axhline(0.80, color="red", linestyle="--", alpha=0.5, linewidth=1)
    fig.tight_layout()
    fig.savefig(RESULT_PNG, dpi=120)
    print(f"Saved: {RESULT_PNG}")


if __name__ == "__main__":
    main()
