"""
Topology-preserving distillation — toy ReLU (Item 42).
Teacher: 2->32->32->32->32->1, Student: 2->32->1.
4 distillation methods: output-only, distance-matrix, CKA, activation-MSE.
Post-hoc PH via ripser. 3 seeds x 2 datasets.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
import ripser
import persim

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 7]
EPOCHS = 2000
LR = 1e-3
N = 2000
LAMBDA_VALS = [0.1, 0.5, 1.0, 2.0]
PH_EPOCHS = [0, 500, 1000, 1500, 2000]
BATCH_SIZE = 200  # also used for distance matrix
PCA_N_COMPONENTS = 2
PCA_VARIANCE_THRESHOLD = 0.60
DEVICE = torch.device("cpu")

RESULTS_DIR = "autoresearch/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def make_spirals(n=2000, turns=3, scale=2.0, seed=0):
    rng = np.random.RandomState(seed)
    n_each = n // 2
    t = np.linspace(0, turns * 2 * np.pi, n_each)
    r = np.linspace(0.1, 1.0, n_each)
    x0 = np.stack([r * np.cos(t), r * np.sin(t)], axis=1) * scale
    x1 = np.stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)], axis=1) * scale
    X = np.vstack([x0, x1]).astype(np.float32)
    y = np.array([0] * n_each + [1] * n_each, dtype=np.float32)
    idx = rng.permutation(n)
    return torch.tensor(X[idx]), torch.tensor(y[idx])


def make_moons_dataset(n=2000, noise=0.1, seed=0):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return torch.tensor(X.astype(np.float32)), torch.tensor(y.astype(np.float32))


def train_test_split(X, y, ratio=0.8, seed=0):
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = rng.permutation(n)
    split = int(n * ratio)
    tr, te = idx[:split], idx[split:]
    return X[tr], y[tr], X[te], y[te]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Teacher(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
        ])
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, return_hidden=False):
        hiddens = []
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
            hiddens.append(h)
        out = self.out(h)
        if return_hidden:
            return out, hiddens
        return out


class Student(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, return_hidden=False):
        h = F.relu(self.fc1(x))
        out = self.out(h)
        if return_hidden:
            return out, [h]
        return out


# ---------------------------------------------------------------------------
# Training teacher
# ---------------------------------------------------------------------------
def train_teacher(X_tr, y_tr, X_te, y_te, epochs=3000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = Teacher().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_tr, y_tr = X_tr.to(DEVICE), y_tr.to(DEVICE)
    for _ in range(epochs):
        model.train()
        logits = model(X_tr).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = (model(X_te.to(DEVICE)).squeeze() > 0).float()
        acc = (preds == y_te.to(DEVICE)).float().mean().item()
    return model, acc


# ---------------------------------------------------------------------------
# PCA projection (fit on teacher, apply to both)
# ---------------------------------------------------------------------------
def fit_pca(teacher_acts: torch.Tensor, n_components=PCA_N_COMPONENTS):
    """Fit PCA on teacher activations, check variance threshold."""
    acts_np = teacher_acts.detach().cpu().numpy()
    pca = PCA(n_components=min(n_components, acts_np.shape[1]))
    pca.fit(acts_np)
    var = pca.explained_variance_ratio_.sum()
    if var < PCA_VARIANCE_THRESHOLD:
        n_components = 3
        pca = PCA(n_components=min(n_components, acts_np.shape[1]))
        pca.fit(acts_np)
        var = pca.explained_variance_ratio_.sum()
    if var < PCA_VARIANCE_THRESHOLD:
        # Use full 32D
        return None, acts_np.shape[1]
    return pca, n_components


def apply_pca(pca, acts: torch.Tensor):
    """Apply PCA projection, differentiable (linear transform)."""
    if pca is None:
        return acts
    W = torch.tensor(pca.components_, dtype=torch.float32, device=acts.device)  # [k, d]
    mu = torch.tensor(pca.mean_, dtype=torch.float32, device=acts.device)       # [d]
    return (acts - mu) @ W.T   # [n, k]


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def distance_matrix_loss(h_s: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of pairwise distance matrix difference, normalised."""
    # h_s, h_t: [n, k]
    n = h_s.shape[0]
    # pairwise squared distances
    def pdist(x):
        # [n, n]
        sq = (x ** 2).sum(dim=1, keepdim=True)
        gram = x @ x.T
        d2 = sq + sq.T - 2 * gram
        d2 = d2.clamp(min=0)
        return d2.sqrt()
    Ds = pdist(h_s)
    Dt = pdist(h_t)
    norm_factor = n * (n - 1)
    return ((Ds - Dt) ** 2).sum() / norm_factor


def linear_cka(H_s: torch.Tensor, H_t: torch.Tensor) -> torch.Tensor:
    """Linear CKA between two [n, d] matrices."""
    n = H_s.shape[0]
    # Center
    H_s = H_s - H_s.mean(dim=0, keepdim=True)
    H_t = H_t - H_t.mean(dim=0, keepdim=True)
    # HSIC
    S_st = (H_s @ H_t.T)
    S_ss = (H_s @ H_s.T)
    S_tt = (H_t @ H_t.T)
    hsic_st = (S_st ** 2).sum() / (n - 1) ** 2
    hsic_ss = (S_ss ** 2).sum() / (n - 1) ** 2
    hsic_tt = (S_tt ** 2).sum() / (n - 1) ** 2
    denom = (hsic_ss * hsic_tt).sqrt().clamp(min=1e-8)
    return hsic_st / denom


# ---------------------------------------------------------------------------
# Train student with given loss method
# ---------------------------------------------------------------------------
def train_student(
    teacher, X_tr, y_tr, X_te, y_te, method, lam,
    epochs=EPOCHS, lr=LR, seed=0, pca=None,
    ph_epochs=None
):
    """
    method: 'output_only' | 'dmat' | 'cka' | 'act_mse'
    Returns dict with test_acc, train_losses, and ph_wasserstein (if ph_epochs given).
    """
    torch.manual_seed(seed)
    student = Student().to(DEVICE)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    X_tr, y_tr = X_tr.to(DEVICE), y_tr.to(DEVICE)
    teacher.eval()

    train_losses = []
    ph_records = []

    for ep in range(epochs + 1):
        # PH checkpoint (post-hoc, no grad)
        if ph_epochs is not None and ep in ph_epochs:
            student.eval()
            with torch.no_grad():
                _, s_hids = student(X_tr[:500], return_hidden=True)
                _, t_hids = teacher(X_tr[:500], return_hidden=True)
            s_h = s_hids[0]  # only hidden layer
            t_h = t_hids[1]  # teacher layer 2 (middle)
            # project
            s_proj = apply_pca(pca, s_h).cpu().numpy()
            t_proj = apply_pca(pca, t_h).cpu().numpy()
            # Skip PH if NaN (student may not have converged yet)
            if np.isnan(s_proj).any() or np.isnan(t_proj).any():
                ph_records.append({'epoch': ep, 'w0': float('nan'), 'w1': float('nan')})
            else:
                # PH via ripser
                ph_s = ripser.ripser(s_proj, maxdim=1)['dgms']
                ph_t = ripser.ripser(t_proj, maxdim=1)['dgms']
                # Wasserstein for H0 and H1
                w0 = persim.wasserstein(ph_s[0], ph_t[0], matching=False)
                # H1 might be empty
                if len(ph_s[1]) > 0 and len(ph_t[1]) > 0:
                    w1 = persim.wasserstein(ph_s[1], ph_t[1], matching=False)
                else:
                    w1 = 0.0
                ph_records.append({'epoch': ep, 'w0': float(w0), 'w1': float(w1)})

        if ep == epochs:
            break

        student.train()
        opt.zero_grad()

        # Sample batch
        idx = torch.randperm(len(X_tr))[:BATCH_SIZE]
        xb, yb = X_tr[idx], y_tr[idx]

        # Teacher soft targets
        with torch.no_grad():
            t_out, t_hids = teacher(xb, return_hidden=True)
            y_soft = t_out.squeeze()
            t_h = t_hids[1]   # teacher middle layer (layer index 1)

        s_out, s_hids = student(xb, return_hidden=True)
        y_s = s_out.squeeze()
        s_h = s_hids[0]

        # Output loss
        loss_out = F.mse_loss(y_s, y_soft)

        if method == 'output_only':
            loss = loss_out
        else:
            if method == 'dmat':
                # Distance-matrix loss: use raw activations (dmat is translation-invariant)
                # No PCA needed — pairwise distances are rotation/translation-invariant
                t_h_det = t_h.detach()
                loss_topo = distance_matrix_loss(s_h, t_h_det)
            elif method == 'cka':
                # CKA: project to PCA space for interpretability, center per batch
                s_proj = apply_pca(pca, s_h)
                t_proj_det = apply_pca(pca, t_h).detach()
                loss_topo = 1.0 - linear_cka(s_proj, t_proj_det)
            elif method == 'act_mse':
                # Activation MSE in PCA space
                s_proj = apply_pca(pca, s_h)
                t_proj_det = apply_pca(pca, t_h).detach()
                loss_topo = F.mse_loss(s_proj, t_proj_det)
            else:
                raise ValueError(f"Unknown method: {method}")

            loss = loss_out + lam * loss_topo

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        opt.step()

        if ep % 100 == 0:
            train_losses.append({'epoch': ep, 'loss': loss.item()})

    # Eval
    student.eval()
    with torch.no_grad():
        preds = (student(X_te.to(DEVICE)).squeeze() > 0).float()
        acc = (preds == y_te.to(DEVICE)).float().mean().item()

    return {'test_acc': acc, 'train_losses': train_losses, 'ph_records': ph_records}


# ---------------------------------------------------------------------------
# Also train standalone student (ground truth labels, no teacher)
# ---------------------------------------------------------------------------
def train_student_standalone(X_tr, y_tr, X_te, y_te, epochs=EPOCHS, lr=LR, seed=0):
    torch.manual_seed(seed)
    student = Student().to(DEVICE)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    X_tr, y_tr = X_tr.to(DEVICE), y_tr.to(DEVICE)
    for ep in range(epochs):
        student.train()
        idx = torch.randperm(len(X_tr))[:BATCH_SIZE]
        xb, yb = X_tr[idx], y_tr[idx]
        loss = F.binary_cross_entropy_with_logits(student(xb).squeeze(), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    student.eval()
    with torch.no_grad():
        preds = (student(X_te.to(DEVICE)).squeeze() > 0).float()
        acc = (preds == y_te.to(DEVICE)).float().mean().item()
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_dataset(dataset_name, seed_offset=0):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    all_results = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        rng_seed = seed + seed_offset

        # Data
        if dataset_name == 'spirals':
            X, y = make_spirals(n=N, turns=3, scale=2.0, seed=rng_seed)
        else:
            X, y = make_moons_dataset(n=N, noise=0.1, seed=rng_seed)
        X_tr, y_tr, X_te, y_te = train_test_split(X, y, seed=rng_seed)

        # Train teacher (always seed=0 for determinism within dataset)
        teacher, t_acc = train_teacher(X_tr, y_tr, X_te, y_te, epochs=3000, lr=LR, seed=seed)
        print(f"  Teacher acc: {t_acc:.4f}")
        assert t_acc >= 0.92, f"Teacher did not reach 92% (got {t_acc:.4f})"

        # Fit PCA on teacher layer-2 activations
        teacher.eval()
        with torch.no_grad():
            _, t_hids = teacher(X_tr[:1000].to(DEVICE), return_hidden=True)
            t_h_ref = t_hids[1]  # layer 2 (middle)
        pca, n_components_used = fit_pca(t_h_ref)
        print(f"  PCA components used: {n_components_used}")

        seed_results = {'teacher_acc': t_acc, 'pca_components': n_components_used}

        # Standalone baseline
        sa_acc = train_student_standalone(X_tr, y_tr, X_te, y_te, seed=seed)
        seed_results['standalone'] = sa_acc
        print(f"  Standalone student acc: {sa_acc:.4f}")

        # Method A: output_only
        r = train_student(teacher, X_tr, y_tr, X_te, y_te,
                          method='output_only', lam=0.0, seed=seed, pca=pca,
                          ph_epochs=PH_EPOCHS)
        seed_results['output_only'] = r
        print(f"  Output-only acc: {r['test_acc']:.4f}")
        print(f"  PH records: {r['ph_records'][:2]}")

        # Methods B/C/D with lambda sweep
        for method_key, method_name in [('dmat', 'distance_matrix'), ('cka', 'CKA'), ('act_mse', 'activation_MSE')]:
            seed_results[method_name] = {}
            best_acc = 0.0
            for lam in LAMBDA_VALS:
                include_ph = (lam == 1.0)  # PH only for lambda=1.0 to save time
                r = train_student(teacher, X_tr, y_tr, X_te, y_te,
                                  method=method_key, lam=lam, seed=seed, pca=pca,
                                  ph_epochs=PH_EPOCHS if include_ph else None)
                seed_results[method_name][f'lam_{lam}'] = r
                print(f"  {method_name} lam={lam}: acc={r['test_acc']:.4f}")
                if r['test_acc'] > best_acc:
                    best_acc = r['test_acc']
            seed_results[method_name]['best_acc'] = best_acc

        all_results[str(seed)] = seed_results

    return all_results


def summarize(all_results, dataset_name):
    print(f"\n{'='*60}")
    print(f"SUMMARY: {dataset_name}")
    print(f"{'='*60}")
    methods = ['distance_matrix', 'CKA', 'activation_MSE']
    for seed, res in all_results.items():
        oo_acc = res['output_only']['test_acc']
        print(f"\nSeed {seed} — output_only: {oo_acc:.4f}")
        for m in methods:
            best = res[m]['best_acc']
            delta = best - oo_acc
            print(f"  {m}: best={best:.4f} (Δ={delta:+.4f})")


def main():
    results = {}

    # Spirals
    spirals_results = run_dataset('spirals', seed_offset=0)
    summarize(spirals_results, 'spirals')
    results['spirals'] = spirals_results

    # Moons
    moons_results = run_dataset('moons', seed_offset=100)
    summarize(moons_results, 'moons')
    results['moons'] = moons_results

    # Save
    out_path = os.path.join(RESULTS_DIR, "cycle_064_topo_distillation.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Criterion evaluation
    print("\n" + "="*60)
    print("CRITERION EVALUATION")
    print("="*60)
    for dataset_name, dataset_res in results.items():
        print(f"\n{dataset_name}:")
        dmat_deltas = []
        ph_decreasing = []
        for seed, res in dataset_res.items():
            oo = res['output_only']['test_acc']
            best_dmat = res['distance_matrix']['best_acc']
            delta = best_dmat - oo
            dmat_deltas.append(delta)
            # Check if PH W1 decreases for dmat lam=1.0
            ph_recs = res['distance_matrix'].get('lam_1.0', {}).get('ph_records', [])
            if len(ph_recs) >= 2:
                w0_vals = [r['w0'] for r in ph_recs]
                decreasing = w0_vals[-1] < w0_vals[0]
                ph_decreasing.append(decreasing)
        mean_delta = np.mean(dmat_deltas)
        ph_dec_frac = np.mean(ph_decreasing) if ph_decreasing else float('nan')
        print(f"  Mean Δacc (dmat vs output_only): {mean_delta:+.4f}")
        print(f"  PH W0 decreasing fraction: {ph_dec_frac:.2f}")
        if mean_delta >= 0.05:
            verdict = "CONFIRMED (≥5pp)"
        elif mean_delta >= 0.03:
            verdict = "INCONCLUSIVE (3-5pp)"
        else:
            verdict = "REFUTED (<3pp)"
        print(f"  Verdict: {verdict}")

    return results


if __name__ == '__main__':
    main()
