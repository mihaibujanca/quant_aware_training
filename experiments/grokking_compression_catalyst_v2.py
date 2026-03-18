"""Grokking as compression catalyst — modular arithmetic.

Cycle 021 (redesigned): The original experiment ran on spirals (continuous task
that doesn't grok) and counted raw activation patterns (wrong metric).

This version uses modular arithmetic (a+b) mod p, which exhibits genuine grokking:
memorization → delayed generalization phase transition. We track:
  - Pattern count (number of distinct linear regions)
  - Region-label purity (weighted: do regions align with task structure?)
  - Weight norm (drops during grokking as WD finds simpler solution)
  - Test accuracy (the grokking signal itself)

Compression comparison: distill pre- and post-grokking teacher into smaller student.
Post-grokking function should be more compressible (algebraic structure vs memorization).
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
P              = 97       # prime modulus (standard grokking benchmark)
FRAC_TRAIN     = 0.4      # 40% of p² pairs for training
HIDDEN         = 128      # hidden dim
DEPTH          = 1        # 1 hidden layer (canonical grokking setup)
EPOCHS         = 50_000   # grokking typically happens at ~10K-30K
CKPT_EVERY     = 500
LR             = 1e-3
WD             = 1.0      # high weight decay is key for grokking
BETAS          = (0.9, 0.98)
SEED           = 42
TARGET_HIDDEN  = 32       # student hidden dim (4× compression)
DISTILL_EPOCHS = 5000

RESULTS_DIR = "autoresearch/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Data: modular arithmetic ─────────────────────────────────────────────────

def make_mod_addition_data(p: int, frac_train: float, seed: int):
    """All (a, b) → (a+b) mod p pairs with one-hot encoding."""
    rng = np.random.RandomState(seed)
    # All p² pairs
    a_vals = np.arange(p).repeat(p)          # [0,0,...,0, 1,1,...,1, ...]
    b_vals = np.tile(np.arange(p), p)         # [0,1,...,p-1, 0,1,...,p-1, ...]
    labels = (a_vals + b_vals) % p            # (p²,)

    # One-hot encode: concat(one_hot(a), one_hot(b)) → dim = 2p
    X = np.zeros((p * p, 2 * p), dtype=np.float32)
    X[np.arange(p * p), a_vals] = 1.0
    X[np.arange(p * p), p + b_vals] = 1.0

    # Random train/test split
    n_total = p * p
    n_train = int(n_total * frac_train)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.long)

    return X_t, y_t, train_idx, test_idx


# ── Model ─────────────────────────────────────────────────────────────────────

def make_model(input_dim: int, hidden: int, n_classes: int, depth: int = 1):
    layers: list[nn.Module] = []
    in_d = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(in_d, hidden))
        layers.append(nn.ReLU())
        in_d = hidden
    layers.append(nn.Linear(in_d, n_classes))
    return nn.Sequential(*layers)


# ── Forward with masks ────────────────────────────────────────────────────────

def forward_with_masks(model: nn.Sequential, X: torch.Tensor):
    """Return (logits, list of (N, H) bool masks per ReLU layer)."""
    masks = []
    a = X
    for module in model:
        if isinstance(module, nn.Linear):
            a = module(a)
        elif isinstance(module, nn.ReLU):
            masks.append(a > 0)
            a = F.relu(a)
    return a, masks


# ── Geometric signals ─────────────────────────────────────────────────────────

def compute_pattern_codes(model: nn.Sequential, X: torch.Tensor) -> list[bytes]:
    """Return packed activation pattern codes for each input."""
    with torch.no_grad():
        _, masks = forward_with_masks(model, X)
    flat = torch.cat([m.to(torch.uint8) for m in masks], dim=1).cpu().numpy()
    packed = np.packbits(flat, axis=1)
    return [bytes(row) for row in packed]


def geometric_signals(model: nn.Sequential, X: torch.Tensor, y: torch.Tensor):
    """Return (n_patterns, weighted_purity, pattern_entropy)."""
    codes = compute_pattern_codes(model, X)
    y_np = y.cpu().numpy()

    # Pattern count
    n_patterns = len(set(codes))

    # Region-label purity (weighted by region size):
    # = fraction of points whose label is the majority label in their region
    region_map: dict[bytes, list[int]] = defaultdict(list)
    for code, label in zip(codes, y_np):
        region_map[code].append(label)

    correct = 0
    total = len(codes)
    for region_labels in region_map.values():
        counter = Counter(region_labels)
        correct += counter.most_common(1)[0][1]
    purity = correct / total

    # Pattern entropy
    freq = Counter(codes)
    probs = np.array([v / total for v in freq.values()])
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    return n_patterns, purity, entropy


def weight_norm(model: nn.Sequential) -> float:
    return float(sum(p.norm().item() for p in model.parameters()))


def eval_acc(model: nn.Sequential, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        logits, _ = forward_with_masks(model, X)
        preds = logits.argmax(dim=1)
        return float((preds == y).float().mean())


# ── Data ──────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)

X_all, y_all, train_idx, test_idx = make_mod_addition_data(P, FRAC_TRAIN, SEED)
X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

print(f"Task: (a+b) mod {P}")
print(f"Total pairs: {P*P}, train: {len(train_idx)}, test: {len(test_idx)}")
print(f"Chance accuracy: {1/P:.4f}")
print()


# ── Training ──────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
model = make_model(2 * P, HIDDEN, P, DEPTH)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=BETAS)

print(f"Training: hidden={HIDDEN}, depth={DEPTH}, epochs={EPOCHS}")
print(f"          lr={LR}, wd={WD}, betas={BETAS}")
print()
print(f"{'epoch':>7s} | {'tr_acc':>6s} {'te_acc':>6s} | {'w_norm':>7s} | "
      f"{'n_pat':>6s} {'purity':>7s} {'entropy':>7s}")
print("-" * 72)

ckpt_epochs: list[int] = []
train_accs: list[float] = []
test_accs: list[float] = []
pattern_counts: list[int] = []
purities: list[float] = []
entropies: list[float] = []
weight_norms: list[float] = []
checkpoints: list[dict] = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    logits, _ = forward_with_masks(model, X_train)
    loss = F.cross_entropy(logits, y_train)
    loss.backward()
    optimizer.step()

    if epoch % CKPT_EVERY == 0:
        model.eval()
        tr_acc = eval_acc(model, X_train, y_train)
        te_acc = eval_acc(model, X_test, y_test)
        # Geometric signals on ALL p² inputs (full partition picture)
        n_pat, pur, ent = geometric_signals(model, X_all, y_all)
        wnorm = weight_norm(model)

        ckpt_epochs.append(epoch)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)
        pattern_counts.append(n_pat)
        purities.append(pur)
        entropies.append(ent)
        weight_norms.append(wnorm)
        checkpoints.append({k: v.clone() for k, v in model.state_dict().items()})

        if epoch % (CKPT_EVERY * 10) == 0 or (epoch <= 5000):
            print(f"  {epoch:5d} | {tr_acc:6.3f} {te_acc:6.3f} | {wnorm:7.2f} | "
                  f"{n_pat:6d} {pur:7.3f} {ent:7.2f}")


# ── Detect grokking ──────────────────────────────────────────────────────────

memorized_at = None  # first ckpt where train acc > 99%
grokked_at = None    # first ckpt AFTER memorized_at where test acc > 80%

for i, (tr, te) in enumerate(zip(train_accs, test_accs)):
    if memorized_at is None and tr > 0.99:
        memorized_at = i
    if memorized_at is not None and grokked_at is None and te > 0.80:
        grokked_at = i

grokking_detected = (memorized_at is not None and grokked_at is not None
                     and grokked_at > memorized_at)

print()
if grokking_detected:
    print(f"GROKKING DETECTED:")
    print(f"  Memorized at epoch {ckpt_epochs[memorized_at]} "
          f"(train={train_accs[memorized_at]:.3f}, test={test_accs[memorized_at]:.3f})")
    print(f"  Grokked at epoch {ckpt_epochs[grokked_at]} "
          f"(train={train_accs[grokked_at]:.3f}, test={test_accs[grokked_at]:.3f})")
    print(f"  Delay: {ckpt_epochs[grokked_at] - ckpt_epochs[memorized_at]} epochs")
    # Pattern stats
    pre_patterns = pattern_counts[memorized_at]
    post_patterns = pattern_counts[grokked_at]
    pre_purity = purities[memorized_at]
    post_purity = purities[grokked_at]
    print(f"  Patterns: {pre_patterns} → {post_patterns} "
          f"(Δ={post_patterns - pre_patterns:+d})")
    print(f"  Purity:   {pre_purity:.3f} → {post_purity:.3f} "
          f"(Δ={post_purity - pre_purity:+.3f})")
elif memorized_at is not None:
    print(f"No grokking: memorized at epoch {ckpt_epochs[memorized_at]} "
          f"but test acc never exceeded 80% (max={max(test_accs):.3f})")
    print("  Try increasing EPOCHS or WD.")
else:
    print(f"No memorization: max train acc = {max(train_accs):.3f}")


# ── Compression comparison ────────────────────────────────────────────────────

def compress_and_eval(
    state_dict: dict,
    X_distill: torch.Tensor,
    y_distill: torch.Tensor,
    X_eval: torch.Tensor,
    y_eval: torch.Tensor,
    target_hidden: int,
    epochs: int,
) -> float:
    """Distill teacher into smaller student on training data, eval on test."""
    teacher = make_model(2 * P, HIDDEN, P, DEPTH)
    teacher.load_state_dict(state_dict)
    teacher.eval()

    with torch.no_grad():
        teacher_logits = teacher(X_distill)  # (N_train, P)

    torch.manual_seed(SEED + 1)
    student = make_model(2 * P, target_hidden, P, DEPTH)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)

    for _ in range(epochs):
        student.train()
        opt.zero_grad()
        s_logits, _ = forward_with_masks(student, X_distill)
        # Distill on logits (MSE)
        loss = F.mse_loss(s_logits, teacher_logits)
        loss.backward()
        opt.step()

    student.eval()
    return eval_acc(student, X_eval, y_eval)


if grokking_detected:
    pre_idx = memorized_at
    post_idx = grokked_at
    # Also find a late checkpoint (well after grokking, maximum test acc)
    best_test_idx = max(range(len(test_accs)), key=lambda i: test_accs[i])
else:
    # No grokking: compare first checkpoint vs last
    pre_idx = 0
    post_idx = len(checkpoints) - 1
    best_test_idx = post_idx

print()
print("Compression comparison (distill teacher → student):")

results_compress = {}
for label, idx in [("pre_grokking", pre_idx), ("post_grokking", post_idx),
                   ("best_test", best_test_idx)]:
    # Inline teacher eval
    tmp = make_model(2 * P, HIDDEN, P, DEPTH)
    tmp.load_state_dict(checkpoints[idx])
    tmp.eval()
    teacher_acc = eval_acc(tmp, X_test, y_test)

    student_acc = compress_and_eval(
        checkpoints[idx], X_train, y_train, X_test, y_test,
        TARGET_HIDDEN, DISTILL_EPOCHS)

    results_compress[label] = {
        "epoch": ckpt_epochs[idx],
        "teacher_test_acc": teacher_acc,
        "student_test_acc": student_acc,
        "n_patterns": pattern_counts[idx],
        "purity": purities[idx],
    }
    print(f"  {label:15s} (epoch {ckpt_epochs[idx]:5d}): "
          f"teacher={teacher_acc:.3f}, student={student_acc:.3f}, "
          f"patterns={pattern_counts[idx]}, purity={purities[idx]:.3f}")


# ── Verdict ───────────────────────────────────────────────────────────────────

if grokking_detected:
    acc_gain = (results_compress["post_grokking"]["student_test_acc"]
                - results_compress["pre_grokking"]["student_test_acc"])
    purity_gain = purities[grokked_at] - purities[memorized_at]
    pattern_drop = pattern_counts[memorized_at] - pattern_counts[grokked_at]
else:
    acc_gain = 0.0
    purity_gain = 0.0
    pattern_drop = 0

print()
if not grokking_detected:
    verdict = "inconclusive"
    print("Verdict: inconclusive (no grokking observed — increase EPOCHS or WD)")
elif acc_gain >= 0.10:
    verdict = "confirmed"
    print(f"Verdict: confirmed (grokking improves compression by {acc_gain:+.3f})")
elif acc_gain < 0.02:
    verdict = "refuted"
    print(f"Verdict: refuted (grokking detected but compression gain {acc_gain:+.3f} < 0.02)")
else:
    verdict = "inconclusive"
    print(f"Verdict: inconclusive (grokking detected, compression gain {acc_gain:+.3f})")

print(f"  Purity change:  {purity_gain:+.3f}")
print(f"  Pattern change: {pattern_drop:+d}")


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f"Grokking as Compression Catalyst — (a+b) mod {P}\n"
             f"hidden={HIDDEN}, depth={DEPTH}, WD={WD}, frac_train={FRAC_TRAIN}",
             fontsize=12)

# Panel 1: Train/Test accuracy
ax = axes[0, 0]
ax.plot(ckpt_epochs, train_accs, label="train", color="steelblue")
ax.plot(ckpt_epochs, test_accs, label="test", color="orange")
if grokking_detected:
    ax.axvline(ckpt_epochs[memorized_at], color="red", ls="--", alpha=0.7, label="memorized")
    ax.axvline(ckpt_epochs[grokked_at], color="green", ls="--", alpha=0.7, label="grokked")
ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
ax.set_title("Train / Test Accuracy"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 2: Pattern count
ax = axes[0, 1]
ax.plot(ckpt_epochs, pattern_counts, color="purple", marker=".", ms=2)
if grokking_detected:
    ax.axvline(ckpt_epochs[memorized_at], color="red", ls="--", alpha=0.5)
    ax.axvline(ckpt_epochs[grokked_at], color="green", ls="--", alpha=0.5)
ax.set_xlabel("epoch"); ax.set_ylabel("distinct patterns")
ax.set_title("Activation Pattern Count"); ax.grid(True, alpha=0.3)

# Panel 3: Region-label purity
ax = axes[0, 2]
ax.plot(ckpt_epochs, purities, color="teal", marker=".", ms=2)
if grokking_detected:
    ax.axvline(ckpt_epochs[memorized_at], color="red", ls="--", alpha=0.5)
    ax.axvline(ckpt_epochs[grokked_at], color="green", ls="--", alpha=0.5)
ax.set_xlabel("epoch"); ax.set_ylabel("weighted purity")
ax.set_title("Region-Label Purity"); ax.grid(True, alpha=0.3)

# Panel 4: Weight norm
ax = axes[1, 0]
ax.plot(ckpt_epochs, weight_norms, color="brown", marker=".", ms=2)
if grokking_detected:
    ax.axvline(ckpt_epochs[memorized_at], color="red", ls="--", alpha=0.5)
    ax.axvline(ckpt_epochs[grokked_at], color="green", ls="--", alpha=0.5)
ax.set_xlabel("epoch"); ax.set_ylabel("total weight L2 norm")
ax.set_title("Weight Norm"); ax.grid(True, alpha=0.3)

# Panel 5: Purity + test accuracy overlay
ax = axes[1, 1]
ax.plot(ckpt_epochs, test_accs, label="test acc", color="orange")
ax.plot(ckpt_epochs, purities, label="purity", color="teal", ls="--")
if grokking_detected:
    ax.axvline(ckpt_epochs[memorized_at], color="red", ls="--", alpha=0.3)
    ax.axvline(ckpt_epochs[grokked_at], color="green", ls="--", alpha=0.3)
ax.set_xlabel("epoch"); ax.set_ylabel("value")
ax.set_title("Test Accuracy vs Purity"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 6: Compression bars
ax = axes[1, 2]
labels_bar = list(results_compress.keys())
teacher_vals = [results_compress[k]["teacher_test_acc"] for k in labels_bar]
student_vals = [results_compress[k]["student_test_acc"] for k in labels_bar]
x_pos = np.arange(len(labels_bar))
w = 0.35
bars1 = ax.bar(x_pos - w/2, teacher_vals, w, label="teacher", color="steelblue", alpha=0.8)
bars2 = ax.bar(x_pos + w/2, student_vals, w, label=f"student (h={TARGET_HIDDEN})",
               color="orange", alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([l.replace("_", "\n") for l in labels_bar], fontsize=8)
ax.set_ylabel("test accuracy")
ax.set_title("Compression Quality"); ax.legend(fontsize=8)
ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3, axis="y")
for bar, val in zip(list(bars1) + list(bars2), teacher_vals + student_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
out_png = f"{RESULTS_DIR}/cycle_031_grokking_trajectory.png"
plt.savefig(out_png, dpi=120)
plt.close()
print(f"\nSaved: {out_png}")


# ── Save JSON ─────────────────────────────────────────────────────────────────

result = {
    "config": {
        "task": f"(a+b) mod {P}", "p": P, "frac_train": FRAC_TRAIN,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "hidden": HIDDEN, "depth": DEPTH, "epochs": EPOCHS,
        "ckpt_every": CKPT_EVERY, "lr": LR, "wd": WD, "betas": list(BETAS),
        "seed": SEED, "target_hidden": TARGET_HIDDEN,
    },
    "trajectory": {
        "epochs": ckpt_epochs,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "pattern_counts": pattern_counts,
        "purities": purities,
        "entropies": entropies,
        "weight_norms": weight_norms,
    },
    "grokking": {
        "detected": grokking_detected,
        "memorized_epoch": ckpt_epochs[memorized_at] if memorized_at is not None else None,
        "grokked_epoch": ckpt_epochs[grokked_at] if grokked_at is not None else None,
        "delay_epochs": (ckpt_epochs[grokked_at] - ckpt_epochs[memorized_at]
                         if grokking_detected else None),
    },
    "compression": results_compress,
    "verdict": verdict,
}

out_json = f"{RESULTS_DIR}/cycle_031_grokking.json"
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {out_json}")
print(f"\n=== DONE === Verdict: {verdict}")
