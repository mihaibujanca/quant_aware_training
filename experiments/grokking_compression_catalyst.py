"""Grokking as compression catalyst — toy ReLU spirals.

Cycle 021: Train spirals MLP well past convergence with weight decay.
Monitor activation pattern count, pattern entropy, single-affine residual, weight norm.
Look for phase transition where partition complexity drops (grokking-style simplification).
Compare compression quality at early-training vs late-training (peak-pattern vs low-pattern).

NOTE: Classic grokking (discrete test-acc jump) does NOT occur on spirals.
We instead observe a weight-norm / partition-complexity interplay.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aleph.datasets import make_spirals

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN    = 200
N_TEST     = 2000
TURNS      = 3
HIDDEN     = 32
DEPTH      = 4      # 4 hidden layers: 2→32→32→32→32→1
EPOCHS     = 30000  # long training to allow any late-stage phase transitions
CKPT_EVERY = 1000
LR         = 0.001
WD         = 0.001  # enough to encourage grokking without killing the network
SEED       = 42

RESULTS_DIR = "autoresearch/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────

def make_mlp(input_dim: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(in_features, hidden))
        layers.append(nn.ReLU())
        in_features = hidden
    layers.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layers)


def linear_layers(model: nn.Sequential) -> list[nn.Linear]:
    return [m for m in model if isinstance(m, nn.Linear)]


# ── Forward pass collecting masks ─────────────────────────────────────────────

def forward_with_masks(model: nn.Sequential, X: torch.Tensor):
    """Return (logits [N,1], masks [list of (N, H) bool tensors])."""
    masks = []
    a = X
    for module in model:
        if isinstance(module, nn.Linear):
            a = module(a)
        elif isinstance(module, nn.ReLU):
            masks.append(a > 0)
            a = F.relu(a)
    return a, masks  # a is logits (N, 1)


# ── Geometric signals ─────────────────────────────────────────────────────────

def pattern_signals(model: nn.Sequential, X: torch.Tensor):
    """Count distinct activation patterns and compute entropy on X."""
    with torch.no_grad():
        logits, masks = forward_with_masks(model, X)

    flat = torch.cat([m.to(torch.uint8) for m in masks], dim=1).cpu().numpy()
    packed = np.packbits(flat, axis=1)
    codes = [bytes(row).hex() for row in packed]
    n_patterns = len(set(codes))

    freq = Counter(codes)
    n = len(codes)
    probs = np.array([v / n for v in freq.values()])
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    return n_patterns, entropy


def single_affine_residual(model: nn.Sequential, X: torch.Tensor) -> float:
    """Fraction of logit variance not explained by a single global affine fit."""
    with torch.no_grad():
        logits, _ = forward_with_masks(model, X)
    logits_cpu = logits.detach().cpu().squeeze(1)
    X_cpu = X.detach().cpu()
    X_aug = torch.cat([X_cpu, torch.ones(len(X_cpu), 1)], dim=1)
    sol = torch.linalg.lstsq(X_aug, logits_cpu.unsqueeze(1)).solution.squeeze(1)
    pred = X_aug @ sol
    resid = float((logits_cpu - pred).pow(2).mean() / (logits_cpu.pow(2).mean() + 1e-8))
    return resid


def weight_norm(model: nn.Sequential) -> float:
    return float(sum(p.norm().item() for p in model.parameters()))


def eval_acc(model: nn.Sequential, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        logits, _ = forward_with_masks(model, X)
        preds = (logits.squeeze(1) > 0).float()
        return float((preds == y).float().mean())


# ── Data ──────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)

X_all, y_all = make_spirals(
    n_samples=N_TRAIN + N_TEST, noise=0.3, n_turns=TURNS, random_state=SEED
)
X_all_t = torch.tensor(X_all, dtype=torch.float32)
y_all_t  = torch.tensor(y_all, dtype=torch.float32)

perm = torch.randperm(len(X_all_t))
train_idx = perm[:N_TRAIN]
test_idx  = perm[N_TRAIN:]
X_train, y_train = X_all_t[train_idx], y_all_t[train_idx]
X_test,  y_test  = X_all_t[test_idx],  y_all_t[test_idx]


# ── Training with checkpoints ─────────────────────────────────────────────────

torch.manual_seed(SEED)
model = make_mlp(2, HIDDEN, DEPTH)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

print(f"Training: n_train={N_TRAIN}, depth={DEPTH}, hidden={HIDDEN}, epochs={EPOCHS}")
print(f"          lr={LR}, wd={WD}, ckpt_every={CKPT_EVERY}")
print()
print(f"{'epoch':>7s} | {'tr_acc':>6s} {'te_acc':>6s} | {'w_norm':>7s} | "
      f"{'n_pat':>6s} {'entropy':>7s} {'resid':>8s}")
print("-" * 68)

checkpoints   = []
train_accs    = []
test_accs     = []
pattern_counts = []
entropies     = []
residuals     = []
weight_norms  = []
ckpt_epochs   = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    logits, _ = forward_with_masks(model, X_train)
    loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), y_train)
    loss.backward()
    optimizer.step()

    if epoch % CKPT_EVERY == 0:
        model.eval()
        tr_acc = eval_acc(model, X_train, y_train)
        te_acc = eval_acc(model, X_test,  y_test)
        n_pat, ent = pattern_signals(model, X_test)
        resid = single_affine_residual(model, X_test)
        wnorm = weight_norm(model)

        checkpoints.append({k: v.clone() for k, v in model.state_dict().items()})
        train_accs.append(tr_acc)
        test_accs.append(te_acc)
        pattern_counts.append(n_pat)
        entropies.append(ent)
        residuals.append(resid)
        weight_norms.append(wnorm)
        ckpt_epochs.append(epoch)

        print(f"  {epoch:5d} | {tr_acc:6.3f} {te_acc:6.3f} | {wnorm:7.2f} | "
              f"{n_pat:6d} {ent:7.2f} {resid:8.4f}")


# ── Detect phase transition: pattern count peak and subsequent drop ────────────

pat_arr = np.array(pattern_counts, dtype=float)
peak_idx  = int(np.argmax(pat_arr))            # checkpoint where patterns are max
trough_idx = int(np.argmin(pat_arr))           # checkpoint where patterns are min

# Drop from peak
max_val   = pat_arr[peak_idx]
# Find minimum AFTER peak (if any)
if peak_idx < len(pat_arr) - 1:
    post_peak_min = np.min(pat_arr[peak_idx:])
    post_peak_min_idx = peak_idx + int(np.argmin(pat_arr[peak_idx:]))
else:
    post_peak_min = max_val
    post_peak_min_idx = peak_idx

drop_from_peak = (max_val - post_peak_min) / max(max_val, 1)

# Grokking-style: train saturated early, test rising late?
train_saturated_at = None
for i, ta in enumerate(train_accs):
    if ta >= 0.90:
        train_saturated_at = i
        break

test_rise = None
if train_saturated_at is not None:
    base_test = test_accs[train_saturated_at]
    for i in range(train_saturated_at + 1, len(test_accs)):
        if test_accs[i] - base_test >= 0.05:
            test_rise = i
            break

grokking_detected = test_rise is not None

print()
print(f"Pattern peak at epoch {ckpt_epochs[peak_idx]} (n={int(max_val)})")
print(f"Pattern min-after-peak at epoch {ckpt_epochs[post_peak_min_idx]} (n={int(post_peak_min)})")
print(f"Drop fraction from peak: {drop_from_peak:.3f}")
print(f"Grokking (test-acc rise ≥0.05 after train saturated): {grokking_detected}")
if train_saturated_at is not None:
    print(f"  Train saturated at epoch {ckpt_epochs[train_saturated_at]}")
if test_rise is not None:
    print(f"  Test rose at epoch {ckpt_epochs[test_rise]}")


# ── Compression comparison: distill from teacher at different checkpoints ──────

def compress_and_eval(
    state_dict: dict,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test:  torch.Tensor,
    y_test:  torch.Tensor,
    target_hidden: int = 8,
    epochs: int = 3000,
) -> float:
    """Distill from teacher checkpoint into smaller student, return test acc."""
    teacher = make_mlp(2, HIDDEN, DEPTH)
    teacher.load_state_dict(state_dict)
    teacher.eval()

    with torch.no_grad():
        tlogits, _ = forward_with_masks(teacher, X_train)
        soft = torch.sigmoid(tlogits.squeeze(1))

    torch.manual_seed(SEED + 1)
    student = make_mlp(2, target_hidden, DEPTH)
    opt = torch.optim.Adam(student.parameters(), lr=0.005)

    for _ in range(epochs):
        student.train()
        opt.zero_grad()
        s_logits, _ = forward_with_masks(student, X_train)
        loss = F.binary_cross_entropy_with_logits(s_logits.squeeze(1), soft)
        loss.backward()
        opt.step()

    student.eval()
    return eval_acc(student, X_test, y_test)


# Compare: (a) pre-grokking checkpoint, (b) post-grokking checkpoint
# If grokking detected: use train-saturated epoch as pre, grokking epoch as post.
# Otherwise: use epoch of max pattern count as pre, epoch of min pattern count as post.
TARGET_HIDDEN = 8  # 8 hidden units per layer vs 32 = 4× compression

if grokking_detected and train_saturated_at is not None and test_rise is not None:
    pre_compress_idx  = train_saturated_at  # last pre-grokking checkpoint
    post_compress_idx = test_rise            # first post-grokking checkpoint
else:
    pre_compress_idx  = peak_idx
    post_compress_idx = post_peak_min_idx

print()
print(f"Compression comparison: "
      f"peak-pattern epoch={ckpt_epochs[pre_compress_idx]} "
      f"vs low-pattern epoch={ckpt_epochs[post_compress_idx]}")

pre_teacher = make_mlp(2, HIDDEN, DEPTH)
pre_teacher.load_state_dict(checkpoints[pre_compress_idx])
pre_teacher_acc  = eval_acc(pre_teacher, X_test, y_test)

post_teacher = make_mlp(2, HIDDEN, DEPTH)
post_teacher.load_state_dict(checkpoints[post_compress_idx])
post_teacher_acc = eval_acc(post_teacher, X_test, y_test)

pre_compressed_acc  = compress_and_eval(
    checkpoints[pre_compress_idx],  X_train, y_train, X_test, y_test, TARGET_HIDDEN
)
post_compressed_acc = compress_and_eval(
    checkpoints[post_compress_idx], X_train, y_train, X_test, y_test, TARGET_HIDDEN
)

pre_patterns  = pattern_counts[pre_compress_idx]
post_patterns = pattern_counts[post_compress_idx]

print(f"  Peak-pattern  (epoch {ckpt_epochs[pre_compress_idx]}): "
      f"teacher={pre_teacher_acc:.3f}, compressed={pre_compressed_acc:.3f}, "
      f"n_patterns={pre_patterns}")
print(f"  Low-pattern   (epoch {ckpt_epochs[post_compress_idx]}): "
      f"teacher={post_teacher_acc:.3f}, compressed={post_compressed_acc:.3f}, "
      f"n_patterns={post_patterns}")


# ── Verdict ───────────────────────────────────────────────────────────────────

acc_improvement = post_compressed_acc - pre_compressed_acc

print()
print(f"Pattern drop fraction (peak→min): {drop_from_peak:.3f} (need ≥0.40 for confirmed)")
print(f"Compression acc improvement:      {acc_improvement:.3f} (need ≥0.10 for confirmed)")

if drop_from_peak >= 0.40 and acc_improvement >= 0.10:
    verdict = "confirmed"
elif drop_from_peak < 0.10 or abs(acc_improvement) < 0.02:
    verdict = "refuted"
else:
    verdict = "inconclusive"

print(f"Verdict: {verdict}")


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Grokking as Compression Catalyst — Toy ReLU Spirals\n"
             f"(n_train={N_TRAIN}, depth={DEPTH}×{HIDDEN}, WD={WD})", fontsize=12)

ax = axes[0, 0]
ax.plot(ckpt_epochs, train_accs, label="train", color="steelblue")
ax.plot(ckpt_epochs, test_accs,  label="test",  color="orange")
ax.axvline(ckpt_epochs[peak_idx], color="purple", linestyle="--", alpha=0.7, label="pat peak")
ax.axvline(ckpt_epochs[post_peak_min_idx], color="red", linestyle="--", alpha=0.7, label="pat min")
ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
ax.set_title("Train/Test Accuracy"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(ckpt_epochs, pattern_counts, color="purple", marker="o", ms=3)
ax.axvline(ckpt_epochs[peak_idx], color="purple", linestyle="--", alpha=0.5)
ax.axvline(ckpt_epochs[post_peak_min_idx], color="red", linestyle="--", alpha=0.5)
ax.set_xlabel("epoch"); ax.set_ylabel("distinct patterns")
ax.set_title(f"Activation Pattern Count (drop={drop_from_peak:.2f})")
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.plot(ckpt_epochs, weight_norms, color="brown", marker="o", ms=3)
ax.axvline(ckpt_epochs[peak_idx], color="purple", linestyle="--", alpha=0.5)
ax.axvline(ckpt_epochs[post_peak_min_idx], color="red", linestyle="--", alpha=0.5)
ax.set_xlabel("epoch"); ax.set_ylabel("total weight L2 norm")
ax.set_title("Weight Norm"); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(ckpt_epochs, entropies, color="green", marker="o", ms=3)
ax.axvline(ckpt_epochs[peak_idx], color="purple", linestyle="--", alpha=0.5)
ax.axvline(ckpt_epochs[post_peak_min_idx], color="red", linestyle="--", alpha=0.5)
ax.set_xlabel("epoch"); ax.set_ylabel("pattern entropy (bits)")
ax.set_title("Pattern Entropy"); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(ckpt_epochs, residuals, color="teal", marker="o", ms=3)
ax.axvline(ckpt_epochs[peak_idx], color="purple", linestyle="--", alpha=0.5)
ax.axvline(ckpt_epochs[post_peak_min_idx], color="red", linestyle="--", alpha=0.5)
ax.set_xlabel("epoch"); ax.set_ylabel("single-affine residual")
ax.set_title("Single-Affine Residual"); ax.grid(True, alpha=0.3)

ax = axes[1, 2]
cats = ["peak-pattern\n(teacher)", "peak-pattern\n(compressed)", "low-pattern\n(teacher)", "low-pattern\n(compressed)"]
vals = [pre_teacher_acc, pre_compressed_acc, post_teacher_acc, post_compressed_acc]
colors = ["steelblue", "lightblue", "orange", "moccasin"]
bars = ax.bar(cats, vals, color=colors)
ax.set_ylabel("test accuracy")
ax.set_title(f"Compression Quality\n(target_hidden={TARGET_HIDDEN}, acc_gain={acc_improvement:+.3f})")
ax.set_ylim(0, 1.05)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/cycle_021_grokking_trajectory.png", dpi=120)
plt.close()
print(f"\nSaved: {RESULTS_DIR}/cycle_021_grokking_trajectory.png")


# ── Save JSON ─────────────────────────────────────────────────────────────────

result = {
    "config": {
        "n_train": N_TRAIN, "n_test": N_TEST, "turns": TURNS,
        "hidden": HIDDEN, "depth": DEPTH, "epochs": EPOCHS,
        "ckpt_every": CKPT_EVERY, "lr": LR, "wd": WD, "seed": SEED,
        "target_hidden_compressed": TARGET_HIDDEN,
    },
    "trajectory": {
        "epochs": ckpt_epochs,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "pattern_counts": pattern_counts,
        "entropies": entropies,
        "single_affine_residuals": residuals,
        "weight_norms": weight_norms,
    },
    "phase_transition": {
        "classic_grokking_detected": grokking_detected,
        "train_saturated_epoch": ckpt_epochs[train_saturated_at] if train_saturated_at is not None else None,
        "test_rise_epoch": ckpt_epochs[test_rise] if test_rise is not None else None,
        "pattern_peak_epoch": ckpt_epochs[peak_idx],
        "pattern_peak_count": int(max_val),
        "post_peak_min_epoch": ckpt_epochs[post_peak_min_idx],
        "post_peak_min_count": int(post_peak_min),
        "drop_from_peak_fraction": float(drop_from_peak),
    },
    "compression": {
        "pre_epoch": ckpt_epochs[pre_compress_idx],
        "post_epoch": ckpt_epochs[post_compress_idx],
        "pre_teacher_acc": pre_teacher_acc,
        "post_teacher_acc": post_teacher_acc,
        "pre_compressed_acc": pre_compressed_acc,
        "post_compressed_acc": post_compressed_acc,
        "pre_patterns": pre_patterns,
        "post_patterns": post_patterns,
        "acc_improvement": float(acc_improvement),
    },
    "verdict": verdict,
}

out_path = f"{RESULTS_DIR}/cycle_021_grokking.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {out_path}")
print()
print("=== DONE ===")
print(f"Verdict:           {verdict}")
print(f"Classic grokking:  {grokking_detected}")
print(f"Pattern drop:      {drop_from_peak:.3f}")
print(f"Compression gain:  {acc_improvement:+.3f}")
