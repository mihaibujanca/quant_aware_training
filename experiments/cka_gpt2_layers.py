"""
Cycle 024: CKA representational similarity matrix across GPT-2 layers.

Computes the 13×13 CKA matrix (embedding + 12 blocks), plots heatmap,
detects representational plateaus, and correlates CKA drops with
single-block swap damage from cycle 005.
"""

import json
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import GPT2Model, GPT2Tokenizer
from scipy.stats import spearmanr

DEVICE = "cpu"
N_TOKENS_MAX = 4096  # max tokens for CKA (O(n^2))
SEQ_LEN = 128
MAX_CHUNKS = 512

RESULTS_DIR = Path("autoresearch/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CYCLE005_DAMAGE_8X = {
    0: 0.1768, 1: 0.0025, 2: 0.0245, 3: 0.0187,
    4: 0.0185, 5: 0.0229, 6: 0.0216, 7: 0.0246,
    8: 0.0291, 9: 0.0261, 10: 0.0307, 11: 0.0673,
}

# ── 1. Load model ──────────────────────────────────────────────────────────────
print("Loading GPT-2 ...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").eval().to(DEVICE)

# ── 2. Register hooks ──────────────────────────────────────────────────────────
# 13 checkpoints: embedding output + 12 block outputs
reps = {i: [] for i in range(13)}

def make_hook(idx):
    def hook(module, input, output):
        # output may be tuple; take first element
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        # h: (batch, seq, 768)
        reps[idx].append(h.detach().cpu())
    return hook

handles = []
# Checkpoint 0: embedding output — hook on model.drop (after wte+wpe+drop)
handles.append(model.drop.register_forward_hook(make_hook(0)))
# Checkpoints 1..12: block outputs
for blk_idx, block in enumerate(model.h):
    handles.append(block.register_forward_hook(make_hook(blk_idx + 1)))

# ── 3. Tokenize Shakespeare ────────────────────────────────────────────────────
print("Tokenizing Shakespeare ...")
text = Path("data/shakespeare.txt").read_text(errors="replace")
tokens = tokenizer.encode(text)
print(f"  Total tokens: {len(tokens)}")

chunks = []
for i in range(0, len(tokens) - SEQ_LEN, SEQ_LEN):
    chunks.append(tokens[i: i + SEQ_LEN])
    if len(chunks) >= MAX_CHUNKS:
        break
print(f"  Chunks: {len(chunks)}")

# ── 4. Forward pass ────────────────────────────────────────────────────────────
print("Running forward passes ...")
with torch.no_grad():
    for chunk_idx, chunk in enumerate(chunks):
        ids = torch.tensor([chunk], dtype=torch.long, device=DEVICE)
        model(ids)
        if (chunk_idx + 1) % 100 == 0:
            print(f"  {chunk_idx + 1}/{len(chunks)}")

# Remove hooks
for h in handles:
    h.remove()

# Concatenate: (N_total_tokens, 768) per checkpoint
print("Assembling representations ...")
X_all = {}
for i in range(13):
    X_all[i] = torch.cat(reps[i], dim=0).reshape(-1, 768)  # (N, 768)
    print(f"  Checkpoint {i}: shape {X_all[i].shape}")

# Subsample if needed
N = X_all[0].shape[0]
if N > N_TOKENS_MAX:
    torch.manual_seed(42)
    idx = torch.randperm(N)[:N_TOKENS_MAX]
    for i in range(13):
        X_all[i] = X_all[i][idx]
    N = N_TOKENS_MAX
print(f"  Using {N} tokens for CKA")

# ── 5. Linear CKA ──────────────────────────────────────────────────────────────
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA (Kornblith et al. 2019), efficient via Frobenius norms."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    # HSIC with linear kernel = ||X^T Y||_F^2
    XtY = X.T @ Y        # (d1, d2)
    XtX = X.T @ X        # (d1, d1)
    YtY = Y.T @ Y        # (d2, d2)
    hsic_xy = XtY.pow(2).sum()
    hsic_xx = XtX.pow(2).sum()
    hsic_yy = YtY.pow(2).sum()
    denom = (hsic_xx * hsic_yy).sqrt()
    if denom < 1e-10:
        return 0.0
    return (hsic_xy / denom).item()

print("Computing 13×13 CKA matrix ...")
cka_matrix = torch.zeros(13, 13)
for i in range(13):
    for j in range(i, 13):
        v = linear_cka(X_all[i], X_all[j])
        cka_matrix[i, j] = v
        cka_matrix[j, i] = v
    print(f"  Row {i} done")

cka_np = cka_matrix.numpy()
print("CKA matrix:")
print(np.round(cka_np, 3))

# ── 6. Save matrix ─────────────────────────────────────────────────────────────
matrix_path = RESULTS_DIR / "cycle_024_cka_matrix.json"
with open(matrix_path, "w") as f:
    json.dump({"cka_matrix": cka_np.tolist(), "n_tokens": N}, f, indent=2)
print(f"Saved matrix to {matrix_path}")

# ── 7. Plot heatmap ────────────────────────────────────────────────────────────
layer_labels = ["emb"] + [f"L{i}" for i in range(12)]

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cka_np, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
plt.colorbar(im, ax=ax, label="Linear CKA")
ax.set_xticks(range(13))
ax.set_yticks(range(13))
ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(layer_labels, fontsize=9)
ax.set_title("GPT-2 CKA Representational Similarity (13×13)\nCycle 024")

# Annotate cells
for i in range(13):
    for j in range(13):
        v = cka_np[i, j]
        color = "white" if v < 0.5 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=color)

plt.tight_layout()
heatmap_path = RESULTS_DIR / "cycle_024_cka_heatmap.png"
plt.savefig(heatmap_path, dpi=150)
plt.close()
print(f"Saved heatmap to {heatmap_path}")

# ── 8. Detect representational plateaus ───────────────────────────────────────
CKA_THRESHOLD = 0.80
MIN_GROUP = 3

plateaus = []
# Check all contiguous groups of size >= MIN_GROUP
for start in range(13):
    for end in range(start + MIN_GROUP - 1, 13):
        group = list(range(start, end + 1))
        # All pairwise CKA >= threshold
        all_above = all(cka_np[i, j] >= CKA_THRESHOLD
                        for i in group for j in group if i != j)
        if all_above:
            # Add if maximal (not already contained in a larger group)
            dominated = any(
                (p["start"] <= start and p["end"] >= end and
                 (p["start"] < start or p["end"] > end))
                for p in plateaus
            )
            if not dominated:
                plateaus.append({"start": start, "end": end,
                                  "size": len(group), "layers": group})

# Keep only maximal plateaus
maximal_plateaus = []
for p in plateaus:
    is_sub = any(
        (q["start"] <= p["start"] and q["end"] >= p["end"] and q != p)
        for q in plateaus
    )
    if not is_sub:
        maximal_plateaus.append(p)

print(f"\nRepresentational plateaus (CKA ≥ {CKA_THRESHOLD}, size ≥ {MIN_GROUP}):")
if maximal_plateaus:
    for p in maximal_plateaus:
        labels = [layer_labels[l] for l in p["layers"]]
        print(f"  Layers {p['start']}–{p['end']} ({labels}), size={p['size']}")
else:
    print("  None found")

# ── 9. CKA drops and Spearman correlation ─────────────────────────────────────
# CKA drop between adjacent checkpoints (12 inter-layer transitions)
# drop[i] = 1 - CKA[i, i+1], i=0..11
# These correspond to "transition from checkpoint i to checkpoint i+1"
# CKA checkpoint indices: 0=emb, 1=block0, 2=block1, ..., 12=block11
# Block i (0-indexed) output is checkpoint i+1
# The "CKA drop from block i to block i+1" = 1 - CKA[i+1, i+2]
# For cycle 005 damage for block i (0..11), compare with transition i+1->i+2

# Actually, let's use: for block i, CKA drop = 1 - CKA[i+1, i+2] (i=0..10, 11 blocks)
# But we only have damage for layers 0..11 (12 blocks)
# Use: drop_for_block_i = 1 - CKA[checkpoint_i+1, checkpoint_i+2]
#   = 1 - CKA[i+1, i+2]  (transition OUT of block i, i.e. how much block i+1 changes things)
# Alternative: use the simpler inter-checkpoint drop
# drop[k] = 1 - cka[k, k+1], k=0..11 (12 values)
# Align with damage[k] for block k (0..11), where block k output is checkpoint k+1
# drop[k] = 1 - cka[k, k+1] measures how much the embedding changed going INTO block k

cka_drops = [1.0 - float(cka_np[k, k + 1]) for k in range(12)]
damage = [CYCLE005_DAMAGE_8X[k] for k in range(12)]

print("\nCKA drops (1 - CKA[i, i+1]) vs cycle 005 damage:")
for k in range(12):
    label = f"emb→L0" if k == 0 else f"L{k-1}→L{k}"
    print(f"  {label}: drop={cka_drops[k]:.4f}, damage={damage[k]*100:.2f}%")

rho, pval = spearmanr(cka_drops, damage)
print(f"\nSpearman(CKA_drop, damage): rho={rho:.4f}, p={pval:.4f}")

# ── 10. Save results ───────────────────────────────────────────────────────────
any_plateau = len(maximal_plateaus) > 0
non_adj_cka = [float(cka_np[i, j]) for i in range(13) for j in range(13)
               if abs(i - j) > 1]
max_non_adj = max(non_adj_cka) if non_adj_cka else 0.0

# Verdict
if any_plateau and rho >= 0.40:
    verdict = "CONFIRMED"
elif max_non_adj < 0.50:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

results = {
    "n_tokens": N,
    "cka_matrix": cka_np.tolist(),
    "plateaus": maximal_plateaus,
    "cka_threshold": CKA_THRESHOLD,
    "cka_drops": cka_drops,
    "damage_8x": damage,
    "spearman_rho": float(rho),
    "spearman_p": float(pval),
    "max_non_adjacent_cka": max_non_adj,
    "verdict": verdict,
}

json_path = RESULTS_DIR / "cycle_024_cka_matrix.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved full results to {json_path}")
print(f"Verdict: {verdict}")

# ── 11. Write summary ──────────────────────────────────────────────────────────
plateau_str = "\n".join(
    f"  - Layers {p['start']}–{p['end']} ({[layer_labels[l] for l in p['layers']]}), size={p['size']}"
    for p in maximal_plateaus
) if maximal_plateaus else "  None found (no group of ≥3 with pairwise CKA ≥ 0.80)"

summary = f"""# Cycle 024 Summary — CKA Representational Similarity (GPT-2)

## Verdict: {verdict}

## Setup
- Model: GPT-2 (12 blocks, 768 hidden dim)
- Corpus: Shakespeare ({N} tokens used for CKA)
- CKA: linear kernel, centered, Frobenius norm formulation
- 13 checkpoints: embedding + 12 block outputs

## CKA Matrix (diagonal and near-diagonal)
- Self-similarity (diagonal): all 1.0 by construction
- Adjacent (mean CKA[i,i+1]): {np.mean([cka_np[i,i+1] for i in range(12)]):.4f}
- Max non-adjacent CKA: {max_non_adj:.4f}

## Representational Plateaus (CKA ≥ {CKA_THRESHOLD}, size ≥ {MIN_GROUP})
{plateau_str}

## Spearman Correlation (CKA drop vs cycle 005 damage)
- Spearman ρ = {rho:.4f}, p = {pval:.4f}
- N = 12 blocks

## Criteria Evaluation
- Plateau criterion (≥1 group ≥3 layers with CKA ≥ 0.80): {"PASS" if any_plateau else "FAIL"}
- Spearman criterion (ρ ≥ 0.40): {"PASS" if rho >= 0.40 else "FAIL"}
- Refuted criterion (all non-adj CKA < 0.50): {"YES" if max_non_adj < 0.50 else "NO"}

## Artifacts
- `autoresearch/results/cycle_024_cka_matrix.json`
- `autoresearch/results/cycle_024_cka_heatmap.png`
"""

summary_path = RESULTS_DIR / "cycle_024_summary.md"
summary_path.write_text(summary)
print(f"Saved summary to {summary_path}")
print("\nDone.")
