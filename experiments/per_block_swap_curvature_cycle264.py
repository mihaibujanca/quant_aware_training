"""
Cycle 264 — Item 274: Pullback curvature κ_l on per-block partial swaps.

For each k ∈ {0..11}, install ReLU + per-channel b_leak ONLY at block k of GPT-2 small,
keep the other 11 blocks GELU, compute the κ profile under G = U^T U on the standard
200×128 wikitext-2 calib slice (seed 42), then build a 12×11 Δκ matrix vs the all-GELU
baseline (rows = swap location k, columns = κ index l ∈ {0..10} → layers 1..11).

Verdict (item 274):
  - CONFIRMED if max(|Δκ|) on the diagonal band ≥ 3× off-diagonal median
    AND late-layer columns show monotonically increasing |Δκ| as swap location k decreases.
  - REFUTED if Δκ heatmap is structureless or anti-diagonal.
  - INCONCLUSIVE otherwise.

Reuses cycle 258 b_leak recipe and ReluPlusBleak module.
"""
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

from aleph.pullback_geometry import (
    collect_residual_trajectory,
    compute_curvature,
    get_unembedding,
)

# ── Config ────────────────────────────────────────────────────────────────────
N_EVAL = 200
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

RESULT_PATH = "autoresearch/results/cycle_264_per_block_swap_curvature.json"
SUMMARY_PATH = "autoresearch/results/cycle_264_summary.md"
PLOT_PATH = "autoresearch/results/cycle_264_swap_kappa_heatmap.png"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}", flush=True)

# ── Load model + data (cycle 258 loader) ─────────────────────────────────────
print("Loading GPT-2 and data...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
all_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        all_ids.append(enc["input_ids"][:, :SEQ_LEN])
    if len(all_ids) >= N_EVAL:
        break
calib_ids = [t.to(DEVICE) for t in all_ids[:N_EVAL]]
print(f"Calib: {len(calib_ids)} seqs × {SEQ_LEN} tokens", flush=True)

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
originals = [model.transformer.h[i].mlp.act for i in range(12)]
print(f"Original activation type: {type(originals[0]).__name__}", flush=True)


def eval_ppl(model, eval_ids):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


baseline_ppl = eval_ppl(model, calib_ids)
print(f"Baseline GELU PPL on calib slice = {baseline_ppl:.4f}", flush=True)


# ── Phase 1: Precompute b_leak per block (cycle 258 recipe) ──────────────────
print("\n──────── Phase 1: Precompute b_leak per block ────────", flush=True)
preact_stash = {l: [] for l in range(12)}
hooks = []


def make_hook(l):
    def hook(module, inputs, output):
        preact_stash[l].append(inputs[0].detach())
    return hook


for l in range(12):
    h = model.transformer.h[l].mlp.c_fc.register_forward_hook(make_hook(l))
    hooks.append(h)

print(f"Hooks registered on {len(hooks)} c_fc modules. Calibration forward pass...", flush=True)
with torch.no_grad():
    for ids in calib_ids:
        _ = model(ids)
for h in hooks:
    h.remove()
print("Hooks removed.", flush=True)

b_leak_dict = {}
b_leak_stats = []
for l in range(12):
    x_l = torch.cat(preact_stash[l], dim=0).reshape(-1, 768)
    c_fc = model.transformer.h[l].mlp.c_fc
    preact = x_l @ c_fc.weight + c_fc.bias  # (n_tokens, 3072)
    gelu_l = F.gelu(preact, approximate='tanh')
    relu_l = F.relu(preact)
    b_leak_l = (gelu_l - relu_l).mean(dim=0)  # (3072,)

    b_leak_dict[l] = b_leak_l.to(DEVICE)
    b_leak_stats.append({
        "block": l,
        "mean": float(b_leak_l.mean().item()),
        "min": float(b_leak_l.min().item()),
        "max": float(b_leak_l.max().item()),
        "abs_mean": float(b_leak_l.abs().mean().item()),
    })
    print(f"  block {l:2d}: b_leak mean={b_leak_stats[l]['mean']:+.5f} "
          f"|b|={b_leak_stats[l]['abs_mean']:.5f}", flush=True)
    preact_stash[l] = None


# ── ReLU + b_leak module (cycle 258 verbatim) ────────────────────────────────
class ReluPlusBleak(nn.Module):
    def __init__(self, b_leak):
        super().__init__()
        self.register_buffer('b_leak', b_leak)

    def forward(self, z):
        return F.relu(z) + self.b_leak


# ── Trajectory collector (cycle 258) ─────────────────────────────────────────
def collect_concat_trajectory(model, calib_ids):
    per_seq_trajectories = []
    for ids in calib_ids:
        traj = collect_residual_trajectory(model, ids)
        per_seq_trajectories.append(traj)
    n_layers_plus = len(per_seq_trajectories[0])
    concat = []
    for li in range(n_layers_plus):
        concat.append(torch.cat([t[li] for t in per_seq_trajectories], dim=0))
    return concat


def restore_originals():
    for l in range(12):
        model.transformer.h[l].mlp.act = originals[l]


U = get_unembedding(model)
print(f"\nU shape = {tuple(U.shape)}", flush=True)


# ── Phase 2: Baseline κ (all-GELU) ───────────────────────────────────────────
print("\n──────── Phase 2: Baseline κ (all-GELU) ────────", flush=True)
restore_originals()
traj_base = collect_concat_trajectory(model, calib_ids)
print(f"  trajectory: {len(traj_base)} layers", flush=True)
K_base = compute_curvature(traj_base, U, reduce='mean')  # length 11
kappa_baseline = K_base.detach().to('cpu').float().tolist()
print(f"  κ_baseline (l=1..11): " + " ".join(f"{v:.4e}" for v in kappa_baseline), flush=True)
del traj_base, K_base


# ── Phase 3: Per-block partial swap κ ────────────────────────────────────────
print("\n──────── Phase 3: Per-block partial swap κ ────────", flush=True)
kappa_per_swap = []  # list of 12 lists of length 11
patched_ppls = []
for k in range(12):
    restore_originals()
    model.transformer.h[k].mlp.act = ReluPlusBleak(b_leak_dict[k]).to(DEVICE)

    other_count = sum(1 for l in range(12) if model.transformer.h[l].mlp.act is originals[l])
    assert other_count == 11, f"swap_at_k={k}: expected 11 untouched, got {other_count}"

    ppl_k = eval_ppl(model, calib_ids)
    print(f"  swap_at_k={k:2d}: patched PPL = {ppl_k:.4f}", flush=True)
    patched_ppls.append(float(ppl_k))

    traj_k = collect_concat_trajectory(model, calib_ids)
    K_k = compute_curvature(traj_k, U, reduce='mean')
    k_list = K_k.detach().to('cpu').float().tolist()
    kappa_per_swap.append(k_list)
    print(f"  swap_at_k={k:2d} κ: " + " ".join(f"{v:.4e}" for v in k_list), flush=True)
    del traj_k, K_k

restore_originals()
post_restore_ppl = eval_ppl(model, calib_ids)
print(f"\nPost-restore baseline PPL = {post_restore_ppl:.4f} "
      f"(should match {baseline_ppl:.4f})", flush=True)
assert abs(post_restore_ppl - baseline_ppl) < 1e-4, "Restore round-trip failed"


# ── Phase 4: Δκ matrix + verdict logic ───────────────────────────────────────
print("\n──────── Phase 4: Δκ matrix + verdict logic ────────", flush=True)
# rows = swap location k (0..11), cols = κ index l (0..10) → layers 1..11
delta_kappa = [
    [kappa_per_swap[k][li] - kappa_baseline[li] for li in range(11)]
    for k in range(12)
]
abs_delta = [[abs(v) for v in row] for row in delta_kappa]

# L1 norm per row (per swap)
l1_per_swap = [sum(row) for row in abs_delta]
ranking = sorted(range(12), key=lambda k: l1_per_swap[k], reverse=True)

# Diagonal band: for swap_at_k, κ indices l ∈ {k-1, k, k+1} ∩ {0..10}.
# (κ index 0 = layer 1, ..., κ index 10 = layer 11.)
def diagonal_indices(k):
    cand = [k - 1, k, k + 1]
    return [li for li in cand if 0 <= li <= 10]


diag_entries = []
offdiag_entries = []
for k in range(12):
    diag_li = set(diagonal_indices(k))
    for li in range(11):
        v = abs_delta[k][li]
        if li in diag_li:
            diag_entries.append(v)
        else:
            offdiag_entries.append(v)

max_diag = max(diag_entries) if diag_entries else 0.0
median_offdiag = (
    sorted(offdiag_entries)[len(offdiag_entries) // 2] if offdiag_entries else 0.0
)
diag_to_offdiag_ratio = (
    max_diag / median_offdiag if median_offdiag > 1e-30 else float('inf')
)

# Late-layer monotonicity: for each late l ∈ {7..10}, check whether |Δκ[k,l]|
# is non-decreasing as k decreases through k = 6, 5, 4, 3, 2, 1, 0
# (i.e., the column should grow leftward through the early-swap rows).
late_cols = [7, 8, 9, 10]
monotonic_results = {}
n_strict_monotone_decreasing_in_k = 0
for li in late_cols:
    col = [abs_delta[k][li] for k in range(7)]  # k=0..6
    # we want col[6] (k=6) <= col[5] <= col[4] <= ... <= col[0] (k=0)
    # i.e., col is non-increasing in k → as k decreases, |Δκ| increases
    rev = col[::-1]  # k=6..0
    is_non_decreasing_as_k_decreases = all(rev[i] <= rev[i + 1] for i in range(len(rev) - 1))
    monotonic_results[li] = {
        "col_k_0_to_6": col,
        "monotone_increasing_as_k_decreases": is_non_decreasing_as_k_decreases,
    }
    if is_non_decreasing_as_k_decreases:
        n_strict_monotone_decreasing_in_k += 1

late_monotone_passes = n_strict_monotone_decreasing_in_k >= 3  # at least 3 of 4 late cols

# CONFIRMED: diag/offdiag ≥ 3× AND late_monotone_passes
clause_diag = diag_to_offdiag_ratio >= 3.0
clause_late = late_monotone_passes
confirmed = clause_diag and clause_late

# REFUTED: heatmap structureless (no diagonal band) — operationalised as
# diag/offdiag ratio < 1.0 (diagonal not even hotter than off-diagonal)
# OR anti-diagonal: max diag < min off-diag.
min_offdiag = min(offdiag_entries) if offdiag_entries else 0.0
refuted_structureless = diag_to_offdiag_ratio < 1.0
refuted_antidiagonal = max_diag < median_offdiag
refuted = refuted_structureless or refuted_antidiagonal

if confirmed:
    verdict = "CONFIRMED"
elif refuted:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"  max|Δκ|_diag = {max_diag:.4e}", flush=True)
print(f"  median|Δκ|_offdiag = {median_offdiag:.4e}", flush=True)
print(f"  diag/offdiag ratio = {diag_to_offdiag_ratio:.3f} (need ≥3.0 for CONFIRMED clause 1)", flush=True)
print(f"  late-layer monotone (k decreasing) cols passing: {n_strict_monotone_decreasing_in_k}/4 "
      f"(need ≥3 for CONFIRMED clause 2)", flush=True)
print(f"  ‖Δκ‖₁ per swap k: " + " ".join(f"{v:.4e}" for v in l1_per_swap), flush=True)
print(f"  L1 ranking (most disruptive first): {ranking}", flush=True)
print(f"  VERDICT: {verdict}", flush=True)


# ── Phase 5: Heatmap ─────────────────────────────────────────────────────────
print("\nPlotting heatmap...", flush=True)
import numpy as np
delta_np = np.array(delta_kappa)  # (12, 11)
abs_max = float(np.max(np.abs(delta_np))) if delta_np.size else 1e-12

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax1, ax2 = axes

im1 = ax1.imshow(delta_np, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, aspect='auto')
ax1.set_xlabel("κ index l (layer l+1)")
ax1.set_ylabel("swap location k (block index)")
ax1.set_title("Δκ[k, l] = κ_swap_at_k − κ_baseline")
ax1.set_xticks(range(11))
ax1.set_xticklabels([str(i) for i in range(11)])
ax1.set_yticks(range(12))
ax1.set_yticklabels([str(i) for i in range(12)])
# Mark diagonal band
for k in range(12):
    for li in diagonal_indices(k):
        ax1.add_patch(plt.Rectangle((li - 0.5, k - 0.5), 1, 1, fill=False,
                                    edgecolor='black', linewidth=0.8))
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

abs_np = np.abs(delta_np)
im2 = ax2.imshow(abs_np, cmap='viridis', aspect='auto')
ax2.set_xlabel("κ index l (layer l+1)")
ax2.set_ylabel("swap location k (block index)")
ax2.set_title(f"|Δκ[k, l]| (verdict: {verdict})")
ax2.set_xticks(range(11))
ax2.set_xticklabels([str(i) for i in range(11)])
ax2.set_yticks(range(12))
ax2.set_yticklabels([str(i) for i in range(12)])
for k in range(12):
    for li in diagonal_indices(k):
        ax2.add_patch(plt.Rectangle((li - 0.5, k - 0.5), 1, 1, fill=False,
                                    edgecolor='white', linewidth=0.8))
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

fig.suptitle(f"Cycle 264 — Per-block partial swap κ heatmap (N={N_EVAL}×{SEQ_LEN}) — {verdict}",
             fontsize=12)
plt.tight_layout()
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
print(f"Plot saved to {PLOT_PATH}", flush=True)


# ── Save JSON ────────────────────────────────────────────────────────────────
result = {
    "cycle": 264,
    "item": 274,
    "verdict": verdict,
    "config": {
        "model": "gpt2",
        "n_eval": N_EVAL,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "device": DEVICE,
    },
    "baseline_ppl_calib_slice": float(baseline_ppl),
    "post_restore_baseline_ppl": float(post_restore_ppl),
    "patched_ppls_per_swap": patched_ppls,
    "kappa_baseline_layers_1_to_11": kappa_baseline,
    "kappa_per_swap_at_k": kappa_per_swap,
    "delta_kappa_matrix": delta_kappa,
    "abs_delta_kappa_matrix": abs_delta,
    "l1_norm_per_swap": l1_per_swap,
    "l1_ranking_descending": ranking,
    "diagonal_band_definition": "for swap k: κ indices l ∈ {k-1, k, k+1} ∩ {0..10}",
    "kappa_index_to_layer": "κ index l corresponds to layer l+1 (κ_1..κ_11)",
    "verdict_logic": {
        "max_abs_delta_diag": float(max_diag),
        "median_abs_delta_offdiag": float(median_offdiag),
        "min_abs_delta_offdiag": float(min_offdiag),
        "diag_over_offdiag_ratio": float(diag_to_offdiag_ratio),
        "diag_threshold_for_clause1": 3.0,
        "clause1_diag_band_dominant": bool(clause_diag),
        "late_cols_indices": late_cols,
        "late_cols_monotone_results": monotonic_results,
        "n_late_cols_monotone_increasing_as_k_decreases": int(n_strict_monotone_decreasing_in_k),
        "n_late_cols_threshold_for_clause2": 3,
        "clause2_late_monotone": bool(clause_late),
        "confirmed_iff": "clause1_diag AND clause2_late",
        "confirmed": bool(confirmed),
        "refuted_structureless_iff": "diag_over_offdiag_ratio<1.0",
        "refuted_antidiagonal_iff": "max_diag<median_offdiag",
        "refuted": bool(refuted),
    },
    "b_leak_stats_per_block": b_leak_stats,
    "plot_path": PLOT_PATH,
}
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved {RESULT_PATH}", flush=True)


# ── Save summary md ──────────────────────────────────────────────────────────
def fmt_kappa_row(row):
    return " ".join(f"{v:+.3e}" for v in row)


summary_lines = [
    f"# Cycle 264 — Per-block partial swap κ heatmap (item 274)",
    "",
    f"**Verdict: {verdict}**",
    "",
    f"For each k ∈ {{0..11}}, install ReLU + per-channel b_leak (cycle 258 recipe) "
    f"ONLY at GPT-2 small block k, leave the other 11 blocks GELU, compute the κ profile "
    f"under G = U^T U on {N_EVAL}×{SEQ_LEN} wikitext-2 calib (seed {SEED}). "
    f"κ via `aleph.pullback_geometry.compute_curvature(reduce='mean')` (11 values for layers 1..11).",
    "",
    f"- Baseline (all-GELU) PPL = {baseline_ppl:.4f}",
    f"- Post-restore PPL = {post_restore_ppl:.4f} (round-trip clean)",
    f"- max|Δκ|_diag = {max_diag:.4e}",
    f"- median|Δκ|_offdiag = {median_offdiag:.4e}",
    f"- diag/offdiag ratio = {diag_to_offdiag_ratio:.3f} (need ≥3.0 for CONFIRMED clause 1)",
    f"- late-layer monotone cols (l∈{{7..10}}): {n_strict_monotone_decreasing_in_k}/4 "
    f"(need ≥3 for CONFIRMED clause 2)",
    "",
    f"**‖Δκ‖₁ per swap location k (descending):**",
    "",
    "| rank | swap_at_k | ‖Δκ‖₁ | patched PPL |",
    "|---|---|---|---|",
]
for r, k in enumerate(ranking):
    summary_lines.append(f"| {r+1} | {k} | {l1_per_swap[k]:.4e} | {patched_ppls[k]:.4f} |")

summary_lines += [
    "",
    "**κ_baseline (layers 1..11):**",
    "",
    f"`{fmt_kappa_row(kappa_baseline)}`",
    "",
    "**Δκ matrix (rows = swap_at_k, columns = κ index l = layer l+1):**",
    "",
    "| k\\l | " + " | ".join(str(i) for i in range(11)) + " |",
    "|" + "---|" * 12,
]
for k in range(12):
    summary_lines.append(
        f"| {k} | " + " | ".join(f"{v:+.2e}" for v in delta_kappa[k]) + " |"
    )

summary_lines += [
    "",
    f"**Verdict logic.** CONFIRMED requires (a) max|Δκ| on the diagonal band ≥ 3× the off-diagonal "
    f"median AND (b) late-layer columns (l ∈ {{7..10}}) show |Δκ| monotonically non-decreasing as "
    f"swap location k decreases through k=6..0. Diagonal band entries: for swap_at_k, the three κ "
    f"indices {{k-1, k, k+1}} ∩ {{0..10}} (k=0 → {{0,1}}; k=11 → {{10,11}} ∩ valid = {{10}}).",
    "",
    f"- Clause 1 (diagonal dominance ≥ 3×): **{'PASS' if clause_diag else 'FAIL'}** "
    f"(ratio={diag_to_offdiag_ratio:.3f})",
    f"- Clause 2 (late-layer monotone): **{'PASS' if clause_late else 'FAIL'}** "
    f"({n_strict_monotone_decreasing_in_k}/4 cols pass)",
    f"- REFUTED triggers if `diag/offdiag<1.0` OR max_diag<median_offdiag.",
    f"- REFUTED structureless: {refuted_structureless}",
    f"- REFUTED antidiagonal:  {refuted_antidiagonal}",
    "",
    f"**Result: {verdict}**.",
    "",
    f"Heatmap: `{PLOT_PATH}`. Full Δκ matrix and per-swap κ profiles saved in JSON.",
]

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"Saved {SUMMARY_PATH}", flush=True)
print("\nDone.", flush=True)
