"""Cycle 039: Inter-block residual-stream lstsq correction — GPT-2 FFN.

Fits affine correction at the residual stream after the last compressed block,
measuring PPL recovery for k=5 at 6x. Zero extra parameters at inference.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnPack,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ── Config ────────────────────────────────────────────────────────────────────
SHAKESPEARE_PATH = str(Path(__file__).parent.parent / "data" / "shakespeare.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K5_BLOCKS = [7, 5, 1, 3, 6]  # best k=5 combo from cycle 038 (drop block 4)
K4_BLOCKS = [7, 5, 4, 1]     # standard k=4 reference
HIDDEN_FRACTION = 1 / 6  # 6x
PROBE_TRAIN_SEQ = 200
PROBE_HOLDOUT_SEQ = 50
BATCH_SIZE = 4
SEED = 42


def eval_ppl(model, teacher, corpus, batch_size: int) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=batch_size)
    return float(metrics["perplexity"])


def collect_hidden_states(model, input_ids, hook_layer_idx):
    """Run model and collect hidden states at a specific block's input."""
    collected = []

    def hook_fn(module, args, kwargs=None):
        # forward pre-hook: args[0] is hidden_states
        h = args[0] if isinstance(args, tuple) else args
        collected.append(h.detach().clone())

    handle = model.transformer.h[hook_layer_idx].register_forward_pre_hook(hook_fn)
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    return torch.cat(collected, dim=0)  # [batch*n_calls, seq_len, hidden_dim]


def make_correction_hook(W_corr, b_corr):
    """Create a forward pre-hook that applies affine correction."""
    def hook_fn(module, args):
        h = args[0]  # [batch, seq, 768]
        h_corrected = h @ W_corr.T + b_corr
        return (h_corrected,) + args[1:]
    return hook_fn


# ── Main ──────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)

print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

print("Building probe corpus...")
corpus_spec = ProbeCorpusSpec(
    teacher=teacher, text_path=SHAKESPEARE_PATH, sequence_length=128,
    probe_train_sequences=PROBE_TRAIN_SEQ, probe_holdout_sequences=PROBE_HOLDOUT_SEQ,
    eval_val_sequences=PROBE_HOLDOUT_SEQ, dataset_name="proxy_shakespeare",
)
corpus = build_probe_corpus(corpus_spec)

print("Computing baseline perplexity...")
baseline_ppl = eval_ppl(teacher.model, teacher, corpus, BATCH_SIZE)
print(f"  Baseline PPL: {baseline_ppl:.2f}")

# Probe and compress all needed blocks
all_blocks = sorted(set(K5_BLOCKS + K4_BLOCKS))
print(f"Probing geometry for blocks {all_blocks}...")
atlas = probe_ffn_geometry(
    teacher, corpus, layer_ids=tuple(all_blocks),
    max_tokens_per_layer=2048, batch_size=BATCH_SIZE, jacobian_samples=8,
)

budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION, selection_mode="topk")
full_pack = derive_ffn_compression(atlas, budget)
compressed_by_id = {layer.layer_id: layer for layer in full_pack.layers}

results = {"baseline_ppl": float(baseline_ppl)}

# ── k=5 uncorrected ──────────────────────────────────────────────────────────
print(f"\n--- k=5 uncorrected (blocks {K5_BLOCKS}) ---")
k5_layers = tuple(compressed_by_id[bid] for bid in K5_BLOCKS)
k5_pack = CompressedFfnPack(
    method=full_pack.method, model_family=full_pack.model_family,
    dataset_name=full_pack.dataset_name, layers=k5_layers, metadata=full_pack.metadata,
)
k5_model = swap_compressed_ffns(teacher, k5_pack, inplace=False)
k5_ppl = eval_ppl(k5_model, teacher, corpus, BATCH_SIZE)
k5_rel = (k5_ppl - baseline_ppl) / baseline_ppl * 100
print(f"  k=5 uncorrected PPL: {k5_ppl:.2f} ({k5_rel:+.2f}%)")
results["k5_uncorrected"] = {"ppl": float(k5_ppl), "rel_pct": float(k5_rel)}

# ── Collect hidden states for correction ──────────────────────────────────────
correction_layer = max(K5_BLOCKS) + 1  # block after last compressed
print(f"\nCollecting hidden states at block {correction_layer} input...")

# Collect from float model and compressed model
probe_ids = corpus.probe_train[:, :-1]  # input tokens

h_float_all = []
h_comp_all = []

for i in range(0, len(probe_ids), BATCH_SIZE):
    batch = probe_ids[i:i+BATCH_SIZE]
    h_f = collect_hidden_states(teacher.model, batch, correction_layer)
    h_c = collect_hidden_states(k5_model, batch, correction_layer)
    h_float_all.append(h_f)
    h_comp_all.append(h_c)

h_float = torch.cat(h_float_all, dim=0).reshape(-1, 768).float()  # [N_tokens, 768]
h_comp = torch.cat(h_comp_all, dim=0).reshape(-1, 768).float()

print(f"  Collected {h_float.shape[0]} token states")
print(f"  Error before correction: MSE={F.mse_loss(h_comp, h_float).item():.6f}")

# ── Fit affine correction ─────────────────────────────────────────────────────
print("\nFitting affine correction (lstsq)...")
# h_float = h_comp @ W.T + b → augment h_comp with ones
H = torch.cat([h_comp, torch.ones(h_comp.shape[0], 1)], dim=1)  # [N, 769]
sol = torch.linalg.lstsq(H, h_float).solution  # [769, 768]
W_corr = sol[:768].T  # [768, 768]
b_corr = sol[768]     # [768]

# Diagnostics
W_delta = W_corr - torch.eye(768)
frob_norm = torch.norm(W_delta, p='fro').item()
b_norm = torch.norm(b_corr).item()
print(f"  ||W_corr - I||_F = {frob_norm:.4f}")
print(f"  ||b_corr|| = {b_norm:.4f}")

# SVD of correction
U, S, Vt = torch.linalg.svd(W_delta, full_matrices=False)
energy = torch.cumsum(S**2, 0) / (S**2).sum()
rank_90 = int((energy < 0.9).sum().item()) + 1
rank_95 = int((energy < 0.95).sum().item()) + 1
print(f"  Correction rank (90% energy): {rank_90}")
print(f"  Correction rank (95% energy): {rank_95}")
print(f"  Top 5 singular values: {S[:5].tolist()}")

# Residual after correction
h_corrected = h_comp @ W_corr.T + b_corr
mse_after = F.mse_loss(h_corrected, h_float).item()
mse_before = F.mse_loss(h_comp, h_float).item()
print(f"  MSE before correction: {mse_before:.6f}")
print(f"  MSE after correction: {mse_after:.6f}")
print(f"  MSE reduction: {(1 - mse_after/mse_before)*100:.1f}%")

results["correction"] = {
    "frob_norm_W_delta": frob_norm,
    "b_norm": b_norm,
    "rank_90": rank_90,
    "rank_95": rank_95,
    "top_5_sv": S[:5].tolist(),
    "mse_before": mse_before,
    "mse_after": mse_after,
    "mse_reduction_pct": (1 - mse_after/mse_before)*100,
}

# ── Apply correction and evaluate k=5 ────────────────────────────────────────
print(f"\n--- k=5 corrected (hook at block {correction_layer}) ---")
hook = k5_model.transformer.h[correction_layer].register_forward_pre_hook(
    make_correction_hook(W_corr, b_corr)
)
k5_corr_ppl = eval_ppl(k5_model, teacher, corpus, BATCH_SIZE)
k5_corr_rel = (k5_corr_ppl - baseline_ppl) / baseline_ppl * 100
hook.remove()

recovery_pp = k5_rel - k5_corr_rel  # positive = improvement
print(f"  k=5 corrected PPL: {k5_corr_ppl:.2f} ({k5_corr_rel:+.2f}%)")
print(f"  Recovery: {recovery_pp:.2f}pp")

results["k5_corrected"] = {
    "ppl": float(k5_corr_ppl), "rel_pct": float(k5_corr_rel),
    "recovery_pp": float(recovery_pp),
}

# ── k=4 reference and correction ─────────────────────────────────────────────
print(f"\n--- k=4 uncorrected (blocks {K4_BLOCKS}) ---")
k4_layers = tuple(compressed_by_id[bid] for bid in K4_BLOCKS)
k4_pack = CompressedFfnPack(
    method=full_pack.method, model_family=full_pack.model_family,
    dataset_name=full_pack.dataset_name, layers=k4_layers, metadata=full_pack.metadata,
)
k4_model = swap_compressed_ffns(teacher, k4_pack, inplace=False)
k4_ppl = eval_ppl(k4_model, teacher, corpus, BATCH_SIZE)
k4_rel = (k4_ppl - baseline_ppl) / baseline_ppl * 100
print(f"  k=4 uncorrected PPL: {k4_ppl:.2f} ({k4_rel:+.2f}%)")

# Also fit correction for k=4 (at block max(K4_BLOCKS)+1 = 8)
k4_correction_layer = max(K4_BLOCKS) + 1  # block 8
h_f4_all, h_c4_all = [], []
for i in range(0, len(probe_ids), BATCH_SIZE):
    batch = probe_ids[i:i+BATCH_SIZE]
    h_f4 = collect_hidden_states(teacher.model, batch, k4_correction_layer)
    h_c4 = collect_hidden_states(k4_model, batch, k4_correction_layer)
    h_f4_all.append(h_f4)
    h_c4_all.append(h_c4)

h_f4 = torch.cat(h_f4_all, dim=0).reshape(-1, 768).float()
h_c4 = torch.cat(h_c4_all, dim=0).reshape(-1, 768).float()
H4 = torch.cat([h_c4, torch.ones(h_c4.shape[0], 1)], dim=1)
sol4 = torch.linalg.lstsq(H4, h_f4).solution
W4 = sol4[:768].T
b4 = sol4[768]

hook4 = k4_model.transformer.h[k4_correction_layer].register_forward_pre_hook(
    make_correction_hook(W4, b4)
)
k4_corr_ppl = eval_ppl(k4_model, teacher, corpus, BATCH_SIZE)
k4_corr_rel = (k4_corr_ppl - baseline_ppl) / baseline_ppl * 100
hook4.remove()
k4_recovery = k4_rel - k4_corr_rel

print(f"  k=4 corrected PPL: {k4_corr_ppl:.2f} ({k4_corr_rel:+.2f}%)")
print(f"  k=4 recovery: {k4_recovery:.2f}pp")

results["k4_uncorrected"] = {"ppl": float(k4_ppl), "rel_pct": float(k4_rel)}
results["k4_corrected"] = {
    "ppl": float(k4_corr_ppl), "rel_pct": float(k4_corr_rel),
    "recovery_pp": float(k4_recovery),
}

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Baseline:       {baseline_ppl:.2f}")
print(f"  k=4 uncorr:     {k4_ppl:.2f} ({k4_rel:+.2f}%)")
print(f"  k=4 corrected:  {k4_corr_ppl:.2f} ({k4_corr_rel:+.2f}%) recovery={k4_recovery:+.2f}pp")
print(f"  k=5 uncorr:     {k5_ppl:.2f} ({k5_rel:+.2f}%)")
print(f"  k=5 corrected:  {k5_corr_ppl:.2f} ({k5_corr_rel:+.2f}%) recovery={recovery_pp:+.2f}pp")

k4_worsened = k4_corr_rel > k4_rel + 1.0  # overcorrection check

if recovery_pp >= 3.0 and not k4_worsened:
    verdict = "confirmed"
    print(f"\nVerdict: CONFIRMED — k=5 correction recovers {recovery_pp:.1f}pp (≥3pp)")
elif recovery_pp < 1.0:
    verdict = "refuted"
    print(f"\nVerdict: REFUTED — correction recovers only {recovery_pp:.1f}pp (<1pp)")
elif k4_worsened:
    verdict = "inconclusive"
    print(f"\nVerdict: INCONCLUSIVE — k=5 recovers {recovery_pp:.1f}pp but k=4 worsens by {k4_rel - k4_corr_rel:.1f}pp")
else:
    verdict = "inconclusive"
    print(f"\nVerdict: INCONCLUSIVE — recovery {recovery_pp:.1f}pp (between 1-3pp)")

results["verdict"] = verdict

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

configs = ["Baseline", "k=4", "k=4\ncorrected", "k=5", "k=5\ncorrected"]
ppls = [baseline_ppl, k4_ppl, k4_corr_ppl, k5_ppl, k5_corr_ppl]
rels = [0, k4_rel, k4_corr_rel, k5_rel, k5_corr_rel]
colors = ["grey", "steelblue", "lightblue", "coral", "salmon"]

x = np.arange(len(configs))
bars = ax.bar(x, rels, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

for i, (r, p) in enumerate(zip(rels, ppls)):
    ax.annotate(f"{r:+.1f}%\n({p:.0f})", (i, r),
                textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9)

ax.axhline(0, color="black", ls="-", alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10)
ax.set_ylabel("Relative PPL change (%)")
ax.set_title(f"Cycle 039: Residual-Stream lstsq Correction\n"
             f"k=5 recovery: {recovery_pp:+.1f}pp  |  k=4 change: {k4_recovery:+.1f}pp  |  "
             f"Verdict: {verdict.upper()}")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_039_residual_correction.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")

out_json = RESULTS_DIR / "cycle_039_residual_correction.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out_json}")

del k5_model, k4_model
