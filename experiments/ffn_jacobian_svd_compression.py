"""
Cycle 071: Spectral compression (Jacobian SVD basis) vs channel selection (topk) — GPT-2 FFN.

Compare four methods at k=512 on single-block swap PPL for blocks [1, 4, 7, 11]:
  (a) topk+lstsq  — existing channel selection pipeline
  (b) SVD-only    — project output onto U_k subspace (no refit)
  (c) SVD+lstsq   — refit output matrix B in U_k basis
  (d) Random+lstsq — random orthogonal basis + refit (null baseline)
"""
import copy
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
    derive_ffn_compression,
    swap_compressed_ffns,
)

OUTPUT_DIR = ROOT / "autoresearch" / "results"
DATA_PATH = ROOT / "data" / "shakespeare.txt"

SEEDS = [42, 123, 7]
K = 512           # budget: keep 512 output directions (out of 768)
SEQ_LEN = 128
PROBE_SEQUENCES = 8    # 8×128 = 1024 probe tokens for lstsq refit
EVAL_SEQUENCES = 200   # eval corpus fixed seed=0
BLOCKS = [1, 4, 7, 11]
DEVICE = "cpu"

# topk+lstsq: hidden_fraction that gives ~512 hidden units out of 3072 → 512/3072 ≈ 0.167
HIDDEN_FRACTION = 512 / 3072  # ≈ 0.1667


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _standard_weight(module: nn.Module) -> torch.Tensor:
    """Return (out_features, in_features) weight regardless of Conv1D vs Linear."""
    w = module.weight.detach().float()
    if module.__class__.__name__ == "Conv1D":
        return w.t().contiguous()
    return w.contiguous()


def _gelu_grad_autograd(z: torch.Tensor) -> torch.Tensor:
    """Element-wise GELU'(z) via autograd — correct for GPT-2 tanh approximation."""
    z_req = z.detach().requires_grad_(True)
    out = F.gelu(z_req, approximate="tanh")
    out.sum().backward()
    return z_req.grad.detach()


def _mean_jacobian_svd(block_idx: int, model: nn.Module, x_probe: torch.Tensor, k: int):
    """
    Compute J̄ = mean_t( W_out @ diag(GELU'(z_t)) @ W_in ) and return U_k (768×k).

    x_probe: (T, 768) float FFN input tokens
    Returns U_k: (768, k) — top-k left singular vectors of J̄
    """
    mlp = model.transformer.h[block_idx].mlp
    W_in = _standard_weight(mlp.c_fc)   # (3072, 768)
    W_out = _standard_weight(mlp.c_proj)  # (768, 3072)

    # Compute z = x @ W_in.T + b_fc for all tokens
    b_fc = mlp.c_fc.bias.detach().float() if mlp.c_fc.bias is not None else None
    z = x_probe @ W_in.t()  # (T, 3072)
    if b_fc is not None:
        z = z + b_fc

    T = x_probe.shape[0]
    J_sum = torch.zeros(768, 768, dtype=torch.float32)

    for t in range(T):
        z_t = z[t]  # (3072,)
        g_t = _gelu_grad_autograd(z_t)  # (3072,)
        # J_t = W_out @ diag(g_t) @ W_in
        # = (W_out * g_t.unsqueeze(0)) @ W_in   — (768, 3072) * (1, 3072) → (768, 768)
        J_t = (W_out * g_t.unsqueeze(0)) @ W_in  # (768, 768)
        J_sum.add_(J_t)

    J_mean = J_sum / T  # (768, 768)

    # SVD — only need left singular vectors
    U, S, Vh = torch.linalg.svd(J_mean, full_matrices=False)  # U: (768, 768)
    U_k = U[:, :k].contiguous()  # (768, k)
    return U_k, S


def _random_orthonormal(n: int, k: int, seed: int) -> torch.Tensor:
    """Return a random orthonormal Q_k (n×k)."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    A = torch.randn(n, k, generator=gen)
    Q, _ = torch.linalg.qr(A)
    return Q.contiguous()  # (n, k)


def _lstsq_refit(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Solve B @ features.T ≈ targets.T via lstsq. Returns B (out_dim × feat_dim)."""
    # features: (N, k), targets: (N, out_dim=768)
    # min ||features @ B.T - targets||_F → B = (features^+ @ targets).T
    sol = torch.linalg.lstsq(features, targets, driver="gelsd")
    return sol.solution.t().contiguous()  # (768, k)


def _collect_ffn_inputs_outputs(block_idx: int, model: nn.Module,
                                 token_ids: torch.Tensor, batch_size: int = 4):
    """Run forward pass, collect FFN inputs and outputs for given block."""
    x_list = []
    y_list = []

    def hook_fn(module, inputs, output):
        x_list.append(inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]))
        y_list.append(output.detach().float().reshape(-1, output.shape[-1]))

    mlp = model.transformer.h[block_idx].mlp
    handle = mlp.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            N = token_ids.shape[0]
            for i in range(0, N, batch_size):
                batch = token_ids[i:i + batch_size].to(DEVICE)
                model(batch, use_cache=False)
    finally:
        handle.remove()

    x = torch.cat(x_list, dim=0)  # (T, 768)
    y = torch.cat(y_list, dim=0)  # (T, 768)
    return x, y


def _eval_ppl_with_hook(model: nn.Module, block_idx: int, hook_fn, eval_tokens: torch.Tensor,
                         batch_size: int = 4) -> float:
    """Evaluate PPL with a forward hook on block block_idx's mlp replacing output."""
    mlp = model.transformer.h[block_idx].mlp
    handle = mlp.register_forward_hook(hook_fn)
    total_loss = 0.0
    total_tokens = 0
    try:
        with torch.no_grad():
            N = eval_tokens.shape[0]
            for i in range(0, N, batch_size):
                batch = eval_tokens[i:i + batch_size].to(DEVICE)  # (B, L)
                targets = batch[:, 1:].contiguous()
                logits = model(batch, use_cache=False).logits  # (B, L, V)
                logits = logits[:, :-1, :].contiguous()
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += targets.numel()
    finally:
        handle.remove()
    mean_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(mean_loss, 20.0))


def _eval_ppl_baseline(model: nn.Module, eval_tokens: torch.Tensor, batch_size: int = 4) -> float:
    """Uncompressed baseline PPL."""
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        N = eval_tokens.shape[0]
        for i in range(0, N, batch_size):
            batch = eval_tokens[i:i + batch_size].to(DEVICE)
            targets = batch[:, 1:].contiguous()
            logits = model(batch, use_cache=False).logits
            logits = logits[:, :-1, :].contiguous()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
    mean_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(mean_loss, 20.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("Loading GPT-2...")
t0 = time.time()
teacher = load_nanochat_teacher(TeacherLoadSpec(
    source="hf_pretrained",
    hf_model_name="gpt2",
    device=DEVICE,
))
model = teacher.model
tokenizer = teacher.tokenizer
model.eval()
print(f"  loaded in {time.time()-t0:.1f}s")

# Fixed eval corpus (seed=0)
print("Building fixed eval corpus...")
eval_corpus = build_probe_corpus(ProbeCorpusSpec(
    text_path=str(DATA_PATH),
    sequence_length=SEQ_LEN,
    probe_train_sequences=PROBE_SEQUENCES,
    probe_holdout_sequences=1,
    eval_val_sequences=EVAL_SEQUENCES,
    tokenizer=tokenizer,
    shuffle_seed=0,
    dataset_name="shakespeare",
))
eval_tokens = eval_corpus.eval_val  # (200, 128)
print(f"  eval_tokens shape: {eval_tokens.shape}")

# Baseline PPL (uncompressed)
print("Computing baseline PPL...")
baseline_ppl = _eval_ppl_baseline(model, eval_tokens)
print(f"  Baseline PPL: {baseline_ppl:.3f}")

# Source FFN param count (single block)
block_0_mlp = model.transformer.h[0].mlp
w_fc = _standard_weight(block_0_mlp.c_fc)   # (3072, 768)
w_proj = _standard_weight(block_0_mlp.c_proj)  # (768, 3072)
source_params_per_block = w_fc.numel() + w_proj.numel()
if block_0_mlp.c_fc.bias is not None:
    source_params_per_block += block_0_mlp.c_fc.bias.numel()
if block_0_mlp.c_proj.bias is not None:
    source_params_per_block += block_0_mlp.c_proj.bias.numel()
print(f"  Source params/block: {source_params_per_block:,}")

# SVD+lstsq / Random compressed params: B(768×k) + [U_k stored as hook state = not model params]
# topk: (k×768) + bias + (768×k) + bias
topk_params = K * 768 + K + 768 * K + 768  # ~789,504
svd_params = 768 * K + 768  # B (768×k) + bias-like offset — no new W_in needed (uses original)
# For fair comparison, report topk params as compressed model params
print(f"  topk k={K}: ~{topk_params:,} params/block")
print(f"  SVD+lstsq k={K}: B matrix = {768*K:,} params/block (+ original W_in, W_out reused)")

# Results storage: per_block[seed][method] = ppl
results = {
    "baseline_ppl": baseline_ppl,
    "source_params_per_block": source_params_per_block,
    "k": K,
    "blocks": BLOCKS,
    "seeds": SEEDS,
    "method_params": {
        "topk_lstsq": topk_params,
        "svd_only": 0,           # no extra params, just projection
        "svd_lstsq": 768 * K,    # B matrix
        "random_lstsq": 768 * K, # B matrix
    },
    "per_block": {},
}

for block_idx in BLOCKS:
    print(f"\n=== Block {block_idx} ===")
    results["per_block"][block_idx] = {}

    for seed in SEEDS:
        print(f"  Seed {seed}...")
        torch.manual_seed(seed)
        seed_t0 = time.time()

        # Build probe corpus for this seed
        probe_corpus = build_probe_corpus(ProbeCorpusSpec(
            text_path=str(DATA_PATH),
            sequence_length=SEQ_LEN,
            probe_train_sequences=PROBE_SEQUENCES,
            probe_holdout_sequences=1,
            eval_val_sequences=4,
            tokenizer=tokenizer,
            shuffle_seed=seed,
            dataset_name="shakespeare",
        ))
        probe_tokens = probe_corpus.probe_train  # (8, 128) → 1024 tokens

        # Collect FFN inputs / outputs on probe set
        x_probe, y_probe = _collect_ffn_inputs_outputs(block_idx, model, probe_tokens)
        print(f"    Probe: x={x_probe.shape}, y={y_probe.shape}")

        # -----------------------------------------------------------------
        # Method (a): topk+lstsq via existing pipeline
        # Budget: hidden_fraction gives k≈512 hidden units
        # -----------------------------------------------------------------
        budget = CompressionBudgetSpec(
            hidden_fraction=HIDDEN_FRACTION,
            min_hidden=K,
            max_hidden=K,
        )
        # Build atlas using the probe corpus
        from aleph.lm_geometric_compression import probe_ffn_geometry, derive_ffn_compression, swap_compressed_ffns, _evaluate_model
        atlas = probe_ffn_geometry(
            teacher,
            probe_corpus,
            layer_ids=(block_idx,),
            max_tokens_per_layer=1024,
            batch_size=4,
        )
        pack = derive_ffn_compression(atlas, budget)
        student_model = swap_compressed_ffns(teacher, pack, inplace=False)
        metrics_a = _evaluate_model(teacher, student_model, eval_corpus, batch_size=4)
        ppl_a = metrics_a["perplexity"]
        print(f"    (a) topk+lstsq PPL: {ppl_a:.3f}")

        # -----------------------------------------------------------------
        # Method (b): SVD-only — compute mean Jacobian SVD, project output
        # -----------------------------------------------------------------
        U_k, sing_vals = _mean_jacobian_svd(block_idx, model, x_probe, K)
        # U_k: (768, k)

        def make_svd_only_hook(U_k):
            def hook_fn(module, inputs, output):
                y = output.float()
                # Project onto U_k subspace
                y_approx = (y @ U_k) @ U_k.t()
                return y_approx.to(output.dtype)
            return hook_fn

        ppl_b = _eval_ppl_with_hook(model, block_idx, make_svd_only_hook(U_k), eval_tokens)
        print(f"    (b) SVD-only PPL: {ppl_b:.3f}")

        # -----------------------------------------------------------------
        # Method (c): SVD+lstsq — refit B such that y ≈ B @ (U_k^T @ y_float)
        # features = y_probe @ U_k  (T×k), targets = y_probe (T×768)
        # -----------------------------------------------------------------
        features = y_probe @ U_k  # (T, k) — project actual FFN outputs
        B = _lstsq_refit(features, y_probe)  # B: (768, k)
        print(f"    (c) SVD+lstsq: B shape = {B.shape}, residual = {(y_probe - features @ B.t()).norm():.4f}")

        def make_svd_lstsq_hook(U_k, B):
            def hook_fn(module, inputs, output):
                y = output.float()
                y_approx = (y @ U_k) @ B.t()
                return y_approx.to(output.dtype)
            return hook_fn

        ppl_c = _eval_ppl_with_hook(model, block_idx, make_svd_lstsq_hook(U_k, B), eval_tokens)
        print(f"    (c) SVD+lstsq PPL: {ppl_c:.3f}")

        # -----------------------------------------------------------------
        # Method (d): Random orthonormal Q_k + lstsq refit
        # -----------------------------------------------------------------
        Q_k = _random_orthonormal(768, K, seed=seed * 100 + block_idx)  # (768, k)
        features_r = y_probe @ Q_k  # (T, k)
        B_r = _lstsq_refit(features_r, y_probe)  # (768, k)

        def make_random_lstsq_hook(Q_k, B_r):
            def hook_fn(module, inputs, output):
                y = output.float()
                y_approx = (y @ Q_k) @ B_r.t()
                return y_approx.to(output.dtype)
            return hook_fn

        ppl_d = _eval_ppl_with_hook(model, block_idx, make_random_lstsq_hook(Q_k, B_r), eval_tokens)
        print(f"    (d) Random+lstsq PPL: {ppl_d:.3f}")

        results["per_block"][block_idx][seed] = {
            "topk_lstsq": ppl_a,
            "svd_only": ppl_b,
            "svd_lstsq": ppl_c,
            "random_lstsq": ppl_d,
            "singular_values_top5": sing_vals[:5].tolist(),
        }
        print(f"    Seed {seed} done in {time.time()-seed_t0:.1f}s")

# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
METHODS = ["topk_lstsq", "svd_only", "svd_lstsq", "random_lstsq"]

print("\n\n=== RESULTS SUMMARY ===")
print(f"Baseline PPL: {baseline_ppl:.3f}")
print(f"\n{'Method':<16} | " + " | ".join(f"Block {b:2d}" for b in BLOCKS) + " | Mean")
print("-" * 80)

agg = {}  # method → {block_idx: mean_ppl}
for method in METHODS:
    block_means = []
    row = []
    for block_idx in BLOCKS:
        ppls = [results["per_block"][block_idx][s][method] for s in SEEDS]
        m = sum(ppls) / len(ppls)
        std = (sum((p - m)**2 for p in ppls) / len(ppls))**0.5
        block_means.append(m)
        row.append(f"{m:.2f}±{std:.2f}")
        agg.setdefault(method, {})[block_idx] = {"mean": m, "std": std, "ppls": ppls}
    mean_all = sum(block_means) / len(block_means)
    print(f"{method:<16} | " + " | ".join(row) + f" | {mean_all:.2f}")

results["aggregated"] = agg

# Verdict
print("\n=== VERDICT ===")
svd_lstsq_means = [agg["svd_lstsq"][b]["mean"] for b in BLOCKS]
topk_means = [agg["topk_lstsq"][b]["mean"] for b in BLOCKS]

wins_svd = sum(1 for s, t in zip(svd_lstsq_means, topk_means)
               if (s - t) / t < -0.03)  # SVD wins by ≥3% relative
wins_topk = sum(1 for s, t in zip(svd_lstsq_means, topk_means)
                if (s - t) / t > 0)  # topk is lower PPL (better)

rel_diffs = [(s - t) / t * 100 for s, t in zip(svd_lstsq_means, topk_means)]
print(f"Relative PPL diff (SVD+lstsq vs topk+lstsq) per block: {[f'{d:+.1f}%' for d in rel_diffs]}")
print(f"SVD+lstsq wins (≥3% relative) in {wins_svd}/4 blocks")
print(f"topk+lstsq wins (lower PPL) in {wins_topk}/4 blocks")

if wins_svd >= 2:
    verdict = "CONFIRMED"
elif wins_topk == 4:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
print(f"Verdict: {verdict}")
results["verdict"] = verdict
results["verdict_details"] = {
    "wins_svd_ge3pct": wins_svd,
    "wins_topk": wins_topk,
    "rel_diffs_pct": rel_diffs,
}

# Save JSON
out_json = OUTPUT_DIR / "cycle_071_jacobian_svd_basis.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

x_pos = range(len(BLOCKS))
width = 0.2
colors = {"topk_lstsq": "#2196F3", "svd_only": "#FF9800", "svd_lstsq": "#4CAF50", "random_lstsq": "#9E9E9E"}
offsets = [-1.5, -0.5, 0.5, 1.5]
labels = {"topk_lstsq": "topk+lstsq", "svd_only": "SVD-only", "svd_lstsq": "SVD+lstsq", "random_lstsq": "Random+lstsq"}

for i, method in enumerate(METHODS):
    means = [agg[method][b]["mean"] for b in BLOCKS]
    stds = [agg[method][b]["std"] for b in BLOCKS]
    xs = [x + offsets[i] * width for x in x_pos]
    ax.bar(xs, means, width=width * 0.9, yerr=stds, capsize=3,
           color=colors[method], label=labels[method], alpha=0.85)

ax.axhline(baseline_ppl, color="black", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline_ppl:.1f})")
ax.set_xticks(list(x_pos))
ax.set_xticklabels([f"Block {b}" for b in BLOCKS])
ax.set_ylabel("Perplexity (lower = better)")
ax.set_title(f"Jacobian SVD vs Channel Selection (k={K}, 3 seeds)\nSingle-block swap PPL — GPT-2 Shakespeare")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

out_png = OUTPUT_DIR / "cycle_071_jacobian_svd_basis.png"
plt.savefig(out_png, dpi=150)
print(f"Saved: {out_png}")
print("\nDone.")
