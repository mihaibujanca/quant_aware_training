"""
Cycle 077: SVD vs topk vs random basis at aggressive compression (k/n <= 0.3) — GPT-2 FFN.

Re-run cycle 071 comparison at k/n in {0.3, 0.2, 0.1} on blocks [1, 4, 7, 11] with 3 seeds.
Three methods per k:
  (a) topk+lstsq  — channel selection pipeline
  (b) SVD+lstsq   — project output onto U_k subspace, refit
  (c) random+lstsq — random orthogonal Q_k basis + refit (null baseline)

k values: k=230 (k/n=0.3), k=154 (k/n=0.2), k=77 (k/n=0.1), where n=768.
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
    _evaluate_model,
)

OUTPUT_DIR = ROOT / "autoresearch" / "results"
DATA_PATH = ROOT / "data" / "shakespeare.txt"

SEEDS = [42, 123, 7]
# k/n ratios: 0.3, 0.2, 0.1 — output dim n=768
K_VALUES = [230, 154, 77]  # round(0.3*768)=230, round(0.2*768)=154, round(0.1*768)=77
KN_RATIOS = [0.3, 0.2, 0.1]
SEQ_LEN = 128
PROBE_SEQUENCES = 8    # 8x128 = 1024 probe tokens for lstsq refit
EVAL_SEQUENCES = 200   # eval corpus fixed seed=0
BLOCKS = [1, 4, 7, 11]
DEVICE = "cpu"


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


def _mean_jacobian_svd(block_idx: int, model: nn.Module, x_probe: torch.Tensor, k_max: int):
    """
    Compute J̄ = mean_t( W_out @ diag(GELU'(z_t)) @ W_in ) and return U (768×768) + S.

    x_probe: (T, 768) float FFN input tokens
    Returns U: (768, 768) full left singular vectors, S: singular values
    Caller slices U[:, :k] for each k.
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
        J_t = (W_out * g_t.unsqueeze(0)) @ W_in  # (768, 768)
        J_sum.add_(J_t)

    J_mean = J_sum / T  # (768, 768)

    # Full SVD — keep all left singular vectors
    U, S, Vh = torch.linalg.svd(J_mean, full_matrices=False)  # U: (768, 768)
    return U.contiguous(), S


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

# Results storage: per_kn[kn_str][block_idx][seed][method] = ppl
results = {
    "baseline_ppl": baseline_ppl,
    "k_values": K_VALUES,
    "kn_ratios": KN_RATIOS,
    "blocks": BLOCKS,
    "seeds": SEEDS,
    "per_kn": {},
}

for k, kn in zip(K_VALUES, KN_RATIOS):
    kn_str = str(kn)
    print(f"\n{'='*60}")
    print(f"=== k/n = {kn} (k={k}) ===")
    print(f"{'='*60}")
    results["per_kn"][kn_str] = {}

    # topk hidden_fraction: k hidden units out of 3072
    hidden_fraction = k / 3072

    for block_idx in BLOCKS:
        print(f"\n--- Block {block_idx} ---")
        results["per_kn"][kn_str][block_idx] = {}

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
            probe_tokens = probe_corpus.probe_train  # (8, 128) -> 1024 tokens

            # Collect FFN inputs / outputs on probe set
            x_probe, y_probe = _collect_ffn_inputs_outputs(block_idx, model, probe_tokens)

            # -----------------------------------------------------------------
            # Method (a): topk+lstsq via existing pipeline
            # -----------------------------------------------------------------
            budget = CompressionBudgetSpec(
                hidden_fraction=hidden_fraction,
                min_hidden=k,
                max_hidden=k,
            )
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
            # Method (b): SVD+lstsq — compute mean Jacobian SVD, refit
            # -----------------------------------------------------------------
            U_full, sing_vals = _mean_jacobian_svd(block_idx, model, x_probe, k)
            U_k = U_full[:, :k].contiguous()  # (768, k)

            features_svd = y_probe @ U_k  # (T, k)
            B_svd = _lstsq_refit(features_svd, y_probe)  # (768, k)

            def make_svd_lstsq_hook(U_k, B):
                def hook_fn(module, inputs, output):
                    y = output.float()
                    y_approx = (y @ U_k) @ B.t()
                    return y_approx.to(output.dtype)
                return hook_fn

            ppl_b = _eval_ppl_with_hook(model, block_idx, make_svd_lstsq_hook(U_k, B_svd), eval_tokens)
            print(f"    (b) SVD+lstsq PPL: {ppl_b:.3f}")

            # -----------------------------------------------------------------
            # Method (c): Random orthonormal Q_k + lstsq refit
            # -----------------------------------------------------------------
            Q_k = _random_orthonormal(768, k, seed=seed * 100 + block_idx)
            features_r = y_probe @ Q_k  # (T, k)
            B_r = _lstsq_refit(features_r, y_probe)  # (768, k)

            def make_random_lstsq_hook(Q_k, B_r):
                def hook_fn(module, inputs, output):
                    y = output.float()
                    y_approx = (y @ Q_k) @ B_r.t()
                    return y_approx.to(output.dtype)
                return hook_fn

            ppl_c = _eval_ppl_with_hook(model, block_idx, make_random_lstsq_hook(Q_k, B_r), eval_tokens)
            print(f"    (c) random+lstsq PPL: {ppl_c:.3f}")

            results["per_kn"][kn_str][block_idx][seed] = {
                "topk_lstsq": ppl_a,
                "svd_lstsq": ppl_b,
                "random_lstsq": ppl_c,
                "singular_values_top5": sing_vals[:5].tolist(),
            }
            print(f"    Seed {seed} done in {time.time()-seed_t0:.1f}s")

# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
METHODS = ["topk_lstsq", "svd_lstsq", "random_lstsq"]

print("\n\n=== RESULTS SUMMARY ===")
print(f"Baseline PPL: {baseline_ppl:.3f}")

agg = {}  # kn_str -> method -> block_idx -> {mean, std, ppls}
for k, kn in zip(K_VALUES, KN_RATIOS):
    kn_str = str(kn)
    agg[kn_str] = {}
    print(f"\n--- k/n={kn} (k={k}) ---")
    print(f"{'Method':<16} | " + " | ".join(f"Block {b:2d}" for b in BLOCKS) + " | Mean")
    print("-" * 75)
    for method in METHODS:
        block_means = []
        row = []
        for block_idx in BLOCKS:
            ppls = [results["per_kn"][kn_str][block_idx][s][method] for s in SEEDS]
            m = sum(ppls) / len(ppls)
            std = (sum((p - m)**2 for p in ppls) / len(ppls))**0.5
            block_means.append(m)
            row.append(f"{m:.1f}±{std:.1f}")
            agg[kn_str].setdefault(method, {})[block_idx] = {"mean": m, "std": std, "ppls": ppls}
        mean_all = sum(block_means) / len(block_means)
        print(f"{method:<16} | " + " | ".join(row) + f" | {mean_all:.1f}")

results["aggregated"] = agg

# Verdict: does SVD+lstsq outperform random+lstsq by >= 5% relative PPL in >= 3/4 blocks
# at k/n <= 0.3?
print("\n=== VERDICT ===")
svd_wins_by_kn = {}
for k, kn in zip(K_VALUES, KN_RATIOS):
    kn_str = str(kn)
    svd_wins = 0
    random_wins = 0
    rel_diffs = []
    for block_idx in BLOCKS:
        svd_m = agg[kn_str]["svd_lstsq"][block_idx]["mean"]
        rnd_m = agg[kn_str]["random_lstsq"][block_idx]["mean"]
        rel_diff = (svd_m - rnd_m) / rnd_m * 100  # negative = SVD better (lower PPL)
        rel_diffs.append(rel_diff)
        if rel_diff <= -5.0:
            svd_wins += 1
        elif rel_diff >= 5.0:
            random_wins += 1
    svd_wins_by_kn[kn_str] = svd_wins
    print(f"k/n={kn}: SVD vs random rel diffs: {[f'{d:+.1f}%' for d in rel_diffs]}")
    print(f"  SVD+lstsq wins (>=5% relative) in {svd_wins}/4 blocks")

# Overall verdict from plan criteria
# Confirmed: SVD outperforms random by >= 5% relative in >= 3/4 blocks at any k/n <= 0.3
confirmed_at = [kn for k, kn in zip(K_VALUES, KN_RATIOS)
                if svd_wins_by_kn[str(kn)] >= 3]
# Refuted: random matches SVD within 3% in all blocks at all k/n
all_within_3pct = True
for k, kn in zip(K_VALUES, KN_RATIOS):
    kn_str = str(kn)
    for block_idx in BLOCKS:
        svd_m = agg[kn_str]["svd_lstsq"][block_idx]["mean"]
        rnd_m = agg[kn_str]["random_lstsq"][block_idx]["mean"]
        rel_diff = abs((svd_m - rnd_m) / rnd_m * 100)
        if rel_diff > 3.0:
            all_within_3pct = False

if len(confirmed_at) >= 1:
    verdict = "CONFIRMED"
elif all_within_3pct:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nOverall Verdict: {verdict}")
if confirmed_at:
    print(f"  SVD clearly wins at k/n: {confirmed_at}")
results["verdict"] = verdict
results["verdict_details"] = {
    "svd_wins_by_kn": svd_wins_by_kn,
    "confirmed_at_kn": confirmed_at,
    "all_within_3pct": all_within_3pct,
}

# Save JSON
out_json = OUTPUT_DIR / "cycle_077_aggressive_basis.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")

# ---------------------------------------------------------------------------
# Plot: 3-panel figure (one per k/n), grouped bars (3 methods x 4 blocks)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
colors = {"topk_lstsq": "#2196F3", "svd_lstsq": "#4CAF50", "random_lstsq": "#9E9E9E"}
labels = {"topk_lstsq": "topk+lstsq", "svd_lstsq": "SVD+lstsq", "random_lstsq": "Random+lstsq"}
offsets = [-1, 0, 1]

for ax, (k, kn) in zip(axes, zip(K_VALUES, KN_RATIOS)):
    kn_str = str(kn)
    x_pos = range(len(BLOCKS))
    width = 0.25
    for i, method in enumerate(METHODS):
        means = [agg[kn_str][method][b]["mean"] for b in BLOCKS]
        stds = [agg[kn_str][method][b]["std"] for b in BLOCKS]
        xs = [x + offsets[i] * width for x in x_pos]
        ax.bar(xs, means, width=width * 0.9, yerr=stds, capsize=3,
               color=colors[method], label=labels[method], alpha=0.85)
    ax.axhline(baseline_ppl, color="black", linestyle="--", linewidth=1.5,
               label=f"Baseline ({baseline_ppl:.1f})")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([f"Block {b}" for b in BLOCKS])
    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title(f"k/n={kn} (k={k})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    f"Aggressive Basis Compression: SVD vs topk vs Random (3 seeds)\n"
    f"Single-block swap PPL — GPT-2 Shakespeare | Baseline={baseline_ppl:.1f}",
    fontsize=12,
)
plt.tight_layout()

out_png = OUTPUT_DIR / "cycle_077_aggressive_basis.png"
plt.savefig(out_png, dpi=150)
print(f"Saved: {out_png}")
print("\nDone.")
