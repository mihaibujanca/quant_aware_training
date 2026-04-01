"""Cycle 184 / Item 197: Residual stream trajectory divergence — GELU vs ReLU+bias.

Measures how the residual stream diverges across 12 GPT-2 blocks when GELU→ReLU+bias.
Five analyses: per-layer norm, SVD rank, growth rate, transition-zone concentration,
alpha interpolation.

Verdict criteria:
  Confirmed if: divergence grows exponentially (R²>0.8) AND rank_90≤32 AND α smooth
  Refuted   if: divergence is diffuse (rank_90>256 at any layer) OR α has sharp transition
  Inconclusive if: linear growth OR rank varies wildly by layer
"""

import copy
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
N_SEQS = 16
SEQ_LEN = 128            # 16 × 128 = 2048 tokens total
BATCH_SIZE = 4
SEED = 42
N_TOKENS_MU = 500        # tokens for mu_neg estimation (alpha sweep)
OWT_PATH = "data/openwebtext_sample.txt"
OUT_JSON = "autoresearch/results/cycle_184_trajectory_divergence.json"
OUT_PNG  = "autoresearch/results/cycle_184_trajectory_divergence.png"
N_BLOCKS = 12
N_CHANNELS = 3072
HIDDEN = 768
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

print(f"Device: {DEVICE}", flush=True)
os.makedirs("autoresearch/results", exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_owt_tokens(tokenizer, n_seqs=N_SEQS, seq_len=SEQ_LEN, seed=SEED):
    with open(OWT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    rng = torch.Generator()
    rng.manual_seed(seed)
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    max_start = len(tokens) - seq_len - 1
    indices = torch.randperm(max_start, generator=rng)[:n_seqs]
    seqs = torch.stack([tokens[i : i + seq_len + 1] for i in sorted(indices.tolist())])
    return seqs  # (n_seqs, seq_len+1)


def eval_ppl(model, seqs, batch_size=BATCH_SIZE):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i : i + batch_size].to(DEVICE)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            out = model(inp)
            logits = out.logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += tgt.numel()
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# Residual stream capture
# ---------------------------------------------------------------------------
@torch.no_grad()
def capture_residuals(model, seqs, batch_size=BATCH_SIZE):
    """Capture residual stream at the OUTPUT of each GPT-2 block.

    Returns: (12, N_tokens, 768) float32 tensor on CPU.
    The output of block l is: x + attn(x) + mlp(ln2(x))
    """
    collected = [[] for _ in range(N_BLOCKS)]
    handles = []

    for l in range(N_BLOCKS):
        def make_hook(layer_id):
            def hook(module, inp, out):
                # GPT-2 block returns (hidden_state, ...) or just hidden_state
                h = out[0] if isinstance(out, tuple) else out
                collected[layer_id].append(h.detach().cpu())
            return hook
        handles.append(model.transformer.h[l].register_forward_hook(make_hook(l)))

    model.eval()
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size, :-1].to(DEVICE)
        model(batch)

    for h in handles:
        h.remove()

    # Stack: each layer -> (N_seqs*seq_len, 768)
    residuals = []
    for l in range(N_BLOCKS):
        r = torch.cat(collected[l], dim=0)  # (N_seqs, T, 768)
        residuals.append(r.reshape(-1, HIDDEN))  # (N_tokens, 768)

    return residuals  # list of 12 tensors, each (N_tokens, 768)


# ---------------------------------------------------------------------------
# Pre-activation capture (z = c_fc output, pre-GELU/ReLU)
# ---------------------------------------------------------------------------
@torch.no_grad()
def capture_z_values(model, seqs, batch_size=BATCH_SIZE):
    """Capture z = W_fc @ LN(x) + b_fc for each block (pre-activation).

    Returns: list of 12 tensors, each (N_tokens, 3072).
    """
    collected = [[] for _ in range(N_BLOCKS)]
    handles = []

    for l in range(N_BLOCKS):
        def make_hook(layer_id):
            def hook(module, inp, out):
                collected[layer_id].append(out.detach().cpu())
            return hook
        handles.append(model.transformer.h[l].mlp.c_fc.register_forward_hook(make_hook(l)))

    model.eval()
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size, :-1].to(DEVICE)
        model(batch)

    for h in handles:
        h.remove()

    z_vals = []
    for l in range(N_BLOCKS):
        z = torch.cat(collected[l], dim=0).reshape(-1, N_CHANNELS)
        z_vals.append(z)
    return z_vals


# ---------------------------------------------------------------------------
# Activation classes
# ---------------------------------------------------------------------------
class ReLUActivation(nn.Module):
    def forward(self, z):
        return F.relu(z)


class AlphaActivation(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, z):
        return self.alpha * F.gelu(z) + (1.0 - self.alpha) * F.relu(z)


# ---------------------------------------------------------------------------
# Iterative bias swap
# ---------------------------------------------------------------------------
def compute_mu_neg(model, block_idx, seqs, n_tokens=N_TOKENS_MU):
    """Compute mu_neg_l = W_down @ E[GELU(z) * 1[z<=0]] for block block_idx.

    Uses the CURRENT model state (upstream blocks may already be swapped).
    """
    accum = torch.zeros(N_CHANNELS, dtype=torch.float64)
    count = 0

    def hook_fn(module, inp, out):
        nonlocal count
        z = inp[0]  # pre-activation
        activated = out  # whatever activation is current
        mask_neg = (z <= 0)
        y_neg = torch.where(mask_neg, activated, torch.zeros_like(activated))
        batch_sum = y_neg.reshape(-1, N_CHANNELS).sum(0)
        accum.add_(batch_sum.cpu().double())
        count += y_neg.reshape(-1, N_CHANNELS).shape[0]

    h = model.transformer.h[block_idx].mlp.act.register_forward_hook(hook_fn)
    n_needed = max(1, n_tokens // SEQ_LEN + 1)
    model.eval()
    with torch.no_grad():
        for i in range(0, min(n_needed, len(seqs)), BATCH_SIZE):
            model(seqs[i : i + BATCH_SIZE, :-1].to(DEVICE))
    h.remove()

    e_y_neg = (accum / count).float().to(DEVICE)
    W_down = model.transformer.h[block_idx].mlp.c_proj.weight.data  # (3072, 768) conv1d style
    mu = e_y_neg @ W_down  # (768,)
    return mu, e_y_neg


def build_relu_iterative(tokenizer, seqs, use_bias=True, activation_cls=None, n_tokens=N_TOKENS_MU):
    """Build GPT-2 with iterative ReLU+bias swap.

    activation_cls: if None, uses ReLUActivation. Otherwise uses activation_cls.
    """
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    mu_norms = []

    for l in range(N_BLOCKS):
        act = activation_cls() if activation_cls is not None else ReLUActivation()
        if use_bias:
            mu, _ = compute_mu_neg(model, l, seqs, n_tokens=n_tokens)
            mu_norms.append(float(mu.norm().item()))
        model.transformer.h[l].mlp.act = act
        if use_bias:
            model.transformer.h[l].mlp.c_proj.bias.data.add_(mu)

    print(f"  Iterative swap done. ||mu|| range: [{min(mu_norms):.3f}, {max(mu_norms):.3f}]" if mu_norms else "", flush=True)
    return model, mu_norms


def build_relu_iterative_alpha(tokenizer, seqs, alpha, n_tokens=N_TOKENS_MU):
    """Build GPT-2 with iterative AlphaActivation(alpha)+bias swap."""
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    for l in range(N_BLOCKS):
        # Compute mu_alpha: bias = E[gelu(z)] - E[gelu_alpha(z)]
        # By linearity: bias = (1-alpha) * E[gelu(z) * 1[z<=0]] = (1-alpha) * e_y_neg @ W_down
        # We compute e_y_neg from the CURRENT model state's GELU activation
        if abs(1.0 - alpha) > 1e-6:  # non-trivial alpha
            _, e_y_neg = compute_mu_neg(model, l, seqs, n_tokens=n_tokens)
            W_down = model.transformer.h[l].mlp.c_proj.weight.data
            mu_alpha = (1.0 - alpha) * (e_y_neg @ W_down)
            model.transformer.h[l].mlp.c_proj.bias.data.add_(mu_alpha)
        model.transformer.h[l].mlp.act = AlphaActivation(alpha)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading tokenizer and data...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    seqs = load_owt_tokens(tokenizer)
    print(f"  Loaded {seqs.shape[0]} seqs × {seqs.shape[1]-1} tokens = {seqs.shape[0]*(seqs.shape[1]-1)} tokens", flush=True)

    # -------------------------------------------------------------------
    # Model A: float GELU
    # -------------------------------------------------------------------
    print("\n--- Model A: float GELU ---", flush=True)
    model_A = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model_A.eval()
    ppl_A = eval_ppl(model_A, seqs)
    print(f"  PPL (float GELU): {ppl_A:.3f}", flush=True)
    if not (15.0 <= ppl_A <= 65.0):
        print(f"ABORT: PPL_A={ppl_A:.2f} outside [15,65]")
        sys.exit(1)

    residuals_A = capture_residuals(model_A, seqs)
    z_values_A  = capture_z_values(model_A, seqs)
    print(f"  Residuals captured: {len(residuals_A)} layers, {residuals_A[0].shape[0]} tokens each", flush=True)

    # -------------------------------------------------------------------
    # Model B: ReLU + iterative bias
    # -------------------------------------------------------------------
    print("\n--- Model B: ReLU + iterative bias ---", flush=True)
    model_B, mu_norms_B = build_relu_iterative(tokenizer, seqs, use_bias=True)
    ppl_B = eval_ppl(model_B, seqs)
    print(f"  PPL (ReLU+iterative bias): {ppl_B:.3f}", flush=True)
    residuals_B = capture_residuals(model_B, seqs)
    del model_B

    # -------------------------------------------------------------------
    # Model C: ReLU, no bias
    # -------------------------------------------------------------------
    print("\n--- Model C: ReLU, no bias ---", flush=True)
    model_C, _ = build_relu_iterative(tokenizer, seqs, use_bias=False)
    ppl_C = eval_ppl(model_C, seqs)
    print(f"  PPL (ReLU, no bias): {ppl_C:.3f}", flush=True)
    residuals_C = capture_residuals(model_C, seqs)
    del model_C

    # -------------------------------------------------------------------
    # Analysis 1: Per-layer divergence norm
    # -------------------------------------------------------------------
    print("\n--- Analysis 1: Per-layer divergence norm ---", flush=True)
    div_norm_B = []  # ReLU+bias vs GELU
    div_norm_C = []  # ReLU-no-bias vs GELU
    for l in range(N_BLOCKS):
        rA = residuals_A[l]  # (N_tok, 768)
        rB = residuals_B[l]
        rC = residuals_C[l]
        norm_A = rA.norm(dim=1)  # (N_tok,)
        div_B = (rA - rB).norm(dim=1) / (norm_A + 1e-8)
        div_C = (rA - rC).norm(dim=1) / (norm_A + 1e-8)
        div_norm_B.append(float(div_B.mean().item()))
        div_norm_C.append(float(div_C.mean().item()))
        print(f"  Layer {l:2d}: div(ReLU+bias)={div_norm_B[-1]:.4f}  div(ReLU-nobias)={div_norm_C[-1]:.4f}", flush=True)

    # -------------------------------------------------------------------
    # Analysis 2: SVD rank of divergence matrix
    # -------------------------------------------------------------------
    print("\n--- Analysis 2: SVD rank ---", flush=True)
    rank_90 = []
    rank_99 = []
    sv_selected = {}  # singular values for selected layers
    for l in range(N_BLOCKS):
        D = (residuals_A[l] - residuals_B[l]).float()  # (N_tok, 768)
        # Subtract mean for cleaner SVD
        D = D - D.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(D, full_matrices=False)
        energy = (S**2).cumsum(0) / (S**2).sum()
        r90 = int((energy < 0.90).sum().item()) + 1
        r99 = int((energy < 0.99).sum().item()) + 1
        rank_90.append(r90)
        rank_99.append(r99)
        if l in {0, 3, 6, 11}:
            sv_selected[l] = S[:64].tolist()
        print(f"  Layer {l:2d}: rank_90={r90:3d}, rank_99={r99:3d}", flush=True)

    # -------------------------------------------------------------------
    # Analysis 3: Growth rate fit
    # -------------------------------------------------------------------
    print("\n--- Analysis 3: Growth rate ---", flush=True)
    D_frob = []
    N_tok = residuals_A[0].shape[0]
    for l in range(N_BLOCKS):
        D = (residuals_A[l] - residuals_B[l]).float()
        frob = float((D.norm().item()) / math.sqrt(N_tok))
        D_frob.append(frob)

    layers = list(range(N_BLOCKS))
    # Exponential fit: log(D_frob) = a*l + b
    log_D = [math.log(max(x, 1e-9)) for x in D_frob]
    n = N_BLOCKS
    sl = sum(layers)
    sl2 = sum(l**2 for l in layers)
    sly = sum(l * y for l, y in zip(layers, log_D))
    sy = sum(log_D)
    a_exp = (n * sly - sl * sy) / (n * sl2 - sl**2)
    b_exp = (sy - a_exp * sl) / n
    pred_exp = [a_exp * l + b_exp for l in layers]
    ss_res_exp = sum((y - p)**2 for y, p in zip(log_D, pred_exp))
    ss_tot = sum((y - sy/n)**2 for y in log_D)
    r2_exp = 1 - ss_res_exp / (ss_tot + 1e-10)

    # Linear fit: D_frob = a*l + b
    sdy = sum(l * d for l, d in zip(layers, D_frob))
    sd = sum(D_frob)
    a_lin = (n * sdy - sl * sd) / (n * sl2 - sl**2)
    b_lin = (sd - a_lin * sl) / n
    pred_lin = [a_lin * l + b_lin for l in layers]
    ss_res_lin = sum((y - p)**2 for y, p in zip(D_frob, pred_lin))
    ss_tot_lin = sum((y - sd/n)**2 for y in D_frob)
    r2_lin = 1 - ss_res_lin / (ss_tot_lin + 1e-10)

    print(f"  Frobenius norm / sqrt(N): {D_frob}", flush=True)
    print(f"  Exponential fit R²={r2_exp:.4f} (a={a_exp:.4f})", flush=True)
    print(f"  Linear fit      R²={r2_lin:.4f} (a={a_lin:.4f})", flush=True)

    # -------------------------------------------------------------------
    # Analysis 4: Transition zone concentration
    # -------------------------------------------------------------------
    print("\n--- Analysis 4: Transition zone ---", flush=True)
    # Compute per-token divergence at each layer
    per_token_div = []
    for l in range(N_BLOCKS):
        rA = residuals_A[l]
        rB = residuals_B[l]
        div = (rA - rB).norm(dim=1)
        per_token_div.append(div)

    # Mean divergence across all layers per token
    total_div = torch.stack(per_token_div).mean(0)  # (N_tok,)
    top50_idx = total_div.topk(50).indices
    bot50_idx = total_div.topk(50, largest=False).indices

    # For each layer, compute transition-zone fraction (|z| < 1) for top vs bottom tokens
    tz_top = []
    tz_bot = []
    for l in range(N_BLOCKS):
        z_l = z_values_A[l]  # (N_tok, 3072)
        tz_frac_top = float((z_l[top50_idx].abs() < 1.0).float().mean().item())
        tz_frac_bot = float((z_l[bot50_idx].abs() < 1.0).float().mean().item())
        tz_top.append(tz_frac_top)
        tz_bot.append(tz_frac_bot)
        print(f"  Layer {l:2d}: tz_top={tz_frac_top:.3f}  tz_bot={tz_frac_bot:.3f}", flush=True)

    # -------------------------------------------------------------------
    # Analysis 5: Alpha interpolation
    # -------------------------------------------------------------------
    print("\n--- Analysis 5: Alpha interpolation ---", flush=True)
    ppl_alpha = {}
    for alpha in ALPHA_VALUES:
        print(f"  alpha={alpha:.2f}:", flush=True)
        model_alpha = build_relu_iterative_alpha(tokenizer, seqs, alpha=alpha, n_tokens=N_TOKENS_MU)
        ppl = eval_ppl(model_alpha, seqs)
        ppl_alpha[alpha] = float(ppl)
        print(f"    PPL={ppl:.3f}", flush=True)
        del model_alpha

    # Smoothness: check if PPL is monotone with alpha (alpha=1 → GELU, alpha=0 → ReLU)
    alpha_ppls = [ppl_alpha[a] for a in ALPHA_VALUES]
    is_monotone = all(alpha_ppls[i] >= alpha_ppls[i+1] for i in range(len(alpha_ppls)-1))
    # Sharp transition: find largest single-step PPL jump
    steps = [abs(alpha_ppls[i+1] - alpha_ppls[i]) for i in range(len(alpha_ppls)-1)]
    max_step = max(steps)
    mean_step = sum(steps) / len(steps)
    sharp_ratio = max_step / (mean_step + 1e-6)
    print(f"  Alpha PPLs: {dict(zip(ALPHA_VALUES, alpha_ppls))}", flush=True)
    print(f"  Monotone: {is_monotone}, max_step: {max_step:.2f}, sharp_ratio: {sharp_ratio:.2f}", flush=True)

    # -------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------
    print("\n--- Verdict ---", flush=True)
    max_rank_90 = max(rank_90)
    exp_ok = r2_exp > 0.8
    rank_ok = max_rank_90 <= 32
    alpha_smooth = sharp_ratio < 3.0  # no single step > 3x mean step

    print(f"  R²_exp={r2_exp:.3f} (>0.8? {exp_ok})", flush=True)
    print(f"  max rank_90={max_rank_90} (≤32? {rank_ok})", flush=True)
    print(f"  alpha smooth? {alpha_smooth} (sharp_ratio={sharp_ratio:.2f})", flush=True)

    if r2_exp > 0.8 and max_rank_90 <= 32 and alpha_smooth:
        verdict = "CONFIRMED"
        verdict_note = f"Exponential divergence (R²={r2_exp:.3f}), low rank (max rank_90={max_rank_90}), smooth α interpolation (sharp_ratio={sharp_ratio:.2f})"
    elif max_rank_90 > 256 or (not alpha_smooth and sharp_ratio > 5.0):
        verdict = "REFUTED"
        verdict_note = f"Diffuse divergence (max rank_90={max_rank_90}) or sharp α transition (sharp_ratio={sharp_ratio:.2f})"
    else:
        verdict = "INCONCLUSIVE"
        verdict_note = f"R²_exp={r2_exp:.3f}, max rank_90={max_rank_90}, sharp_ratio={sharp_ratio:.2f}"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  {verdict_note}", flush=True)

    # -------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------
    results = {
        "cycle": 184, "item": 197, "verdict": verdict, "verdict_note": verdict_note,
        "ppl_A_float_gelu": float(ppl_A),
        "ppl_B_relu_iterbias": float(ppl_B),
        "ppl_C_relu_nobias": float(ppl_C),
        "analysis1_div_norm_B": div_norm_B,
        "analysis1_div_norm_C": div_norm_C,
        "analysis2_rank_90": rank_90,
        "analysis2_rank_99": rank_99,
        "analysis2_sv_selected": {str(k): v for k, v in sv_selected.items()},
        "analysis3_D_frob": D_frob,
        "analysis3_r2_exp": float(r2_exp),
        "analysis3_r2_lin": float(r2_lin),
        "analysis3_a_exp": float(a_exp),
        "analysis4_tz_top50": tz_top,
        "analysis4_tz_bot50": tz_bot,
        "analysis5_ppl_alpha": {str(k): v for k, v in ppl_alpha.items()},
        "analysis5_monotone": is_monotone,
        "analysis5_sharp_ratio": float(sharp_ratio),
        "config": {
            "n_seqs": N_SEQS, "seq_len": SEQ_LEN, "n_tokens_mu": N_TOKENS_MU,
            "alpha_values": ALPHA_VALUES, "seed": SEED,
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_JSON}", flush=True)

    # -------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Residual Stream Trajectory Divergence: GELU vs ReLU+bias", fontsize=13)

    # (0,0) Per-layer divergence
    ax = axes[0, 0]
    ax.plot(range(N_BLOCKS), div_norm_B, "b-o", label="ReLU+bias vs GELU")
    ax.plot(range(N_BLOCKS), div_norm_C, "r--s", label="ReLU-nobias vs GELU", alpha=0.7)
    ax.set_xlabel("Block")
    ax.set_ylabel("Rel. divergence ||rA-rB|| / ||rA||")
    ax.set_title("A1: Per-layer divergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) SVD rank
    ax = axes[0, 1]
    ax.plot(range(N_BLOCKS), rank_90, "g-o", label="rank_90 (90% energy)")
    ax.plot(range(N_BLOCKS), rank_99, "m--s", label="rank_99 (99% energy)")
    ax.axhline(32, color="r", linestyle=":", label="threshold 32")
    ax.axhline(256, color="k", linestyle=":", label="threshold 256")
    ax.set_xlabel("Block")
    ax.set_ylabel("Rank")
    ax.set_title("A2: SVD rank of divergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,2) Growth rate
    ax = axes[0, 2]
    ax.semilogy(range(N_BLOCKS), D_frob, "b-o", label="Observed")
    ax.semilogy(range(N_BLOCKS), [math.exp(p) for p in pred_exp], "r--", label=f"Exp fit R²={r2_exp:.3f}")
    ax.set_xlabel("Block")
    ax.set_ylabel("||D||_F / sqrt(N) (log scale)")
    ax.set_title("A3: Growth rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) Transition zone
    ax = axes[1, 0]
    ax.plot(range(N_BLOCKS), tz_top, "r-o", label="Top-50 divergent tokens")
    ax.plot(range(N_BLOCKS), tz_bot, "b--s", label="Bot-50 divergent tokens")
    ax.set_xlabel("Block")
    ax.set_ylabel("Fraction |z|<1 (transition zone)")
    ax.set_title("A4: Transition zone fraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Alpha interpolation
    ax = axes[1, 1]
    ax.plot(ALPHA_VALUES, [ppl_alpha[a] for a in ALPHA_VALUES], "g-o")
    ax.axhline(ppl_A, color="b", linestyle=":", label=f"Float GELU ({ppl_A:.1f})")
    ax.set_xlabel("α (1=GELU, 0=ReLU)")
    ax.set_ylabel("PPL")
    ax.set_title("A5: Alpha interpolation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) SV spectra
    ax = axes[1, 2]
    colors = ["b", "g", "r", "m"]
    for (l, color) in zip([0, 3, 6, 11], colors):
        sv = sv_selected[l]
        ax.semilogy(range(len(sv)), sv, color=color, label=f"Layer {l}")
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular value (log scale)")
    ax.set_title("SV spectra (selected layers)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"Plot saved to {OUT_PNG}", flush=True)

    return results


if __name__ == "__main__":
    results = main()
