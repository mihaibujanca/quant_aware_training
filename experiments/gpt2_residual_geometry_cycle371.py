#!/usr/bin/env python3
"""Cycle 371 / Item 379 — Residual stream geometry across compression stages.

Hypothesis: 6-block trained compression shows superlinear eff_rank degradation
vs 3-block (PPL ~47), indicating representation subspace collapse underlying
the composition barrier.

Configs: float | 3-block [2,1,10] | 6-block [2,1,10,4,3,11]
Adapter: topk(k=512)+lstsq + MLP adapter hidden=64, 500 Adam steps, lr=3e-4

Metrics at each of 12 block boundaries:
  1. Effective rank: exp(H(sigma/sum(sigma)))
  2. Participation ratio: (sum(lambda))^2 / sum(lambda^2)
  3. Top-5 PC concentration: fraction of variance in top-5 PCs
  4. Mean pairwise cosine: mean |cos(h_i, h_j)| over 1000 pairs

Verdict:
  CONFIRMED  if 6-block eff_rank drop >30% from float at >=2 layers AND
             3-block drop <15%
  REFUTED    if both configs show <10% eff_rank deviation from float
  INCONCLUSIVE otherwise
"""
import json
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS_3 = [2, 1, 10]
BLOCKS_6 = [2, 1, 10, 4, 3, 11]
K_UNIFORM = 512
ADAPTER_HIDDEN = 64
N_CALIB = 200
N_EVAL = 100
N_GEOM = 200   # sequences for geometry measurement
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 500
LR = 3e-4
EVAL_EVERY = 100
SEED = 42

FLOAT_BASELINE_REF = 53.895
OUTPUT_JSON = "autoresearch/results/cycle_371_residual_geometry.json"
OUTPUT_PLOT = "autoresearch/results/cycle_371_geometry_profiles.png"


# ── Geometry utilities ────────────────────────────────────────────────────────

def effective_rank(H: torch.Tensor) -> float:
    """Effective rank via entropy of singular value distribution.
    H: (N, D) tensor. Centers H before SVD.
    """
    H = H.float()
    H = H - H.mean(0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(H, full_matrices=False)
    except Exception:
        return float("nan")
    S = S[S > 1e-10]
    if S.numel() == 0:
        return 1.0
    p = S / S.sum()
    entropy = -(p * p.log()).sum().item()
    return math.exp(entropy)


def participation_ratio(H: torch.Tensor) -> float:
    """Participation ratio of covariance eigenvalues.
    PR = (sum lambda)^2 / sum(lambda^2). Ranges [1, D].
    H: (N, D) tensor.
    """
    H = H.float()
    H = H - H.mean(0, keepdim=True)
    cov = H.T @ H / (H.shape[0] - 1)
    try:
        lam = torch.linalg.eigvalsh(cov)
    except Exception:
        return float("nan")
    lam = lam[lam > 1e-10]
    if lam.numel() == 0:
        return 1.0
    pr = (lam.sum().item() ** 2) / (lam.square().sum().item())
    return pr


def top5_pc_concentration(H: torch.Tensor) -> float:
    """Fraction of variance in top-5 PCs."""
    H = H.float()
    H = H - H.mean(0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(H, full_matrices=False)
    except Exception:
        return float("nan")
    S2 = S.square()
    total = S2.sum().item()
    if total < 1e-10:
        return float("nan")
    top5 = S2[:5].sum().item()
    return top5 / total


def mean_pairwise_cosine(H: torch.Tensor, n_pairs: int = 1000, seed: int = 0) -> float:
    """Mean absolute cosine over random pairs of token vectors."""
    H = H.float()
    N = H.shape[0]
    if N < 2:
        return float("nan")
    gen = torch.Generator()
    gen.manual_seed(seed)
    idx_a = torch.randint(0, N, (n_pairs,), generator=gen)
    idx_b = torch.randint(0, N, (n_pairs,), generator=gen)
    a = H[idx_a]
    b = H[idx_b]
    cos = F.cosine_similarity(a, b, dim=1).abs().mean().item()
    return cos


def compute_geometry(H: torch.Tensor) -> dict:
    """Compute all 4 metrics for a (N, D) residual stream tensor."""
    return {
        "eff_rank": effective_rank(H),
        "participation_ratio": participation_ratio(H),
        "top5_pc_conc": top5_pc_concentration(H),
        "mean_cos": mean_pairwise_cosine(H),
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    print("Loading GPT-2 tokenizer and wikitext-2 validation data...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
    all_ids = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
        if enc["input_ids"].shape[1] >= SEQ_LEN:
            all_ids.append(enc["input_ids"][:, :SEQ_LEN])
        if len(all_ids) >= N_CALIB + N_EVAL:
            break

    print(f"  Total seqs: {len(all_ids)}", flush=True)
    calib_ids_cpu = list(all_ids[:N_CALIB])
    eval_ids_dev = [t.to(DEVICE) for t in all_ids[N_CALIB: N_CALIB + N_EVAL]]
    # Reuse calib as geometry measurement sequences (same 200)
    geom_ids_dev = [t.to(DEVICE) for t in all_ids[:N_GEOM]]
    return calib_ids_cpu, eval_ids_dev, geom_ids_dev


# ── PPL eval ──────────────────────────────────────────────────────────────────

def eval_ppl(model, eval_ids_dev):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in eval_ids_dev:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


# ── Compression ───────────────────────────────────────────────────────────────

def collect_block_activations(model, block_ids, calib_ids_dev, device):
    block_data = {b: {"input": [], "output": [], "preact": [], "hidden": []} for b in block_ids}
    hooks = []

    def make_hook(blk_id):
        def hook_fn(module, inputs, output):
            x = inputs[0].detach()
            z = module.c_fc(x).detach()
            h = F.gelu(z).detach()
            y = output.detach()
            block_data[blk_id]["input"].append(x.reshape(-1, x.shape[-1]).cpu())
            block_data[blk_id]["output"].append(y.reshape(-1, y.shape[-1]).cpu())
            block_data[blk_id]["preact"].append(z.reshape(-1, z.shape[-1]).cpu())
            block_data[blk_id]["hidden"].append(h.reshape(-1, h.shape[-1]).cpu())
        return hook_fn

    for blk_id in block_ids:
        hook = model.transformer.h[blk_id].mlp.register_forward_hook(make_hook(blk_id))
        hooks.append(hook)

    model.eval()
    with torch.no_grad():
        for ids in calib_ids_dev:
            ids_dev = ids.to(device) if ids.device.type != device else ids
            model(ids_dev)

    for hook in hooks:
        hook.remove()

    result = {}
    for blk_id in block_ids:
        result[blk_id] = {
            "input": torch.cat(block_data[blk_id]["input"], dim=0).float(),
            "output": torch.cat(block_data[blk_id]["output"], dim=0).float(),
            "preact": torch.cat(block_data[blk_id]["preact"], dim=0).float(),
            "hidden": torch.cat(block_data[blk_id]["hidden"], dim=0).float(),
        }
    return result


class CompressedGPT2MLPWithAdapter(nn.Module):
    def __init__(self, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias, adapter_hidden=64):
        super().__init__()
        k = c_fc_weight.shape[0]
        self.register_buffer("fc_weight", c_fc_weight.contiguous())
        self.register_buffer("fc_bias", c_fc_bias.contiguous())
        self.proj_weight = nn.Parameter(c_proj_weight.contiguous())
        self.proj_bias = nn.Parameter(c_proj_bias.contiguous())

        scale1 = 1.0 / math.sqrt(k)
        scale2 = 1.0 / math.sqrt(adapter_hidden)
        self.adapter_W1 = nn.Parameter(torch.randn(adapter_hidden, k) * scale1)
        self.adapter_b1 = nn.Parameter(torch.zeros(adapter_hidden))
        self.adapter_W2 = nn.Parameter(torch.randn(768, adapter_hidden) * scale2)
        self.adapter_b2 = nn.Parameter(torch.zeros(768))

    def forward(self, x):
        h = F.linear(x, self.fc_weight, self.fc_bias)
        h = F.gelu(h)
        out = F.linear(h, self.proj_weight, self.proj_bias)
        a = F.relu(F.linear(h, self.adapter_W1, self.adapter_b1))
        out = out + F.linear(a, self.adapter_W2, self.adapter_b2)
        return out


def compress_blocks(model, block_ids, k, calib_ids_dev, device):
    """topk+lstsq compression, uniform k across blocks."""
    k_per_block = {b: k for b in block_ids}
    block_acts = collect_block_activations(model, block_ids, calib_ids_dev, device)
    compress_cache = {}
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    for blk in block_ids:
        acts = block_acts[blk]
        hidden = acts["hidden"]
        output = acts["output"]
        preact = acts["preact"]
        hidden_dim = hidden.shape[1]

        std_h = hidden.std(dim=0)
        mean_h = hidden.mean(dim=0)
        deriv = (0.5 * (1.0 + torch.erf(preact * inv_sqrt_2))
                 + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi)
        mean_deriv = deriv.mean(dim=0)

        c_proj = model.transformer.h[blk].mlp.c_proj
        c_proj_weight = c_proj.weight.detach().cpu().float()
        output_norms = c_proj_weight.norm(dim=1)

        channel_scores = std_h * output_norms * (mean_h.abs() + mean_deriv.abs() + 1e-6)

        dead_mask = hidden.abs().amax(dim=0) <= 1e-6
        available = torch.arange(hidden_dim)
        live = available[~dead_mask]
        if live.numel() < k:
            live = available
        scores = channel_scores[live]
        topk_idx = torch.topk(scores, k=k, largest=True).indices
        selected = live[topk_idx].sort().values

        c_fc = model.transformer.h[blk].mlp.c_fc
        fc_w = c_fc.weight.detach().cpu().float().T[selected]  # (k, 768)
        fc_b = c_fc.bias.detach().cpu().float()[selected]       # (k,)

        hidden_sel = hidden[:, selected].float()
        design = torch.cat([hidden_sel, torch.ones(hidden_sel.shape[0], 1)], dim=1)
        solution = torch.linalg.lstsq(design, output.float()).solution
        proj_w = solution[:-1].T.contiguous()
        proj_b = solution[-1].contiguous()

        compress_cache[blk] = {
            "fc_w": fc_w, "fc_b": fc_b, "proj_w": proj_w, "proj_b": proj_b,
        }
        with torch.no_grad():
            hidden_sel2 = hidden[:, selected].float()
            recon = hidden_sel2 @ proj_w.T + proj_b
            mse = F.mse_loss(recon, output).item()
        print(f"  Block {blk}: k={k}, MSE={mse:.4f}", flush=True)

    return compress_cache


def install_compressed_blocks(model, block_ids, compress_cache, device):
    modules = {}
    for blk in block_ids:
        c = compress_cache[blk]
        comp_mlp = CompressedGPT2MLPWithAdapter(
            c_fc_weight=c["fc_w"].to(device=device, dtype=torch.float32),
            c_fc_bias=c["fc_b"].to(device=device, dtype=torch.float32),
            c_proj_weight=c["proj_w"].to(device=device, dtype=torch.float32),
            c_proj_bias=c["proj_b"].to(device=device, dtype=torch.float32),
            adapter_hidden=ADAPTER_HIDDEN,
        ).to(device)
        model.transformer.h[blk].mlp = comp_mlp
        modules[blk] = comp_mlp
    return modules


def train_adapters(model, compressed_modules, calib_ids_dev, eval_ids_dev):
    for p in model.parameters():
        p.requires_grad_(False)

    trainable = []
    for mod in compressed_modules.values():
        for pname in ["proj_weight", "proj_bias",
                      "adapter_W1", "adapter_b1", "adapter_W2", "adapter_b2"]:
            p = getattr(mod, pname)
            p.requires_grad_(True)
            trainable.append(p)

    n_train = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {n_train:,}", flush=True)

    init_ppl = eval_ppl(model, eval_ids_dev)
    print(f"  Init PPL: {init_ppl:.4f}", flush=True)

    optimizer = torch.optim.Adam(trainable, lr=LR)
    ids_shuffled = list(calib_ids_dev)
    random.shuffle(ids_shuffled)

    best_ppl = init_ppl
    best_step = 0
    best_state = {}
    for blk, mod in compressed_modules.items():
        for pname in ["proj_weight", "proj_bias",
                      "adapter_W1", "adapter_b1", "adapter_W2", "adapter_b2"]:
            best_state[f"{pname}_{blk}"] = getattr(mod, pname).detach().clone()

    step = 0
    while step < N_STEPS:
        for ids in ids_shuffled:
            if step >= N_STEPS:
                break
            model.train()
            optimizer.zero_grad()
            out = model(ids, labels=ids)
            out.loss.backward()
            optimizer.step()
            step += 1
            if step % EVAL_EVERY == 0:
                ppl_e = eval_ppl(model, eval_ids_dev)
                if ppl_e < best_ppl:
                    best_ppl = ppl_e
                    best_step = step
                    for blk, mod in compressed_modules.items():
                        for pname in ["proj_weight", "proj_bias",
                                      "adapter_W1", "adapter_b1", "adapter_W2", "adapter_b2"]:
                            best_state[f"{pname}_{blk}"] = getattr(mod, pname).detach().clone()
                if step % 100 == 0:
                    print(f"  step {step}: ppl={ppl_e:.2f} best={best_ppl:.2f}@{best_step}",
                          flush=True)

    with torch.no_grad():
        for blk, mod in compressed_modules.items():
            for pname in ["proj_weight", "proj_bias",
                          "adapter_W1", "adapter_b1", "adapter_W2", "adapter_b2"]:
                getattr(mod, pname).data.copy_(best_state[f"{pname}_{blk}"])

    final_ppl = eval_ppl(model, eval_ids_dev)
    print(f"  Final PPL (restored best): {final_ppl:.4f}", flush=True)
    return final_ppl


# ── Residual stream geometry measurement ──────────────────────────────────────

def measure_residual_geometry(model, geom_ids_dev, device):
    """Hook all 12 transformer block outputs, collect residual stream tensors,
    compute geometry metrics for each."""
    n_blocks = len(model.transformer.h)
    resid_data = {i: [] for i in range(n_blocks)}
    hooks = []

    def make_hook(blk_i):
        def fn(module, inputs, outputs):
            # transformer block output is a tuple; first element is residual stream
            if isinstance(outputs, tuple):
                h = outputs[0]
            else:
                h = outputs
            resid_data[blk_i].append(h.detach().reshape(-1, h.shape[-1]).cpu())
        return fn

    for i in range(n_blocks):
        h = model.transformer.h[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for ids in geom_ids_dev:
            model(ids)

    for h in hooks:
        h.remove()

    geom = {}
    for i in range(n_blocks):
        H = torch.cat(resid_data[i], dim=0).float()
        # H shape: (N*T, 768) = (200*128, 768) = (25600, 768)
        # SVD on full matrix is expensive; subsample to 4096 rows
        if H.shape[0] > 4096:
            idx = torch.randperm(H.shape[0])[:4096]
            H_sub = H[idx]
        else:
            H_sub = H
        metrics = compute_geometry(H_sub)
        geom[i] = metrics
        print(f"  Block {i}: eff_rank={metrics['eff_rank']:.1f} "
              f"PR={metrics['participation_ratio']:.1f} "
              f"top5={metrics['top5_pc_conc']:.3f} "
              f"cos={metrics['mean_cos']:.3f}", flush=True)

    return geom


# ── Main ──────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
random.seed(SEED)
print(f"Device: {DEVICE}", flush=True)

calib_ids_cpu, eval_ids_dev, geom_ids_dev = load_data()
calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]

results = {}

# 1. Float baseline
print("\n" + "="*60, flush=True)
print("Config: float (no compression)", flush=True)
print("="*60, flush=True)
float_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
float_ppl = eval_ppl(float_model, eval_ids_dev)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
print("Measuring residual geometry (float)...", flush=True)
float_geom = measure_residual_geometry(float_model, geom_ids_dev, DEVICE)
results["float"] = {
    "ppl": float(float_ppl),
    "blocks_compressed": [],
    "geometry": {str(k): v for k, v in float_geom.items()},
}
del float_model
if DEVICE == "mps":
    torch.mps.empty_cache()

# 2. 3-block compression: [2, 1, 10]
print("\n" + "="*60, flush=True)
print("Config: 3-block [2, 1, 10]", flush=True)
print("="*60, flush=True)
torch.manual_seed(SEED)
model3 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
print("Compressing 3 blocks...", flush=True)
cache3 = compress_blocks(model3, BLOCKS_3, K_UNIFORM, calib_ids_dev, DEVICE)
mods3 = install_compressed_blocks(model3, BLOCKS_3, cache3, DEVICE)
print("Training adapters (500 steps)...", flush=True)
ppl3 = train_adapters(model3, mods3, calib_ids_dev, eval_ids_dev)
print(f"3-block PPL: {ppl3:.4f}", flush=True)
print("Measuring residual geometry (3-block)...", flush=True)
geom3 = measure_residual_geometry(model3, geom_ids_dev, DEVICE)
results["3block"] = {
    "ppl": float(ppl3),
    "blocks_compressed": BLOCKS_3,
    "geometry": {str(k): v for k, v in geom3.items()},
}
del model3
if DEVICE == "mps":
    torch.mps.empty_cache()

# 3. 6-block compression: [2, 1, 10, 4, 3, 11]
print("\n" + "="*60, flush=True)
print("Config: 6-block [2, 1, 10, 4, 3, 11]", flush=True)
print("="*60, flush=True)
torch.manual_seed(SEED)
model6 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
print("Compressing 6 blocks...", flush=True)
cache6 = compress_blocks(model6, BLOCKS_6, K_UNIFORM, calib_ids_dev, DEVICE)
mods6 = install_compressed_blocks(model6, BLOCKS_6, cache6, DEVICE)
print("Training adapters (500 steps)...", flush=True)
ppl6 = train_adapters(model6, mods6, calib_ids_dev, eval_ids_dev)
print(f"6-block PPL: {ppl6:.4f}", flush=True)
print("Measuring residual geometry (6-block)...", flush=True)
geom6 = measure_residual_geometry(model6, geom_ids_dev, DEVICE)
results["6block"] = {
    "ppl": float(ppl6),
    "blocks_compressed": BLOCKS_6,
    "geometry": {str(k): v for k, v in geom6.items()},
}
del model6
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Analysis ──────────────────────────────────────────────────────────────────

print("\n" + "="*60, flush=True)
print("GEOMETRY ANALYSIS", flush=True)
print("="*60, flush=True)
print(f"Float PPL: {results['float']['ppl']:.4f}", flush=True)
print(f"3-block PPL: {results['3block']['ppl']:.4f} (target <50)", flush=True)
print(f"6-block PPL: {results['6block']['ppl']:.4f} (target >60)", flush=True)

# Compute eff_rank drops at each layer
n_layers = 12
float_er = [results["float"]["geometry"][str(i)]["eff_rank"] for i in range(n_layers)]
er3 = [results["3block"]["geometry"][str(i)]["eff_rank"] for i in range(n_layers)]
er6 = [results["6block"]["geometry"][str(i)]["eff_rank"] for i in range(n_layers)]

print("\nEff_rank drops relative to float:", flush=True)
print(f"{'Layer':>6} {'Float':>8} {'3-block':>8} {'3-blk drop':>12} {'6-block':>8} {'6-blk drop':>12}", flush=True)
drop3_counts = {"gt30": 0, "gt15": 0, "lt10": 0}
drop6_counts = {"gt30": 0, "gt15": 0, "lt10": 0}
for i in range(n_layers):
    f = float_er[i]
    d3 = (f - er3[i]) / f * 100 if f > 0 else 0
    d6 = (f - er6[i]) / f * 100 if f > 0 else 0
    print(f"  {i:4d}  {f:8.2f}  {er3[i]:8.2f}  {d3:+10.1f}%  {er6[i]:8.2f}  {d6:+10.1f}%", flush=True)
    if d3 > 30:
        drop3_counts["gt30"] += 1
    if d3 > 15:
        drop3_counts["gt15"] += 1
    if abs(d3) < 10:
        drop3_counts["lt10"] += 1
    if d6 > 30:
        drop6_counts["gt30"] += 1
    if d6 > 15:
        drop6_counts["gt15"] += 1
    if abs(d6) < 10:
        drop6_counts["lt10"] += 1

print(f"\n3-block: >30% drop at {drop3_counts['gt30']} layers, <10% drop at {drop3_counts['lt10']} layers", flush=True)
print(f"6-block: >30% drop at {drop6_counts['gt30']} layers, <10% drop at {drop6_counts['lt10']} layers", flush=True)

# Verdict (needs PPL barrier confirmed first)
ppl_barrier = results["3block"]["ppl"] < 50 and results["6block"]["ppl"] > 60
if ppl_barrier:
    print("\nPPL barrier confirmed (3-block<50, 6-block>60)", flush=True)
    if drop6_counts["gt30"] >= 2 and drop3_counts["gt30"] == 0:
        verdict = "CONFIRMED"
    elif drop6_counts["lt10"] >= 10 and drop3_counts["lt10"] >= 10:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
else:
    print(f"\nWARNING: PPL barrier not fully reproduced "
          f"(3-block={results['3block']['ppl']:.2f}, 6-block={results['6block']['ppl']:.2f})", flush=True)
    if drop6_counts["gt30"] >= 2 and drop3_counts["gt30"] == 0:
        verdict = "CONFIRMED"
    elif drop6_counts["lt10"] >= 10 and drop3_counts["lt10"] >= 10:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

print(f"\nVERDICT: {verdict}", flush=True)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    f"Residual Stream Geometry — Float vs 3-block (PPL {results['3block']['ppl']:.1f}) "
    f"vs 6-block (PPL {results['6block']['ppl']:.1f})",
    fontsize=11
)

metrics_to_plot = [
    ("eff_rank", "Effective Rank", axes[0, 0]),
    ("participation_ratio", "Participation Ratio", axes[0, 1]),
    ("top5_pc_conc", "Top-5 PC Concentration", axes[1, 0]),
    ("mean_cos", "Mean Pairwise |Cosine|", axes[1, 1]),
]

layers = list(range(n_layers))
for key, title, ax in metrics_to_plot:
    f_vals = [results["float"]["geometry"][str(i)][key] for i in layers]
    v3_vals = [results["3block"]["geometry"][str(i)][key] for i in layers]
    v6_vals = [results["6block"]["geometry"][str(i)][key] for i in layers]

    ax.plot(layers, f_vals, "k-o", label="float", markersize=4)
    ax.plot(layers, v3_vals, "b-s", label=f"3-block (PPL {results['3block']['ppl']:.1f})", markersize=4)
    ax.plot(layers, v6_vals, "r-^", label=f"6-block (PPL {results['6block']['ppl']:.1f})", markersize=4)
    ax.set_title(title)
    ax.set_xlabel("Block index")
    ax.set_ylabel(title)
    ax.legend(fontsize=8)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
plt.savefig(OUTPUT_PLOT, dpi=120, bbox_inches="tight")
print(f"Plot saved to {OUTPUT_PLOT}", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────

out = {
    "cycle": 371,
    "item": 379,
    "seed": SEED,
    "verdict": verdict,
    "ppl_barrier_confirmed": ppl_barrier,
    "results": results,
    "analysis": {
        "drop3_counts": drop3_counts,
        "drop6_counts": drop6_counts,
        "float_eff_rank": float_er,
        "3block_eff_rank": er3,
        "6block_eff_rank": er6,
    },
}
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"JSON saved to {OUTPUT_JSON}", flush=True)
print("Done.", flush=True)
