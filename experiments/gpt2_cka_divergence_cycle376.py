#!/usr/bin/env python3
"""Cycle 376 / Item 384 — Per-token CKA and trajectory divergence: 3-block vs 6-block.

Hypothesis: CKA gap > 0.1 at some layer between 3-block and 6-block compressed models
            compared to float, indicating representation divergence underlying composition barrier.

Configs: float | 3-block [2,1,10] | 6-block [2,1,10,4,3,11]
Adapter: topk(k=512)+lstsq + MLP adapter hidden=64, 500 Adam steps, lr=3e-4

Metrics at each of 12 block boundaries:
  1. Linear CKA(float_h[l], comp_h[l]) — how similar compressed repr is to float
  2. Per-token L2 divergence: mean, p95, max of ||float_h[l] - comp_h[l]||_2

Verdict:
  CONFIRMED   if max(cka_3block[l] - cka_6block[l]) > 0.1 at any layer
  REFUTED     if max gap < 0.05
  INCONCLUSIVE otherwise
"""
import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
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
N_CKA = 200   # sequences for CKA measurement (200 * 128 = 25600 tokens, subsample to 4096)
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 500
LR = 3e-4
EVAL_EVERY = 100
SEED = 42

FLOAT_BASELINE_REF = 53.895
OUTPUT_JSON = "autoresearch/results/cycle_376_cka_divergence.json"
OUTPUT_PLOT = "autoresearch/results/cycle_376_cka_divergence.png"


# ── CKA and L2 utilities ──────────────────────────────────────────────────────

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between (N, D) tensors X and Y (centered).
    CKA(X, Y) = ||Y'X||^2_F / (||X'X||_F * ||Y'Y||_F)
    """
    X = X.float()
    Y = Y.float()
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = (X.T @ Y).pow(2).sum()
    hsic_xx = (X.T @ X).pow(2).sum()
    hsic_yy = (Y.T @ Y).pow(2).sum()
    denom = (hsic_xx * hsic_yy).sqrt()
    if denom < 1e-20:
        return float("nan")
    return (hsic_xy / denom).item()


def l2_stats(float_h: torch.Tensor, comp_h: torch.Tensor) -> dict:
    """Per-token L2 divergence statistics."""
    diffs = (float_h.float() - comp_h.float()).norm(dim=1)
    return {
        "mean": diffs.mean().item(),
        "p95": diffs.quantile(0.95).item(),
        "max": diffs.max().item(),
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
    # Reuse calib seqs for CKA measurement
    cka_ids_dev = [t.to(DEVICE) for t in all_ids[:N_CKA]]
    return calib_ids_cpu, eval_ids_dev, cka_ids_dev


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


# ── Hidden collection for CKA ─────────────────────────────────────────────────

def collect_all_hidden(model, cka_ids_dev, max_tokens=4096, seed=0):
    """Hook all 12 transformer block outputs, collect residual stream (N_tokens, 768) per layer.
    Subsample to max_tokens for CKA efficiency.
    Returns: dict {layer_idx: Tensor(max_tokens, 768)} on CPU.
    """
    n_blocks = len(model.transformer.h)
    layer_data = {i: [] for i in range(n_blocks)}
    hooks = []

    def make_hook(blk_i):
        def fn(module, inputs, outputs):
            if isinstance(outputs, tuple):
                h = outputs[0]
            else:
                h = outputs
            layer_data[blk_i].append(h.detach().reshape(-1, h.shape[-1]).cpu())
        return fn

    for i in range(n_blocks):
        hook = model.transformer.h[i].register_forward_hook(make_hook(i))
        hooks.append(hook)

    model.eval()
    with torch.no_grad():
        for ids in cka_ids_dev:
            model(ids)

    for hook in hooks:
        hook.remove()

    result = {}
    gen = torch.Generator()
    gen.manual_seed(seed)
    for i in range(n_blocks):
        H = torch.cat(layer_data[i], dim=0).float()
        if H.shape[0] > max_tokens:
            idx = torch.randperm(H.shape[0], generator=gen)[:max_tokens]
            H = H[idx]
        result[i] = H
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
random.seed(SEED)
print(f"Device: {DEVICE}", flush=True)

calib_ids_cpu, eval_ids_dev, cka_ids_dev = load_data()
calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]

results = {}
n_layers = 12

# 1. Float baseline
print("\n" + "="*60, flush=True)
print("Config: float (no compression)", flush=True)
print("="*60, flush=True)
float_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
float_ppl = eval_ppl(float_model, eval_ids_dev)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
print("Collecting float hidden states...", flush=True)
float_hidden = collect_all_hidden(float_model, cka_ids_dev)
del float_model
if DEVICE == "mps":
    torch.mps.empty_cache()

# 2. 3-block compression
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
print("Collecting 3-block hidden states...", flush=True)
hidden3 = collect_all_hidden(model3, cka_ids_dev)
del model3
if DEVICE == "mps":
    torch.mps.empty_cache()

# 3. 6-block compression
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
print("Collecting 6-block hidden states...", flush=True)
hidden6 = collect_all_hidden(model6, cka_ids_dev)
del model6
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── CKA and L2 computation ────────────────────────────────────────────────────

print("\n" + "="*60, flush=True)
print("CKA and L2 divergence analysis", flush=True)
print("="*60, flush=True)

cka_3block = []
cka_6block = []
l2_3block = []
l2_6block = []

print(f"{'Layer':>6} {'CKA-3blk':>10} {'CKA-6blk':>10} {'gap':>8} {'L2-3blk-mean':>14} {'L2-6blk-mean':>14}", flush=True)
for i in range(n_layers):
    fh = float_hidden[i]
    h3 = hidden3[i]
    h6 = hidden6[i]

    c3 = linear_cka(fh, h3)
    c6 = linear_cka(fh, h6)
    d3 = l2_stats(fh, h3)
    d6 = l2_stats(fh, h6)

    cka_3block.append(c3)
    cka_6block.append(c6)
    l2_3block.append(d3)
    l2_6block.append(d6)

    gap = (c3 - c6) if (not math.isnan(c3) and not math.isnan(c6)) else float("nan")
    print(f"  {i:4d}  {c3:10.4f}  {c6:10.4f}  {gap:8.4f}  {d3['mean']:14.4f}  {d6['mean']:14.4f}",
          flush=True)

# Verdict
valid_gaps = [(c3 - c6) for c3, c6 in zip(cka_3block, cka_6block)
              if not math.isnan(c3) and not math.isnan(c6)]
max_gap = max(valid_gaps) if valid_gaps else float("nan")
gap_layer = valid_gaps.index(max_gap) if valid_gaps else -1

print(f"\nMax CKA gap (3-block - 6-block): {max_gap:.4f} at layer {gap_layer}", flush=True)

if math.isnan(max_gap):
    verdict = "INCONCLUSIVE"
elif max_gap > 0.1:
    verdict = "CONFIRMED"
elif max_gap < 0.05:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"VERDICT: {verdict}", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────

output = {
    "float_ppl": float(float_ppl),
    "ppl_3block": float(ppl3),
    "ppl_6block": float(ppl6),
    "cka_3block": [float(v) for v in cka_3block],
    "cka_6block": [float(v) for v in cka_6block],
    "l2_3block": l2_3block,
    "l2_6block": l2_6block,
    "max_cka_gap": float(max_gap),
    "gap_layer": int(gap_layer),
    "verdict": verdict,
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved JSON: {OUTPUT_JSON}", flush=True)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"CKA vs Float — 3-block (PPL {ppl3:.1f}) vs 6-block (PPL {ppl6:.1f}) | "
    f"Max gap={max_gap:.3f} @ layer {gap_layer} → {verdict}",
    fontsize=11
)

layers = list(range(n_layers))

# Top panel: CKA
ax0 = axes[0]
ax0.plot(layers, cka_3block, "b-o", label=f"3-block (PPL {ppl3:.1f})", markersize=5)
ax0.plot(layers, cka_6block, "r-^", label=f"6-block (PPL {ppl6:.1f})", markersize=5)
ax0.set_xlabel("Layer")
ax0.set_ylabel("Linear CKA (float hidden, compressed hidden)")
ax0.set_title("CKA vs Float")
ax0.set_ylim(0, 1.05)
ax0.legend()
ax0.grid(True, alpha=0.3)
# Mark gap layer
if gap_layer >= 0:
    ax0.axvline(x=gap_layer, color="gray", linestyle="--", alpha=0.5, label=f"max gap layer {gap_layer}")

# Bottom panel: L2 divergence
ax1 = axes[1]
l2_3_mean = [d["mean"] for d in l2_3block]
l2_3_p95 = [d["p95"] for d in l2_3block]
l2_6_mean = [d["mean"] for d in l2_6block]
l2_6_p95 = [d["p95"] for d in l2_6block]

ax1.plot(layers, l2_3_mean, "b-o", label="3-block mean", markersize=5)
ax1.fill_between(layers, l2_3_mean, l2_3_p95, alpha=0.15, color="blue", label="3-block p95")
ax1.plot(layers, l2_6_mean, "r-^", label="6-block mean", markersize=5)
ax1.fill_between(layers, l2_6_mean, l2_6_p95, alpha=0.15, color="red", label="6-block p95")
ax1.set_xlabel("Layer")
ax1.set_ylabel("Per-token L2 divergence from float")
ax1.set_title("L2 Divergence vs Float")
ax1.legend()
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=120, bbox_inches="tight")
print(f"Saved plot: {OUTPUT_PLOT}", flush=True)
print("Done.", flush=True)
