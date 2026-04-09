#!/usr/bin/env python3
"""Cycle 366 / Item 375 — Residual stream error propagation monitor.

Trace per-token error through GPT-2's full 12-block forward pass.
Compare float model vs 6-block compressed model (blocks [2,1,10,4,3,11],
uniform topk+lstsq+adapter, k=512, adapter h=64).

Measures:
  - ||h_compressed - h_float|| (absolute norm) at each block output
  - Additive baseline: cumsum of local FFN errors (non-compressed blocks = 0)
  - Growth rate g_i = err_{i+1} - err_i
  - Inflection block: first block where g_i >= 2 * median(g)

Verdict:
  CONFIRMED    if inflection block exists (g_i >= 2x median) at same index in >= 2/3 seeds
  REFUTED      if no inflection block in any seed, OR block differs across seeds
  INCONCLUSIVE if error is non-monotonic (decreases at 2+ blocks)

Usage:
  MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/gpt2_residual_error_monitor_cycle375.py
"""
import copy
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
ALL_BLOCKS = [2, 1, 10, 4, 3, 11]
K_UNIFORM = 512
ADAPTER_HIDDEN = 64
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 1500
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]

FLOAT_BASELINE_REF = 53.895
OUTPUT_JSON = "autoresearch/results/cycle_366_residual_error_monitor.json"
OUTPUT_PNG  = "autoresearch/results/cycle_366_error_profile.png"


# ── Compression infrastructure (from cycle374 template) ──────────────────────

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


def collect_block_activations(model, block_ids, calib_ids, device):
    inv_sqrt_2    = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi  = 1.0 / math.sqrt(2.0 * math.pi)
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
        for ids in calib_ids:
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
    def __init__(self, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias,
                 adapter_hidden=64):
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


def compress_blocks(model, block_ids, k_per_block, calib_ids_dev, device):
    inv_sqrt_2    = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi  = 1.0 / math.sqrt(2.0 * math.pi)
    block_acts = collect_block_activations(model, block_ids, calib_ids_dev, device)
    compress_cache = {}

    for blk in block_ids:
        acts = block_acts[blk]
        hidden = acts["hidden"]
        output = acts["output"]
        preact = acts["preact"]

        k = k_per_block[blk]
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
        c_fc_weight_full = c_fc.weight.detach().cpu().float()
        c_fc_bias_full = c_fc.bias.detach().cpu().float()

        fc_w = c_fc_weight_full.T[selected]
        fc_b = c_fc_bias_full[selected]

        hidden_sel = hidden[:, selected].float()
        design = torch.cat([hidden_sel, torch.ones(hidden_sel.shape[0], 1)], dim=1)
        solution = torch.linalg.lstsq(design, output.float()).solution
        proj_w = solution[:-1].T.contiguous()
        proj_b = solution[-1].contiguous()

        with torch.no_grad():
            recon = hidden_sel @ proj_w.T + proj_b
            frob_recon = (recon - output).norm().item()

        compress_cache[blk] = {
            "fc_w": fc_w, "fc_b": fc_b,
            "proj_w": proj_w, "proj_b": proj_b,
            "selected": selected, "k": k,
            "frob_recon": frob_recon,
        }
        print(f"  Block {blk}: k={k}, frob_recon={frob_recon:.2f}", flush=True)

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


def train_joint(model, compressed_modules, n_steps, lr, calib_ids_dev, eval_ids_dev,
                phase_name="joint"):
    for p in model.parameters():
        p.requires_grad_(False)

    trainable = []
    for mod in compressed_modules.values():
        for pname in ["proj_weight", "proj_bias",
                      "adapter_W1", "adapter_b1", "adapter_W2", "adapter_b2"]:
            p = getattr(mod, pname)
            p.requires_grad_(True)
            trainable.append(p)

    init_ppl = eval_ppl(model, eval_ids_dev)
    print(f"  [{phase_name}] init PPL: {init_ppl:.4f}", flush=True)

    optimizer = torch.optim.Adam(trainable, lr=lr)
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
    while step < n_steps:
        for ids in ids_shuffled:
            if step >= n_steps:
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
                if step % 500 == 0:
                    print(f"  [{phase_name}] step {step}: ppl={ppl_e:.2f} "
                          f"best={best_ppl:.2f}@{best_step}", flush=True)

    with torch.no_grad():
        for blk, mod in compressed_modules.items():
            for pname in ["proj_weight", "proj_bias",
                          "adapter_W1", "adapter_b1", "adapter_W2", "adapter_b2"]:
                getattr(mod, pname).data.copy_(best_state[f"{pname}_{blk}"])

    restored_ppl = eval_ppl(model, eval_ids_dev)
    print(f"  [{phase_name}] init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
          f"restored={restored_ppl:.2f}", flush=True)
    return best_ppl, restored_ppl


# ── Error measurement infrastructure ─────────────────────────────────────────

def measure_residual_errors(float_model, comp_model, eval_ids_dev, device):
    """
    Measure per-block residual stream error ||h_comp - h_float|| and
    local FFN error at compressed blocks, averaged over all eval tokens.

    Returns:
      cumulative_error: list of 12 floats (block 0..11, block output norms)
      additive_baseline: list of 12 floats (cumsum of local FFN errors)
      local_ffn_errors: dict {block_idx: float}
    """
    n_blocks = 12

    # Storage for float and compressed block outputs
    float_block_outputs  = {i: [] for i in range(n_blocks)}  # (T, 768)
    comp_block_outputs   = {i: [] for i in range(n_blocks)}

    # Storage for local FFN errors at compressed blocks
    # We capture: float_ffn_out, comp_ffn_out  at each compressed block
    float_ffn_out = {b: [] for b in ALL_BLOCKS}
    comp_ffn_out  = {b: [] for b in ALL_BLOCKS}

    float_hooks = []
    comp_hooks  = []

    def make_block_hook(storage, blk_idx):
        def hook_fn(module, inputs, output):
            # output is a tuple (tensor, ...) for transformer blocks
            if isinstance(output, tuple):
                h = output[0].detach()
            else:
                h = output.detach()
            storage[blk_idx].append(h.reshape(-1, h.shape[-1]).cpu())
        return hook_fn

    def make_ffn_hook(storage, blk_idx):
        def hook_fn(module, inputs, output):
            y = output.detach() if not isinstance(output, tuple) else output[0].detach()
            storage[blk_idx].append(y.reshape(-1, y.shape[-1]).cpu())
        return hook_fn

    for i in range(n_blocks):
        float_hooks.append(
            float_model.transformer.h[i].register_forward_hook(
                make_block_hook(float_block_outputs, i)))
        comp_hooks.append(
            comp_model.transformer.h[i].register_forward_hook(
                make_block_hook(comp_block_outputs, i)))

    for b in ALL_BLOCKS:
        float_hooks.append(
            float_model.transformer.h[b].mlp.register_forward_hook(
                make_ffn_hook(float_ffn_out, b)))
        comp_hooks.append(
            comp_model.transformer.h[b].mlp.register_forward_hook(
                make_ffn_hook(comp_ffn_out, b)))

    float_model.eval()
    comp_model.eval()
    with torch.no_grad():
        for ids in eval_ids_dev:
            ids_dev = ids.to(device)
            float_model(ids_dev)
            comp_model(ids_dev)

    for h in float_hooks + comp_hooks:
        h.remove()

    # Concatenate across batches
    float_blk = {i: torch.cat(float_block_outputs[i], dim=0) for i in range(n_blocks)}
    comp_blk  = {i: torch.cat(comp_block_outputs[i], dim=0)  for i in range(n_blocks)}
    float_ffn = {b: torch.cat(float_ffn_out[b], dim=0) for b in ALL_BLOCKS}
    comp_ffn_c = {b: torch.cat(comp_ffn_out[b], dim=0) for b in ALL_BLOCKS}

    # Validation: float vs float should be < 1e-5
    # (We're running two separate float_model calls; actually the same model, so error=0)

    # Per-block cumulative error norm (mean over tokens)
    cumulative_error = []
    for i in range(n_blocks):
        diff = comp_blk[i] - float_blk[i]
        norm_per_token = diff.norm(dim=-1)      # (T,)
        cumulative_error.append(norm_per_token.mean().item())

    # Local FFN error at compressed blocks (mean over tokens)
    local_ffn_errors = {}
    for b in ALL_BLOCKS:
        diff_ffn = comp_ffn_c[b] - float_ffn[b]
        local_ffn_errors[b] = diff_ffn.norm(dim=-1).mean().item()

    # Additive baseline: cumsum of local FFN errors in block order (0..11)
    # Non-compressed blocks contribute 0
    additive_baseline = []
    running = 0.0
    for i in range(n_blocks):
        if i in ALL_BLOCKS:
            running += local_ffn_errors[i]
        additive_baseline.append(running)

    return cumulative_error, additive_baseline, local_ffn_errors


def find_inflection_block(cumulative_error):
    """
    g_i = err_{i+1} - err_i for i in 0..10
    inflection = first i where g_i >= 2 * median(g)
    Returns (inflection_block_idx or None, growth_rates, is_non_monotonic)
    """
    errs = cumulative_error
    growth = [errs[i+1] - errs[i] for i in range(len(errs)-1)]

    # Check non-monotonic: decreases at >= 2 blocks
    decreases = sum(1 for g in growth if g < 0)
    is_non_monotonic = decreases >= 2

    sorted_g = sorted(growth)
    n = len(sorted_g)
    if n % 2 == 1:
        median_g = sorted_g[n // 2]
    else:
        median_g = (sorted_g[n//2 - 1] + sorted_g[n//2]) / 2.0

    inflection = None
    for i, g in enumerate(growth):
        if g >= 2.0 * median_g:
            inflection = i  # block i+1 is the block after the big jump
            break

    return inflection, growth, is_non_monotonic


# ── Main experiment ───────────────────────────────────────────────────────────

print(f"Device: {DEVICE}", flush=True)
print("Loading GPT-2 tokenizer and wikitext-2 data...", flush=True)

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

calib_ids_cpu = list(all_ids[:N_CALIB])
eval_ids_dev  = [t.to(DEVICE) for t in all_ids[N_CALIB: N_CALIB + N_EVAL]]
print(f"Calibration: {len(calib_ids_cpu)} seqs, Eval: {len(eval_ids_dev)} seqs", flush=True)

# Float baseline PPL check
print("Computing float baseline PPL...", flush=True)
_check_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
_check_model.eval()
float_ppl_check = eval_ppl(_check_model, eval_ids_dev)
print(f"Float PPL: {float_ppl_check:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_ppl_check - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl_check} differs from reference by > 0.5"
)
del _check_model
if DEVICE == "mps":
    torch.mps.empty_cache()

k_per_block = {b: K_UNIFORM for b in ALL_BLOCKS}

all_seed_results = []

for seed in SEEDS:
    print(f"\n{'='*60}", flush=True)
    print(f"Seed {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(seed)
    random.seed(seed)

    # Load fresh float model (kept for error measurement)
    float_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    float_model.eval()

    # Load compressed model
    comp_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]

    print("  Compressing 6 blocks...", flush=True)
    compress_cache = compress_blocks(comp_model, ALL_BLOCKS, k_per_block, calib_ids_dev, DEVICE)
    compressed_modules = install_compressed_blocks(comp_model, ALL_BLOCKS, compress_cache, DEVICE)

    print("  Training adapters...", flush=True)
    best_ppl, restored_ppl = train_joint(
        comp_model, compressed_modules, N_STEPS, LR, calib_ids_dev, eval_ids_dev,
        phase_name=f"seed{seed}"
    )

    # Measure residual stream errors
    print("  Measuring residual stream errors...", flush=True)
    cumulative_error, additive_baseline, local_ffn_errors = measure_residual_errors(
        float_model, comp_model, eval_ids_dev, DEVICE
    )

    print("  Cumulative error per block:", flush=True)
    for i, (e, b) in enumerate(zip(cumulative_error, additive_baseline)):
        tag = "*" if i in ALL_BLOCKS else " "
        print(f"    {tag}Block {i:2d}: cumul={e:.4f}  additive_baseline={b:.4f}", flush=True)

    inflection, growth_rates, is_non_monotonic = find_inflection_block(cumulative_error)
    print(f"  Growth rates: {[f'{g:.4f}' for g in growth_rates]}", flush=True)
    print(f"  Inflection block: {inflection}  non-monotonic: {is_non_monotonic}", flush=True)

    all_seed_results.append({
        "seed": seed,
        "best_ppl": float(best_ppl),
        "restored_ppl": float(restored_ppl),
        "cumulative_error": [float(x) for x in cumulative_error],
        "additive_baseline": [float(x) for x in additive_baseline],
        "local_ffn_errors": {str(b): float(v) for b, v in local_ffn_errors.items()},
        "growth_rates": [float(g) for g in growth_rates],
        "inflection_block": inflection,
        "is_non_monotonic": is_non_monotonic,
    })

    del float_model, comp_model
    if DEVICE == "mps":
        torch.mps.empty_cache()

# ── Verdict ───────────────────────────────────────────────────────────────────

non_mono_count = sum(1 for r in all_seed_results if r["is_non_monotonic"])
inflection_blocks = [r["inflection_block"] for r in all_seed_results if r["inflection_block"] is not None]

print(f"\n{'='*60}", flush=True)
print("VERDICT ANALYSIS", flush=True)
print(f"Non-monotonic seeds: {non_mono_count}/{len(SEEDS)}", flush=True)
print(f"Inflection blocks found: {inflection_blocks}", flush=True)

if non_mono_count >= 2:
    verdict = "INCONCLUSIVE"
    verdict_reason = "Error non-monotonic in >= 2 seeds"
elif len(inflection_blocks) == 0:
    verdict = "REFUTED"
    verdict_reason = "No inflection block found in any seed"
else:
    # Check if same block in >= 2/3 seeds
    from collections import Counter
    counts = Counter(inflection_blocks)
    most_common_block, count = counts.most_common(1)[0]
    if count >= 2:
        verdict = "CONFIRMED"
        verdict_reason = f"Inflection block {most_common_block} at same index in {count}/{len(SEEDS)} seeds"
    else:
        verdict = "REFUTED"
        verdict_reason = f"Inflection blocks differ across seeds: {inflection_blocks}"

print(f"\nVERDICT: {verdict}", flush=True)
print(f"Reason: {verdict_reason}", flush=True)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax2 = axes[1]

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
block_range = list(range(12))

for i, r in enumerate(all_seed_results):
    seed = r["seed"]
    ax1.plot(block_range, r["cumulative_error"], "o-", color=colors[i],
             label=f"Seed {seed} actual", linewidth=2)
    ax1.plot(block_range, r["additive_baseline"], "--", color=colors[i],
             label=f"Seed {seed} additive", linewidth=1.5, alpha=0.7)

for blk in ALL_BLOCKS:
    ax1.axvline(x=blk, color="gray", alpha=0.3, linestyle=":")

ax1.set_xlabel("Block index")
ax1.set_ylabel("Mean ||h_comp - h_float|| per token")
ax1.set_title("Residual Stream Error vs Block\n(solid=actual, dashed=additive baseline)")
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)

# Growth rate plot
for i, r in enumerate(all_seed_results):
    seed = r["seed"]
    ax2.plot(range(len(r["growth_rates"])), r["growth_rates"], "o-",
             color=colors[i], label=f"Seed {seed}", linewidth=2)

ax2.axhline(y=0, color="black", linewidth=0.8)
ax2.set_xlabel("Block transition (i → i+1)")
ax2.set_ylabel("g_i = err_{i+1} - err_i")
ax2.set_title(f"Error Growth Rate\nVerdict: {verdict}")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight")
print(f"Saved plot to {OUTPUT_PNG}", flush=True)

# ── Summary stats ──────────────────────────────────────────────────────────────

mean_ppl = sum(r["best_ppl"] for r in all_seed_results) / len(all_seed_results)
std_ppl  = (sum((r["best_ppl"] - mean_ppl) ** 2 for r in all_seed_results) / len(all_seed_results)) ** 0.5

# Mean cumulative error per block
mean_cumul = [
    sum(r["cumulative_error"][i] for r in all_seed_results) / len(all_seed_results)
    for i in range(12)
]
mean_additive = [
    sum(r["additive_baseline"][i] for r in all_seed_results) / len(all_seed_results)
    for i in range(12)
]

print(f"\nMean compressed PPL: {mean_ppl:.4f} ± {std_ppl:.4f}", flush=True)
print(f"Mean cumulative error (last block): {mean_cumul[-1]:.4f}", flush=True)
print(f"Mean additive baseline (last block): {mean_additive[-1]:.4f}", flush=True)
print(f"Amplification ratio (actual/additive at block 11): "
      f"{mean_cumul[-1]/max(mean_additive[-1], 1e-9):.3f}", flush=True)

# ── Save JSON ──────────────────────────────────────────────────────────────────

output = {
    "float_ppl": float(float_ppl_check),
    "mean_compressed_ppl": float(mean_ppl),
    "std_compressed_ppl": float(std_ppl),
    "verdict": verdict,
    "verdict_reason": verdict_reason,
    "inflection_blocks_per_seed": inflection_blocks,
    "non_monotonic_seeds": non_mono_count,
    "compressed_blocks": ALL_BLOCKS,
    "k_uniform": K_UNIFORM,
    "mean_cumulative_error_per_block": [float(x) for x in mean_cumul],
    "mean_additive_baseline_per_block": [float(x) for x in mean_additive],
    "amplification_ratio_block11": float(mean_cumul[-1] / max(mean_additive[-1], 1e-9)),
    "seed_results": all_seed_results,
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved JSON to {OUTPUT_JSON}", flush=True)
print("Done.", flush=True)
