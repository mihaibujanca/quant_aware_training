#!/usr/bin/env python3
"""Cycle 365 / Item 374 — Adaptive k per block for GPT-2 6-block compression.

Allocate k proportional to per-block Frobenius error from cycle 362, with
k_min=256 floor, holding total budget at 3072 (= 6×512 uniform).

Adaptive k (proportional to frob, from cycle 362 JSON):
  block 1:  256 (floored from 254)
  block 2:  271
  block 3:  302
  block 4:  308
  block 10: 897
  block 11: 1040
  (adjusted to sum=3072: +2 to block 10 → 899, or similar rounding)

Sqrt-proportional fallback (automatic if adaptive PPL > 65.54):
  k_i ~ sqrt(frob_i), same floor=256, same budget=3072

Verdict:
  CONFIRMED    if adaptive mean PPL < 60.0
  REFUTED      if 60.0 <= adaptive mean PPL <= 65.54 (marginal gain) or
               adaptive mean PPL > 65.54 (worse than uniform)

Usage:
  python experiments/gpt2_adaptive_k_cycle374.py [--uniform] [--sqrt]
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
ALL_BLOCKS = [2, 1, 10, 4, 3, 11]

# Frobenius norms from cycle 362
FROB_NORMS = {2: 1637.14, 1: 1385.25, 10: 5423.82, 4: 1860.59, 3: 1822.63, 11: 6293.04}

TOTAL_BUDGET = 6 * 512  # 3072
K_MIN = 256

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
UNIFORM_6BLOCK_REF = 65.54  # cycle 361

OUTPUT_PATH = "autoresearch/results/cycle_365_adaptive_k.json"


def compute_adaptive_k(frob_dict, budget, k_min, use_sqrt=False):
    """Compute per-block k proportional to frob (or sqrt(frob)), with floor and budget constraint."""
    if use_sqrt:
        scores = {b: math.sqrt(f) for b, f in frob_dict.items()}
    else:
        scores = dict(frob_dict)

    total_score = sum(scores.values())
    raw = {b: budget * s / total_score for b, s in scores.items()}

    # Apply floor — this may increase some values
    floored = {b: max(k_min, round(v)) for b, v in raw.items()}
    current_sum = sum(floored.values())
    diff = budget - current_sum

    if diff > 0:
        # Budget surplus: distribute extra to high-score blocks
        # Only blocks not clamped to floor
        eligible = {b: scores[b] for b in floored if floored[b] > k_min}
        if not eligible:
            eligible = dict(scores)
        total_e = sum(eligible.values())
        # Integer distribute
        final_k = dict(floored)
        remainder = diff
        for b in sorted(eligible, key=lambda b: eligible[b], reverse=True):
            add = round(diff * eligible[b] / total_e)
            add = min(add, remainder)
            final_k[b] += add
            remainder -= add
        if remainder != 0:
            max_b = max(scores, key=lambda b: scores[b])
            final_k[max_b] += remainder
    elif diff < 0:
        # Budget deficit (flooring pushed things over): reduce high-score non-floored blocks
        final_k = dict(floored)
        excess = -diff  # positive
        for b in sorted(scores, key=lambda b: scores[b], reverse=True):
            if final_k[b] > k_min and excess > 0:
                reduce = min(final_k[b] - k_min, excess)
                final_k[b] -= reduce
                excess -= reduce
        if excess > 0:
            # Edge case: reduce the largest block further
            max_b = max(scores, key=lambda b: scores[b])
            final_k[max_b] -= excess
    else:
        final_k = floored

    # Final floor enforcement
    final_k = {b: max(k_min, v) for b, v in final_k.items()}
    # Adjust to hit exact budget
    final_sum = sum(final_k.values())
    if final_sum != budget:
        max_b = max(scores, key=lambda b: scores[b])
        final_k[max_b] += budget - final_sum

    assert sum(final_k.values()) == budget, f"k sum={sum(final_k.values())} != {budget}"
    assert min(final_k.values()) >= k_min, f"k below floor: {final_k}"
    return final_k


# Pre-compute allocations
ADAPTIVE_K = compute_adaptive_k(FROB_NORMS, TOTAL_BUDGET, K_MIN, use_sqrt=False)
SQRT_K = compute_adaptive_k(FROB_NORMS, TOTAL_BUDGET, K_MIN, use_sqrt=True)
UNIFORM_K = {b: 512 for b in ALL_BLOCKS}


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
    """Compressed MLP with MLP adapter."""

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
    """Run topk+lstsq compression. Returns per-block stats and compressed module dicts."""
    block_acts = collect_block_activations(model, block_ids, calib_ids_dev, device)
    compress_cache = {}
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

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

        fc_w = c_fc_weight_full.T[selected]   # (k, 768)
        fc_b = c_fc_bias_full[selected]        # (k,)

        hidden_sel = hidden[:, selected].float()
        design = torch.cat([hidden_sel, torch.ones(hidden_sel.shape[0], 1)], dim=1)
        solution = torch.linalg.lstsq(design, output.float()).solution
        proj_w = solution[:-1].T.contiguous()  # (768, k)
        proj_b = solution[-1].contiguous()      # (768,)

        with torch.no_grad():
            recon = hidden_sel @ proj_w.T + proj_b
            mse = F.mse_loss(recon, output).item()
            frob_recon = (recon - output).norm().item()

        compress_cache[blk] = {
            "fc_w": fc_w, "fc_b": fc_b,
            "proj_w": proj_w, "proj_b": proj_b,
            "selected": selected, "k": k,
            "mse": mse, "frob_recon": frob_recon,
        }
        print(f"  Block {blk}: k={k} (budget frac={k/hidden_dim:.3f}), "
              f"MSE={mse:.4f}, frob_recon={frob_recon:.2f}",
              flush=True)

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
    """Train proj_weight + adapter params for all compressed_modules jointly."""
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
    print(f"  [{phase_name}] trainable params: {n_train:,}", flush=True)

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
    t0 = time.time()
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

    # Restore best
    with torch.no_grad():
        for blk, mod in compressed_modules.items():
            for pname in ["proj_weight", "proj_bias",
                          "adapter_W1", "adapter_b1", "adapter_W2", "adapter_b2"]:
                getattr(mod, pname).data.copy_(best_state[f"{pname}_{blk}"])

    restored_ppl = eval_ppl(model, eval_ids_dev)
    dt = time.time() - t0
    print(f"  [{phase_name}] init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
          f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)
    return best_ppl, restored_ppl


def run_experiment(k_per_block, seeds, calib_ids_cpu, eval_ids_dev, mode_name):
    """Run compression with given k_per_block over multiple seeds. Returns results dict."""
    print(f"\n{'='*60}", flush=True)
    print(f"Mode: {mode_name}", flush=True)
    print(f"k per block: {k_per_block}", flush=True)
    print(f"k sum: {sum(k_per_block.values())} (budget: {TOTAL_BUDGET})", flush=True)
    print(f"{'='*60}", flush=True)

    seed_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        random.seed(seed)

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]

        print("  Compressing all 6 blocks...", flush=True)
        compress_cache = compress_blocks(model, ALL_BLOCKS, k_per_block, calib_ids_dev, DEVICE)
        compressed_modules = install_compressed_blocks(model, ALL_BLOCKS, compress_cache, DEVICE)

        per_block_frob = {b: compress_cache[b]["frob_recon"] for b in ALL_BLOCKS}

        best_ppl, restored_ppl = train_joint(
            model, compressed_modules, N_STEPS, LR, calib_ids_dev, eval_ids_dev,
            phase_name=f"{mode_name}-seed{seed}"
        )

        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()

        seed_results.append({
            "seed": seed,
            "best_ppl": float(best_ppl),
            "restored_ppl": float(restored_ppl),
            "per_block_frob_recon": {str(b): float(v) for b, v in per_block_frob.items()},
            "per_block_k": {str(b): int(k_per_block[b]) for b in ALL_BLOCKS},
        })
        print(f"  Seed {seed}: best_ppl={best_ppl:.4f} restored={restored_ppl:.4f}", flush=True)

    ppls = [r["best_ppl"] for r in seed_results]
    mean_ppl = sum(ppls) / len(ppls)
    std_ppl = (sum((p - mean_ppl) ** 2 for p in ppls) / len(ppls)) ** 0.5

    print(f"\n{mode_name} summary: mean PPL={mean_ppl:.4f} ± {std_ppl:.4f}", flush=True)
    return {
        "mode": mode_name,
        "k_per_block": {str(b): int(k_per_block[b]) for b in ALL_BLOCKS},
        "k_sum": sum(k_per_block.values()),
        "seed_results": seed_results,
        "mean_ppl": float(mean_ppl),
        "std_ppl": float(std_ppl),
    }


# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--uniform", action="store_true", help="Run uniform k=512 baseline")
parser.add_argument("--sqrt", action="store_true", help="Run sqrt-proportional allocation")
args = parser.parse_args()

print(f"Device: {DEVICE}", flush=True)
print(f"Adaptive k: {ADAPTIVE_K}", flush=True)
print(f"Sqrt k:     {SQRT_K}", flush=True)
print(f"Uniform k:  {UNIFORM_K}", flush=True)

# ── Load data ─────────────────────────────────────────────────────────────────
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
eval_ids_dev = [t.to(DEVICE) for t in all_ids[N_CALIB: N_CALIB + N_EVAL]]
print(f"Calibration: {len(calib_ids_cpu)} seqs, Eval: {len(eval_ids_dev)} seqs", flush=True)

# ── Float baseline ────────────────────────────────────────────────────────────
print("Computing float baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
float_ppl = eval_ppl(base_model, eval_ids_dev)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference by > 0.5"
)
del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Run experiments ───────────────────────────────────────────────────────────
all_results = {"float_ppl": float(float_ppl)}

if args.uniform:
    uniform_results = run_experiment(UNIFORM_K, SEEDS, calib_ids_cpu, eval_ids_dev, "uniform")
    all_results["uniform"] = uniform_results
elif args.sqrt:
    sqrt_results = run_experiment(SQRT_K, SEEDS, calib_ids_cpu, eval_ids_dev, "sqrt_proportional")
    all_results["sqrt_proportional"] = sqrt_results
else:
    # Main: adaptive run
    adaptive_results = run_experiment(ADAPTIVE_K, SEEDS, calib_ids_cpu, eval_ids_dev, "adaptive")
    all_results["adaptive"] = adaptive_results

    adaptive_mean = adaptive_results["mean_ppl"]
    adaptive_std = adaptive_results["std_ppl"]

    # Verdict
    if adaptive_mean < 60.0:
        verdict = "CONFIRMED"
    elif adaptive_mean <= UNIFORM_6BLOCK_REF:
        verdict = "REFUTED"
    else:
        verdict = "REFUTED-WORSE"

    all_results["verdict"] = verdict
    print(f"\n{'='*60}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"  Adaptive mean PPL: {adaptive_mean:.4f} ± {adaptive_std:.4f}", flush=True)
    print(f"  Thresholds: <60.0 CONFIRMED, <=65.54 REFUTED, >65.54 REFUTED-WORSE", flush=True)
    print(f"{'='*60}", flush=True)

    if adaptive_mean > UNIFORM_6BLOCK_REF:
        print("\nAuto-running sqrt-proportional fallback (adaptive PPL > uniform ref)...", flush=True)
        sqrt_results = run_experiment(SQRT_K, SEEDS, calib_ids_cpu, eval_ids_dev, "sqrt_proportional")
        all_results["sqrt_proportional"] = sqrt_results

    # Also run uniform baseline for comparison
    print("\nRunning uniform k=512 control baseline...", flush=True)
    uniform_results = run_experiment(UNIFORM_K, SEEDS, calib_ids_cpu, eval_ids_dev, "uniform")
    all_results["uniform"] = uniform_results

    print(f"\n{'='*60}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"  Float PPL:             {float_ppl:.4f}", flush=True)
    print(f"  Adaptive (mean ± std): {adaptive_mean:.4f} ± {adaptive_std:.4f}", flush=True)
    print(f"  Uniform ref:           {UNIFORM_6BLOCK_REF}", flush=True)
    if "sqrt_proportional" in all_results:
        sq = all_results["sqrt_proportional"]
        print(f"  Sqrt-prop (mean±std):  {sq['mean_ppl']:.4f} ± {sq['std_ppl']:.4f}", flush=True)
    if "uniform" in all_results:
        uni = all_results["uniform"]
        print(f"  Uniform (mean ± std):  {uni['mean_ppl']:.4f} ± {uni['std_ppl']:.4f}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"{'='*60}", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out = {
    "cycle": 365,
    "item": 374,
    "float_ppl": float(float_ppl),
    "adaptive_k": {str(b): int(ADAPTIVE_K[b]) for b in ALL_BLOCKS},
    "sqrt_k": {str(b): int(SQRT_K[b]) for b in ALL_BLOCKS},
    "uniform_k": {str(b): 512 for b in ALL_BLOCKS},
    "frob_norms": {str(b): FROB_NORMS[b] for b in ALL_BLOCKS},
    "results": all_results,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {OUTPUT_PATH}", flush=True)
print("Done.", flush=True)
