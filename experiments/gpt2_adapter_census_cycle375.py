#!/usr/bin/env python3
"""Cycle 375 — Item 383: Adapter dead-neuron census and exact pruning.

Train a 6-block topk+lstsq+adapter model (k=512, h=64, seed=42, 1500 steps),
then census each adapter's 64 ReLU hidden neurons:
  - dead: never activate on calibration data
  - always-active: always in linear regime
  - mixed: crosses ReLU boundary

Exact pruning:
  - Dead neurons: delete rows/cols (zero contribution — exact)
  - Always-active neurons: fold into linear term in proj (exact)

Verdict:
  CONFIRMED   if pct_prunable >= 10 AND ppl_delta < 0.1
  REFUTED     if pct_prunable < 5 OR ppl_delta > 0.5
  INCONCLUSIVE otherwise
"""
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
ALL_BLOCKS = [2, 1, 10, 4, 3, 11]
K = 512
ADAPTER_HIDDEN = 64
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 1500
LR = 1e-4
EVAL_EVERY = 100
SEED = 42

FLOAT_BASELINE_REF = 53.895
UNIFORM_6BLOCK_REF = 65.54

OUTPUT_PATH = "autoresearch/results/cycle_375_adapter_census.json"


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


def compress_blocks(model, block_ids, k_val, calib_ids_dev, device):
    k_per_block = {b: k_val for b in block_ids}
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

        compress_cache[blk] = {
            "fc_w": fc_w, "fc_b": fc_b,
            "proj_w": proj_w, "proj_b": proj_b,
            "selected": selected, "k": k,
        }
        print(f"  Block {blk}: k={k}", flush=True)

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


def train_joint(model, compressed_modules, n_steps, lr, calib_ids_dev, eval_ids_dev):
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
    print(f"  [train] trainable params: {n_train:,}", flush=True)

    init_ppl = eval_ppl(model, eval_ids_dev)
    print(f"  [train] init PPL: {init_ppl:.4f}", flush=True)

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
                    print(f"  [train] step {step}: ppl={ppl_e:.2f} best={best_ppl:.2f}@{best_step}", flush=True)

    # Restore best
    with torch.no_grad():
        for blk, mod in compressed_modules.items():
            for pname in ["proj_weight", "proj_bias",
                          "adapter_W1", "adapter_b1", "adapter_W2", "adapter_b2"]:
                getattr(mod, pname).data.copy_(best_state[f"{pname}_{blk}"])

    restored_ppl = eval_ppl(model, eval_ids_dev)
    dt = time.time() - t0
    print(f"  [train] init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
          f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)
    return restored_ppl


def census_adapters(model, block_ids, calib_ids_dev, device):
    """Census adapter neurons: dead / always-active / mixed."""
    census = {}

    # Collect pre-relu adapter activations via hooks
    pre_relu_data = {b: [] for b in block_ids}
    hooks = []

    def make_hook(blk_id):
        def hook_fn(module, inputs, output):
            # Recompute adapter pre-relu from inside the module
            # h = gelu(fc_weight @ x + fc_bias), then adapter_W1 @ h + adapter_b1
            h_gelu = inputs[0].detach()  # this is the gelu output passed into the forward
            # We need to intercept the adapter pre-relu from the module directly
            pass
        return hook_fn

    # Better: intercept at the adapter level by temporarily wrapping forward
    # Instead, hook adapter_W1 output directly
    # We run inference and manually collect pre_relu for each block

    model.eval()
    with torch.no_grad():
        for ids in calib_ids_dev:
            ids_dev = ids.to(device)
            # Use forward hooks on CompressedGPT2MLPWithAdapter
            batch_pre_relu = {}

            def make_pre_relu_hook(blk_id):
                def hook_fn(module, inputs, output):
                    # inputs[0] is x (the residual stream input to MLP)
                    x = inputs[0].detach()
                    h_gelu = F.gelu(F.linear(x, module.fc_weight, module.fc_bias))
                    pre_relu = F.linear(h_gelu, module.adapter_W1, module.adapter_b1)
                    batch_pre_relu[blk_id] = pre_relu.reshape(-1, pre_relu.shape[-1]).cpu()
                return hook_fn

            local_hooks = []
            for blk_id in block_ids:
                h = model.transformer.h[blk_id].mlp.register_forward_hook(
                    make_pre_relu_hook(blk_id)
                )
                local_hooks.append(h)

            model(ids_dev)

            for h in local_hooks:
                h.remove()

            for blk_id in block_ids:
                if blk_id in batch_pre_relu:
                    pre_relu_data[blk_id].append(batch_pre_relu[blk_id])

    for blk_id in block_ids:
        all_pre_relu = torch.cat(pre_relu_data[blk_id], dim=0).float()  # (N_tokens, 64)
        n_neurons = all_pre_relu.shape[1]

        max_val = all_pre_relu.max(dim=0).values   # (64,)
        min_val = all_pre_relu.min(dim=0).values   # (64,)

        dead_mask = max_val <= 0.0           # never positive → always zero post-relu
        always_active_mask = min_val > 0.0   # always positive → linear regime
        mixed_mask = ~dead_mask & ~always_active_mask

        dead_count = dead_mask.sum().item()
        always_active_count = always_active_mask.sum().item()
        mixed_count = mixed_mask.sum().item()

        census[blk_id] = {
            "dead": int(dead_count),
            "always_active": int(always_active_count),
            "mixed": int(mixed_count),
            "total": int(n_neurons),
            "dead_indices": dead_mask.nonzero(as_tuple=True)[0].tolist(),
            "always_active_indices": always_active_mask.nonzero(as_tuple=True)[0].tolist(),
        }
        print(f"  Block {blk_id}: dead={dead_count} always_active={always_active_count} "
              f"mixed={mixed_count} / {n_neurons}", flush=True)

    return census


def prune_adapters(model, block_ids, census, device):
    """Exact pruning: remove dead neurons, fold always-active into linear."""
    for blk_id in block_ids:
        mod = model.transformer.h[blk_id].mlp
        info = census[blk_id]

        dead_idx = torch.tensor(info["dead_indices"], dtype=torch.long)
        always_idx = torch.tensor(info["always_active_indices"], dtype=torch.long)

        W1 = mod.adapter_W1.data.clone()   # (h, k)
        b1 = mod.adapter_b1.data.clone()   # (h,)
        W2 = mod.adapter_W2.data.clone()   # (768, h)
        b2 = mod.adapter_b2.data.clone()   # (768,)

        with torch.no_grad():
            # Step 1: Fold always-active neurons into linear correction
            # For always-active neuron j: contribution = W2[:,j] * (W1[j,:] @ h_gelu + b1[j])
            # This equals W2[:,j] @ W1[j,:] @ h_gelu + W2[:,j] * b1[j]
            # Add to proj_weight and proj_bias
            if always_idx.numel() > 0:
                W2_always = W2[:, always_idx]    # (768, n_always)
                W1_always = W1[always_idx, :]    # (n_always, k)
                b1_always = b1[always_idx]        # (n_always,)

                # Linear weight correction: (768, k)
                delta_proj_w = W2_always @ W1_always   # (768, k)
                # Bias correction: W2_always @ b1_always = (768,)
                delta_proj_b = W2_always @ b1_always + b2  # wait, b2 stays separate

                mod.proj_weight.data += delta_proj_w
                mod.proj_bias.data += W2_always @ b1_always

            # Step 2: Remove dead and always-active neurons from adapter
            prunable_idx = torch.cat([dead_idx, always_idx]).sort().values
            all_idx = torch.arange(W1.shape[0])
            keep_mask = torch.ones(W1.shape[0], dtype=torch.bool)
            if prunable_idx.numel() > 0:
                keep_mask[prunable_idx] = False
            keep_idx = all_idx[keep_mask]

            if keep_idx.numel() < W1.shape[0]:
                new_W1 = W1[keep_idx, :]        # (n_keep, k)
                new_b1 = b1[keep_idx]            # (n_keep,)
                new_W2 = W2[:, keep_idx]         # (768, n_keep)
                new_b2 = b2                       # unchanged

                mod.adapter_W1 = nn.Parameter(new_W1)
                mod.adapter_b1 = nn.Parameter(new_b1)
                mod.adapter_W2 = nn.Parameter(new_W2)
                mod.adapter_b2 = nn.Parameter(new_b2)

        n_pruned = W1.shape[0] - keep_idx.numel()
        print(f"  Block {blk_id}: pruned {n_pruned} neurons "
              f"({info['dead']} dead + {info['always_active']} always-active), "
              f"remaining={keep_idx.numel()}", flush=True)


# ── Data loading ──────────────────────────────────────────────────────────────
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
eval_ids_dev = [t.to(DEVICE) for t in all_ids[N_CALIB: N_CALIB + N_EVAL]]
calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
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

# ── Phase 1: Train ────────────────────────────────────────────────────────────
print("\n=== Phase 1: Training ===", flush=True)
torch.manual_seed(SEED)
random.seed(SEED)

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

print("  Compressing all 6 blocks...", flush=True)
compress_cache = compress_blocks(model, ALL_BLOCKS, K, calib_ids_dev, DEVICE)
compressed_modules = install_compressed_blocks(model, ALL_BLOCKS, compress_cache, DEVICE)

baseline_ppl = train_joint(model, compressed_modules, N_STEPS, LR, calib_ids_dev, eval_ids_dev)
print(f"\nBaseline PPL (post-training): {baseline_ppl:.4f}", flush=True)

assert 63 <= baseline_ppl <= 68, (
    f"Baseline PPL {baseline_ppl:.2f} out of expected range [63, 68]"
)

# ── Phase 2: Census ───────────────────────────────────────────────────────────
print("\n=== Phase 2: Adapter neuron census ===", flush=True)
census = census_adapters(model, ALL_BLOCKS, calib_ids_dev, DEVICE)

total_neurons = sum(info["total"] for info in census.values())
total_dead = sum(info["dead"] for info in census.values())
total_always_active = sum(info["always_active"] for info in census.values())
total_prunable = total_dead + total_always_active
pct_prunable = 100.0 * total_prunable / total_neurons

print(f"\nCensus summary:", flush=True)
print(f"  Total neurons: {total_neurons}", flush=True)
print(f"  Dead: {total_dead} ({100.0*total_dead/total_neurons:.1f}%)", flush=True)
print(f"  Always-active: {total_always_active} ({100.0*total_always_active/total_neurons:.1f}%)", flush=True)
print(f"  Prunable: {total_prunable} ({pct_prunable:.1f}%)", flush=True)

# ── Phase 3: Pruning ──────────────────────────────────────────────────────────
print("\n=== Phase 3: Exact pruning ===", flush=True)
prune_adapters(model, ALL_BLOCKS, census, DEVICE)

pruned_ppl = eval_ppl(model, eval_ids_dev)
ppl_delta = pruned_ppl - baseline_ppl
print(f"\nPruned PPL: {pruned_ppl:.4f}", flush=True)
print(f"Baseline PPL: {baseline_ppl:.4f}", flush=True)
print(f"PPL delta: {ppl_delta:+.6f}", flush=True)

# ── Verdict ────────────────────────────────────────────────────────────────────
if pct_prunable >= 10 and abs(ppl_delta) < 0.1:
    verdict = "CONFIRMED"
elif pct_prunable < 5 or ppl_delta > 0.5:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"  pct_prunable={pct_prunable:.1f}%  ppl_delta={ppl_delta:+.6f}", flush=True)
print(f"{'='*60}", flush=True)

# ── Save JSON ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out = {
    "cycle": 375,
    "item": 383,
    "seed": SEED,
    "float_ppl": float(float_ppl),
    "baseline_ppl": float(baseline_ppl),
    "census": {
        str(b): {
            "dead": info["dead"],
            "always_active": info["always_active"],
            "mixed": info["mixed"],
            "total": info["total"],
        }
        for b, info in census.items()
    },
    "total_neurons": total_neurons,
    "total_dead": total_dead,
    "total_always_active": total_always_active,
    "total_prunable": total_prunable,
    "pct_prunable": float(pct_prunable),
    "pruned_ppl": float(pruned_ppl),
    "ppl_delta": float(ppl_delta),
    "verdict": verdict,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {OUTPUT_PATH}", flush=True)
print("Done.", flush=True)
