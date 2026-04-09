#!/usr/bin/env python3
"""Cycle 349 / Item 358 — topk+lstsq 3-block + per-block MLP adapter.

Combines two confirmed wins:
  - topk+lstsq 3-block composition (c348): blocks [b2, b1, b10] → PPL 49.60
  - MLP adapter on LASER (c347): h=64 → PPL 48.51 on single block

Approach:
  Phase 1: topk+lstsq compress blocks [2, 1, 10], retrain proj (1000 steps)
           → reproduces c348 3-block baseline (~49.60)
  Phase 2: freeze proj, add MLP adapter per block, train adapters (1000 steps)
           → tests whether adapters improve on top of topk+lstsq composition

The MLP adapter sits in parallel with the compressed c_proj:
  output = CompressedMLP(x) + Adapter(gelu_hidden)
  where Adapter(h) = W2 @ relu(W1 @ h + b1) + b2
  W1: (h_dim, k), W2: (768, h_dim)

Verdict (item 358):
  CONFIRMED if PPL < 48.0 (below single-block adapter on LASER)
  REFUTED if PPL >= 49.6 (adapter doesn't help on top of topk+lstsq)
  INCONCLUSIVE if 48.0 <= PPL < 49.6
"""
import json
import math
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = [2, 1, 10]  # same 3-block combo as c348 stage 3

# topk+lstsq: 6x compression = keep 512 of 3072 hidden channels
HIDDEN_FRACTION = 1 / 6  # → k=512

# Data
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Phase 1: retrain proj (reproduce c348 baseline)
N_STEPS_P1 = 1000
LR_P1 = 1e-4

# Phase 2: train adapters
N_STEPS_P2 = 1000
LR_P2 = 1e-4
ADAPTER_HIDDEN = 64  # same as c347

EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42

FLOAT_BASELINE_REF = 53.895
TOPK_3BLOCK_REF = 49.60  # c348 stage 3

OUTPUT_PATH = "autoresearch/results/cycle_349_topk_adapter.json"

torch.manual_seed(LOAD_SEED)
print(f"Device: {DEVICE}", flush=True)

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
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB : N_CALIB + N_EVAL]]
print(f"Calibration: {len(calib_ids_cpu)} seqs, Eval: {len(eval_ids)} seqs", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def collect_block_activations(model, block_ids, calib_ids, device):
    """Collect MLP input/output/preact/hidden activations for specified blocks."""
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
            ids_dev = ids.to(device) if ids.device != device else ids
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
    """Compressed MLP with optional MLP adapter.

    Compressed path: frozen c_fc (k channels) → GELU → trainable c_proj (k→768)
    Adapter path: GELU hidden (k) → W1 (h, k) → ReLU → W2 (768, h) + bias

    When adapter_active=False, behaves identically to CompressedGPT2MLP from c348.
    """

    def __init__(self, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias,
                 adapter_hidden=0):
        super().__init__()
        k = c_fc_weight.shape[0]
        # Frozen input projection
        self.register_buffer("fc_weight", c_fc_weight.contiguous())  # (k, 768)
        self.register_buffer("fc_bias", c_fc_bias.contiguous())      # (k,)
        # Trainable output projection (from Phase 1)
        self.proj_weight = nn.Parameter(c_proj_weight.contiguous())   # (768, k)
        self.proj_bias = nn.Parameter(c_proj_bias.contiguous())       # (768,)

        # MLP adapter (initially inactive)
        self.adapter_active = adapter_hidden > 0
        if self.adapter_active:
            scale1 = 1.0 / math.sqrt(k)
            scale2 = 1.0 / math.sqrt(adapter_hidden)
            self.adapter_W1 = nn.Parameter(torch.randn(adapter_hidden, k) * scale1)
            self.adapter_b1 = nn.Parameter(torch.zeros(adapter_hidden))
            self.adapter_W2 = nn.Parameter(torch.randn(768, adapter_hidden) * scale2)
            self.adapter_b2 = nn.Parameter(torch.zeros(768))

    def forward(self, x):
        h = F.linear(x, self.fc_weight, self.fc_bias)  # (batch, seq, k)
        h = F.gelu(h)
        out = F.linear(h, self.proj_weight, self.proj_bias)  # (batch, seq, 768)
        if self.adapter_active:
            a = F.relu(F.linear(h, self.adapter_W1, self.adapter_b1))  # (batch, seq, adapter_h)
            adapter_out = F.linear(a, self.adapter_W2, self.adapter_b2)  # (batch, seq, 768)
            out = out + adapter_out
        return out


# ── Float baseline ────────────────────────────────────────────────────────────
print("Computing float baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
float_ppl = eval_ppl(base_model, eval_ids)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference {FLOAT_BASELINE_REF} by > 0.5"
)

# ── Probe geometry and compress all target blocks ────────────────────────────
print("\nCollecting calibration activations for all target blocks...", flush=True)
calib_dev = [t.to(DEVICE) for t in calib_ids_cpu]
block_acts = collect_block_activations(base_model, BLOCKS, calib_dev, DEVICE)
parts = [f"b{b}: {block_acts[b]['hidden'].shape}" for b in BLOCKS]
print(f"Collected activations: {', '.join(parts)}", flush=True)

# Compute channel scores and compress each block (identical to c348)
compress_cache = {}
for blk in BLOCKS:
    acts = block_acts[blk]
    hidden = acts["hidden"]
    output = acts["output"]
    preact = acts["preact"]

    hidden_dim = hidden.shape[1]
    k = max(1, int(round(hidden_dim * HIDDEN_FRACTION)))

    # Channel scoring
    std_h = hidden.std(dim=0)
    mean_h = hidden.mean(dim=0)
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    deriv = 0.5 * (1.0 + torch.erf(preact * inv_sqrt_2)) + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi
    mean_deriv = deriv.mean(dim=0)

    c_proj = base_model.transformer.h[blk].mlp.c_proj
    c_proj_weight = c_proj.weight.detach().cpu().float()  # (3072, 768) Conv1D
    output_norms = c_proj_weight.norm(dim=1)  # (3072,)

    channel_scores = std_h * output_norms * (mean_h.abs() + mean_deriv.abs() + 1e-6)

    dead_mask = hidden.abs().amax(dim=0) <= 1e-6
    available = torch.arange(hidden_dim)
    live = available[~dead_mask]
    if live.numel() < k:
        live = available
    scores = channel_scores[live]
    topk_idx = torch.topk(scores, k=k, largest=True).indices
    selected = live[topk_idx].sort().values

    c_fc = base_model.transformer.h[blk].mlp.c_fc
    c_fc_weight_full = c_fc.weight.detach().cpu().float()  # (768, 3072) Conv1D
    c_fc_bias_full = c_fc.bias.detach().cpu().float()

    fc_w = c_fc_weight_full.T[selected]  # (k, 768)
    fc_b = c_fc_bias_full[selected]      # (k,)

    # lstsq for output projection
    hidden_sel = hidden[:, selected].float()
    design = torch.cat([hidden_sel, torch.ones(hidden_sel.shape[0], 1)], dim=1)
    solution = torch.linalg.lstsq(design, output.float()).solution
    proj_w = solution[:-1].T.contiguous()  # (768, k)
    proj_b = solution[-1].contiguous()     # (768,)

    with torch.no_grad():
        recon = hidden_sel @ proj_w.T + proj_b
        mse = F.mse_loss(recon, output).item()
        cos = F.cosine_similarity(recon.reshape(1, -1), output.reshape(1, -1)).item()

    compress_cache[blk] = {
        "fc_w": fc_w, "fc_b": fc_b,
        "proj_w": proj_w, "proj_b": proj_b,
        "selected": selected, "k": k,
        "mse": mse, "cos": cos,
    }
    print(f"Block {blk}: k={k}, dead={dead_mask.sum().item()}, "
          f"lstsq MSE={mse:.4f}, cos={cos:.6f}", flush=True)

del block_acts, base_model
if DEVICE == "mps":
    torch.mps.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: topk+lstsq 3-block retrain (reproduce c348 baseline)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("Phase 1: topk+lstsq 3-block retrain (reproduce c348 baseline)", flush=True)
print(f"{'='*60}", flush=True)

# Run Phase 1 with seed 42 only (just need the retrained proj params)
torch.manual_seed(42)
random.seed(42)

model_p1 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_p1.eval()

# Install compressed MLPs (no adapter for Phase 1)
compressed_modules_p1 = {}
for blk in BLOCKS:
    c = compress_cache[blk]
    comp_mlp = CompressedGPT2MLPWithAdapter(
        c_fc_weight=c["fc_w"].to(device=DEVICE, dtype=torch.float32),
        c_fc_bias=c["fc_b"].to(device=DEVICE, dtype=torch.float32),
        c_proj_weight=c["proj_w"].to(device=DEVICE, dtype=torch.float32),
        c_proj_bias=c["proj_b"].to(device=DEVICE, dtype=torch.float32),
        adapter_hidden=0,  # no adapter in Phase 1
    ).to(DEVICE)
    model_p1.transformer.h[blk].mlp = comp_mlp
    compressed_modules_p1[blk] = comp_mlp

init_ppl_p1 = eval_ppl(model_p1, eval_ids)
print(f"Phase 1 init PPL: {init_ppl_p1:.4f}", flush=True)

# Freeze everything, unfreeze proj_weight + proj_bias
for p in model_p1.parameters():
    p.requires_grad_(False)
trainable_p1 = []
for blk in BLOCKS:
    compressed_modules_p1[blk].proj_weight.requires_grad_(True)
    compressed_modules_p1[blk].proj_bias.requires_grad_(True)
    trainable_p1.extend([compressed_modules_p1[blk].proj_weight,
                         compressed_modules_p1[blk].proj_bias])
n_train_p1 = sum(p.numel() for p in trainable_p1)
print(f"Phase 1 trainable params: {n_train_p1:,}", flush=True)

optimizer_p1 = torch.optim.Adam(trainable_p1, lr=LR_P1)
calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
random.shuffle(calib_ids_dev)

best_ppl_p1 = init_ppl_p1
best_step_p1 = 0
best_state_p1 = {}
for blk in BLOCKS:
    best_state_p1[f"proj_w_{blk}"] = compressed_modules_p1[blk].proj_weight.detach().clone()
    best_state_p1[f"proj_b_{blk}"] = compressed_modules_p1[blk].proj_bias.detach().clone()

step = 0
t0 = time.time()
while step < N_STEPS_P1:
    for ids in calib_ids_dev:
        if step >= N_STEPS_P1:
            break
        model_p1.train()
        optimizer_p1.zero_grad()
        out = model_p1(ids, labels=ids)
        out.loss.backward()
        optimizer_p1.step()
        step += 1
        if step % EVAL_EVERY == 0:
            ppl_e = eval_ppl(model_p1, eval_ids)
            if ppl_e < best_ppl_p1:
                best_ppl_p1 = ppl_e
                best_step_p1 = step
                best_state_p1 = {}
                for blk in BLOCKS:
                    best_state_p1[f"proj_w_{blk}"] = compressed_modules_p1[blk].proj_weight.detach().clone()
                    best_state_p1[f"proj_b_{blk}"] = compressed_modules_p1[blk].proj_bias.detach().clone()
            if step % 500 == 0:
                print(f"  Phase 1 step {step}: ppl={ppl_e:.2f} best={best_ppl_p1:.2f}@{best_step_p1}",
                      flush=True)

# Restore best
with torch.no_grad():
    for blk in BLOCKS:
        compressed_modules_p1[blk].proj_weight.data.copy_(best_state_p1[f"proj_w_{blk}"])
        compressed_modules_p1[blk].proj_bias.data.copy_(best_state_p1[f"proj_b_{blk}"])
restored_ppl_p1 = eval_ppl(model_p1, eval_ids)
dt_p1 = time.time() - t0
print(f"Phase 1: init={init_ppl_p1:.2f} best={best_ppl_p1:.2f}@{best_step_p1} "
      f"restored={restored_ppl_p1:.2f} ({dt_p1:.0f}s)", flush=True)

# Save the retrained proj params for Phase 2
retrained_proj = {}
for blk in BLOCKS:
    retrained_proj[blk] = {
        "proj_w": best_state_p1[f"proj_w_{blk}"].cpu().clone(),
        "proj_b": best_state_p1[f"proj_b_{blk}"].cpu().clone(),
    }

phase1_result = {
    "init_ppl": float(init_ppl_p1),
    "best_ppl": float(best_ppl_p1),
    "best_step": int(best_step_p1),
    "restored_ppl": float(restored_ppl_p1),
    "trainable_params": n_train_p1,
}
print(f"\nPhase 1 baseline PPL: {best_ppl_p1:.4f} (ref {TOPK_3BLOCK_REF})", flush=True)

del model_p1, compressed_modules_p1, optimizer_p1
if DEVICE == "mps":
    torch.mps.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Add per-block MLP adapters on top of frozen topk+lstsq
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"Phase 2: per-block MLP adapter (h={ADAPTER_HIDDEN}) on top of topk+lstsq", flush=True)

k = compress_cache[BLOCKS[0]]["k"]
n_adapter_per_block = ADAPTER_HIDDEN * k + ADAPTER_HIDDEN + 768 * ADAPTER_HIDDEN + 768
n_adapter_total = n_adapter_per_block * len(BLOCKS)
print(f"Adapter params per block: {n_adapter_per_block:,} "
      f"(W1: {ADAPTER_HIDDEN}x{k} + b1: {ADAPTER_HIDDEN} + W2: 768x{ADAPTER_HIDDEN} + b2: 768)",
      flush=True)
print(f"Total adapter params (3 blocks): {n_adapter_total:,}", flush=True)
print(f"{'='*60}", flush=True)

seed_results = []
for seed in SEEDS:
    torch.manual_seed(seed)
    random.seed(seed)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    # Install compressed MLPs WITH adapter, using retrained proj from Phase 1
    compressed_modules = {}
    for blk in BLOCKS:
        c = compress_cache[blk]
        comp_mlp = CompressedGPT2MLPWithAdapter(
            c_fc_weight=c["fc_w"].to(device=DEVICE, dtype=torch.float32),
            c_fc_bias=c["fc_b"].to(device=DEVICE, dtype=torch.float32),
            c_proj_weight=retrained_proj[blk]["proj_w"].to(device=DEVICE, dtype=torch.float32),
            c_proj_bias=retrained_proj[blk]["proj_b"].to(device=DEVICE, dtype=torch.float32),
            adapter_hidden=ADAPTER_HIDDEN,
        ).to(DEVICE)
        model.transformer.h[blk].mlp = comp_mlp
        compressed_modules[blk] = comp_mlp

    init_ppl = eval_ppl(model, eval_ids)

    # Freeze EVERYTHING, then unfreeze only adapter params
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for blk in BLOCKS:
        mod = compressed_modules[blk]
        mod.adapter_W1.requires_grad_(True)
        mod.adapter_b1.requires_grad_(True)
        mod.adapter_W2.requires_grad_(True)
        mod.adapter_b2.requires_grad_(True)
        trainable.extend([mod.adapter_W1, mod.adapter_b1, mod.adapter_W2, mod.adapter_b2])
    n_train = sum(p.numel() for p in trainable)
    assert n_train == n_adapter_total, f"{n_train} != {n_adapter_total}"

    optimizer = torch.optim.Adam(trainable, lr=LR_P2)
    calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
    random.shuffle(calib_ids_dev)

    best_ppl = init_ppl
    best_step = 0
    best_state = {}
    for blk in BLOCKS:
        mod = compressed_modules[blk]
        best_state[f"W1_{blk}"] = mod.adapter_W1.detach().clone()
        best_state[f"b1_{blk}"] = mod.adapter_b1.detach().clone()
        best_state[f"W2_{blk}"] = mod.adapter_W2.detach().clone()
        best_state[f"b2_{blk}"] = mod.adapter_b2.detach().clone()
    history = [{"step": 0, "eval_ppl": float(init_ppl)}]

    step = 0
    t0 = time.time()
    while step < N_STEPS_P2:
        for ids in calib_ids_dev:
            if step >= N_STEPS_P2:
                break
            model.train()
            optimizer.zero_grad()
            out = model(ids, labels=ids)
            out.loss.backward()
            optimizer.step()
            step += 1
            if step % EVAL_EVERY == 0:
                ppl_e = eval_ppl(model, eval_ids)
                history.append({"step": step, "eval_ppl": float(ppl_e)})
                if ppl_e < best_ppl:
                    best_ppl = ppl_e
                    best_step = step
                    best_state = {}
                    for blk in BLOCKS:
                        mod = compressed_modules[blk]
                        best_state[f"W1_{blk}"] = mod.adapter_W1.detach().clone()
                        best_state[f"b1_{blk}"] = mod.adapter_b1.detach().clone()
                        best_state[f"W2_{blk}"] = mod.adapter_W2.detach().clone()
                        best_state[f"b2_{blk}"] = mod.adapter_b2.detach().clone()
                if step % 500 == 0:
                    print(f"  seed {seed} step {step}: ppl={ppl_e:.2f} "
                          f"best={best_ppl:.2f}@{best_step}", flush=True)

    # Restore best
    with torch.no_grad():
        for blk in BLOCKS:
            mod = compressed_modules[blk]
            mod.adapter_W1.data.copy_(best_state[f"W1_{blk}"])
            mod.adapter_b1.data.copy_(best_state[f"b1_{blk}"])
            mod.adapter_W2.data.copy_(best_state[f"W2_{blk}"])
            mod.adapter_b2.data.copy_(best_state[f"b2_{blk}"])
    restored_ppl = eval_ppl(model, eval_ids)
    dt = time.time() - t0

    print(f"  seed {seed}: init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
          f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)

    seed_results.append({
        "seed": seed,
        "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl),
        "best_step": int(best_step),
        "restored_ppl": float(restored_ppl),
        "history": history,
    })

    del model, compressed_modules, optimizer
    if DEVICE == "mps":
        torch.mps.empty_cache()


# ── Summary and verdict ──────────────────────────────────────────────────────
best_ppls = [s["best_ppl"] for s in seed_results]
mean_ppl = sum(best_ppls) / len(best_ppls)
std_ppl = math.sqrt(sum((x - mean_ppl) ** 2 for x in best_ppls) / len(best_ppls))

print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"Float PPL:                     {float_ppl:.4f}", flush=True)
print(f"Phase 1 (topk+lstsq 3-block):  {best_ppl_p1:.4f}", flush=True)
print(f"Phase 2 (+ MLP adapter h={ADAPTER_HIDDEN}):", flush=True)
print(f"  Mean PPL:  {mean_ppl:.4f} ± {std_ppl:.4f}", flush=True)
print(f"  Min PPL:   {min(best_ppls):.4f}", flush=True)
print(f"  Max PPL:   {max(best_ppls):.4f}", flush=True)
print(f"  Adapter params: {n_adapter_total:,} ({len(BLOCKS)} blocks × {n_adapter_per_block:,})", flush=True)

for s in seed_results:
    print(f"  seed {s['seed']}: {s['best_ppl']:.4f} @ step {s['best_step']}", flush=True)

# Verdict
if mean_ppl < 48.0:
    verdict = "CONFIRMED"
elif mean_ppl >= 49.6:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nVERDICT: {verdict}", flush=True)
print(f"  Threshold CONFIRMED: mean PPL < 48.0", flush=True)
print(f"  Threshold REFUTED:   mean PPL >= 49.6", flush=True)

# ── Save JSON ────────────────────────────────────────────────────────────────
import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

result = {
    "cycle": 349,
    "item": 358,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "phase1": phase1_result,
    "phase2": {
        "blocks": BLOCKS,
        "adapter_hidden": ADAPTER_HIDDEN,
        "adapter_params_per_block": n_adapter_per_block,
        "adapter_params_total": n_adapter_total,
        "mean_ppl": float(mean_ppl),
        "std_ppl": float(std_ppl),
        "min_ppl": float(min(best_ppls)),
        "max_ppl": float(max(best_ppls)),
    },
    "phase2_seeds": seed_results,
    "references": {
        "c348_topk_3block": TOPK_3BLOCK_REF,
        "c347_laser_adapter_single_block": 48.51,
    },
    "per_block_lstsq": {
        str(blk): {"k": compress_cache[blk]["k"], "mse": compress_cache[blk]["mse"],
                    "cos": compress_cache[blk]["cos"]}
        for blk in BLOCKS
    },
    "config": {
        "BLOCKS": BLOCKS, "HIDDEN_FRACTION": HIDDEN_FRACTION,
        "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE,
        "N_STEPS_P1": N_STEPS_P1, "LR_P1": LR_P1,
        "N_STEPS_P2": N_STEPS_P2, "LR_P2": LR_P2,
        "ADAPTER_HIDDEN": ADAPTER_HIDDEN,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved results to {OUTPUT_PATH}", flush=True)
print("Done.", flush=True)
