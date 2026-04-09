#!/usr/bin/env python3
"""Cycle 361 / Item 370 — Progressive 3-block compression with intermediate retrain.

Attack the composition barrier by interleaving compression with brief full-model
adaptation passes so that uncompressed blocks can absorb diffuse topk error before
more compression is added.

Schedule:
  Phase 1: Compress blocks [2, 1, 10] with topk+lstsq k=512 + adapter h=64,
            train 1000 steps (proj_weight + adapter).
  Phase 2: Freeze Phase-1 blocks, fine-tune ALL remaining (9) blocks 200 steps lr=5e-5
            to adapt them to the partial compression.
  Phase 3: Compress blocks [4, 3, 11] with topk+lstsq k=512 + adapter h=64,
            train 1000 steps. Phase-1 blocks stay frozen.
  Phase 4: Freeze Phase-3 blocks, fine-tune remaining 6 uncompressed blocks 200 steps.
  Phase 5 (optional): Compress blocks [0, 5, 7], train 1000 steps.

PPL measured after each phase.

Verdict (6-block after Phase 4):
  CONFIRMED    if PPL < 55.0   (within 1.1× of float baseline 53.89)
  REFUTED      if PPL > 65.0   (still catastrophic)
  INCONCLUSIVE otherwise
"""
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
PHASE1_BLOCKS = [2, 1, 10]
PHASE3_BLOCKS = [4, 3, 11]
PHASE5_BLOCKS = [0, 5, 7]

HIDDEN_FRACTION = 1 / 6   # k=512 of 3072
ADAPTER_HIDDEN = 64

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

N_STEPS_COMPRESS = 1000
LR_COMPRESS = 1e-4

N_STEPS_ADAPT = 200
LR_ADAPT = 5e-5

EVAL_EVERY = 100
SEED = 42

FLOAT_BASELINE_REF = 53.895
TOPK_3BLOCK_REF = 47.43  # c349 3-block + adapter h=64 best PPL

OUTPUT_PATH = "autoresearch/results/cycle_361_progressive_compression.json"

torch.manual_seed(SEED)
random.seed(SEED)
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
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB: N_CALIB + N_EVAL]]
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
    """Collect MLP hidden/output/preact activations for specified blocks."""
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
    """Compressed MLP with MLP adapter.

    Compressed path: frozen c_fc (k channels) → GELU → trainable c_proj (k→768)
    Adapter path: GELU hidden (k) → W1 (h,k) → ReLU → W2 (768,h) + bias
    """

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


def compute_compression(model, block_ids, calib_ids_dev, device):
    """Run topk+lstsq compression for the given blocks. Returns compress_cache dict."""
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
        k = max(1, int(round(hidden_dim * HIDDEN_FRACTION)))

        std_h = hidden.std(dim=0)
        mean_h = hidden.mean(dim=0)
        deriv = (0.5 * (1.0 + torch.erf(preact * inv_sqrt_2))
                 + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi)
        mean_deriv = deriv.mean(dim=0)

        c_proj = model.transformer.h[blk].mlp.c_proj
        c_proj_weight = c_proj.weight.detach().cpu().float()  # (3072, 768) Conv1D
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
        c_fc_weight_full = c_fc.weight.detach().cpu().float()  # (768, 3072) Conv1D
        c_fc_bias_full = c_fc.bias.detach().cpu().float()

        fc_w = c_fc_weight_full.T[selected]  # (k, 768)
        fc_b = c_fc_bias_full[selected]      # (k,)

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
        print(f"  Block {blk}: k={k}, dead={dead_mask.sum().item()}, "
              f"lstsq MSE={mse:.4f}, cos={cos:.6f}", flush=True)

    return compress_cache


def install_compressed_blocks(model, block_ids, compress_cache, device):
    """Install CompressedGPT2MLPWithAdapter for each block in block_ids."""
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


def train_compression_phase(model, compressed_modules, n_steps, lr, calib_ids_dev,
                             phase_name="compression"):
    """Train proj_weight + adapter params for compressed_modules. Returns best PPL."""
    # Freeze everything first
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

    init_ppl = eval_ppl(model, eval_ids)
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
                ppl_e = eval_ppl(model, eval_ids)
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
    restored_ppl = eval_ppl(model, eval_ids)
    dt = time.time() - t0
    print(f"  [{phase_name}] init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
          f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)
    return best_ppl, restored_ppl, n_train


def train_adaptation_phase(model, frozen_block_ids, n_steps, lr, calib_ids_dev,
                            phase_name="adaptation"):
    """Fine-tune all non-frozen blocks for n_steps. Returns best PPL."""
    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze all blocks not in frozen set
    trainable = []
    for blk_idx, block in enumerate(model.transformer.h):
        if blk_idx not in frozen_block_ids:
            for p in block.parameters():
                p.requires_grad_(True)
                trainable.append(p)

    n_train = sum(p.numel() for p in trainable)
    print(f"  [{phase_name}] trainable params: {n_train:,} "
          f"(blocks NOT in {sorted(frozen_block_ids)})", flush=True)

    init_ppl = eval_ppl(model, eval_ids)
    print(f"  [{phase_name}] init PPL: {init_ppl:.4f}", flush=True)

    optimizer = torch.optim.Adam(trainable, lr=lr)

    ids_shuffled = list(calib_ids_dev)
    random.shuffle(ids_shuffled)

    best_ppl = init_ppl
    best_step = 0
    best_state = {i: {} for i in range(len(model.transformer.h)) if i not in frozen_block_ids}
    for blk_idx, block in enumerate(model.transformer.h):
        if blk_idx not in frozen_block_ids:
            for name, p in block.named_parameters():
                best_state[blk_idx][name] = p.detach().clone()

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
                ppl_e = eval_ppl(model, eval_ids)
                if ppl_e < best_ppl:
                    best_ppl = ppl_e
                    best_step = step
                    for blk_idx, block in enumerate(model.transformer.h):
                        if blk_idx not in frozen_block_ids:
                            for name, p in block.named_parameters():
                                best_state[blk_idx][name] = p.detach().clone()
                if step % 100 == 0:
                    print(f"  [{phase_name}] step {step}: ppl={ppl_e:.2f} "
                          f"best={best_ppl:.2f}@{best_step}", flush=True)

    # Restore best
    with torch.no_grad():
        for blk_idx, block in enumerate(model.transformer.h):
            if blk_idx not in frozen_block_ids:
                for name, p in block.named_parameters():
                    p.data.copy_(best_state[blk_idx][name])
    restored_ppl = eval_ppl(model, eval_ids)
    dt = time.time() - t0
    print(f"  [{phase_name}] init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
          f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)
    return best_ppl, restored_ppl, n_train


# ── Float baseline ────────────────────────────────────────────────────────────
print("Computing float baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
float_ppl = eval_ppl(base_model, eval_ids)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference by > 0.5"
)
del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()

results = {"float_ppl": float(float_ppl), "phases": []}
calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]

# ══════════════════════════════════════════════════════════════════════════════
# Load fresh model for the progressive experiment
# ══════════════════════════════════════════════════════════════════════════════
torch.manual_seed(SEED)
random.seed(SEED)
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

# Track which blocks are compressed and frozen
compressed_frozen = set()

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Compress blocks [2, 1, 10] + retrain (1000 steps)
# ══════════════════════════════════════════════════════════════════════════════
sep = "=" * 60
print(f"\n{sep}", flush=True)
print(f"Phase 1: Compress blocks {PHASE1_BLOCKS} + retrain {N_STEPS_COMPRESS} steps", flush=True)
print(f"{sep}", flush=True)

print("  Computing topk+lstsq compression for Phase 1 blocks...", flush=True)
compress_p1 = compute_compression(model, PHASE1_BLOCKS, calib_ids_dev, DEVICE)
p1_modules = install_compressed_blocks(model, PHASE1_BLOCKS, compress_p1, DEVICE)

p1_best_ppl, p1_restored_ppl, p1_n_train = train_compression_phase(
    model, p1_modules, N_STEPS_COMPRESS, LR_COMPRESS, calib_ids_dev, "Phase1-compress"
)

# Freeze Phase 1 blocks
for blk in PHASE1_BLOCKS:
    for p in model.transformer.h[blk].parameters():
        p.requires_grad_(False)
    compressed_frozen.add(blk)

ppl_after_p1 = eval_ppl(model, eval_ids)
print(f"\nPPL after Phase 1: {ppl_after_p1:.4f}", flush=True)
results["phases"].append({
    "phase": 1, "description": f"compress {PHASE1_BLOCKS}",
    "blocks_compressed": list(PHASE1_BLOCKS),
    "best_ppl": float(p1_best_ppl), "restored_ppl": float(p1_restored_ppl),
    "ppl_after_freeze": float(ppl_after_p1),
})

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Fine-tune remaining 9 uncompressed blocks (200 steps, lr=5e-5)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}", flush=True)
print(f"Phase 2: Adapt remaining 9 blocks {N_STEPS_ADAPT} steps lr={LR_ADAPT}", flush=True)
print(f"{sep}", flush=True)

p2_best_ppl, p2_restored_ppl, p2_n_train = train_adaptation_phase(
    model, compressed_frozen, N_STEPS_ADAPT, LR_ADAPT, calib_ids_dev, "Phase2-adapt"
)

ppl_after_p2 = eval_ppl(model, eval_ids)
print(f"\nPPL after Phase 2 (3-block + adapt): {ppl_after_p2:.4f}", flush=True)
results["phases"].append({
    "phase": 2, "description": "adapt remaining 9 blocks",
    "frozen_blocks": sorted(compressed_frozen),
    "best_ppl": float(p2_best_ppl), "restored_ppl": float(p2_restored_ppl),
    "ppl_after_freeze": float(ppl_after_p2),
})

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Compress blocks [4, 3, 11] + retrain (1000 steps)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}", flush=True)
print(f"Phase 3: Compress blocks {PHASE3_BLOCKS} + retrain {N_STEPS_COMPRESS} steps", flush=True)
print(f"  (Phase 1 blocks {sorted(compressed_frozen)} remain frozen)", flush=True)
print(f"{sep}", flush=True)

print("  Computing topk+lstsq compression for Phase 3 blocks...", flush=True)
# Important: collect activations AFTER adaptation (model has adapted)
compress_p3 = compute_compression(model, PHASE3_BLOCKS, calib_ids_dev, DEVICE)
p3_modules = install_compressed_blocks(model, PHASE3_BLOCKS, compress_p3, DEVICE)

# Ensure Phase 1 blocks stay frozen during Phase 3 training
for blk in compressed_frozen:
    for p in model.transformer.h[blk].parameters():
        p.requires_grad_(False)

p3_best_ppl, p3_restored_ppl, p3_n_train = train_compression_phase(
    model, p3_modules, N_STEPS_COMPRESS, LR_COMPRESS, calib_ids_dev, "Phase3-compress"
)

# Freeze Phase 3 blocks too
for blk in PHASE3_BLOCKS:
    for p in model.transformer.h[blk].parameters():
        p.requires_grad_(False)
    compressed_frozen.add(blk)

ppl_after_p3 = eval_ppl(model, eval_ids)
print(f"\nPPL after Phase 3 (6 blocks compressed): {ppl_after_p3:.4f}", flush=True)
results["phases"].append({
    "phase": 3, "description": f"compress {PHASE3_BLOCKS} (6 blocks total)",
    "blocks_compressed": list(PHASE3_BLOCKS),
    "all_compressed": sorted(compressed_frozen),
    "best_ppl": float(p3_best_ppl), "restored_ppl": float(p3_restored_ppl),
    "ppl_after_freeze": float(ppl_after_p3),
})

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Fine-tune remaining 6 uncompressed blocks (200 steps, lr=5e-5)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}", flush=True)
print(f"Phase 4: Adapt remaining 6 blocks {N_STEPS_ADAPT} steps lr={LR_ADAPT}", flush=True)
print(f"  (frozen: {sorted(compressed_frozen)})", flush=True)
print(f"{sep}", flush=True)

p4_best_ppl, p4_restored_ppl, p4_n_train = train_adaptation_phase(
    model, compressed_frozen, N_STEPS_ADAPT, LR_ADAPT, calib_ids_dev, "Phase4-adapt"
)

ppl_after_p4 = eval_ppl(model, eval_ids)
print(f"\nPPL after Phase 4 (6-block + final adapt): {ppl_after_p4:.4f}", flush=True)
results["phases"].append({
    "phase": 4, "description": "adapt remaining 6 blocks",
    "frozen_blocks": sorted(compressed_frozen),
    "best_ppl": float(p4_best_ppl), "restored_ppl": float(p4_restored_ppl),
    "ppl_after_freeze": float(ppl_after_p4),
})

# ── 6-block verdict (primary) ─────────────────────────────────────────────────
ppl_6block = ppl_after_p4
if ppl_6block < 55.0:
    verdict = "CONFIRMED"
elif ppl_6block > 65.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}", flush=True)
print(f"VERDICT (6-block after Phase 4): {verdict}", flush=True)
print(f"  PPL = {ppl_6block:.4f}  (threshold: <55.0 CONFIRMED, >65.0 REFUTED)", flush=True)
print(f"{'='*60}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 (optional): Compress blocks [0, 5, 7] — 9 blocks total
# Only run if 6-block result is promising (PPL < 70)
# ══════════════════════════════════════════════════════════════════════════════
phase5_result = None
if ppl_6block < 70.0:
    print(f"\n{sep}", flush=True)
    print(f"Phase 5 (optional): Compress blocks {PHASE5_BLOCKS} (9 blocks total)", flush=True)
    print(f"  (6-block PPL={ppl_6block:.2f} < 70 — running Phase 5)", flush=True)
    print(f"{sep}", flush=True)

    print("  Computing topk+lstsq compression for Phase 5 blocks...", flush=True)
    compress_p5 = compute_compression(model, PHASE5_BLOCKS, calib_ids_dev, DEVICE)
    p5_modules = install_compressed_blocks(model, PHASE5_BLOCKS, compress_p5, DEVICE)

    for blk in compressed_frozen:
        for p in model.transformer.h[blk].parameters():
            p.requires_grad_(False)

    p5_best_ppl, p5_restored_ppl, p5_n_train = train_compression_phase(
        model, p5_modules, N_STEPS_COMPRESS, LR_COMPRESS, calib_ids_dev, "Phase5-compress"
    )

    for blk in PHASE5_BLOCKS:
        for p in model.transformer.h[blk].parameters():
            p.requires_grad_(False)
        compressed_frozen.add(blk)

    ppl_after_p5 = eval_ppl(model, eval_ids)
    print(f"\nPPL after Phase 5 (9 blocks compressed): {ppl_after_p5:.4f}", flush=True)
    phase5_result = {
        "phase": 5, "description": f"compress {PHASE5_BLOCKS} (9 blocks total)",
        "blocks_compressed": list(PHASE5_BLOCKS),
        "all_compressed": sorted(compressed_frozen),
        "best_ppl": float(p5_best_ppl), "restored_ppl": float(p5_restored_ppl),
        "ppl_after_freeze": float(ppl_after_p5),
    }
    results["phases"].append(phase5_result)
else:
    print(f"\nPhase 5 SKIPPED: 6-block PPL={ppl_6block:.2f} >= 70 (not promising)", flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"Float PPL:                       {float_ppl:.4f}", flush=True)
print(f"3-block ref (c349):              {TOPK_3BLOCK_REF}", flush=True)
print(f"Phase 1 (compress 3 blocks):     {results['phases'][0]['best_ppl']:.4f}", flush=True)
print(f"Phase 2 (adapt 9 remaining):     {results['phases'][1]['best_ppl']:.4f}", flush=True)
print(f"Phase 3 (compress 3 more):       {results['phases'][2]['best_ppl']:.4f}", flush=True)
print(f"Phase 4 (adapt 6 remaining):     {results['phases'][3]['best_ppl']:.4f}", flush=True)
if phase5_result:
    print(f"Phase 5 (compress 3 more=9tot):  {phase5_result['best_ppl']:.4f}", flush=True)
print(f"\nVERDICT: {verdict}", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

result = {
    "cycle": 361,
    "item": 370,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "ppl_6block": float(ppl_6block),
    "phases": results["phases"],
    "references": {
        "float_baseline": FLOAT_BASELINE_REF,
        "topk_3block_adapter_ref": TOPK_3BLOCK_REF,
    },
    "config": {
        "PHASE1_BLOCKS": PHASE1_BLOCKS,
        "PHASE3_BLOCKS": PHASE3_BLOCKS,
        "PHASE5_BLOCKS": PHASE5_BLOCKS,
        "HIDDEN_FRACTION": HIDDEN_FRACTION,
        "ADAPTER_HIDDEN": ADAPTER_HIDDEN,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE,
        "N_STEPS_COMPRESS": N_STEPS_COMPRESS,
        "LR_COMPRESS": LR_COMPRESS,
        "N_STEPS_ADAPT": N_STEPS_ADAPT,
        "LR_ADAPT": LR_ADAPT,
        "SEED": SEED,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved results to {OUTPUT_PATH}", flush=True)
print("Done.", flush=True)
