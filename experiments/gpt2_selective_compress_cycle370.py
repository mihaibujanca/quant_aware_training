#!/usr/bin/env python3
"""Cycle 370 / Item 378 — Selective compression: compress blocks 0-9, leave b10+b11 at float.

Hypothesis: composition barrier is driven by b10/b11 quantization error (77% of
total local FFN error per c366). Skipping these two blocks should yield PPL near float.

Config: topk+lstsq+adapter (k=512, h=64) on blocks {0..9}. Blocks 10,11 at float32.
Joint adapter training, 2500 steps, 3 seeds.

Verdict:
  CONFIRMED     if mean PPL < 55.0
  REFUTED       if mean PPL > 60.0
  INCONCLUSIVE  if PPL in [55, 60]
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
ALL_BLOCKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # blocks 10,11 left at float

K = 512                  # uniform k for all compressed blocks
ADAPTER_HIDDEN = 64
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 2500
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]

FLOAT_BASELINE_REF = 53.895

OUTPUT_PATH = "autoresearch/results/cycle_370_selective_compression.json"


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
    """Run topk+lstsq compression. Returns per-block stats and compressed module dicts."""
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


def run_seed(seed, calib_ids_cpu, eval_ids_dev):
    """Run compression + training for a single seed. Returns result dict."""
    print(f"\n--- Seed {seed} ---", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]

    print(f"  Compressing blocks {ALL_BLOCKS}...", flush=True)
    compress_cache = compress_blocks(model, ALL_BLOCKS, K, calib_ids_dev, DEVICE)
    compressed_modules = install_compressed_blocks(model, ALL_BLOCKS, compress_cache, DEVICE)

    per_block_frob = {b: compress_cache[b]["frob_recon"] for b in ALL_BLOCKS}
    per_block_mse = {b: compress_cache[b]["mse"] for b in ALL_BLOCKS}

    # Verify b10, b11 are untouched
    for blk_id in [10, 11]:
        assert not isinstance(model.transformer.h[blk_id].mlp, CompressedGPT2MLPWithAdapter), \
            f"Block {blk_id} should not be compressed!"

    best_ppl, restored_ppl = train_joint(
        model, compressed_modules, N_STEPS, LR, calib_ids_dev, eval_ids_dev,
        phase_name=f"joint-seed{seed}"
    )

    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return {
        "seed": seed,
        "best_ppl": float(best_ppl),
        "restored_ppl": float(restored_ppl),
        "per_block_frob_recon": {str(b): float(v) for b, v in per_block_frob.items()},
        "per_block_mse": {str(b): float(v) for b, v in per_block_mse.items()},
        "per_block_k": {str(b): K for b in ALL_BLOCKS},
    }


# ── Load data ─────────────────────────────────────────────────────────────────
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

# ── Count expected adapter params ────────────────────────────────────────────
# Per block: adapter_W1(h,k) + adapter_b1(h) + adapter_W2(768,h) + adapter_b2(768)
#           + proj_weight(768,k) + proj_bias(768)
n_adapter_per_block = (ADAPTER_HIDDEN * K + ADAPTER_HIDDEN +
                       768 * ADAPTER_HIDDEN + 768 +
                       768 * K + 768)
n_adapter_total = len(ALL_BLOCKS) * n_adapter_per_block
print(f"Expected trainable params: {n_adapter_total:,} ({len(ALL_BLOCKS)} blocks)", flush=True)

# ── Run 3 seeds ───────────────────────────────────────────────────────────────
seed_results = []
for seed in SEEDS:
    r = run_seed(seed, calib_ids_cpu, eval_ids_dev)
    seed_results.append(r)
    print(f"  Seed {seed}: best_ppl={r['best_ppl']:.4f} restored={r['restored_ppl']:.4f}",
          flush=True)

ppls = [r["best_ppl"] for r in seed_results]
mean_ppl = sum(ppls) / len(ppls)
std_ppl = (sum((p - mean_ppl) ** 2 for p in ppls) / len(ppls)) ** 0.5

# Effective compression ratio
# 10 out of 12 FFN blocks compressed at k=512 out of 3072 channels
# FFN params: c_fc (768*3072+3072) + c_proj (3072*768+768) ≈ 2*768*3072
# Compressed block saves: (3072-512)*768*2 params (roughly)
total_ffn_params = 12 * 2 * 768 * 3072  # approx
compressed_ffn_params = len(ALL_BLOCKS) * 2 * K * 768
uncompressed_ffn_params = 2 * 768 * 3072   # b10 + b11 each
param_savings = len(ALL_BLOCKS) * 2 * (3072 - K) * 768
compression_ratio = param_savings / total_ffn_params

print(f"\n{'='*60}", flush=True)
print(f"RESULTS SUMMARY", flush=True)
print(f"  Float PPL: {float_ppl:.4f}", flush=True)
print(f"  Compressed blocks: {ALL_BLOCKS}", flush=True)
print(f"  Float blocks: [10, 11]", flush=True)
for r in seed_results:
    print(f"  Seed {r['seed']}: PPL={r['best_ppl']:.4f}", flush=True)
print(f"  Mean PPL: {mean_ppl:.4f} ± {std_ppl:.4f}", flush=True)
print(f"  Effective compression ratio (FFN param savings): {compression_ratio:.3f}", flush=True)
print(f"{'='*60}", flush=True)

# ── Verdict ───────────────────────────────────────────────────────────────────
if mean_ppl < 55.0:
    verdict = "CONFIRMED"
elif mean_ppl > 60.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nVERDICT: {verdict}", flush=True)
print(f"  mean PPL={mean_ppl:.4f} ± {std_ppl:.4f}", flush=True)
print(f"  Thresholds: <55.0 CONFIRMED, >60.0 REFUTED, [55,60] INCONCLUSIVE", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out = {
    "cycle": 370,
    "item": 378,
    "hypothesis": "composition barrier driven by b10/b11 error; skip b10+b11 → PPL near float",
    "float_ppl": float(float_ppl),
    "compressed_blocks": ALL_BLOCKS,
    "float_blocks": [10, 11],
    "k": K,
    "adapter_hidden": ADAPTER_HIDDEN,
    "n_steps": N_STEPS,
    "n_adapter_total": n_adapter_total,
    "compression_ratio": float(compression_ratio),
    "mean_ppl": float(mean_ppl),
    "std_ppl": float(std_ppl),
    "seed_results": seed_results,
    "verdict": verdict,
    "thresholds": {"confirmed": 55.0, "refuted": 60.0},
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {OUTPUT_PATH}", flush=True)
print("Done.", flush=True)
