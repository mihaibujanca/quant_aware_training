#!/usr/bin/env python3
"""Cycle 350 / Item 359 — MLP adapter hidden-size sweep on GPT-2 b2 topk+lstsq.

Sweeps adapter hidden sizes {16, 32, 64, 128} on top of topk+lstsq-compressed b2.
Maps the Pareto frontier: smallest adapter that crosses below float PPL (53.89).

Context:
  Cycle 348: topk+lstsq b2 alone → PPL 48.47
  Cycle 347: LASER b2 + adapter h=64 → PPL 48.51 (246K adapter params)
  This experiment: sweep adapter h on topk+lstsq b2

Adapter architecture:
  MLP residual: x → x + W2·ReLU(W1·x + b1) + b2
  W1: (h, 768), W2: (768, h), initialized small (std=0.01)

Verdict:
  CONFIRMED if >= 1 h gives PPL < 53.89 (float) with clear Pareto structure
  REFUTED if adapter never beats topk+lstsq baseline (48.47) at any h
  INCONCLUSIVE otherwise
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
BLOCK_ID = 2          # b2: highest output-covariance concentration (99.6%)
HIDDEN_FRACTION = 1 / 6  # → k=512 of 3072

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 500
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42

ADAPTER_HIDDEN_SIZES = [16, 32, 64, 128]

FLOAT_BASELINE_REF = 53.895
TOPK_BASELINE_REF = 48.47   # cycle 348 result
CONFIRMED_LT = 53.89        # must beat float to CONFIRM

OUTPUT_PATH = "autoresearch/results/cycle_350_adapter_sweep.json"

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


class CompressedGPT2MLP(nn.Module):
    """Compressed MLP: keeps k of 3072 hidden channels, with trainable W_out."""

    def __init__(self, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias):
        super().__init__()
        # Frozen input projection
        self.register_buffer("fc_weight", c_fc_weight.contiguous())  # (k, 768)
        self.register_buffer("fc_bias", c_fc_bias.contiguous())       # (k,)
        # Trainable output projection
        self.proj_weight = nn.Parameter(c_proj_weight.contiguous())    # (768, k)
        self.proj_bias = nn.Parameter(c_proj_bias.contiguous())        # (768,)

    def forward(self, x):
        h = F.linear(x, self.fc_weight, self.fc_bias)
        h = F.gelu(h)
        out = F.linear(h, self.proj_weight, self.proj_bias)
        return out


class MLPAdapter(nn.Module):
    """Residual MLP adapter: x → x + W2·ReLU(W1·x + b1) + b2.

    Initialized with small weights so it starts as near-identity.
    """

    def __init__(self, d_model: int, h: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(h, d_model) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(h))
        self.W2 = nn.Parameter(torch.randn(d_model, h) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return x + F.linear(F.relu(F.linear(x, self.W1, self.b1)), self.W2, self.b2)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class CompressedMLPWithAdapter(nn.Module):
    """Compressed MLP + residual MLP adapter."""

    def __init__(self, compressed_mlp: CompressedGPT2MLP, adapter: MLPAdapter):
        super().__init__()
        self.compressed = compressed_mlp
        self.adapter = adapter

    def forward(self, x):
        out = self.compressed(x)
        out = self.adapter(out)
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

# ── Compress block b2 ────────────────────────────────────────────────────────
print(f"\nCollecting calibration activations for block {BLOCK_ID}...", flush=True)
calib_dev = [t.to(DEVICE) for t in calib_ids_cpu]
block_acts = collect_block_activations(base_model, [BLOCK_ID], calib_dev, DEVICE)
acts = block_acts[BLOCK_ID]

hidden = acts["hidden"]   # (N, 3072)
output = acts["output"]   # (N, 768)
preact = acts["preact"]   # (N, 3072)

hidden_dim = hidden.shape[1]  # 3072
k = max(1, int(round(hidden_dim * HIDDEN_FRACTION)))
print(f"Block {BLOCK_ID}: hidden_dim={hidden_dim}, k={k}", flush=True)

# Channel scoring (same formula as cycle 348)
std_h = hidden.std(dim=0)
mean_h = hidden.mean(dim=0)
inv_sqrt_2 = 1.0 / math.sqrt(2.0)
inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
deriv = 0.5 * (1.0 + torch.erf(preact * inv_sqrt_2)) + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi
mean_deriv = deriv.mean(dim=0)

c_proj = base_model.transformer.h[BLOCK_ID].mlp.c_proj
c_proj_weight = c_proj.weight.detach().cpu().float()  # Conv1D: (3072, 768)
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

# Extract c_fc weights at selected indices
c_fc = base_model.transformer.h[BLOCK_ID].mlp.c_fc
c_fc_weight_full = c_fc.weight.detach().cpu().float()  # Conv1D: (768, 3072)
c_fc_bias_full = c_fc.bias.detach().cpu().float()      # (3072,)

fc_w = c_fc_weight_full.T[selected]   # (k, 768)
fc_b = c_fc_bias_full[selected]       # (k,)

# lstsq for output projection
hidden_sel = hidden[:, selected].float()   # (N, k)
design = torch.cat([hidden_sel, torch.ones(hidden_sel.shape[0], 1)], dim=1)
solution = torch.linalg.lstsq(design, output.float()).solution
proj_w = solution[:-1].T.contiguous()   # (768, k)
proj_b = solution[-1].contiguous()      # (768,)

with torch.no_grad():
    recon = hidden_sel @ proj_w.T + proj_b
    mse = F.mse_loss(recon, output).item()
    cos = F.cosine_similarity(recon.reshape(1, -1), output.reshape(1, -1)).item()

print(f"Block {BLOCK_ID}: k={k}, dead={dead_mask.sum().item()}, "
      f"lstsq MSE={mse:.4f}, cos={cos:.6f}", flush=True)

compress_cache = {
    "fc_w": fc_w,
    "fc_b": fc_b,
    "proj_w": proj_w,
    "proj_b": proj_b,
    "selected": selected,
    "k": k,
    "mse": mse,
    "cos": cos,
}

del block_acts, acts, hidden, output, preact
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Adapter sweep ─────────────────────────────────────────────────────────────
D_MODEL = 768

sweep_results = []
for h in ADAPTER_HIDDEN_SIZES:
    # Count params
    n_adapter = 2 * h * D_MODEL + h + D_MODEL
    n_proj = proj_w.numel() + proj_b.numel()
    n_total_trainable = n_adapter + n_proj

    print(f"\n{'='*60}", flush=True)
    print(f"Adapter h={h}: adapter_params={n_adapter}, "
          f"total_trainable={n_total_trainable}", flush=True)
    print(f"{'='*60}", flush=True)

    seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        model.eval()

        # Build compressed MLP + adapter
        comp_mlp = CompressedGPT2MLP(
            c_fc_weight=compress_cache["fc_w"].to(device=DEVICE, dtype=torch.float32),
            c_fc_bias=compress_cache["fc_b"].to(device=DEVICE, dtype=torch.float32),
            c_proj_weight=compress_cache["proj_w"].to(device=DEVICE, dtype=torch.float32),
            c_proj_bias=compress_cache["proj_b"].to(device=DEVICE, dtype=torch.float32),
        )
        adapter = MLPAdapter(D_MODEL, h).to(DEVICE)
        combined = CompressedMLPWithAdapter(comp_mlp, adapter).to(DEVICE)
        model.transformer.h[BLOCK_ID].mlp = combined

        init_ppl = eval_ppl(model, eval_ids)

        # Freeze all except proj_weight/bias and adapter
        for p in model.parameters():
            p.requires_grad_(False)
        combined.compressed.proj_weight.requires_grad_(True)
        combined.compressed.proj_bias.requires_grad_(True)
        for p in combined.adapter.parameters():
            p.requires_grad_(True)

        trainable = ([combined.compressed.proj_weight, combined.compressed.proj_bias]
                     + list(combined.adapter.parameters()))
        n_train = sum(p.numel() for p in trainable)
        assert n_train == n_total_trainable, f"{n_train} != {n_total_trainable}"

        optimizer = torch.optim.Adam(trainable, lr=LR)
        calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
        random.shuffle(calib_ids_dev)

        best_ppl = init_ppl
        best_step = 0
        best_state = {
            "proj_w": combined.compressed.proj_weight.detach().clone(),
            "proj_b": combined.compressed.proj_bias.detach().clone(),
            "adapter": {n: p.detach().clone() for n, p in combined.adapter.named_parameters()},
        }
        history = [{"step": 0, "eval_ppl": float(init_ppl)}]

        step = 0
        t0 = time.time()
        while step < N_STEPS:
            for ids in calib_ids_dev:
                if step >= N_STEPS:
                    break
                model.train()
                optimizer.zero_grad()
                out = model(ids, labels=ids)
                loss = out.loss
                loss.backward()
                optimizer.step()
                step += 1

                if step % EVAL_EVERY == 0:
                    ppl_e = eval_ppl(model, eval_ids)
                    history.append({"step": step, "eval_ppl": float(ppl_e)})
                    if ppl_e < best_ppl:
                        best_ppl = ppl_e
                        best_step = step
                        best_state = {
                            "proj_w": combined.compressed.proj_weight.detach().clone(),
                            "proj_b": combined.compressed.proj_bias.detach().clone(),
                            "adapter": {n: p.detach().clone() for n, p in combined.adapter.named_parameters()},
                        }
                    if step % 200 == 0:
                        print(f"  h={h} seed={seed} step={step}: ppl={ppl_e:.2f} "
                              f"best={best_ppl:.2f}@{best_step}", flush=True)

        # Restore best
        with torch.no_grad():
            combined.compressed.proj_weight.data.copy_(best_state["proj_w"])
            combined.compressed.proj_bias.data.copy_(best_state["proj_b"])
            for n, p in combined.adapter.named_parameters():
                p.data.copy_(best_state["adapter"][n])
        restored_ppl = eval_ppl(model, eval_ids)
        dt = time.time() - t0

        print(f"  h={h} seed={seed}: init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
              f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)

        seed_results.append({
            "seed": seed,
            "init_ppl": float(init_ppl),
            "best_ppl": float(best_ppl),
            "best_step": int(best_step),
            "restored_ppl": float(restored_ppl),
            "history": history,
        })

        del model, comp_mlp, adapter, combined, optimizer
        if DEVICE == "mps":
            torch.mps.empty_cache()

    best_ppls = [s["best_ppl"] for s in seed_results]
    best_mean = sum(best_ppls) / len(best_ppls)
    best_std = math.sqrt(sum((x - best_mean) ** 2 for x in best_ppls) / len(best_ppls))

    sweep_results.append({
        "h": h,
        "n_adapter_params": n_adapter,
        "n_total_trainable": n_total_trainable,
        "init_ppl": float(seed_results[0]["init_ppl"]),
        "best_mean": float(best_mean),
        "best_std": float(best_std),
        "best_min": float(min(best_ppls)),
        "best_max": float(max(best_ppls)),
        "seeds": seed_results,
    })

    print(f"\nh={h} summary: best_mean={best_mean:.2f} ± {best_std:.2f} "
          f"(min={min(best_ppls):.2f}, max={max(best_ppls):.2f})", flush=True)


# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("ADAPTER HIDDEN-SIZE SWEEP — topk+lstsq b2", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'h':>6s}  {'n_adapter':>10s}  {'PPL mean':>10s}  {'PPL std':>8s}", flush=True)
for r in sweep_results:
    print(f"{r['h']:>6d}  {r['n_adapter_params']:>10d}  {r['best_mean']:>10.2f}  {r['best_std']:>8.2f}", flush=True)

print(f"\nFloat PPL: {float_ppl:.4f}")
print(f"topk+lstsq baseline (c348): {TOPK_BASELINE_REF}")
print(f"CONFIRMED threshold (PPL < {CONFIRMED_LT}): beat float")

# Pareto check: does Pareto structure exist (monotone with h)?
all_means = [r["best_mean"] for r in sweep_results]
any_beats_topk = any(m < TOPK_BASELINE_REF for m in all_means)
any_beats_float = any(m < CONFIRMED_LT for m in all_means)
pareto_monotone = all(all_means[i] >= all_means[i+1] for i in range(len(all_means)-1))

if any_beats_float:
    verdict = "CONFIRMED"
elif any_beats_topk:
    verdict = "INCONCLUSIVE"
else:
    verdict = "REFUTED"

print(f"\nAny h beats topk baseline ({TOPK_BASELINE_REF}): {any_beats_topk}")
print(f"Any h beats float ({CONFIRMED_LT}): {any_beats_float}")
print(f"Monotone Pareto (larger h → lower PPL): {pareto_monotone}")
print(f"VERDICT: {verdict}", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────
import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
result = {
    "cycle": 350,
    "item": 359,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "topk_baseline_ref": TOPK_BASELINE_REF,
    "confirmed_lt": CONFIRMED_LT,
    "method": "topk+lstsq+adapter",
    "block_id": BLOCK_ID,
    "hidden_fraction": HIDDEN_FRACTION,
    "k": k,
    "any_beats_topk": any_beats_topk,
    "any_beats_float": any_beats_float,
    "pareto_monotone": pareto_monotone,
    "sweep": [{k2: v for k2, v in r.items() if k2 != "seeds"} for r in sweep_results],
    "sweep_seeds": {r["h"]: r["seeds"] for r in sweep_results},
    "compress_stats": {
        "k": compress_cache["k"],
        "mse": compress_cache["mse"],
        "cos": compress_cache["cos"],
        "n_dead": int(dead_mask.sum().item()),
    },
    "config": {
        "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE, "N_STEPS": N_STEPS, "LR": LR,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
        "ADAPTER_HIDDEN_SIZES": ADAPTER_HIDDEN_SIZES,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved results to {OUTPUT_PATH}", flush=True)
