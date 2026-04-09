#!/usr/bin/env python3
"""Cycle 360 / Item 368 — LoRA on compressed c_proj vs MLP adapter on GPT-2 b2.

Question: does the adapter's PPL gain come from nonlinear residual learning (ReLU)
or just extra parameters? LoRA replaces the nonlinear adapter with a linear
low-rank correction to W_proj.

Setup (GPT-2 b2, topk+lstsq k=512):
  Baseline:        PPL ~48.47 (c348)
  MLP adapter ref: h=16 ~47.81 (25K params), h=32 ~47.43 (50K), h=64 ~46.82 (99K)
                   h=128 ~46.22 (197K) (c350)
  LoRA:            r=8 (12K params), r=32 (49K), r=64 (98K)

LoRA: W_proj = W_proj_frozen + B @ A   where A: (r, 768), B: (768, r)
Initialized: A ~ N(0, 0.01), B = 0

Verdict:
  REFUTED-linear-suffices     if LoRA matches MLP adapter within ±0.5 PPL @ matched count
  CONFIRMED-nonlinearity-essential if MLP adapter beats LoRA by ≥1.0 PPL @ matched count
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
BLOCK_ID = 2
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

LORA_RANKS = [8, 32, 64]

FLOAT_BASELINE_REF = 53.895
TOPK_BASELINE_REF = 48.47

# MLP adapter references from cycle 350 (mean PPL across seeds)
MLP_ADAPTER_REF = {
    16:  {"ppl": 47.81, "params": 25344},
    32:  {"ppl": 47.43, "params": 50432},
    64:  {"ppl": 46.82, "params": 100608},
    128: {"ppl": 46.22, "params": 197376},
}

OUTPUT_PATH = "autoresearch/results/cycle_360_lora_vs_adapter.json"

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


class CompressedGPT2MLP(nn.Module):
    """Compressed MLP: keeps k of 3072 hidden channels, trainable W_out."""

    def __init__(self, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias):
        super().__init__()
        self.register_buffer("fc_weight", c_fc_weight.contiguous())
        self.register_buffer("fc_bias", c_fc_bias.contiguous())
        self.proj_weight = nn.Parameter(c_proj_weight.contiguous())
        self.proj_bias = nn.Parameter(c_proj_bias.contiguous())

    def forward(self, x):
        h = F.linear(x, self.fc_weight, self.fc_bias)
        h = F.gelu(h)
        return F.linear(h, self.proj_weight, self.proj_bias)


class CompressedMLPWithLoRA(nn.Module):
    """Compressed MLP + LoRA correction on W_proj.

    W_eff = W_proj_frozen + B @ A
    A: (r, d_in=k), B: (d_out=768, r)
    Init: A ~ N(0, 0.01), B = 0  → starts at zero delta.
    """

    def __init__(self, compressed_mlp: CompressedGPT2MLP, r: int):
        super().__init__()
        self.compressed = compressed_mlp
        d_out, d_in = compressed_mlp.proj_weight.shape  # (768, k)
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.r = r

    def forward(self, x):
        h = F.linear(x, self.compressed.fc_weight, self.compressed.fc_bias)
        h = F.gelu(h)
        # Effective weight: W_proj + B @ A
        w_eff = self.compressed.proj_weight + self.lora_B @ self.lora_A
        return F.linear(h, w_eff, self.compressed.proj_bias)

    @property
    def n_lora_params(self):
        return self.lora_A.numel() + self.lora_B.numel()


class MLPAdapter(nn.Module):
    """Residual MLP adapter: x → x + W2·ReLU(W1·x + b1) + b2."""

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
    def __init__(self, compressed_mlp: CompressedGPT2MLP, adapter: MLPAdapter):
        super().__init__()
        self.compressed = compressed_mlp
        self.adapter = adapter

    def forward(self, x):
        return self.adapter(self.compressed(x))


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

hidden = acts["hidden"]
output = acts["output"]
preact = acts["preact"]

hidden_dim = hidden.shape[1]
k = max(1, int(round(hidden_dim * HIDDEN_FRACTION)))
print(f"Block {BLOCK_ID}: hidden_dim={hidden_dim}, k={k}", flush=True)

# Channel scoring (same formula as cycle 348/350)
std_h = hidden.std(dim=0)
mean_h = hidden.mean(dim=0)
inv_sqrt_2 = 1.0 / math.sqrt(2.0)
inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
deriv = 0.5 * (1.0 + torch.erf(preact * inv_sqrt_2)) + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi
mean_deriv = deriv.mean(dim=0)

c_proj = base_model.transformer.h[BLOCK_ID].mlp.c_proj
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

c_fc = base_model.transformer.h[BLOCK_ID].mlp.c_fc
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
    mse = F.mse_loss(recon, output).item()
    cos = F.cosine_similarity(recon.reshape(1, -1), output.reshape(1, -1)).item()

print(f"Block {BLOCK_ID}: k={k}, dead={dead_mask.sum().item()}, "
      f"lstsq MSE={mse:.4f}, cos={cos:.6f}", flush=True)

compress_cache = {
    "fc_w": fc_w, "fc_b": fc_b, "proj_w": proj_w, "proj_b": proj_b,
    "selected": selected, "k": k, "mse": mse, "cos": cos,
}

del block_acts, acts, hidden, output, preact
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Generic training loop ─────────────────────────────────────────────────────
D_MODEL = 768

def run_training(model, trainable_params, seed, label):
    init_ppl = eval_ppl(model, eval_ids)

    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
    random.shuffle(calib_ids_dev)

    best_ppl = init_ppl
    best_step = 0
    best_state = {n: p.detach().clone() for n, p in
                  [(str(i), p) for i, p in enumerate(trainable_params)]}
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
            out.loss.backward()
            optimizer.step()
            step += 1

            if step % EVAL_EVERY == 0:
                ppl_e = eval_ppl(model, eval_ids)
                history.append({"step": step, "eval_ppl": float(ppl_e)})
                if ppl_e < best_ppl:
                    best_ppl = ppl_e
                    best_step = step
                    best_state = {str(i): p.detach().clone()
                                  for i, p in enumerate(trainable_params)}
                if step % 200 == 0:
                    print(f"  {label} seed={seed} step={step}: ppl={ppl_e:.2f} "
                          f"best={best_ppl:.2f}@{best_step}", flush=True)

    # Restore best
    with torch.no_grad():
        for i, p in enumerate(trainable_params):
            p.data.copy_(best_state[str(i)])
    restored_ppl = eval_ppl(model, eval_ids)
    dt = time.time() - t0
    print(f"  {label} seed={seed}: init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
          f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)

    return {
        "seed": seed, "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl), "best_step": int(best_step),
        "restored_ppl": float(restored_ppl), "history": history,
    }


# ── LoRA sweep ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("LoRA SWEEP", flush=True)
print(f"{'='*60}", flush=True)

lora_results = []
for r in LORA_RANKS:
    # LoRA params: A=(r, k) + B=(768, r)  plus proj_bias (768)
    n_lora = r * k + D_MODEL * r
    n_proj_bias = D_MODEL
    n_total = n_lora + n_proj_bias

    print(f"\nLoRA r={r}: lora_params={n_lora}, total_trainable={n_total}", flush=True)

    seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        model.eval()

        comp_mlp = CompressedGPT2MLP(
            c_fc_weight=compress_cache["fc_w"].to(device=DEVICE, dtype=torch.float32),
            c_fc_bias=compress_cache["fc_b"].to(device=DEVICE, dtype=torch.float32),
            c_proj_weight=compress_cache["proj_w"].to(device=DEVICE, dtype=torch.float32),
            c_proj_bias=compress_cache["proj_b"].to(device=DEVICE, dtype=torch.float32),
        )
        lora_mlp = CompressedMLPWithLoRA(comp_mlp, r=r).to(DEVICE)
        model.transformer.h[BLOCK_ID].mlp = lora_mlp

        # Freeze everything; train only LoRA params + proj_bias
        for p in model.parameters():
            p.requires_grad_(False)
        lora_mlp.lora_A.requires_grad_(True)
        lora_mlp.lora_B.requires_grad_(True)
        lora_mlp.compressed.proj_bias.requires_grad_(True)

        trainable = [lora_mlp.lora_A, lora_mlp.lora_B, lora_mlp.compressed.proj_bias]
        n_train = sum(p.numel() for p in trainable)
        assert n_train == n_total, f"{n_train} != {n_total}"

        res = run_training(model, trainable, seed, label=f"LoRA-r{r}")
        seed_results.append(res)

        del model, comp_mlp, lora_mlp
        if DEVICE == "mps":
            torch.mps.empty_cache()

    best_ppls = [s["best_ppl"] for s in seed_results]
    best_mean = sum(best_ppls) / len(best_ppls)
    best_std = math.sqrt(sum((x - best_mean) ** 2 for x in best_ppls) / len(best_ppls))

    lora_results.append({
        "r": r, "n_lora_params": n_lora, "n_total_trainable": n_total,
        "init_ppl": float(seed_results[0]["init_ppl"]),
        "best_mean": float(best_mean), "best_std": float(best_std),
        "best_min": float(min(best_ppls)), "best_max": float(max(best_ppls)),
        "seeds": seed_results,
    })
    print(f"\nLoRA r={r} summary: best_mean={best_mean:.2f} ± {best_std:.2f} "
          f"(min={min(best_ppls):.2f}, max={max(best_ppls):.2f})", flush=True)


# ── MLP adapter at matched param counts ──────────────────────────────────────
# We need matched-param adapters: h that gives ~same param count as each LoRA r.
# LoRA r=8: 12K → closest adapter h=16 (25K) or below
# LoRA r=32: 49K → adapter h=32 (50K)  -- near-exact match
# LoRA r=64: 98K → adapter h=64 (99K)  -- near-exact match
# Run adapter h=16, 32, 64 here for direct comparison in same run.

ADAPTER_HS = [16, 32, 64]

print(f"\n{'='*60}", flush=True)
print("MLP ADAPTER SWEEP (matched param counts)", flush=True)
print(f"{'='*60}", flush=True)

adapter_results = []
for h in ADAPTER_HS:
    n_adapter = 2 * h * D_MODEL + h + D_MODEL
    n_proj = compress_cache["proj_w"].numel() + compress_cache["proj_b"].numel()
    n_total = n_adapter + n_proj

    print(f"\nAdapter h={h}: adapter_params={n_adapter}, total_trainable={n_total}", flush=True)

    seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        model.eval()

        comp_mlp = CompressedGPT2MLP(
            c_fc_weight=compress_cache["fc_w"].to(device=DEVICE, dtype=torch.float32),
            c_fc_bias=compress_cache["fc_b"].to(device=DEVICE, dtype=torch.float32),
            c_proj_weight=compress_cache["proj_w"].to(device=DEVICE, dtype=torch.float32),
            c_proj_bias=compress_cache["proj_b"].to(device=DEVICE, dtype=torch.float32),
        )
        adapter = MLPAdapter(D_MODEL, h).to(DEVICE)
        combined = CompressedMLPWithAdapter(comp_mlp, adapter).to(DEVICE)
        model.transformer.h[BLOCK_ID].mlp = combined

        for p in model.parameters():
            p.requires_grad_(False)
        combined.compressed.proj_weight.requires_grad_(True)
        combined.compressed.proj_bias.requires_grad_(True)
        for p in combined.adapter.parameters():
            p.requires_grad_(True)

        trainable = ([combined.compressed.proj_weight, combined.compressed.proj_bias]
                     + list(combined.adapter.parameters()))
        n_train = sum(p.numel() for p in trainable)
        assert n_train == n_total, f"{n_train} != {n_total}"

        res = run_training(model, trainable, seed, label=f"Adapter-h{h}")
        seed_results.append(res)

        del model, comp_mlp, adapter, combined
        if DEVICE == "mps":
            torch.mps.empty_cache()

    best_ppls = [s["best_ppl"] for s in seed_results]
    best_mean = sum(best_ppls) / len(best_ppls)
    best_std = math.sqrt(sum((x - best_mean) ** 2 for x in best_ppls) / len(best_ppls))

    adapter_results.append({
        "h": h, "n_adapter_params": n_adapter, "n_total_trainable": n_total,
        "init_ppl": float(seed_results[0]["init_ppl"]),
        "best_mean": float(best_mean), "best_std": float(best_std),
        "best_min": float(min(best_ppls)), "best_max": float(max(best_ppls)),
        "seeds": seed_results,
    })
    print(f"\nAdapter h={h} summary: best_mean={best_mean:.2f} ± {best_std:.2f} "
          f"(min={min(best_ppls):.2f}, max={max(best_ppls):.2f})", flush=True)


# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("COMPARISON TABLE", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Method':>16s}  {'Params':>8s}  {'PPL mean':>10s}  {'PPL std':>8s}", flush=True)
print(f"{'baseline':>16s}  {'---':>8s}  {TOPK_BASELINE_REF:>10.2f}  {'---':>8s}", flush=True)
for r in lora_results:
    print(f"{'LoRA r='+str(r['r']):>16s}  {r['n_lora_params']:>8d}  "
          f"{r['best_mean']:>10.2f}  {r['best_std']:>8.2f}", flush=True)
for r in adapter_results:
    print(f"{'Adapter h='+str(r['h']):>16s}  {r['n_adapter_params']:>8d}  "
          f"{r['best_mean']:>10.2f}  {r['best_std']:>8.2f}", flush=True)

# Matched comparisons (by approximate param count)
# r=32 ↔ h=32 (~50K params): exact match
# r=64 ↔ h=64 (~99K params): near-exact match
comparisons = []
for lora_r in lora_results:
    # Find closest adapter by param count
    best_match = min(adapter_results,
                     key=lambda a: abs(a["n_adapter_params"] - lora_r["n_lora_params"]))
    gap = best_match["best_mean"] - lora_r["best_mean"]
    comparisons.append({
        "lora_r": lora_r["r"],
        "lora_params": lora_r["n_lora_params"],
        "lora_ppl": lora_r["best_mean"],
        "adapter_h": best_match["h"],
        "adapter_params": best_match["n_adapter_params"],
        "adapter_ppl": best_match["best_mean"],
        "gap_ppl": float(gap),  # positive = adapter is better
    })

print(f"\n{'Matched comparisons (adapter PPL - LoRA PPL, neg=LoRA better)':}", flush=True)
for c in comparisons:
    dir_str = "adapter BETTER" if c["gap_ppl"] < -0.5 else (
              "LoRA BETTER" if c["gap_ppl"] > 0.5 else "SIMILAR")
    print(f"  LoRA r={c['lora_r']} ({c['lora_params']}p, PPL={c['lora_ppl']:.2f}) vs "
          f"Adapter h={c['adapter_h']} ({c['adapter_params']}p, PPL={c['adapter_ppl']:.2f}): "
          f"gap={c['gap_ppl']:.2f} → {dir_str}", flush=True)

# Determine exact-match pairs for verdict (r=32 vs h=32, r=64 vs h=64)
exact_gaps = [c["gap_ppl"] for c in comparisons
              if c["lora_r"] in [32, 64]]

if len(exact_gaps) > 0:
    mean_gap = sum(exact_gaps) / len(exact_gaps)
else:
    mean_gap = comparisons[0]["gap_ppl"] if comparisons else 0.0

if mean_gap <= -1.0:
    verdict = "CONFIRMED-nonlinearity-essential"
elif mean_gap >= -0.5:
    verdict = "REFUTED-linear-suffices"
else:
    verdict = "INCONCLUSIVE"

print(f"\nFloat PPL: {float_ppl:.4f}")
print(f"topk+lstsq baseline: {TOPK_BASELINE_REF}")
print(f"Mean gap (adapter - LoRA) at matched params: {mean_gap:.3f}")
print(f"VERDICT: {verdict}", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
result = {
    "cycle": 360,
    "item": 368,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "topk_baseline_ref": TOPK_BASELINE_REF,
    "block_id": BLOCK_ID,
    "k": k,
    "lora_results": lora_results,
    "adapter_results": adapter_results,
    "comparisons": comparisons,
    "mean_gap_at_matched_params": float(mean_gap),
    "mlp_adapter_ref_c350": MLP_ADAPTER_REF,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {OUTPUT_PATH}", flush=True)
