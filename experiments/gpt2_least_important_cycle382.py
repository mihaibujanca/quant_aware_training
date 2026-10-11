#!/usr/bin/env python3
"""Cycle 382 / Item 390 — Compress only least-important blocks [b1,b7,b2] + adapter.

Train float+adapter AND compressed+adapter on wikitext-2 VALIDATION.
Evaluate BOTH on wikitext-2 TEST (held-out, never seen during training).
Select best checkpoint by VAL loss, then evaluate on test ONCE.
Also eval bare float GPT-2 on test as baseline.

Config: 3 blocks [b1, b7, b2] (3 least-important by c367), k=512,
        h=64 for BOTH float and compressed adapters.
Seeds: 42, 137, 2024

Penalty = mean_comp_test - mean_float_adapter_test
  CONFIRMED: penalty < 3.0    (minimal compression damage on least-important blocks)
  REFUTED:   penalty >= 5.0   (compression damage comparable to important blocks)
  INCONCLUSIVE: [3.0, 5.0)

c378 reference penalty (blocks [2,1,10], h=64/h=128): 7.49
This script uses SAME h=64 on BOTH sides, blocks [1,7,2] instead.
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
BLOCKS = [1, 7, 2]
HIDDEN_FRACTION = 1 / 6  # → k=512
ADAPTER_H = 64            # h=64 for BOTH float and compressed configs
LR = 1e-4
N_STEPS = 2000
SEEDS = [42, 137, 2024]
N_CALIB = 200
N_EVAL = 200
SEQ_LEN = 128
EVAL_EVERY = 100

FLOAT_BASELINE_REF = 53.895
C378_PENALTY_REF = 7.49   # c378 penalty with blocks [2,1,10] and h=64/h=128

CONFIRMED_LT_PENALTY = 3.0   # penalty < 3 → CONFIRMED
REFUTED_GE_PENALTY = 5.0     # penalty >= 5 → REFUTED

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_382_least_important.json"

torch.manual_seed(42)
print(f"Device: {DEVICE}", flush=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading GPT-2 tokenizer and wikitext-2 data...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def tokenize_split(split_name, n_sequences):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split_name)
    texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
    ids_list = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
        if enc["input_ids"].shape[1] >= SEQ_LEN:
            ids_list.append(enc["input_ids"][:, :SEQ_LEN])
        if len(ids_list) >= n_sequences:
            break
    return ids_list


# Validation split: for training and checkpoint selection
val_ids_cpu = tokenize_split("validation", N_CALIB + 50)  # extra 50 for eval on val
calib_ids_cpu = val_ids_cpu[:N_CALIB]
val_eval_ids = [t.to(DEVICE) for t in val_ids_cpu[N_CALIB: N_CALIB + 50]]
print(f"Val calib: {len(calib_ids_cpu)} seqs, Val eval: {len(val_eval_ids)} seqs", flush=True)

# Test split: held-out, touched only once per config at the very end
test_ids_cpu = tokenize_split("test", N_EVAL)
test_eval_ids = [t.to(DEVICE) for t in test_ids_cpu[:N_EVAL]]
print(f"Test eval: {len(test_eval_ids)} seqs", flush=True)


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


def run_steps_with_val_selection(model, trainable_params, n_steps, calib_dev, val_ids, label=""):
    """Train on calib, select checkpoint by val PPL, return best_val_ppl."""
    opt = torch.optim.Adam(trainable_params, lr=LR)
    shuffled = list(calib_dev)
    random.shuffle(shuffled)

    init_ppl = eval_ppl(model, val_ids)
    best_val_ppl = init_ppl
    best_state = {id(p): p.detach().clone() for p in trainable_params}
    print(f"  {label} init val PPL: {init_ppl:.4f}", flush=True)

    step = 0
    while step < n_steps:
        for ids in shuffled:
            if step >= n_steps:
                break
            model.train()
            opt.zero_grad()
            model(ids, labels=ids).loss.backward()
            opt.step()
            step += 1
            if step % EVAL_EVERY == 0:
                ppl_e = eval_ppl(model, val_ids)
                if ppl_e < best_val_ppl:
                    best_val_ppl = ppl_e
                    best_state = {id(p): p.detach().clone() for p in trainable_params}
                if step % 500 == 0:
                    print(f"  {label} step {step}/{n_steps}: val_ppl={ppl_e:.2f} "
                          f"best_val={best_val_ppl:.2f}", flush=True)

    # Restore best val checkpoint
    with torch.no_grad():
        for p in trainable_params:
            p.data.copy_(best_state[id(p)])

    return best_val_ppl


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


# ── Float baseline on test ────────────────────────────────────────────────────
print("Computing float baseline PPL on test...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
float_val_ppl = eval_ppl(base_model, val_eval_ids)
float_test_ppl = eval_ppl(base_model, test_eval_ids)
print(f"Float val PPL: {float_val_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
print(f"Float test PPL: {float_test_ppl:.4f}", flush=True)

# Collect activations for compression (reuse base_model)
print("\nCollecting calibration activations for compression...", flush=True)
calib_dev = [t.to(DEVICE) for t in calib_ids_cpu]
block_acts = collect_block_activations(base_model, BLOCKS, calib_dev, DEVICE)

# Build compress_cache (topk+lstsq, k=512)
compress_cache = {}
for blk in BLOCKS:
    acts = block_acts[blk]
    hidden = acts["hidden"]
    output = acts["output"]
    preact = acts["preact"]

    hidden_dim = hidden.shape[1]
    k = max(1, int(round(hidden_dim * HIDDEN_FRACTION)))

    std_h = hidden.std(dim=0)
    mean_h = hidden.mean(dim=0)
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    deriv = (0.5 * (1.0 + torch.erf(preact * inv_sqrt_2))
             + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi)
    mean_deriv = deriv.mean(dim=0)

    c_proj = base_model.transformer.h[blk].mlp.c_proj
    c_proj_weight = c_proj.weight.detach().cpu().float()   # Conv1D: (3072, 768)
    output_norms = c_proj_weight.norm(dim=1)               # (3072,)

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
    c_fc_weight_full = c_fc.weight.detach().cpu().float()  # Conv1D: (768, 3072)
    c_fc_bias_full = c_fc.bias.detach().cpu().float()

    fc_w = c_fc_weight_full.T[selected]   # (k, 768)
    fc_b = c_fc_bias_full[selected]       # (k,)

    hidden_sel = hidden[:, selected].float()
    design = torch.cat([hidden_sel, torch.ones(hidden_sel.shape[0], 1)], dim=1)
    solution = torch.linalg.lstsq(design, output.float()).solution
    proj_w = solution[:-1].T.contiguous()   # (768, k)
    proj_b = solution[-1].contiguous()      # (768,)

    compress_cache[blk] = {
        "fc_w": fc_w, "fc_b": fc_b,
        "proj_w": proj_w, "proj_b": proj_b,
        "selected": selected, "k": k,
    }
    print(f"Block {blk}: k={k}", flush=True)

del block_acts, base_model
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Model classes ─────────────────────────────────────────────────────────────
class FloatGPT2MLPWithAdapter(nn.Module):
    """Uncompressed GPT-2 MLP with residual adapter on post-FFN output. h=64."""

    def __init__(self, fc_weight, fc_bias, proj_weight, proj_bias, h=64):
        super().__init__()
        self.register_buffer("fc_weight", fc_weight.contiguous())
        self.register_buffer("fc_bias", fc_bias.contiguous())
        self.register_buffer("proj_weight", proj_weight.contiguous())
        self.register_buffer("proj_bias", proj_bias.contiguous())

        d_model = proj_weight.shape[0]
        self.adapter_W1 = nn.Parameter(torch.randn(h, d_model) * 0.01)
        self.adapter_b1 = nn.Parameter(torch.zeros(h))
        self.adapter_W2 = nn.Parameter(torch.randn(d_model, h) * 0.01)
        self.adapter_b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        h = F.linear(x, self.fc_weight, self.fc_bias)
        h = F.gelu(h)
        out = F.linear(h, self.proj_weight, self.proj_bias)
        a = F.relu(F.linear(out, self.adapter_W1, self.adapter_b1))
        out = out + F.linear(a, self.adapter_W2, self.adapter_b2)
        return out

    def adapter_params(self):
        return [self.adapter_W1, self.adapter_b1, self.adapter_W2, self.adapter_b2]


class CompressedGPT2MLPWithAdapter(nn.Module):
    """Compressed MLP (topk k=512 + lstsq) with adapter on GELU hidden. h=64."""

    def __init__(self, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias, h=64):
        super().__init__()
        k = c_fc_weight.shape[0]
        self.register_buffer("fc_weight", c_fc_weight.contiguous())
        self.register_buffer("fc_bias", c_fc_bias.contiguous())
        self.proj_weight = nn.Parameter(c_proj_weight.contiguous())
        self.proj_bias = nn.Parameter(c_proj_bias.contiguous())

        scale1 = 1.0 / math.sqrt(k)
        scale2 = 1.0 / math.sqrt(h)
        self.adapter_W1 = nn.Parameter(torch.randn(h, k) * scale1)
        self.adapter_b1 = nn.Parameter(torch.zeros(h))
        self.adapter_W2 = nn.Parameter(torch.randn(768, h) * scale2)
        self.adapter_b2 = nn.Parameter(torch.zeros(768))

    def forward(self, x):
        h = F.linear(x, self.fc_weight, self.fc_bias)
        h = F.gelu(h)
        out = F.linear(h, self.proj_weight, self.proj_bias)
        a = F.relu(F.linear(h, self.adapter_W1, self.adapter_b1))
        out = out + F.linear(a, self.adapter_W2, self.adapter_b2)
        return out

    def all_trainable_params(self):
        return [self.proj_weight, self.proj_bias,
                self.adapter_W1, self.adapter_b1, self.adapter_W2, self.adapter_b2]

    def adapter_params(self):
        return [self.adapter_W1, self.adapter_b1, self.adapter_W2, self.adapter_b2]


# ── Per-seed training ─────────────────────────────────────────────────────────
# Float adapter: 3 blocks × (W1: 768×64 + b1: 64 + W2: 64×768 + b2: 768) = 3 × 99,136
EXPECTED_FLOAT_ADAPTER_PARAMS = 3 * (2 * ADAPTER_H * 768 + ADAPTER_H + 768)
# Compressed adapter: 3 blocks × (W1: 512×64 + b1: 64 + W2: 64×768 + b2: 768) = 3 × (32768+64+49152+768)
EXPECTED_COMP_ADAPTER_PARAMS = 3 * (ADAPTER_H * 512 + ADAPTER_H + ADAPTER_H * 768 + 768)

print(f"\nFloat adapter trainable params: {EXPECTED_FLOAT_ADAPTER_PARAMS}", flush=True)
print(f"Compressed adapter params: {EXPECTED_COMP_ADAPTER_PARAMS} (+ proj weights)", flush=True)

float_adapter_results = []
compressed_adapter_results = []

for seed in SEEDS:
    print(f"\n{'='*60}", flush=True)
    print(f"SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(seed)
    random.seed(seed)

    # ── Config A: Float + adapter ──
    print(f"\n[Seed {seed}] Config A: Float + adapter (h={ADAPTER_H})", flush=True)
    model_a = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model_a.eval()

    for blk in BLOCKS:
        mlp = model_a.transformer.h[blk].mlp
        fc_w = mlp.c_fc.weight.detach().float().T.contiguous()    # (3072, 768)
        fc_b = mlp.c_fc.bias.detach().float()
        proj_w = mlp.c_proj.weight.detach().float().T.contiguous()  # (768, 3072)
        proj_b = mlp.c_proj.bias.detach().float()
        new_mlp = FloatGPT2MLPWithAdapter(
            fc_weight=fc_w, fc_bias=fc_b,
            proj_weight=proj_w, proj_bias=proj_b,
            h=ADAPTER_H
        ).to(DEVICE)
        model_a.transformer.h[blk].mlp = new_mlp

    for p in model_a.parameters():
        p.requires_grad_(False)
    trainable_a = []
    for blk in BLOCKS:
        adapter_ps = model_a.transformer.h[blk].mlp.adapter_params()
        for p in adapter_ps:
            p.requires_grad_(True)
        trainable_a.extend(adapter_ps)

    n_trainable_a = sum(p.numel() for p in trainable_a)
    print(f"  Trainable params: {n_trainable_a} (expected {EXPECTED_FLOAT_ADAPTER_PARAMS})",
          flush=True)

    t0 = time.time()
    best_val_ppl_a = run_steps_with_val_selection(
        model_a, trainable_a, N_STEPS, calib_dev, val_eval_ids,
        label=f"float_adapter seed={seed}"
    )
    elapsed_a = time.time() - t0

    # Final eval on TEST (held-out) — only done once after checkpoint selection
    test_ppl_a = eval_ppl(model_a, test_eval_ids)
    print(f"  [Seed {seed}] Float+adapter: best_val={best_val_ppl_a:.4f}, "
          f"TEST={test_ppl_a:.4f}, time={elapsed_a:.1f}s", flush=True)

    float_adapter_results.append({
        "seed": seed,
        "best_val_ppl": float(best_val_ppl_a),
        "test_ppl": float(test_ppl_a),
    })

    del model_a, trainable_a
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # ── Config B: Compressed + adapter ──
    torch.manual_seed(seed)
    random.seed(seed)

    print(f"\n[Seed {seed}] Config B: Compressed + adapter (h={ADAPTER_H})", flush=True)
    model_b = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model_b.eval()

    for blk in BLOCKS:
        c = compress_cache[blk]
        m = CompressedGPT2MLPWithAdapter(
            c_fc_weight=c["fc_w"].to(DEVICE, torch.float32),
            c_fc_bias=c["fc_b"].to(DEVICE, torch.float32),
            c_proj_weight=c["proj_w"].to(DEVICE, torch.float32),
            c_proj_bias=c["proj_b"].to(DEVICE, torch.float32),
            h=ADAPTER_H,
        ).to(DEVICE)
        model_b.transformer.h[blk].mlp = m

    for p in model_b.parameters():
        p.requires_grad_(False)
    trainable_b = []
    for blk in BLOCKS:
        mlp_b = model_b.transformer.h[blk].mlp
        for p in mlp_b.all_trainable_params():
            p.requires_grad_(True)
        trainable_b.extend(mlp_b.all_trainable_params())

    n_trainable_b = sum(p.numel() for p in trainable_b)
    print(f"  Trainable params: {n_trainable_b}", flush=True)

    t0 = time.time()
    best_val_ppl_b = run_steps_with_val_selection(
        model_b, trainable_b, N_STEPS, calib_dev, val_eval_ids,
        label=f"comp_adapter seed={seed}"
    )
    elapsed_b = time.time() - t0

    # Final eval on TEST (held-out) — only done once after checkpoint selection
    test_ppl_b = eval_ppl(model_b, test_eval_ids)
    print(f"  [Seed {seed}] Comp+adapter: best_val={best_val_ppl_b:.4f}, "
          f"TEST={test_ppl_b:.4f}, time={elapsed_b:.1f}s", flush=True)

    compressed_adapter_results.append({
        "seed": seed,
        "best_val_ppl": float(best_val_ppl_b),
        "test_ppl": float(test_ppl_b),
    })

    del model_b, trainable_b
    if DEVICE == "mps":
        torch.mps.empty_cache()

# ── Verdict ───────────────────────────────────────────────────────────────────
float_test_ppls = [r["test_ppl"] for r in float_adapter_results]
comp_test_ppls = [r["test_ppl"] for r in compressed_adapter_results]

mean_float = sum(float_test_ppls) / len(float_test_ppls)
mean_comp = sum(comp_test_ppls) / len(comp_test_ppls)

std_float = math.sqrt(sum((p - mean_float) ** 2 for p in float_test_ppls) / len(float_test_ppls))
std_comp = math.sqrt(sum((p - mean_comp) ** 2 for p in comp_test_ppls) / len(comp_test_ppls))

penalty = mean_comp - mean_float  # positive = compressed+adapter is worse

if penalty < CONFIRMED_LT_PENALTY:
    verdict = "CONFIRMED"
elif penalty >= REFUTED_GE_PENALTY:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"Float baseline test PPL:        {float_test_ppl:.4f}", flush=True)
print(f"Float+adapter test PPL:         {mean_float:.4f} ± {std_float:.4f}", flush=True)
print(f"Compressed+adapter test PPL:    {mean_comp:.4f} ± {std_comp:.4f}", flush=True)
print(f"Penalty (comp - float_adapter): {penalty:.4f}", flush=True)
print(f"c378 reference penalty:         {C378_PENALTY_REF} (blocks [2,1,10], h=64/128)", flush=True)
print(f"  Individual float_adapter:  {[f'{p:.2f}' for p in float_test_ppls]}", flush=True)
print(f"  Individual comp_adapter:   {[f'{p:.2f}' for p in comp_test_ppls]}", flush=True)

result = {
    "float_test_ppl": float(float_test_ppl),
    "float_adapter": {
        "mean_test_ppl": float(mean_float),
        "std_test_ppl": float(std_float),
        "seeds": float_adapter_results,
    },
    "compressed_adapter": {
        "mean_test_ppl": float(mean_comp),
        "std_test_ppl": float(std_comp),
        "seeds": compressed_adapter_results,
    },
    "penalty": float(penalty),
    "penalty_c378_ref": C378_PENALTY_REF,
    "blocks": BLOCKS,
    "verdict": verdict,
    "config": {
        "blocks": BLOCKS,
        "k": 512,
        "adapter_h": ADAPTER_H,
        "n_steps": N_STEPS,
        "seeds": SEEDS,
        "confirmed_lt_penalty": CONFIRMED_LT_PENALTY,
        "refuted_ge_penalty": REFUTED_GE_PENALTY,
        "val_used_for": "training and checkpoint selection",
        "test_used_for": "final evaluation only",
    },
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {OUTPUT_PATH}", flush=True)
